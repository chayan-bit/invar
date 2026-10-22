use invar_cuda::{Device, Error, Runtime, SubmitError};
use std::{cell::Cell, rc::Rc};

#[derive(Clone)]
struct Mock(Rc<Cell<usize>>);

struct Work(Rc<Cell<usize>>);

impl Drop for Work {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

impl Device for Mock {
    type Work = Work;

    fn prepare(&mut self, _: &[u8]) -> Result<Self::Work, Error> {
        Ok(Work(self.0.clone()))
    }

    fn submit(&mut self, _: &mut Self::Work) -> Result<(), SubmitError> {
        Ok(())
    }

    fn complete(&mut self, _: &mut Self::Work) -> Result<bool, Error> {
        Ok(false)
    }

    fn read(&mut self, _: &mut Self::Work) -> Result<Vec<u8>, Error> {
        Ok(vec![1, 2, 3])
    }
}

#[test]
fn issue_m3_01_cuda_runtime_timeout_drop_quarantines_until_completion() {
    let freed = Rc::new(Cell::new(0));
    let mut runtime = Runtime::new(Mock(freed.clone()), 1);
    let submitted = runtime.prepare(&[1, 2, 3]).unwrap().submit().unwrap();
    drop(submitted);
    assert_eq!(freed.get(), 0);
    assert_eq!(runtime.quarantine_len(), 1);
    assert!(runtime.prepare(&[4]).is_err());
    assert_eq!(freed.get(), 0);
}

#[test]
fn issue_m3_01_cuda_runtime_completed_output_releases_once() {
    struct Ready(Rc<Cell<usize>>);
    impl Device for Ready {
        type Work = Work;
        fn prepare(&mut self, _: &[u8]) -> Result<Work, Error> {
            Ok(Work(self.0.clone()))
        }
        fn submit(&mut self, _: &mut Work) -> Result<(), SubmitError> {
            Ok(())
        }
        fn complete(&mut self, _: &mut Work) -> Result<bool, Error> {
            Ok(true)
        }
        fn read(&mut self, _: &mut Work) -> Result<Vec<u8>, Error> {
            Ok(vec![9])
        }
    }
    let freed = Rc::new(Cell::new(0));
    let mut runtime = Runtime::new(Ready(freed.clone()), 1);
    let mut submitted = runtime.prepare(&[9]).unwrap().submit().unwrap();
    assert_eq!(submitted.poll().unwrap(), Some(vec![9]));
    assert_eq!(submitted.poll(), Err(Error::Poisoned));
    drop(submitted);
    assert_eq!(freed.get(), 1);
    assert_eq!(runtime.quarantine_len(), 0);
}

#[test]
fn issue_m3_01_cuda_runtime_unknown_submit_keeps_work() {
    struct Unknown(Rc<Cell<usize>>);
    impl Device for Unknown {
        type Work = Work;
        fn prepare(&mut self, _: &[u8]) -> Result<Work, Error> {
            Ok(Work(self.0.clone()))
        }
        fn submit(&mut self, _: &mut Work) -> Result<(), SubmitError> {
            Err(SubmitError::InFlightUnknown(Error::Driver("launch".into())))
        }
        fn complete(&mut self, _: &mut Work) -> Result<bool, Error> {
            Ok(false)
        }
        fn read(&mut self, _: &mut Work) -> Result<Vec<u8>, Error> {
            panic!("unknown completion must not publish")
        }
    }
    let freed = Rc::new(Cell::new(0));
    let mut runtime = Runtime::new(Unknown(freed.clone()), 1);
    assert_eq!(
        runtime.prepare(&[1]).unwrap().submit().err(),
        Some(Error::Driver("launch".into()))
    );
    assert_eq!(runtime.quarantine_len(), 1);
    runtime.reap();
    assert_eq!(freed.get(), 0);
}

#[test]
fn issue_m3_01_cuda_runtime_reap_releases_exactly_once() {
    struct Later {
        freed: Rc<Cell<usize>>,
        done: Rc<Cell<bool>>,
    }
    impl Device for Later {
        type Work = Work;
        fn prepare(&mut self, _: &[u8]) -> Result<Work, Error> {
            Ok(Work(self.freed.clone()))
        }
        fn submit(&mut self, _: &mut Work) -> Result<(), SubmitError> {
            Ok(())
        }
        fn complete(&mut self, _: &mut Work) -> Result<bool, Error> {
            Ok(self.done.get())
        }
        fn read(&mut self, _: &mut Work) -> Result<Vec<u8>, Error> {
            panic!("quarantined output must not publish")
        }
    }
    let freed = Rc::new(Cell::new(0));
    let done = Rc::new(Cell::new(false));
    let mut runtime = Runtime::new(
        Later {
            freed: freed.clone(),
            done: done.clone(),
        },
        1,
    );
    drop(runtime.prepare(&[7]).unwrap().submit().unwrap());
    runtime.reap();
    assert_eq!(freed.get(), 0);
    done.set(true);
    runtime.reap();
    runtime.reap();
    assert_eq!(freed.get(), 1);
    assert_eq!(runtime.quarantine_len(), 0);
    drop(runtime.prepare(&[8]).unwrap());
    assert_eq!(freed.get(), 2);
}

#[test]
fn issue_m3_01_cuda_runtime_rejects_limits_and_typed_prepare_errors() {
    struct Reject(Error);
    impl Device for Reject {
        type Work = ();
        fn prepare(&mut self, _: &[u8]) -> Result<(), Error> {
            Err(self.0.clone())
        }
        fn submit(&mut self, _: &mut ()) -> Result<(), SubmitError> {
            unreachable!()
        }
        fn complete(&mut self, _: &mut ()) -> Result<bool, Error> {
            unreachable!()
        }
        fn read(&mut self, _: &mut ()) -> Result<Vec<u8>, Error> {
            unreachable!()
        }
    }
    for expected in [
        Error::Unavailable,
        Error::InvalidHandle,
        Error::OutOfMemory,
        Error::StaleBorrow,
        Error::Driver("driver failure".into()),
    ] {
        let mut runtime = Runtime::new(Reject(expected.clone()), 1);
        assert_eq!(runtime.prepare(&[1]).err(), Some(expected));
    }
    let mut runtime = Runtime::new(Reject(Error::Unavailable), 0);
    assert_eq!(runtime.prepare(&[1]).err(), Some(Error::Poisoned));
    assert_eq!(
        invar_cuda::validate_transfer_len(invar_cuda::MAX_TRANSFER_BYTES + 1),
        Err(Error::ResourceLimit(
            "CUDA transfer or owned staging exceeds 8 GiB"
        ))
    );
}

#[test]
fn issue_m3_01_cuda_runtime_timeout_and_cancel_retain_loans() {
    let freed = Rc::new(Cell::new(0));
    let mut runtime = Runtime::new(Mock(freed.clone()), 2);
    let submitted = runtime.prepare(&[1]).unwrap().submit().unwrap();
    assert_eq!(submitted.timeout(), Error::Timeout);
    assert_eq!(runtime.quarantine_len(), 1);
    assert_eq!(freed.get(), 0);
    // Cancellation on a separate runtime has the same ownership result.
    let mut other = Runtime::new(Mock(freed.clone()), 2);
    let submitted = other.prepare(&[2]).unwrap().submit().unwrap();
    assert_eq!(submitted.cancel(), Error::Canceled);
    assert_eq!(other.quarantine_len(), 1);
    assert_eq!(freed.get(), 0);
}

#[test]
fn issue_m3_01_cuda_runtime_quiescent_launch_failure_can_release() {
    struct Failed(Rc<Cell<usize>>);
    impl Device for Failed {
        type Work = Work;
        fn prepare(&mut self, _: &[u8]) -> Result<Work, Error> {
            Ok(Work(self.0.clone()))
        }
        fn submit(&mut self, _: &mut Work) -> Result<(), SubmitError> {
            Err(SubmitError::FailedQuiescent(Error::InvalidHandle))
        }
        fn complete(&mut self, _: &mut Work) -> Result<bool, Error> {
            unreachable!()
        }
        fn read(&mut self, _: &mut Work) -> Result<Vec<u8>, Error> {
            unreachable!()
        }
    }
    let freed = Rc::new(Cell::new(0));
    let mut runtime = Runtime::new(Failed(freed.clone()), 1);
    assert_eq!(
        runtime.prepare(&[1]).unwrap().submit().err(),
        Some(Error::InvalidHandle)
    );
    assert_eq!(freed.get(), 1);
    assert_eq!(runtime.quarantine_len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m3_01_gpu_round_trip() {
    use invar_cuda::cuda::CudaDevice;
    use std::time::{Duration, Instant};

    let device = CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    eprintln!("device={} caps={:?}", device.name(), device.capabilities());
    let mut runtime = Runtime::new(device, 8);
    let bytes = (0u16..4096).map(|v| v as u8).collect::<Vec<_>>();
    let mut submitted = runtime.prepare(&bytes).unwrap().submit().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(actual) = submitted.poll().unwrap() {
            assert_eq!(actual, bytes);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "CUDA_PENDING: transfer did not complete"
        );
        std::thread::yield_now();
    }
}

#[test]
fn issue_m3_01_cuda_runtime_quarantine_bounds_reaps_and_poisons_on_overflow() {
    use invar_cuda::Quarantine;
    let freed = Rc::new(Cell::new(0));
    let done = Rc::new(Cell::new(false));
    let mut quarantine = Quarantine::new(2);
    assert!(!quarantine.blocks_dispatch());
    quarantine.push(Work(freed.clone()));
    assert!(quarantine.blocks_dispatch());
    // Unconfirmed work is never released by a reap.
    quarantine.reap(|_| done.get());
    assert_eq!((quarantine.len(), freed.get()), (1, 0));
    // Confirmed completion releases each entry exactly once and re-enables dispatch.
    done.set(true);
    quarantine.reap(|_| done.get());
    quarantine.reap(|_| done.get());
    assert_eq!((quarantine.len(), freed.get()), (0, 1));
    assert!(!quarantine.blocks_dispatch());
    // Past the bound, work is leaked (never freed) and dispatch stays disabled.
    done.set(false);
    for _ in 0..3 {
        quarantine.push(Work(freed.clone()));
    }
    assert_eq!((quarantine.len(), freed.get()), (2, 1));
    done.set(true);
    quarantine.reap(|_| done.get());
    assert_eq!((quarantine.len(), freed.get()), (0, 3));
    assert!(quarantine.blocks_dispatch());
    // Dropping a quarantine never runs destructors for unconfirmed work.
    let mut dropped = Quarantine::new(1);
    dropped.push(Work(freed.clone()));
    drop(dropped);
    assert_eq!(freed.get(), 3);
}
