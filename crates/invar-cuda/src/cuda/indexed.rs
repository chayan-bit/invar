//! Guarded irregular CUDA work with pinned ownership and stream-ordered
//! frontier phases. Submission failures quarantine the entire ABI allocation set.
use super::*;
use cudarc::driver::CudaModule;
use invar_device::indexed::{self as ir, Bound};

struct Work {
    host: Vec<PinnedHostSlice<u32>>,
    device: Vec<CudaSlice<u32>>,
    functions: Vec<CudaFunction>,
    _module: Arc<CudaModule>,
    fence: CudaEvent,
    stream: Arc<CudaStream>,
}
impl CudaDevice {
    pub fn run_indexed(&mut self, bound: &Bound, compiler: &Path) -> Result<ir::Receipt, Error> {
        self.run_indexed_with_fault(bound, compiler, CollectiveFault::None)
    }
    #[doc(hidden)]
    pub fn run_indexed_with_fault(
        &mut self,
        bound: &Bound,
        compiler: &Path,
        fault: CollectiveFault,
    ) -> Result<ir::Receipt, Error> {
        self.admit()?;
        if bound.plan().capabilities() != self.caps {
            return Err(Error::Limit);
        }
        if fault == CollectiveFault::OutOfMemory {
            return Err(Error::OutOfMemory);
        }
        let (blocks, lanes) = ir::launch_dimensions(bound.plan());
        for (requested, attribute) in [
            (
                blocks,
                driver_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
            ),
            (
                lanes,
                driver_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
            ),
            (
                lanes,
                driver_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            ),
        ] {
            if i64::from(requested)
                > i64::from(self.context.attribute(attribute).map_err(driver_error)?)
            {
                return Err(Error::ResourceLimit(
                    "indexed launch exceeds CUDA device dimensions",
                ));
            }
        }
        let bytes = bound
            .buffer_bytes()
            .iter()
            .try_fold(0usize, |a, b| a.checked_add(*b))
            .ok_or(Error::ResourceLimit("indexed allocation byte sum overflow"))?;
        if bytes > self.context.mem_get_info().map_err(driver_error)?.0 {
            return Err(Error::ResourceLimit(
                "indexed buffers exceed free CUDA device memory",
            ));
        }
        let source = ir::emit_cuda(bound.plan())
            .map_err(|e| Error::Compile(format!("indexed IR: {e:?}")))?;
        let compiled = compile_cubin_bounded(&source, compiler, self.caps)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let functions = ir::kernel_names(bound.plan())
            .iter()
            .map(|name| module.load_function(name).map_err(driver_error))
            .collect::<Result<Vec<_>, _>>()?;
        let initial = ir::initial_buffers(bound).map_err(|_| Error::OutOfMemory)?;
        let mut host = Vec::new();
        let mut device = Vec::new();
        for words in initial {
            let len = words.len().max(1);
            // SAFETY: freshly owned pinned memory; all slots initialized before
            // async copies or safe slices can observe them.
            let mut pinned =
                unsafe { self.context.alloc_pinned::<u32>(len) }.map_err(driver_error)?;
            let ptr = pinned.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: ptr owns len disjoint writable slots, words.len()<=len.
            unsafe {
                std::ptr::write_bytes(ptr, 0, len);
                std::ptr::copy_nonoverlapping(words.as_ptr(), ptr, words.len());
            }
            host.push(pinned);
            // SAFETY: the ordered H2D below initializes all slots before use.
            device.push(unsafe { self.stream.alloc::<u32>(len) }.map_err(driver_error)?);
        }
        let fence = self.context.new_event(None).map_err(driver_error)?;
        let mut work = Box::new(Work {
            host,
            device,
            functions,
            _module: module,
            fence,
            stream: self.stream.clone(),
        });
        if fault == CollectiveFault::LaunchFailure {
            return Err(Error::InvalidHandle);
        }
        let submitted = (|| -> Result<u32, Error> {
            for (h, d) in work.host.iter().zip(&mut work.device) {
                work.stream.memcpy_htod(h, d).map_err(driver_error)?;
            }
            let mut phases = 0;
            for (round, kernel) in ir::schedule(bound.plan()) {
                let mut launch = work.stream.launch_builder(&work.functions[kernel]);
                for buffer in &mut work.device {
                    launch.arg(buffer);
                }
                launch.arg(&round);
                // SAFETY: Bound validated every indirect read/write and owns
                // immutable copied inputs; the verified emitter's eight-slot
                // ABI matches these retained allocations and scalar. Successive
                // kernels in this stream establish every phase boundary without
                // a host wait; only terminal completion permits publication.
                let (phase_blocks, phase_lanes) = ir::phase_launch_dimensions(bound.plan(), kernel);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (phase_blocks, 1, 1),
                        block_dim: (phase_lanes, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(driver_error)?;
                phases += 1;
            }
            for slot in [2, 6] {
                work.stream
                    .memcpy_dtoh(&work.device[slot], &mut work.host[slot])
                    .map_err(driver_error)?;
            }
            if fault == CollectiveFault::PartialCompletion {
                return Err(Error::Timeout);
            }
            work.fence.record(&work.stream).map_err(driver_error)?;
            let begun = Instant::now();
            while !work.fence.is_complete() {
                if begun.elapsed() >= KERNEL_DEADLINE {
                    return Err(Error::Timeout);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(phases)
        })();
        let phases = match submitted {
            Ok(n) => n,
            Err(e) => {
                self.hold(work);
                return Err(e);
            }
        };
        let words = ir::copy(
            &work.host[2].as_slice().map_err(driver_error)?
                [..bound.plan().ir().output.elements as usize],
        )
        .map_err(|_| Error::OutOfMemory)?;
        let status = work.host[6].as_slice().map_err(driver_error)?;
        let output = ir::decode(bound.plan(), words, status)
            .map_err(|e| Error::Driver(format!("indexed output refusal: {e:?}")))?;
        Ok(ir::Receipt {
            target: "cuda",
            reason: "verified-u32-completed",
            output,
            source_hash: source_hash(&source),
            transfer_bytes: bound.transfer_bytes(),
            transfer_mode: "pinned-h2d-d2h",
            completed_phases: phases,
        })
    }
}
