//! One source run publishing both reached transpose and reached stencil loops
//! through a single `invar_cuda::Runtime`, hence one shared quarantine.
//!
//! Each family keeps its own contract matching, packet domain, header, fuel
//! and publication checks unchanged; this module only chooses which family's
//! existing decision applies to a reached loop. A family whose contracts do
//! not match a loop records nothing. Unknown or pending work from either family
//! quarantines in the shared runtime and blocks dispatch and source replay for
//! both, in this run and later runs, until the backend confirms quiescence and
//! the runtime is reaped. Tickets are synchronous and fenced; there is no
//! asynchronous ticket or device residency. CPU only.
//!
//! Both families match a reached loop by exact function name and loop span.
//! A key claimed by both families' contracts is ambiguous and is resolved
//! before either family runs, so no family is preferred and nothing is
//! prepared or submitted. While the runtime holds unconfirmed work it stops
//! with the same unconfirmed-work error either family would return;
//! otherwise it records `combined_ambiguous_family` and the loop runs as
//! ordinary source.
use super::{Error, Run};
use crate::{eval::EvalError, front::Span, mir_eval::LiveLoop, transpose_source};
use invar_cuda::{Device, Runtime, SubmitError};
use invar_device::Capabilities;

const AMBIGUOUS: &str = "combined_ambiguous_family";

/// Run `main`, publishing reached stencil and transpose loops.
pub fn run_source<D: Device>(
    source: &[u8],
    fuel: u64,
    runtime: &mut Runtime<D>,
    caps: Capabilities,
) -> Result<Run, Error> {
    run_source_bounded(source, fuel, runtime, caps, super::STAGING_LIMIT)
}

/// As `run_source`, with the stencil per-ticket staging bound explicit.
pub fn run_source_bounded<D: Device>(
    source: &[u8],
    fuel: u64,
    runtime: &mut Runtime<D>,
    caps: Capabilities,
    stencil_staging_limit: usize,
) -> Result<Run, Error> {
    let checked = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, true))
        .map_err(Error::Source)?;
    let program =
        crate::mir::lower(&checked).map_err(|e| Error::Net(crate::general_net::Error::Mir(e)))?;
    let stencils = super::contracts(&checked);
    let transposes = transpose_source::host::contracts(&checked);
    let ambiguous = overlap(
        &stencils
            .iter()
            .map(|c| (c.function(), c.loop_span()))
            .collect::<Vec<_>>(),
        &transposes
            .iter()
            .map(|c| (c.function(), c.loop_span()))
            .collect::<Vec<_>>(),
    );
    let mut refusals = Vec::new();
    let mut provider = |live: &LiveLoop<'_>| {
        route(
            &ambiguous,
            (&live.function, live.span),
            runtime,
            &mut refusals,
            |runtime, refusals| {
                super::provide(
                    source,
                    &stencils,
                    runtime,
                    caps,
                    stencil_staging_limit,
                    refusals,
                    live,
                )
            },
            |runtime, refusals| {
                transpose_source::host::provide(source, &transposes, runtime, caps, refusals, live)
            },
        )
    };
    let (observation, completed) =
        crate::mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
    Ok(Run {
        observation,
        completed,
        refusals,
    })
}

/// Exact function/span keys claimed by both families.
fn overlap(stencil: &[(&str, Span)], transpose: &[(&str, Span)]) -> Vec<(String, Span)> {
    stencil
        .iter()
        .filter(|key| transpose.contains(key))
        .map(|&(function, span)| (function.to_owned(), span))
        .collect()
}

/// Decide one reached loop. An ambiguous key never reaches either family.
/// Otherwise stencil decides first; each family returns `Ok(None)` without
/// recording or dispatching when its own contracts do not match.
fn route<D: Device, P>(
    ambiguous: &[(String, Span)],
    (function, span): (&str, Span),
    runtime: &mut Runtime<D>,
    refusals: &mut Vec<&'static str>,
    stencil: impl FnOnce(&mut Runtime<D>, &mut Vec<&'static str>) -> Result<Option<P>, EvalError>,
    transpose: impl FnOnce(&mut Runtime<D>, &mut Vec<&'static str>) -> Result<Option<P>, EvalError>,
) -> Result<Option<P>, EvalError> {
    if ambiguous.iter().any(|(f, s)| f == function && *s == span) {
        if runtime.blocks_dispatch() {
            return Err(EvalError::Tool {
                code: "CUDA_UNAVAILABLE",
                message: "unconfirmed work blocks an ambiguous stencil/transpose loop".into(),
            });
        }
        refusals.push(AMBIGUOUS);
        return Ok(None);
    }
    if let Some(plan) = stencil(runtime, refusals)? {
        return Ok(Some(plan));
    }
    transpose(runtime, refusals)
}

/// CPU backend routing each family's packets by magic to its own CPU device.
/// Never a GPU claim.
#[derive(Default)]
pub struct CpuDevice {
    stencil: super::CpuDevice,
    transpose: transpose_source::host::CpuDevice,
}
pub enum CpuWork {
    Stencil(super::CpuWork),
    Transpose(transpose_source::host::CpuWork),
}
impl Device for CpuDevice {
    type Work = CpuWork;
    fn prepare(&mut self, bytes: &[u8]) -> Result<CpuWork, invar_cuda::Error> {
        if bytes.starts_with(super::MAGIC) {
            self.stencil.prepare(bytes).map(CpuWork::Stencil)
        } else if bytes.starts_with(b"itr1") {
            self.transpose.prepare(bytes).map(CpuWork::Transpose)
        } else {
            Err(invar_cuda::Error::InvalidHandle)
        }
    }
    fn submit(&mut self, work: &mut CpuWork) -> Result<(), SubmitError> {
        match work {
            CpuWork::Stencil(w) => self.stencil.submit(w),
            CpuWork::Transpose(w) => self.transpose.submit(w),
        }
    }
    fn complete(&mut self, work: &mut CpuWork) -> Result<bool, invar_cuda::Error> {
        match work {
            CpuWork::Stencil(w) => self.stencil.complete(w),
            CpuWork::Transpose(w) => self.transpose.complete(w),
        }
    }
    fn read(&mut self, work: &mut CpuWork) -> Result<Vec<u8>, invar_cuda::Error> {
        match work {
            CpuWork::Stencil(w) => self.stencil.read(w),
            CpuWork::Transpose(w) => self.transpose.read(w),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use invar_cuda::Error as DeviceError;
    use std::{cell::Cell, rc::Rc};

    /// Counts prepares; every submit leaves its work unknown and unconfirmed.
    struct Unknown(Rc<Cell<usize>>);
    impl Device for Unknown {
        type Work = ();
        fn prepare(&mut self, _: &[u8]) -> Result<(), DeviceError> {
            self.0.set(self.0.get() + 1);
            Ok(())
        }
        fn submit(&mut self, _: &mut ()) -> Result<(), SubmitError> {
            Err(SubmitError::InFlightUnknown(DeviceError::Timeout))
        }
        fn complete(&mut self, _: &mut ()) -> Result<bool, DeviceError> {
            Ok(false)
        }
        fn read(&mut self, _: &mut ()) -> Result<Vec<u8>, DeviceError> {
            Err(DeviceError::Timeout)
        }
    }

    fn span(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    fn runtime(blocked: bool) -> (Runtime<Unknown>, Rc<Cell<usize>>) {
        let prepares = Rc::new(Cell::new(0));
        let mut runtime = Runtime::new(Unknown(prepares.clone()), 2);
        if blocked {
            let prepared = runtime.prepare(b"setup").unwrap();
            assert!(prepared.submit_classified().is_err());
            assert!(runtime.blocks_dispatch());
        }
        (runtime, prepares)
    }

    type Decision = Result<Option<u32>, EvalError>;

    fn never(_: &mut Runtime<Unknown>, _: &mut Vec<&'static str>) -> Decision {
        panic!("an ambiguous or already decided loop reached a family")
    }

    #[test]
    fn overlap_is_exact_function_and_span() {
        let stencil = [("f", span(1, 9)), ("g", span(10, 20)), ("k", span(5, 6))];
        let transpose = [("f", span(1, 9)), ("g", span(10, 21)), ("h", span(10, 20))];
        assert_eq!(
            overlap(&stencil, &transpose),
            vec![("f".to_owned(), span(1, 9))]
        );
        assert!(overlap(&stencil, &[]).is_empty());
        assert!(overlap(&[], &transpose).is_empty());
    }

    #[test]
    fn ambiguous_loop_falls_back_before_either_family_or_any_dispatch() {
        let ambiguous = overlap(&[("f", span(1, 9))], &[("f", span(1, 9))]);
        let (mut rt, prepares) = runtime(false);
        let mut refusals = vec!["earlier"];
        let got = route(
            &ambiguous,
            ("f", span(1, 9)),
            &mut rt,
            &mut refusals,
            never,
            never,
        );
        assert_eq!(got, Ok(None));
        assert_eq!(refusals, vec!["earlier", AMBIGUOUS]);
        assert_eq!(prepares.get(), 0);
        assert_eq!(rt.quarantine_len(), 0);
    }

    #[test]
    fn ambiguous_loop_with_unconfirmed_work_stops_without_fallback() {
        let ambiguous = overlap(&[("f", span(1, 9))], &[("f", span(1, 9))]);
        let (mut rt, prepares) = runtime(true);
        let mut refusals = Vec::new();
        let got = route(
            &ambiguous,
            ("f", span(1, 9)),
            &mut rt,
            &mut refusals,
            never,
            never,
        );
        assert!(
            matches!(
                got,
                Err(EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    ..
                })
            ),
            "{got:?}"
        );
        assert!(refusals.is_empty());
        assert_eq!(prepares.get(), 1, "only the setup ticket");
        assert_eq!(rt.quarantine_len(), 1);
    }

    #[test]
    fn unambiguous_loops_reach_each_family_at_most_once_in_order() {
        // An ambiguous key elsewhere does not affect this loop.
        let ambiguous = overlap(&[("f", span(1, 9))], &[("f", span(1, 9))]);
        let key = ("f", span(1, 10));
        let (mut rt, _) = runtime(false);
        let calls = Cell::new((0, 0));
        let stencil = |result: Decision| {
            let calls = &calls;
            move |_: &mut Runtime<Unknown>, _: &mut Vec<&'static str>| {
                calls.set((calls.get().0 + 1, calls.get().1));
                result
            }
        };
        let transpose = |_: &mut Runtime<Unknown>, refusals: &mut Vec<&'static str>| {
            calls.set((calls.get().0, calls.get().1 + 1));
            refusals.push("transpose_tag");
            Ok(Some(2))
        };
        let mut refusals = Vec::new();
        // Stencil publishes: transpose is never consulted.
        let got = route(
            &ambiguous,
            key,
            &mut rt,
            &mut refusals,
            stencil(Ok(Some(1))),
            never,
        );
        assert_eq!((got, calls.get()), (Ok(Some(1)), (1, 0)));
        // Stencil does not match: transpose decides, once.
        let got = route(
            &ambiguous,
            key,
            &mut rt,
            &mut refusals,
            stencil(Ok(None)),
            transpose,
        );
        assert_eq!((got, calls.get()), (Ok(Some(2)), (2, 1)));
        // A stencil error stops before transpose.
        let err = || {
            Err(EvalError::Tool {
                code: "CUDA_UNAVAILABLE",
                message: "x".into(),
            })
        };
        let got = route(
            &ambiguous,
            key,
            &mut rt,
            &mut refusals,
            stencil(err()),
            never,
        );
        assert_eq!((got, calls.get()), (err(), (3, 1)));
        assert_eq!(refusals, vec!["transpose_tag"]);
    }
}
