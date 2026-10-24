//! Spatial signatures, built-in result records, and sorted-grid execution.

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod metal;
pub mod placement;
mod runtime;
pub use runtime::Grid;
pub(crate) use runtime::invoke;

use crate::front::Type;

pub(crate) const GRID: &str = "VoxelGrid";
pub(crate) const NEIGHBORS: &str = "Neighbors";
pub(crate) const KNN: &str = "Knn";

pub(crate) fn result(name: &str) -> Option<Type> {
    Some(match name {
        "voxel_grid_build" => Type::Named(GRID.to_owned()),
        "radius_neighbors" => Type::Named(NEIGHBORS.to_owned()),
        "knn" => Type::Named(KNN.to_owned()),
        "grid_excluded" => Type::U32,
        _ => return None,
    })
}

pub(crate) fn records() -> [(String, Vec<(String, Type)>); 2] {
    [
        (
            NEIGHBORS.to_owned(),
            vec![
                ("counts".to_owned(), Type::ArrayU32),
                ("idx".to_owned(), Type::ArrayU32),
            ],
        ),
        (
            KNN.to_owned(),
            vec![
                ("idx".to_owned(), Type::ArrayU32),
                ("d2".to_owned(), Type::ArrayF32),
            ],
        ),
    ]
}

pub(crate) fn signature(name: &str) -> Option<(Vec<Type>, Type)> {
    let result = result(name)?;
    if name == "grid_excluded" {
        return Some((vec![Type::Named(GRID.to_owned())], result));
    }
    let mut params = vec![Type::ArrayF32; 3];
    if name == "voxel_grid_build" {
        params.push(Type::F32);
    } else {
        params.insert(0, Type::Named(GRID.to_owned()));
        if name == "radius_neighbors" {
            params.push(Type::F32);
        }
        params.push(Type::U32);
    }
    Some((params, result))
}

pub(crate) fn read_count(name: &str) -> usize {
    match name {
        "voxel_grid_build" => 3,
        "radius_neighbors" | "knn" => 4,
        "grid_excluded" => 1,
        _ => 0,
    }
}

pub(crate) fn mode(name: &str, position: usize) -> Option<crate::front::Mode> {
    (position < read_count(name)).then_some(crate::front::Mode::Read)
}

pub(crate) fn is_grid(ty: &Type) -> bool {
    matches!(ty, Type::Named(name) if name == GRID)
}

pub(crate) fn is_builtin(name: &str) -> bool {
    matches!(name, GRID | NEIGHBORS | KNN)
}

pub(crate) fn contains(program: &crate::mir::Program) -> bool {
    program.functions.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.instructions)
        .any(|i| matches!(&i.kind, crate::mir::Operation::Call { name, .. } if result(name).is_some()))
}

pub(crate) fn run_cpu(
    source: &[u8],
    program: &crate::mir::Program,
    fuel: u64,
    workers: usize,
    hooks: &crate::schedule::Hooks,
) -> Result<crate::schedule::Run, crate::schedule::Error> {
    let (mut reductions, mut fault) = (Vec::new(), None);
    let mut reducer = |call: &crate::mir_eval::ReduceCall<'_>| {
        crate::sorting::cpu_reduce(call, workers, &mut reductions, &mut fault)
    };
    let run = run_cpu_with(source, program, fuel, workers, hooks, &mut reducer)?;
    match fault {
        Some(error) => Err(error),
        None => Ok(crate::schedule::Run { reductions, ..run }),
    }
}

/// [`run_cpu`] with the caller's float reducer (auto passes its cost-model
/// choice); the caller keeps the reduction receipts.
pub(crate) fn run_cpu_with(
    source: &[u8],
    program: &crate::mir::Program,
    fuel: u64,
    workers: usize,
    hooks: &crate::schedule::Hooks,
    reducer: &mut crate::mir_eval::Reducer<'_>,
) -> Result<crate::schedule::Run, crate::schedule::Error> {
    use crate::{
        plan::PlanKind,
        schedule::{Error, Run},
    };
    let checked = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, true))
        .map_err(|_| Error::Candidate)?;
    let lowered = crate::mir::lower(&checked).map_err(|_| Error::Candidate)?;
    if lowered.contract_digest() != program.contract_digest() {
        return Err(Error::Candidate);
    }
    let mut cpu = CpuIntrinsics::new(program, workers, hooks);
    let mut provider =
        |function: &str, name: &str, args: &[crate::eval::Value], span, live: &mut usize| {
            cpu.invoke(function, name, args, span, live)
        };
    let observation =
        crate::mir_eval::observe_all(program, fuel, None, Some(&mut provider), Some(reducer)).0;
    let CpuIntrinsics {
        joined, executed, ..
    } = cpu;
    let plan = if executed.iter().any(|step| step.kind == PlanKind::Spatial) {
        PlanKind::Spatial
    } else if executed.is_empty() {
        PlanKind::Serial
    } else {
        PlanKind::Sort
    };
    Ok(Run {
        observation,
        plan,
        workers_used: workers.min(joined.get()),
        chunks: Vec::new(),
        joined: joined.get(),
        fallback_reason: None,
        executed,
        reductions: Vec::new(),
    })
}

/// The CPU engines of the explicit intrinsics as one MIR provider, as the
/// established CPU run of the program uses them: spatial queries on `workers`,
/// sorts serially in a spatial program and on the counted sort engine
/// otherwise. Each reached site is recorded with the implementation that ran
/// it (placement records them too, #50).
pub(crate) struct CpuIntrinsics<'a> {
    workers: usize,
    spatial: bool,
    hooks: &'a crate::schedule::Hooks,
    pub joined: std::cell::Cell<usize>,
    pub executed: Vec<crate::schedule::Executed>,
}

impl<'a> CpuIntrinsics<'a> {
    pub(crate) fn new(
        program: &crate::mir::Program,
        workers: usize,
        hooks: &'a crate::schedule::Hooks,
    ) -> Self {
        Self {
            workers,
            spatial: contains(program),
            hooks,
            joined: std::cell::Cell::new(0),
            executed: Vec::new(),
        }
    }

    pub(crate) fn invoke(
        &mut self,
        function: &str,
        name: &str,
        args: &[crate::eval::Value],
        span: crate::front::Span,
        live: &mut usize,
    ) -> Result<crate::eval::Value, crate::eval::EvalError> {
        use crate::plan::PlanKind;
        let (before, joined, workers) = (self.joined.get(), &self.joined, self.workers);
        let spatial = result(name).is_some();
        let (value, implementation) = if spatial {
            let value =
                runtime::invoke_counted(name, args, span, live, workers, joined, self.hooks)?;
            let parallel = joined.get() != before;
            (
                value,
                if parallel {
                    "cpu-sorted-grid-parallel-queries"
                } else {
                    "cpu-sorted-grid-serial"
                },
            )
        } else if self.spatial {
            (
                crate::sorting::invoke(name, args, span, live, 1)?,
                "cpu-lsd-radix-4x8",
            )
        } else {
            let value = crate::sorting::invoke_counted(name, args, span, live, workers, joined)?;
            let parallel = joined.get() != before;
            (
                value,
                if parallel {
                    "cpu-parallel-lsd-radix-4x8"
                } else {
                    "cpu-lsd-radix-4x8"
                },
            )
        };
        self.executed.push(crate::schedule::Executed {
            function: function.into(),
            span,
            kind: if spatial {
                PlanKind::Spatial
            } else {
                PlanKind::Sort
            },
            published_array: None,
            implementation: Some(implementation),
        });
        Ok(value)
    }
}
