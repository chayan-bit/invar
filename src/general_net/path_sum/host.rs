//! Host side of one path-sum call: the licence guard, the shape cost model,
//! the stage list every target encodes, publication of the out words, and
//! the exact fallback.

use super::{
    Combine, Enumerator, FLAG_BUDGET, FLAG_DEPTH, FLAG_GEOMETRY, FLAG_GUARD, FLAG_RAW, GROUP,
    Lowering, MAX_LANES, emit::MAX_PARAMS,
};
use crate::{
    eval::{EvalError, Value},
    general_net::{Net, Proof, oracle},
};

/// Kernel shape of one launch (#140 size-dependent fusion).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// One lane in one group: the root is a leaf, a chain or serial.
    Root,
    /// One group loops over every lane and reduces once.
    Fused,
    /// One lane per thread across groups, then one reducing group.
    Parallel,
}

/// The shape cost model: launch latency against waves of path replay.
/// A path costs `depth` node evaluations; a wave is one group of lanes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub fused_launch_us: f64,
    pub parallel_launch_us: f64,
    /// One node evaluation of one wave.
    pub node_us: f64,
    /// Groups the device runs at once.
    pub cores: u32,
    /// `(shape, min lanes, max lanes)` launches that a host call refuses:
    /// the D5 gate measured them losing to the handwritten kernels, so the
    /// native path answers instead (`Plan::Fallback("cost")`).
    pub host_refusal: &'static [(Shape, u32, u32)],
}

impl CostModel {
    /// Apple M4 (10 GPU cores), from the #140 pilot (`policy.tsv`).
    pub const METAL: Self = Self {
        fused_launch_us: 13.75,
        parallel_launch_us: 20.25,
        node_us: 0.37,
        cores: 10,
        // Host-scope 256-lane cells missed by up to 0.32% in the exclusive
        // runs (`tests/fixtures/archived-evidence/p4a-path-sum/metal-gate-*.txt`); 32 lanes passed.
        host_refusal: &[(Shape::Fused, 33, MAX_LANES)],
    };
    /// RTX 5070 Ti (sm_120, 70 SMs), calibrated on a Linux CUDA host.
    pub const CUDA: Self = Self {
        fused_launch_us: 6.0,
        parallel_launch_us: 9.0,
        node_us: 0.25,
        cores: 70,
        // Every host-scope cell missed on an idle GPU (two-speed launch
        // latency, `tests/fixtures/archived-evidence/p4a-path-sum/cuda-gate-idle.txt`).
        host_refusal: &[
            (Shape::Root, 0, MAX_LANES),
            (Shape::Fused, 0, MAX_LANES),
            (Shape::Parallel, 0, MAX_LANES),
        ],
    };

    pub fn shape(&self, lanes: u32) -> Shape {
        if lanes <= 1 {
            return Shape::Root;
        }
        let waves = f64::from(lanes.div_ceil(GROUP));
        let depth = f64::from(u32::BITS - lanes.leading_zeros());
        let fused = self.fused_launch_us + self.node_us * waves * depth;
        let parallel = self.parallel_launch_us
            + self.node_us * (waves / f64::from(self.cores.max(1))).ceil() * depth;
        if fused <= parallel {
            Shape::Fused
        } else {
            Shape::Parallel
        }
    }

    /// Whether a host call routes `launch` to the native path.
    pub fn refuses_host(&self, launch: &Launch) -> bool {
        self.host_refusal
            .iter()
            .any(|&(shape, lo, hi)| shape == launch.shape && (lo..=hi).contains(&launch.lanes))
    }
}

/// Who waits for the answer (the two D5 scopes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// A host call: plan, launch, wait and publish.
    Host,
    /// A device-resident caller that enqueues the launch without a host
    /// round trip; the differential tests run here to exercise every kernel.
    Resident,
}

/// One guarded launch: `Args` words and the stages to encode in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// `struct Args { int root[8]; uint lanes; uint groups; }`.
    pub args: [u32; MAX_PARAMS + 2],
    pub lanes: u32,
    pub groups: u32,
    pub shape: Shape,
}

impl Launch {
    /// `(kernel, groups)` per stage; every group has `GROUP` threads.
    pub fn stages(&self) -> Vec<(&'static str, u32)> {
        match self.shape {
            Shape::Root => vec![("pathsum_root", 1)],
            Shape::Fused => vec![("pathsum_fused", 1)],
            Shape::Parallel => vec![("pathsum_map", self.groups), ("pathsum_reduce", 1)],
        }
    }

    pub fn arg_bytes(&self) -> Vec<u8> {
        self.args.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    Launch(Launch),
    /// The exact path decides; the reason names the failed guard.
    Fallback(&'static str),
}

/// What a fast launch published, or why the exact path must decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Fast(i32),
    Fallback(&'static str),
}

/// A target that runs the stages of a lowering's source.
pub trait Launcher {
    /// Encode `launch.stages()` and return the published out words.
    fn launch(&mut self, lowering: &Lowering, launch: &Launch) -> Result<Vec<u32>, String>;
}

impl Lowering {
    /// The host guard and shape choice for entry arguments `args`.
    pub fn plan(&self, args: &[i64], model: &CostModel) -> Plan {
        let Some(root) = self.entry.root_args(args) else {
            return Plan::Fallback("entry");
        };
        if root.len() != self.licence.len()
            || root
                .iter()
                .zip(&self.licence)
                .any(|(v, g)| *v < g.low || *v > g.high)
        {
            return Plan::Fallback("licence");
        }
        let (lanes, shape) = match self.enumerator {
            Enumerator::Interval { lo, hi } => {
                let lanes = (root[hi] - root[lo]).max(1);
                match u32::try_from(lanes) {
                    Ok(n) if n <= MAX_LANES => (n, model.shape(n)),
                    _ => return Plan::Fallback("lanes"),
                }
            }
            Enumerator::Serial => (1, Shape::Root),
        };
        let mut words = [0u32; MAX_PARAMS + 2];
        for (slot, v) in words.iter_mut().zip(&root) {
            // i32 and u32 arguments travel as their 32-bit patterns.
            *slot = *v as u32;
        }
        let groups = lanes.div_ceil(GROUP);
        words[MAX_PARAMS] = lanes;
        words[MAX_PARAMS + 1] = groups;
        Plan::Launch(Launch {
            args: words,
            lanes,
            groups,
            shape,
        })
    }

    /// `plan`, then the cost model's refusal for `scope`.
    pub fn route(&self, args: &[i64], model: &CostModel, scope: Scope) -> Plan {
        match self.plan(args, model) {
            Plan::Launch(launch) if scope == Scope::Host && model.refuses_host(&launch) => {
                Plan::Fallback("cost")
            }
            plan => plan,
        }
    }

    /// Publish out words `[flags, raw, value..]`, or name the lane flag.
    pub fn finish(&self, out: &[u32]) -> Result<i32, &'static str> {
        let flags = *out.first().ok_or("out")?;
        if flags & FLAG_GEOMETRY != 0 {
            return Err("geometry");
        }
        if flags & FLAG_DEPTH != 0 {
            return Err("depth");
        }
        if flags & FLAG_BUDGET != 0 {
            return Err("budget");
        }
        if flags & FLAG_GUARD != 0 {
            return Err("guard");
        }
        if out.len() < self.out_words() {
            return Err("out");
        }
        if flags & FLAG_RAW != 0 {
            return Ok(out[1] as i32);
        }
        match self.combine {
            Combine::Residue(p) if out[2] < p => Ok(out[2] as i32),
            Combine::Residue(_) => Err("residue"),
            Combine::Sum => {
                let sum = i64::from(out[2]) | (i64::from(out[3]) << 32);
                let magnitude = u64::from(out[4]) | (u64::from(out[5]) << 32);
                // No partial sum can exceed the sum of magnitudes.
                if magnitude > i32::MAX as u64 {
                    return Err("sum");
                }
                i32::try_from(sum).map_err(|_| "sum")
            }
        }
    }

    /// Guard, launch and publish; the caller runs the exact path on fallback.
    pub fn execute(
        &self,
        args: &[i64],
        model: &CostModel,
        scope: Scope,
        launcher: &mut dyn Launcher,
    ) -> Result<Outcome, String> {
        let launch = match self.route(args, model, scope) {
            Plan::Launch(launch) => launch,
            Plan::Fallback(reason) => return Ok(Outcome::Fallback(reason)),
        };
        let out = launcher.launch(self, &launch)?;
        Ok(match self.finish(&out) {
            Ok(v) => Outcome::Fast(v),
            Err(reason) => Outcome::Fallback(reason),
        })
    }

    /// The observable result of a host call: the fast path, or the exact net
    /// oracle, which owns every source-order trap, tool limit and refusal.
    pub fn run(
        &self,
        net: &Net,
        proof: &Proof,
        args: &[i64],
        model: &CostModel,
        launcher: &mut dyn Launcher,
        limits: oracle::Limits,
    ) -> Result<(Result<Value, EvalError>, Outcome), String> {
        let outcome = self.execute(args, model, Scope::Host, launcher)?;
        if let Outcome::Fast(v) = outcome {
            return Ok((Ok(Value::I32(v)), outcome));
        }
        let values = self.entry.values(args).ok_or("entry argument types")?;
        let observed = oracle::observe_call(net, proof, &self.entry.name, &values, limits)
            .map_err(|e| format!("oracle: {e:?}"))?;
        Ok((observed.semantic.result, outcome))
    }
}
