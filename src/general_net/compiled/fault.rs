//! Test-only device-side fault injection for the ABI5 child protocol.
//!
//! With the `gnet-fault-injection` feature, an otherwise idle thread of the
//! single group (thread 31; children use threads 1..=8) rewrites shared slot
//! words between group barriers. Every hook call is bracketed by two group
//! barriers, so no root or child thread touches a slot while the hook runs:
//! the injected state is adversarial but the accesses are data-race free under
//! the CUDA, Metal and C++ memory models (a racing plain write is undefined
//! behaviour, not a test of the protocol). Without the feature `Fault` is
//! uninhabited: production code cannot construct one, and the hook text below
//! is not compiled into the binary at all.
use super::{Error, Program, Target, emit::Writer};

/// Name of the non-inlined device hook. The GNET-03 suites check that it is
/// absent from production source, CUBIN and the `invar` binary.
#[cfg(feature = "gnet-fault-injection")]
pub const HOOK: &str = "gnet_fault_inject";
/// The group thread that injects; never a root or child lane.
#[cfg(feature = "gnet-fault-injection")]
const THREAD: u32 = 31;

#[cfg(feature = "gnet-fault-injection")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// After the completion barrier and before the root's next commit check,
    /// stamp every done slot with the previous generation's completion.
    StaleCompletion,
    /// After a child quantum, take every slot whose child has fully published
    /// (done, completion equal to generation), stamp it with the previous
    /// generation and corrupt its value word, so only a commit that ignores
    /// the generation could publish the corrupt value.
    PoisonedStale,
    /// After a child quantum, claim every free used lane with an identity no
    /// call site can match, releasing it one child phase later; the root's
    /// discovery meanwhile finds the lanes contested.
    ContestedClaim,
    /// After a child quantum, cancel lane 0 while its child is unfinished.
    CancelRace,
}
#[cfg(not(feature = "gnet-fault-injection"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {}

/// Group phases at which the hook runs, alone between two group barriers.
#[derive(Clone, Copy)]
pub(super) enum Phase {
    Children = 1,
    Publication = 2,
}

/// The hook call for one phase of the group body; empty without a fault.
/// Emitted at loop level, so every group thread reaches both barriers.
#[cfg(not(feature = "gnet-fault-injection"))]
pub(super) fn call(p: &Program<'_>, _: Phase) -> String {
    match p.fault {
        None => String::new(),
        Some(fault) => match fault {},
    }
}

#[cfg(feature = "gnet-fault-injection")]
pub(super) fn call(p: &Program<'_>, phase: Phase) -> String {
    if p.fault.is_none() {
        return String::new();
    }
    format!(
        "\n        GNET_GROUP_BARRIER; if (index=={THREAD}) {HOOK}(slots,config[5],{}); GNET_GROUP_BARRIER;",
        phase as u32
    )
}

#[cfg(not(feature = "gnet-fault-injection"))]
pub(super) fn hook(_: &mut Writer, _: Target, fault: Option<Fault>) -> Result<(), Error> {
    match fault {
        None => Ok(()),
        Some(fault) => match fault {},
    }
}

#[cfg(feature = "gnet-fault-injection")]
pub(super) fn hook(w: &mut Writer, target: Target, fault: Option<Fault>) -> Result<(), Error> {
    let Some(fault) = fault else {
        return Ok(());
    };
    let attribute = match target {
        Target::Cuda => "__device__ __attribute__((noinline))",
        Target::Metal => "__attribute__((noinline))",
        Target::Cpu => return Err(Error::SpeculationTarget),
    };
    // Every write targets identity/generation/validity words only; a commit
    // still needs an exact identity match, equal generations and a done task.
    let body = match fault {
        Fault::StaleCompletion => {
            "if (phase!=2) return; for (gnet_u lane=0;lane<capacity;++lane) { GNET_MEM GnetSpecSlot& s=slots[lane]; if (s.valid && s.task.done && s.generation) s.completion_generation=s.generation-1; }"
        }
        Fault::PoisonedStale => {
            "if (phase!=1) return; for (gnet_u lane=0;lane<capacity;++lane) { GNET_MEM GnetSpecSlot& s=slots[lane]; if (s.valid && s.task.done && s.generation && s.completion_generation==s.generation) { s.completion_generation=s.generation-1; s.output[1]^=(gnet_w)0x5a5a5a5a; } }"
        }
        Fault::ContestedClaim => {
            "if (phase!=1) return; gnet_u forged=~(gnet_u)0;
    for (gnet_u lane=0;lane<capacity;++lane) { GNET_MEM GnetSpecSlot& s=slots[lane];
        if (s.valid && s.parent_frame==forged && s.call_pc==forged) s.valid=0;
        else if (!s.valid && s.task.started) { s.parent_frame=forged; s.parent_activation=forged; s.call_pc=forged; s.valid=1; } }"
        }
        Fault::CancelRace => {
            "if (phase!=1 || capacity==0) return; GNET_MEM GnetSpecSlot& s=slots[0]; if (s.valid && !s.task.done) s.valid=0;"
        }
    };
    w.push(&format!(
        "{attribute} void {HOOK}(GNET_MEM GnetSpecSlot* slots,gnet_u capacity,gnet_u phase) {{\n    {body}\n}}"
    ))
}
