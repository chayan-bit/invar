//! P6(a): `invar run` through the general-net parallel normal form (#54).
//!
//! A program that no family, f32, sort, spatial or reduction path claims would
//! otherwise run as serial MIR. [`route`] derives its normal form from the
//! run's own MIR, re-checks every certificate with the independent checker
//! (`certificate::check_all`) and executes `main` through the law executor the
//! P1b differential tests use (`normal_form::exec::run`). Its result is
//! published only when a law was applied, no law state was violated, and the
//! run provably stays inside the source limits the reference evaluator
//! enforces (fuel, call depth and live array elements); anything else is a [`Route::Fallback`] and
//! the caller runs the unchanged serial path. There is no opt-out (#54).

use crate::{
    eval::{EvalError, Value},
    general_net::{
        Limits, Net,
        normal_form::{
            self, LawId, Rep, Ring,
            certificate::{self},
            exec::{self, Mutation},
        },
    },
    mir,
};

/// Calls the reference evaluator allows, with margin; the law executor's own
/// limit is larger, so a deeper run is not published.
const MAX_DEPTH: usize = 32;

/// A run whose result came from the law executor.
#[derive(Debug)]
pub struct Routed {
    pub result: Result<i32, EvalError>,
    /// Laws the executor applied, in law-table order.
    pub laws: Vec<LawId>,
    /// Certificates the independent checker accepted.
    pub certificates: usize,
    /// Compiled-MIR contract digest the certificates are bound to.
    pub fingerprint: [u8; 32],
    /// SHA-256 of the checked certificate set.
    pub certificate_digest: String,
    pub lanes: u64,
}

#[derive(Debug)]
pub enum Route {
    Routed(Routed),
    /// The serial path runs; the reason is recorded in the receipt.
    Fallback(&'static str),
}

impl Routed {
    /// The receipt's `normal_form` object for a routed run.
    pub fn document(&self) -> serde_json::Value {
        serde_json::json!({
            "status": "routed",
            "laws": self.laws.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "certificates": self.certificates,
            "fingerprint": crate::hex::hex(&self.fingerprint),
            "certificate_digest": self.certificate_digest,
            "lanes": self.lanes,
        })
    }
}

impl Route {
    /// The receipt's optional `normal_form` object. A program without a
    /// certified law gets none, so its documents stay byte-identical.
    pub fn document(&self) -> Option<serde_json::Value> {
        match self {
            Self::Routed(routed) => Some(routed.document()),
            Self::Fallback(reason) if UNCERTIFIED.contains(reason) => None,
            Self::Fallback(reason) => {
                Some(serde_json::json!({"status": "fallback", "reason": reason}))
            }
        }
    }
}

/// Fallbacks before any certificate exists: not recorded in receipts.
const UNCERTIFIED: [&str; 3] = ["net-unavailable", "regions-unavailable", "no-certified-law"];

/// Run `main` of `program` through its checked normal form under `fuel`.
pub fn route(program: &mir::Program, fuel: u64) -> Route {
    let Ok(net) = Net::encode(program, Limits::default()) else {
        return Route::Fallback("net-unavailable");
    };
    let Ok(proof) = net.verify() else {
        return Route::Fallback("net-unavailable");
    };
    let Ok(form) = normal_form::extract(&net, &proof) else {
        return Route::Fallback("regions-unavailable");
    };
    if form.certificates().next().is_none() {
        return Route::Fallback("no-certified-law");
    }
    // L9/L10 path mode replaces self calls by holes, so the call depth the
    // reference evaluator limits is not observed; those programs stay serial.
    if form.functions.iter().any(|f| f.recursion.is_some()) {
        return Route::Fallback("recursion-law-not-routed");
    }
    let Ok(certificates) = certificate::check_all(&net, &proof, &form) else {
        return Route::Fallback("certificate-rejected");
    };
    // Lemma B regroups checked `(a s + b) % P` updates in `Z/P`, which does
    // not see an overflow trap of `a s + b` itself (#54 differential run).
    if form.certificates().any(|c| {
        matches!(
            c.rep,
            Some(
                Rep::Affine(Ring::Residue { .. })
                    | Rep::Matrix {
                        ring: Ring::Residue { .. },
                        ..
                    }
            )
        )
    }) {
        return Route::Fallback("residue-law-not-routed");
    }
    // Every source statement the reference executes is executed (and ticked)
    // at least once by the law executor, and one statement or loop iteration
    // spends at most its function's tick count of source fuel. So a run of at
    // most `fuel / ticks` executor ticks stays inside the reference's fuel.
    let ticks = program
        .functions
        .iter()
        .map(|function| {
            function
                .blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .filter(|instruction| matches!(instruction.kind, mir::Operation::Tick))
                .count() as u64
        })
        .max()
        .unwrap_or(0)
        .max(1);
    let budget = fuel / ticks;
    let Ok(outcome) = exec::run_bounded(&net, &proof, &form, "main", &[], Mutation::None, budget)
    else {
        return Route::Fallback("executor-refused");
    };
    let receipt = outcome.receipt;
    if receipt.steps > budget {
        return Route::Fallback("fuel-unproven");
    }
    // Every array the reference allocates is allocated (and counted) by the
    // executor, so a total inside the ceiling bounds the reference's live set.
    if receipt.allocated > crate::array_limits::MAX_LIVE_ELEMENTS as u64 {
        return Route::Fallback("live-elements-unproven");
    }
    if receipt.applied.is_empty() {
        return Route::Fallback("no-law-applied");
    }
    if !receipt.violations.is_empty() {
        return Route::Fallback("law-state-violation");
    }
    if receipt.max_depth > MAX_DEPTH {
        return Route::Fallback("call-depth-unproven");
    }
    let result = match outcome.result {
        Ok(Value::I32(value)) => Ok(value),
        Ok(_) => return Route::Fallback("executor-refused"),
        Err(error @ EvalError::Trap { .. }) => Err(error),
        // The executor's tool limits are not the reference evaluator's.
        Err(_) => return Route::Fallback("executor-tool-limit"),
    };
    let checked: Vec<_> = form.certificates().collect();
    Route::Routed(Routed {
        result,
        laws: receipt.applied.keys().copied().collect(),
        certificates,
        fingerprint: form.fingerprint,
        certificate_digest: crate::hex::sha256_hex(format!("{checked:?}").as_bytes()),
        lanes: receipt.lanes,
    })
}
