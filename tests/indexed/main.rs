#[allow(dead_code)]
#[path = "../support/language_limits.rs"]
mod language_limits;

mod device;
mod frontier_source;
mod gather_preflight;
mod gather_runtime;
mod gather_scatter_frontier;
mod histogram_runtime;
mod histogram_source;
mod static_scatter_proof;
mod static_scatter_runtime;

/// The MIR evaluator and the net oracle on `Net::encode(m)` agree with the reference evaluator on
/// result, steps and cleanup at every fuel `0..=last_fuel`; returns the number of fuels observed.
fn agree_on_every_prefix(
    p: &invar::check::CheckedProgram,
    m: &invar::mir::Program,
    last_fuel: u64,
    context: &str,
) -> u64 {
    use invar::{eval, general_net, mir_eval};
    let net = general_net::Net::encode(m, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    for fuel in 0..=last_fuel {
        let reference = eval::observe(p, fuel);
        let extracted = mir_eval::observe(m, fuel);
        let interaction = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                ..general_net::oracle::Limits::default()
            },
        )
        .unwrap();
        for observed in [&extracted, &interaction.semantic] {
            assert_eq!(observed.result, reference.result, "{context} fuel={fuel}");
            assert_eq!(
                observed.steps_used, reference.steps_used,
                "{context} fuel={fuel}"
            );
            assert_eq!(
                observed.cleanup_trace, reference.cleanup_trace,
                "{context} fuel={fuel}"
            );
        }
    }
    last_fuel + 1
}
