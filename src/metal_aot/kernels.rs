//! The Metal kernels a program's `invar run` route can dispatch, known before
//! it runs. Generated MSL does not depend on the input length or the Apple
//! GPU family, so one build covers every run; a kernel missing at run time is
//! refused, never compiled.
use invar_device::{Capabilities, Provenance, fsum};

use crate::{admission, device_ir, mir, plan::Execution, plan::PlanKind};

/// Any planned length gives the same MSL; this one plans every region kind.
const BUILD_LENGTH: u32 = 4096;
const BUILD_CAPS: Capabilities = Capabilities::Metal {
    apple_family: 7,
    msl_major: 3,
    msl_minor: 1,
};

/// Every library the admitted GNET artifact can dispatch, including its D1
/// preflight. Build and load use this same input-independent inventory.
pub fn gnet_kernels(
    artifact: &crate::general_net::compiled::Artifact,
) -> Vec<(String, Vec<String>)> {
    let msl = artifact.source();
    let mut kernels = vec![(msl.to_owned(), super::msl_entry_points(msl))];
    if artifact.abi_version() == 13 {
        kernels.push((
            invar_device::f32_probe::emit_msl(),
            vec![invar_device::f32_probe::KERNEL_NAME.to_owned()],
        ));
    }
    kernels
}

/// The generated MSL of every Metal region of `program` at `length`, with the
/// functions it exports, followed by the placement calibration probe, then
/// the explicit sort kernels when the program sorts, the spatial kernels
/// with the strict f32 probe when it runs a spatial operation, then the fsum
/// reduction kernels when the program calls one.
pub fn run_kernels(
    source: &[u8],
    program: &mir::Program,
) -> Result<Vec<(String, Vec<String>)>, String> {
    let report = admission::analyze(source, program).map_err(|e| format!("admission:{e:?}"))?;
    let mut kernels = vec![(
        invar_device::PROBE_MSL.to_owned(),
        vec![invar_device::PROBE_FUNCTION.to_owned()],
    )];
    for region in report.regions() {
        if region.plan_kind() == PlanKind::Serial
            || !matches!(
                region.execution(),
                Execution::Map(_)
                    | Execution::FusedProducer(_)
                    | Execution::DagProducer(_)
                    | Execution::Reduction(_)
                    | Execution::PairReduction(_)
                    | Execution::Scan(_)
            )
        {
            continue;
        }
        // A region without a Metal plan is refused at run time as well.
        let Some((msl, functions)) = region_msl(region, BUILD_LENGTH, BUILD_CAPS) else {
            continue;
        };
        if !kernels.iter().any(|(known, _)| *known == msl) {
            kernels.push((msl, functions.iter().map(|f| (*f).to_owned()).collect()));
        }
    }
    // Spatial grid construction orders cell keys with the sort kernels.
    let spatial = crate::spatial::contains(program);
    if spatial || crate::sorting::contains(program) {
        let functions = invar_device::sorting::KERNELS.map(str::to_owned);
        kernels.push((invar_device::sorting::METAL.to_owned(), functions.to_vec()));
    }
    if spatial {
        let functions = invar_device::spatial::KERNELS.map(str::to_owned);
        kernels.push((invar_device::spatial::metal_source(), functions.to_vec()));
        // The query kernel enables its normal window only after this probe.
        let probe = invar_device::f32_probe::KERNEL_NAME.to_owned();
        kernels.push((invar_device::f32_probe::emit_msl(), vec![probe]));
    }
    if calls_fsum(program) {
        let functions = [fsum::Op::SumTree, fsum::Op::SumExact].map(|op| op.kernel().to_owned());
        kernels.push((fsum::msl(), functions.to_vec()));
    }
    Ok(kernels)
}

/// Whether any instruction of `program` calls a v0.2 float reduction.
fn calls_fsum(program: &mir::Program) -> bool {
    program
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .any(|instruction| {
            matches!(&instruction.kind, mir::Operation::Call { name, .. }
                if crate::fsum::Kind::from_name(name).is_some())
        })
}

/// The MSL `schedule::metal` emits for `region` at `length` on `caps`.
pub fn region_msl(
    region: &admission::Region,
    length: u32,
    caps: Capabilities,
) -> Option<(String, &'static [&'static str])> {
    let mut ir = device_ir::from_region(region, length).ok()?;
    device_ir::plan_metal(&mut ir).ok()?;
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let plan = invar_device::verify(ir, origin, caps).ok()?;
    let msl = invar_device::emit_msl(&plan).ok()?;
    Some((msl, invar_device::metal_functions(plan.ir())))
}
