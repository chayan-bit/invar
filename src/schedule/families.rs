//! Ordered, closed family dispatch. One file and one registration per family.
//! A family is a lowering of checked normal-form nodes (#54). Only a
//! completed Step can reach publication.
use super::*;
use crate::general_net::normal_form::NormalForm;
use lowering::{Admitted, Live};

// The list is precedence order: the first lowered candidate owns that loop.
// Keep the existing core first so new adapters cannot steal its fusion links.
macro_rules! families {
    ($( $variant:ident => $module:ident ),+ $(,)?) => {
        $(mod $module;)+

        enum Candidate { $( $variant(Box<$module::Candidate>), )+ }

        pub(super) struct Selected {
            function: String,
            span: Span,
            candidate: Candidate,
            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            region: Region,
        }

        pub(super) fn analyze(source: &[u8], program: &mir::Program) -> Result<Vec<Selected>, admission::Error> {
            dispatch(source, program, None).map(|dispatch| dispatch.selected)
        }

        /// Regions from the checked normal form only: no form, no region.
        pub(super) fn dispatch(source: &[u8], program: &mir::Program, forge: Option<fn(&mut NormalForm)>) -> Result<Dispatch, admission::Error> {
            let report = admission::analyze(source, program)?;
            let checked = match lowering::check(program, forge) {
                Ok(checked) => checked,
                Err(reason) => return Ok(Dispatch { selected: Vec::new(), gaps: Vec::new(), refusal: Some(reason) }),
            };
            let mut selected = Vec::new();
            for region in report.regions() {
                let admitted: Admitted = checked.admitted(program, region);
                let live = Live { source, program, region };
                let candidate = 'lower: {
                    $(if let Some(candidate) = $module::lower(&admitted, &live)? {
                        break 'lower Some(Candidate::$variant(Box::new(candidate)));
                    })+
                    None
                };
                if let Some(candidate) = candidate {
                    selected.push(Selected {function: region.function().to_owned(), span: region.span(), candidate, #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))] region: region.clone()});
                }
            }
            let claimed = selected.iter().map(|s| (s.function.clone(), s.span)).collect::<Vec<_>>();
            Ok(Dispatch { gaps: checked.gaps(program, &claimed), selected, refusal: None })
        }

        impl Selected {
            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            pub(super) fn cost_record(&self, live: &LiveLoop<'_>, caps: invar_device::Capabilities, model: crate::plan::DeviceCostModel) -> Option<tuning::CostRecord> {
                let mut ir = crate::device_ir::from_region(&self.region, u32::try_from(live.iterations).ok()?).ok()?;
                if matches!(caps, invar_device::Capabilities::Metal { .. }) { crate::device_ir::plan_metal(&mut ir).ok()?; }
                let origin = invar_device::Provenance { source_identity: ir.source_identity, graph_revision: ir.graph_revision };
                let plan = invar_device::verify(ir, origin, caps).ok()?;
                tuning::CostRecord::from_plan(&plan, tuning::Residency::HOST, model).ok()
            }

            pub(super) fn function(&self) -> &str { &self.function }
            pub(super) fn span(&self) -> Span { self.span }
            pub(super) fn is_core(&self) -> bool { matches!(self.candidate, Candidate::Core(_)) }
            pub(super) fn family(&self) -> &'static str {
                match &self.candidate { $( Candidate::$variant(_) => stringify!($module), )+ }
            }

            pub(super) fn cpu(&self, live: &LiveLoop<'_>, pending: &mut VecDeque<Pending>, workers: usize, hooks: &Hooks) -> Result<Option<Step>, Error> {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::cpu(candidate, live, pending, workers, hooks), )+ }
            }

            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            pub(super) fn is_consumer(&self) -> bool {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::is_consumer(candidate), )+ }
            }

            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            pub(super) fn is_producer(&self) -> bool {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::is_producer(candidate), )+ }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            pub(super) fn metal_cost(&self, live: &LiveLoop<'_>, caps: invar_device::Capabilities, model: crate::plan::DeviceCostModel) -> Option<crate::plan::DeviceCost> {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::metal::cost(candidate, live, caps, model), )+ }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            pub(super) fn metal_ineligible_reason(&self, len: usize) -> &'static str {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::metal::ineligible_reason(candidate, len), )+ }
            }

            /// The compiled-ahead kernels `metal_step` would load at `live` on a
            /// device with `caps`; core regions are built from admission instead.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            pub(super) fn metal_kernels(&self, live: &LiveLoop<'_>, caps: invar_device::Capabilities) -> Vec<(String, Vec<String>)> {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::metal::kernels(candidate, live, caps), )+ }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            pub(super) fn metal_step(&self, source: &[u8], live: &LiveLoop<'_>, pending: &mut VecDeque<Pending>, device: &mut invar_metal::MetalDevice, fault: metal::Fault) -> Result<(Step, bool, crate::device_cache::HintState), metal::Error> {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::metal::step(source, candidate, live, pending, device, fault), )+ }
            }

            #[cfg(feature = "cuda")]
            pub(super) fn cuda_cost(&self, live: &LiveLoop<'_>, caps: invar_device::Capabilities, model: crate::plan::DeviceCostModel) -> Option<crate::plan::DeviceCost> {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::cuda::cost(candidate, live, caps, model), )+ }
            }

            #[cfg(feature = "cuda")]
            pub(super) fn cuda_step(&self, source: &[u8], live: &LiveLoop<'_>, pending: &mut VecDeque<Pending>, device: &mut invar_cuda::cuda::CudaDevice, compiler: &std::path::Path, fault: cuda::Fault) -> Result<(Step, bool, crate::device_cache::HintState), cuda::Error> {
                match &self.candidate { $( Candidate::$variant(candidate) => $module::cuda::step(source, candidate, live, pending, device, compiler, fault), )+ }
            }
        }
    }
}

/// The dispatcher's outcome: selected regions, certified loops no family
/// lowers, and why no form was available (then nothing is selected).
pub(super) struct Dispatch {
    pub(super) selected: Vec<Selected>,
    pub(super) gaps: Vec<lowering::Gap>,
    pub(super) refusal: Option<&'static str>,
}

families! {
    Core => core,
    Tensor => tensor,
    Grid => grid,
    Segmented => segmented,
    Indexed => indexed,
}
