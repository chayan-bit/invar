//! One dispatcher (#263): `run`, `auto`, `explain`, `calibrate` and `build`
//! get regions only from the checked parallel normal form. The form is
//! extracted from the run's own MIR, every certificate is re-checked
//! (`regions::family::analyze` runs `certificate::check_all` first), and each
//! A0 loop is offered, with the PNF admissions at its span, to the family
//! lowerings in precedence order. The A0 region only binds the reached loop.
//! A form that is unavailable or refused yields no region at all.

use crate::{
    admission::Region,
    front::Span,
    general_net::{
        Limits, Net,
        normal_form::{self, ArrayLaw, LoopNode, Node, NormalForm, Rep},
        regions::family::{self, Family},
    },
    mir,
};

/// The PNF facts at one A0 loop: the static family admissions the checked
/// form grants there.
pub(super) struct Admitted {
    pub(super) families: Vec<Family>,
}

/// The admissions the core family lowers: maps, fusions, reductions and scans.
pub(super) const CORE: &[Family] = &[
    Family::Map,
    Family::FusedMap,
    Family::Dag,
    Family::Reduce,
    Family::Scan,
];

impl Admitted {
    pub(super) fn admits(&self, families: &[Family]) -> bool {
        self.families.iter().any(|f| families.contains(f))
    }
}

/// The reached-loop binding a lowering may read: the A0 region and the source
/// and MIR its contracts are bound to. It never selects a family.
pub(super) struct Live<'a> {
    pub(super) source: &'a [u8],
    pub(super) program: &'a mir::Program,
    pub(super) region: &'a Region,
}

/// A certified parallel loop no family lowers: it runs on the existing CPU path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gap {
    pub function: String,
    pub span: Span,
    /// `no-lowering:<shape>/<rep>`.
    pub reason: String,
}

/// The checked form and its family admissions, or why there is none.
pub(super) struct Checked {
    form: NormalForm,
    report: family::Report,
}

/// Why no region was offered: the form could not be built or was refused.
pub(super) fn check(
    program: &mir::Program,
    forge: Option<fn(&mut NormalForm)>,
) -> Result<Checked, &'static str> {
    let net = Net::encode(program, Limits::default()).map_err(|_| "net-unavailable")?;
    let proof = net.verify().map_err(|_| "net-unavailable")?;
    let mut form = normal_form::extract(&net, &proof).map_err(|_| "regions-unavailable")?;
    if let Some(forge) = forge {
        forge(&mut form);
    }
    let report = family::analyze(&net, &proof, &form).map_err(|error| match error {
        family::Error::Certificate(_) => "certificate-rejected",
        family::Error::FormMismatch => "form-mismatch",
        family::Error::Regions(_) | family::Error::Net(_) => "regions-unavailable",
    })?;
    Ok(Checked { form, report })
}

impl Checked {
    pub(super) fn admitted(&self, program: &mir::Program, region: &Region) -> Admitted {
        let function = program
            .functions
            .iter()
            .position(|f| f.name == region.function());
        Admitted {
            families: self
                .report
                .admissions
                .iter()
                .filter(|a| Some(a.region.function) == function && a.span == region.span())
                .map(|a| a.family)
                .collect(),
        }
    }

    /// Certified parallel loops whose span no lowering claimed.
    pub(super) fn gaps(&self, program: &mir::Program, claimed: &[(String, Span)]) -> Vec<Gap> {
        self.parallel(program)
            .filter_map(|(f, span, node)| {
                let function = program.functions[f].name.clone();
                (!claimed
                    .iter()
                    .any(|(name, s)| *name == function && *s == span))
                .then(|| Gap {
                    function,
                    span,
                    reason: format!("no-lowering:{}/{}", node.shape(), rep(node)),
                })
            })
            .collect()
    }

    fn parallel(&self, program: &mir::Program) -> std::vec::IntoIter<(usize, Span, &LoopNode)> {
        let loops = self.form.functions.iter().flat_map(|function| {
            function.loops.iter().filter_map(move |form| {
                let Node::Parallel(node) = &form.node else {
                    return None;
                };
                let head = form.region.head?;
                let block = program.functions.get(function.index)?.blocks.get(head)?;
                let Some(mir::Terminator::Branch { span, .. }) = &block.end else {
                    return None;
                };
                (!form.certificates.is_empty()).then_some((function.index, *span, node))
            })
        });
        loops.collect::<Vec<_>>().into_iter()
    }
}

/// The representation a lowering keys on: the first carried value's, else the
/// first array law, else `none`.
fn rep(node: &LoopNode) -> &'static str {
    if let Some(carried) = node.carried.first() {
        return match carried.rep {
            Rep::Sealed(_) => "sealed",
            Rep::LiftedSum(_) => "lifted-sum",
            Rep::SaturatedProduct(_) => "saturated-product",
            Rep::Affine(_) => "affine",
            Rep::Matrix { .. } => "matrix",
            Rep::Tropical { .. } => "tropical",
            Rep::Select { .. } => "select",
            Rep::Last => "last",
            Rep::Counter(_) => "counter",
            Rep::Lifted(_) => "lifted",
        };
    }
    match node.arrays.first().map(|(_, law)| law) {
        Some(ArrayLaw::Own) => "own",
        Some(ArrayLaw::Combine { .. }) => "combine",
        Some(ArrayLaw::Ordered) => "ordered",
        Some(ArrayLaw::Compaction { .. }) => "compaction",
        Some(ArrayLaw::Monotone { .. }) => "monotone",
        None => "none",
    }
}

/// One loop the dispatcher selected, by family module (`core`, `tensor`, ...).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub function: String,
    pub span: Span,
    pub family: &'static str,
}

/// What the dispatcher offers the scheduler for one program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatched {
    pub selected: Vec<Selection>,
    pub gaps: Vec<Gap>,
    /// Why no form was available (`certificate-rejected`, ...); then nothing
    /// is selected and every loop runs on the existing CPU path.
    pub refusal: Option<&'static str>,
}

/// The dispatcher's selection for `program`, with `hooks.forge` applied to
/// the extracted form before it is checked.
pub fn selection(
    source: &[u8],
    program: &mir::Program,
    hooks: &super::Hooks,
) -> Result<Dispatched, crate::admission::Error> {
    let dispatch = super::families::dispatch(source, program, hooks.forge)?;
    Ok(Dispatched {
        selected: dispatch
            .selected
            .iter()
            .map(|s| Selection {
                function: s.function().to_owned(),
                span: s.span(),
                family: s.family(),
            })
            .collect(),
        gaps: dispatch.gaps,
        refusal: dispatch.refusal,
    })
}

/// Whether `program` has a family region (grid, segmented, tensor) whose
/// device kernels depend on the live state its loop reaches.
pub fn has_family_region(
    source: &[u8],
    program: &mir::Program,
) -> Result<bool, crate::admission::Error> {
    Ok(super::families::analyze(source, program)?
        .iter()
        .any(|region| !region.is_core()))
}

/// Whether `run_source` would run `program` as plain serial MIR: no spatial,
/// sort or float-reduction call, no f32 local and no family region. Only such
/// programs are offered to the normal-form route, so every existing parallel
/// path keeps its exact behaviour.
pub fn serial_only(source: &[u8], program: &mir::Program) -> bool {
    !crate::spatial::contains(program)
        && !crate::sorting::contains(program)
        && !super::has_reduction(program)
        && !program.functions.iter().any(|f| {
            f.locals
                .values()
                .any(|l| matches!(l.ty, crate::front::Type::F32 | crate::front::Type::ArrayF32))
        })
        && super::families::analyze(source, program).is_ok_and(|regions| regions.is_empty())
}

/// The A0 report `explain`, `auto` and `calibrate` describe, restricted to
/// the dispatcher: a core plan no lowering selected is reported serial.
pub fn regions(
    source: &[u8],
    program: &mir::Program,
) -> Result<crate::admission::Report, crate::admission::Error> {
    let mut report = crate::admission::analyze(source, program)?;
    let selected = super::families::analyze(source, program)?;
    report.demote(source, |region| {
        selected
            .iter()
            .any(|s| s.is_core() && s.function() == region.function() && s.span() == region.span())
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    #[test]
    fn issue_263_a_plan_the_dispatcher_did_not_select_is_described_serial() {
        let source = b"fn main() -> i32 { let a = filled_u32(4u32, 1u32); var out = filled_u32(4u32, 0u32); for i in 0u32..4u32 { out[i] = wrapping_add(a[i], 3u32); } return 0; }";
        let checked = crate::check::check(crate::front::parse(source).unwrap(), false).unwrap();
        let program = crate::mir::lower(&checked).unwrap();
        let selected = super::regions(source, &program).unwrap().to_json();
        assert_eq!(selected["regions"][0]["status"], "admitted");
        crate::admission::validate_serialized(&selected, source, &program).unwrap();
        let mut report = crate::admission::analyze(source, &program).unwrap();
        report.demote(source, |_| false);
        let demoted = report.to_json();
        let region = &demoted["regions"][0];
        assert_eq!(region["status"], "rejected");
        assert_eq!(region["first_rejected_obligation"], "DEPENDENCE");
        assert_eq!(region["chosen_plan"], "mir-serial");
        assert_eq!(region["runtime_guard"], serde_json::Value::Null);
        let dependence = region["obligations"].as_array().unwrap().iter();
        let dependence: Vec<_> = dependence.filter(|o| o["name"] == "DEPENDENCE").collect();
        assert_eq!(dependence.len(), 1);
        assert_eq!(dependence[0]["evidence"], "unknown");
        // A record the dispatcher would not produce for this program is stale.
        assert!(crate::admission::validate_serialized(&demoted, source, &program).is_err());
    }
}
