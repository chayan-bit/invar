//! Private source-binding and sealed-model tests. No native/ROS qualification.
use super::model::*;
use super::*;
use crate::front;
use std::path::Path;
use std::sync::OnceLock;

#[path = "substep_admission_cases/binding.rs"]
mod binding;
#[path = "substep_admission_cases/ledger.rs"]
mod ledger;
#[path = "substep_admission_cases/options.rs"]
mod options;
#[path = "substep_admission_cases/replay.rs"]
mod replay;
#[path = "substep_admission_cases/residual.rs"]
mod residual;

fn source(n: usize) -> String {
    let original = if n == 7 {
        include_str!("../../examples/cells/reach_place/cell.inv")
    } else {
        include_str!("../../examples/cells/reach_place/cell-ur5e.inv")
    };
    let start = original.find("component mujoco_sim {").unwrap();
    let end = start + original[start..].find("\n}").unwrap() + 2;
    let component = original[start..end].replace(
        "    impl invar run;",
        include_str!("../fixtures/substep_admission/policy-clause.inv").trim_end(),
    );
    format!("{}{}{}", &original[..start], component, &original[end..])
}
fn compile(source: &str) -> Result<SystemIr, crate::front::FrontError> {
    super::super::compile_at(
        front::parse_system(source.as_bytes())?,
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place"),
    )
}
fn binding(n: usize) -> &'static Binding {
    static SIX: OnceLock<Binding> = OnceLock::new();
    static SEVEN: OnceLock<Binding> = OnceLock::new();
    let storage = if n == 6 { &SIX } else { &SEVEN };
    storage.get_or_init(|| {
        bind(&compile(&source(n)).expect("ordinary source policy binding"))
            .unwrap()
            .unwrap()
    })
}
fn request(run: &Run<'_>) -> Request {
    Request {
        run: run.id,
        identity: run.binding.identity,
        next_step: run.live.step + 1,
        arm_sequence: 1,
        arm: run.live.applied.arm(),
        equality: Equality {
            sequence: 1,
            attached: run.live.attached,
        },
    }
}
fn new_run(binding: &Binding, required: u64, capacity: u32) -> Run<'_> {
    Run::new(
        binding,
        RunId([1; 16]),
        State::home(binding),
        required,
        capacity,
    )
    .unwrap()
}
fn drain(run: &mut Run<'_>) {
    if !run.queue.is_empty() {
        run.acknowledge(run.last_record).unwrap();
    }
    assert!(run.credits_consistent());
}
fn capture(value: &Value, binding: &Binding) -> State {
    let floats = |key: &str| {
        value[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let witness = value["witness"].as_str().unwrap();
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&witness[index * 2..index * 2 + 2], 16).unwrap();
    }
    State {
        step: value["step"].as_u64().unwrap(),
        time: value["time"].as_f64().unwrap(),
        q: floats("q"),
        dq: floats("dq"),
        forces: floats("forces"),
        contacts: value["contacts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                (
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                )
            })
            .collect(),
        applied: Controls::from_raw(binding.layout, &floats("controls")).unwrap(),
        attached: false,
        latent: [0; 8],
        witness: Digest(bytes),
    }
}

fn same_floats(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn same_arm(a: &Arm, b: &Arm) -> bool {
    match (a, b) {
        (Arm::Seven(a), Arm::Seven(b)) => {
            a.iter().zip(b).all(|(x, y)| x.0.to_bits() == y.0.to_bits())
        }
        (Arm::Six(a), Arm::Six(b)) => a.iter().zip(b).all(|(x, y)| x.0.to_bits() == y.0.to_bits()),
        _ => false,
    }
}
fn same_state(a: &State, b: &State) -> bool {
    a.step == b.step
        && a.time.to_bits() == b.time.to_bits()
        && same_floats(&a.q, &b.q)
        && same_floats(&a.dq, &b.dq)
        && same_floats(&a.forces, &b.forces)
        && same_floats(&a.applied.raw(), &b.applied.raw())
        && a.contacts == b.contacts
        && a.attached == b.attached
        && a.latent == b.latent
        && a.witness == b.witness
}

// Independently walk persisted records, without consulting Run::finish or its
// counters/credits. This checks model evidence only, not native host attestation.
fn verify_complete_trace(
    records: &[Record],
    binding: &Binding,
    id: RunId,
    required: u64,
) -> Result<(), Fault> {
    if required == 0 || required > MAX_STEPS {
        return Err(Fault::Incomplete);
    }
    let Some(Record {
        kind: Kind::Snapshot(initial),
        ..
    }) = records.first()
    else {
        return Err(Fault::Incomplete);
    };
    if initial.step != 0 || initial.time.to_bits() != 0.0_f64.to_bits() {
        return Err(Fault::Order);
    }
    binding.gate(initial)?;
    let mut current = initial.clone();
    let mut evaluations = 0;
    let mut rank_expected = 0;
    let mut frozen: Option<State> = None;
    let mut frozen_arm: Option<Arm> = None;
    let mut accepted: Option<(Controls, State)> = None;
    let mut command: Option<Request> = None;
    let mut previous: Option<Request> = None;
    let mut finalized = false;
    for (index, record) in records.iter().enumerate() {
        if record.sequence != index as u64 + 1 || finalized {
            return Err(Fault::Order);
        }
        if index == 0 {
            continue;
        }
        match &record.kind {
            Kind::Candidate {
                step,
                evaluation,
                rank,
                controls,
                prospective,
                predicted,
                result,
            } => {
                if accepted.is_some()
                    || command.is_some()
                    || *step != current.step + 1
                    || *step > required
                    || *rank != rank_expected
                    || *rank >= binding.candidates()
                    || *evaluation != evaluations + 1
                    || *evaluation > MAX_RUN_CANDIDATES
                {
                    return Err(Fault::Order);
                }
                let expected_tendon = match binding.layout {
                    Layout::SevenArmTendon => binding
                        .tendon_start()
                        .and_then(|start| start.checked_sub(u8::try_from(*rank).ok()?)),
                    Layout::SixArm => None,
                };
                if controls.tendon() != expected_tendon {
                    return Err(Fault::Order);
                }
                let mut expected_input = current.clone();
                expected_input.attached = prospective.attached;
                if !same_state(prospective, &expected_input)
                    || frozen.as_ref().is_some_and(|v| !same_state(v, prospective))
                {
                    return Err(Fault::PredictionMismatch);
                }
                frozen.get_or_insert_with(|| prospective.clone());
                let arm = controls.arm();
                if frozen_arm
                    .as_ref()
                    .is_some_and(|first| !same_arm(first, &arm))
                {
                    return Err(Fault::PredictionMismatch);
                }
                frozen_arm.get_or_insert(arm);
                if predicted.step != *step
                    || predicted.time.to_bits() != (current.time + DT).to_bits()
                    || !same_floats(&predicted.applied.raw(), &controls.raw())
                    || predicted.attached != prospective.attached
                {
                    return Err(Fault::PredictionMismatch);
                }
                let observed = binding.gate(predicted);
                if observed != *result {
                    return Err(Fault::PredictionMismatch);
                }
                match observed {
                    Ok(()) => accepted = Some((controls.clone(), predicted.as_ref().clone())),
                    Err(Fault::Bounds | Fault::Contact) => (),
                    Err(fault) => return Err(fault),
                }
                rank_expected += 1;
                evaluations += 1;
            }
            Kind::Command {
                step,
                request,
                controls,
            } => {
                let (selected, _) = accepted.as_ref().ok_or(Fault::Order)?;
                if command.is_some()
                    || *step != current.step + 1
                    || request.next_step != *step
                    || request.run != id
                    || request.identity != binding.identity
                {
                    return Err(Fault::Identity);
                }
                if !same_floats(&selected.raw(), &controls.raw())
                    || !same_arm(&selected.arm(), &request.arm)
                    || frozen
                        .as_ref()
                        .is_none_or(|s| s.attached != request.equality.attached)
                {
                    return Err(Fault::PredictionMismatch);
                }
                if previous.as_ref().is_some_and(|last| {
                    request.arm_sequence < last.arm_sequence
                        || (request.arm_sequence == last.arm_sequence
                            && !same_arm(&request.arm, &last.arm))
                        || request.equality.sequence < last.equality.sequence
                        || (request.equality.sequence == last.equality.sequence
                            && request.equality.attached != last.equality.attached)
                }) {
                    return Err(Fault::Order);
                }
                command = Some(request.clone());
            }
            Kind::Snapshot(state) => {
                let (_, predicted) = accepted.take().ok_or(Fault::Order)?;
                previous = Some(command.take().ok_or(Fault::Order)?);
                if state.step != current.step + 1 || !same_state(state, &predicted) {
                    return Err(Fault::PredictionMismatch);
                }
                binding.gate(state)?;
                current = state.clone();
                rank_expected = 0;
                frozen = None;
                frozen_arm = None;
            }
            Kind::Halt {
                step,
                reason,
                evaluations: charged,
            } => {
                if *step != current.step || *charged < evaluations || *charged > MAX_RUN_CANDIDATES
                {
                    return Err(Fault::Order);
                }
                return Err(*reason);
            }
            Kind::Final {
                step,
                witness,
                outcome,
            } => {
                (*outcome)?;
                if accepted.is_some()
                    || command.is_some()
                    || frozen.is_some()
                    || *step != required
                    || current.step != required
                    || *witness != current.witness
                {
                    return Err(Fault::Incomplete);
                }
                finalized = true;
            }
        }
    }
    if !finalized {
        return Err(Fault::Incomplete);
    }
    Ok(())
}

fn vendored(name: &str) -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/cells/reach_place/assets/menagerie")
            .join(name),
    )
    .unwrap()
}
fn rendered(xml: &str) -> Result<(), Fault> {
    check_rendered(&roxmltree::Document::parse(xml).unwrap())
}

type Forge = Option<Box<dyn Fn(&str) -> String>>;
thread_local! {
    static FORGE: std::cell::RefCell<Forge> = const { std::cell::RefCell::new(None) };
}
/// Test-only seam: replaces the binder's own rendered model text, so P5
/// forgeries reach the real `bind` path with every other input unchanged.
pub(super) fn forge_rendered(xml: String) -> String {
    FORGE.with(|forge| match forge.borrow().as_ref() {
        Some(forge) => forge(&xml),
        None => xml,
    })
}
fn bind_forged(ir: &SystemIr, forge: impl Fn(&str) -> String + 'static) -> Result<(), Fault> {
    FORGE.with(|f| *f.borrow_mut() = Some(Box::new(forge)));
    let result = bind(ir).map(|_| ());
    FORGE.with(|f| *f.borrow_mut() = None);
    result
}
fn rendered_model(ir: &SystemIr) -> String {
    let captured = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
    let sink = captured.clone();
    bind_forged(ir, move |xml| {
        *sink.borrow_mut() = xml.to_owned();
        xml.to_owned()
    })
    .unwrap();
    captured.take()
}
fn in_body(node: roxmltree::Node<'_, '_>, name: &str) -> bool {
    node.ancestors()
        .any(|a| a.has_tag_name("body") && a.attribute("name") == Some(name))
}
/// Rewrites the attributes of the first selected `<geom>` (always childless).
fn retag(
    xml: &str,
    select: impl Fn(roxmltree::Node<'_, '_>) -> bool,
    changes: &[(&str, &str)],
) -> String {
    let doc = roxmltree::Document::parse(xml).unwrap();
    let geom = doc
        .descendants()
        .find(|n| n.has_tag_name("geom") && select(*n))
        .expect("selected geom");
    assert!(geom.children().next().is_none());
    let mut attrs: Vec<(String, String)> = geom
        .attributes()
        .map(|a| (a.name().to_owned(), a.value().to_owned()))
        .collect();
    for (name, value) in changes {
        match attrs.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = (*value).to_owned(),
            None => attrs.push(((*name).to_owned(), (*value).to_owned())),
        }
    }
    let tag: String = attrs.iter().map(|(k, v)| format!(" {k}=\"{v}\"")).collect();
    let range = geom.range();
    format!("{}<geom{tag}/>{}", &xml[..range.start], &xml[range.end..])
}
/// Independent effective-mask resolution of a vendored model: nested
/// `<default>` classes (top level is `main`), `class`, inherited `childclass`
/// and MuJoCo's built-in 1, in binder walk order from `root`.
fn vendored_roles(xml: &str, root: &str) -> String {
    type Masks = (Option<String>, Option<String>);
    fn defaults(
        node: roxmltree::Node<'_, '_>,
        inherited: &Masks,
        out: &mut BTreeMap<String, Masks>,
    ) {
        let mut masks = inherited.clone();
        for geom in node.children().filter(|n| n.has_tag_name("geom")) {
            if let Some(v) = geom.attribute("contype") {
                masks.0 = Some(v.to_owned());
            }
            if let Some(v) = geom.attribute("conaffinity") {
                masks.1 = Some(v.to_owned());
            }
        }
        out.insert(
            node.attribute("class").unwrap_or("main").to_owned(),
            masks.clone(),
        );
        for child in node.children().filter(|n| n.has_tag_name("default")) {
            defaults(child, &masks, out);
        }
    }
    fn walk(
        body: roxmltree::Node<'_, '_>,
        active: &str,
        classes: &BTreeMap<String, Masks>,
        out: &mut String,
    ) {
        let active = body.attribute("childclass").unwrap_or(active);
        for geom in body.children().filter(|n| n.has_tag_name("geom")) {
            let class = &classes[geom.attribute("class").unwrap_or(active)];
            let contype = geom
                .attribute("contype")
                .or(class.0.as_deref())
                .unwrap_or("1");
            let conaffinity = geom
                .attribute("conaffinity")
                .or(class.1.as_deref())
                .unwrap_or("1");
            out.push(match (contype, conaffinity) {
                ("1", "1") => 'c',
                ("0", "0") => 'v',
                _ => '?',
            });
        }
        for child in body.children().filter(|n| n.has_tag_name("body")) {
            walk(child, active, classes, out);
        }
    }
    let doc = roxmltree::Document::parse(xml).unwrap();
    let mut classes = BTreeMap::new();
    for top in doc
        .root_element()
        .children()
        .filter(|n| n.has_tag_name("default"))
    {
        defaults(top, &(None, None), &mut classes);
    }
    let body = doc
        .descendants()
        .find(|n| n.has_tag_name("body") && n.attribute("name") == Some(root))
        .unwrap();
    let mut out = String::new();
    walk(body, "main", &classes, &mut out);
    out
}

fn selected_robot_id(ir: &SystemIr) -> String {
    let document = ir.checked_document().unwrap();
    let product = &document["products"][0]["robot"];
    document["robots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == *product)
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned()
}
