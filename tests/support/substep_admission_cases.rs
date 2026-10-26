//! Private source-binding tests. No native/ROS qualification.
use super::*;
use crate::front;
use std::path::Path;

#[path = "substep_admission_cases/binding.rs"]
mod binding;
#[path = "substep_admission_cases/options.rs"]
mod options;
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
