//! Pure, exact checks over a checked system IR document.

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::BigInt;
use num_rational::BigRational;
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    pub code: &'static str,
    pub message: String,
    pub subject: String,
}

fn issue(code: &'static str, message: impl Into<String>, subject: &str) -> Issue {
    Issue {
        code,
        message: message.into(),
        subject: subject.to_owned(),
    }
}

pub(super) fn decimal(value: &str) -> Option<BigRational> {
    let negative = value.starts_with('-');
    let value = value.strip_prefix('-').unwrap_or(value);
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    let digits = format!("{whole}{fraction}");
    let mut numerator = BigInt::parse_bytes(digits.as_bytes(), 10)?;
    if negative {
        numerator = -numerator;
    }
    Some(BigRational::new(
        numerator,
        BigInt::from(10u8).pow(fraction.len() as u32),
    ))
}

fn quantity(value: &Value, expected: &str) -> Option<BigRational> {
    let number = decimal(value["value"].as_str()?)?;
    let scale = match (expected, value["unit"].as_str()?) {
        ("rate", "Hz") | ("cores", "one") => BigInt::from(1),
        ("seconds", "s") => BigInt::from(1),
        ("seconds", "ms") => BigInt::from(1_000),
        ("seconds", "us") => BigInt::from(1_000_000),
        ("seconds", "ns") => BigInt::from(1_000_000_000),
        ("bytes", "MiB") => BigInt::from(1u64 << 20),
        ("bytes", "GiB") => BigInt::from(1u64 << 30),
        ("bits_per_second", "Mbit/s") => BigInt::from(1_000_000),
        _ => return None,
    };
    if expected == "seconds" && value["unit"] != "s" {
        Some(number / BigRational::from_integer(scale))
    } else {
        Some(number * BigRational::from_integer(scale))
    }
}

fn components(document: &Value) -> BTreeMap<&str, &Value> {
    document["components"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(
            document["extern_components"]
                .as_array()
                .into_iter()
                .flatten(),
        )
        .filter_map(|component| Some((component["id"].as_str()?, component)))
        .collect()
}

fn instances(document: &Value) -> BTreeMap<&str, &str> {
    document["system"]["instances"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|instance| Some((instance["name"].as_str()?, instance["component"].as_str()?)))
        .collect()
}

fn port<'a>(
    path: &str,
    instances: &BTreeMap<&str, &str>,
    components: &BTreeMap<&str, &'a Value>,
) -> Option<&'a Value> {
    let (instance, port_name) = path.split_once('.')?;
    components.get(instances.get(instance)?)?["ports"]
        .as_array()?
        .iter()
        .find(|port| port["name"] == port_name)
}

fn time_checks(document: &Value, issues: &mut Vec<Issue>) {
    let components = components(document);
    let instances = instances(document);
    let zero = BigRational::from_integer(0.into());
    let one = BigRational::from_integer(1.into());
    for connection in document["connections"].as_array().into_iter().flatten() {
        let Some(id) = connection["id"].as_str() else {
            continue;
        };
        let (Some(source), Some(target), Some(policy)) = (
            connection["source"]
                .as_str()
                .and_then(|p| port(p, &instances, &components)),
            connection["target"]
                .as_str()
                .and_then(|p| port(p, &instances, &components)),
            connection["policy"].as_str(),
        ) else {
            issues.push(issue("S1100", "connection port or policy is invalid", id));
            continue;
        };
        let (Some(producer), Some(consumer)) = (
            quantity(&source["rate"], "rate"),
            quantity(&target["rate"], "rate"),
        ) else {
            issues.push(issue("S1100", "connection rate has the wrong unit", id));
            continue;
        };
        if producer <= zero || consumer <= zero {
            issues.push(issue("S1100", "connection rate must be positive", id));
        } else if let Some(count) = policy
            .strip_prefix("decimate(")
            .and_then(|s| s.strip_suffix(')'))
        {
            let count = count.parse::<u32>().ok().filter(|n| (1..=1024).contains(n));
            if count.is_none_or(|n| producer != consumer * BigRational::from_integer(n.into())) {
                issues.push(issue("S1100", "decimation rate is not exact", id));
            }
        } else if policy.starts_with("queue(") {
            if producer > consumer {
                issues.push(issue("S1100", "queue producer exceeds consumer rate", id));
            }
        } else if policy != "sample-and-hold" {
            issues.push(issue("S1100", "connection policy is not finite", id));
        }
    }

    for (instance, component_id) in &instances {
        let Some(component) = components.get(component_id) else {
            continue;
        };
        let (Some(wcet), Some(deadline)) = (
            quantity(&component["wcet"], "seconds"),
            quantity(&component["deadline"], "seconds"),
        ) else {
            issues.push(issue(
                "S1101",
                "component requires WCET and deadline budgets",
                component_id,
            ));
            continue;
        };
        if wcet <= zero || deadline <= zero || wcet > deadline {
            issues.push(issue("S1101", "WCET exceeds deadline", component_id));
        }
        for item in component["ports"].as_array().into_iter().flatten() {
            if let Some(rate) = quantity(&item["rate"], "rate")
                && rate > zero
                && deadline > &one / rate
            {
                issues.push(issue(
                    "S1101",
                    format!(
                        "deadline exceeds period at {instance}.{}",
                        item["name"].as_str().unwrap_or("?")
                    ),
                    component_id,
                ));
            }
        }
    }

    for flow in document["flows"].as_array().into_iter().flatten() {
        let Some(id) = flow["id"].as_str() else {
            continue;
        };
        let Some(budget) = quantity(&flow["within"], "seconds") else {
            issues.push(issue("S1102", "flow budget has the wrong unit", id));
            continue;
        };
        let mut sum = zero.clone();
        let mut visited = BTreeSet::new();
        let paths: Vec<_> = flow["ports"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        for path in &paths {
            if let Some((instance, _)) = path.split_once('.')
                && visited.insert(instance)
                && let Some(component) = instances
                    .get(instance)
                    .and_then(|name| components.get(name))
                && let Some(wcet) = quantity(&component["wcet"], "seconds")
            {
                sum += wcet;
            }
        }
        for pair in paths.windows(2) {
            let source_instance = pair[0].split_once('.').map(|(name, _)| name);
            let target_instance = pair[1].split_once('.').map(|(name, _)| name);
            let connection = document["connections"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|connection| {
                    connection["source"] == pair[0] && connection["target"] == pair[1]
                });
            let Some(connection) = connection else {
                if source_instance != target_instance {
                    issues.push(issue(
                        "S1102",
                        "flow crosses a disconnected component hop",
                        id,
                    ));
                } else if let (Some(source), Some(target)) = (
                    port(pair[0], &instances, &components),
                    port(pair[1], &instances, &components),
                ) && (source["direction"] != "in" || target["direction"] != "out")
                {
                    issues.push(issue(
                        "S1102",
                        "flow internal hop must be input to output",
                        id,
                    ));
                }
                continue;
            };
            if let Some(source) = port(pair[0], &instances, &components)
                && let Some(latency) = quantity(&source["latency"], "seconds")
            {
                sum += latency;
            }
            if connection["policy"].as_str().is_some_and(|policy| {
                policy == "sample-and-hold" || policy.starts_with("decimate(")
            }) && let Some(target) = port(pair[1], &instances, &components)
                && let Some(rate) = quantity(&target["rate"], "rate")
                && rate > zero
            {
                sum += &one / rate;
            }
        }
        if sum > budget {
            issues.push(issue("S1102", "flow exceeds declared latency budget", id));
        }
    }
}

fn resource_checks(document: &Value, issues: &mut Vec<Issue>) {
    let components = components(document);
    let instances = instances(document);
    let Some(product_id) = document["system"]["product"].as_str() else {
        return;
    };
    let Some(product) = document["products"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|product| product["id"] == product_id)
    else {
        return;
    };
    let keys = [
        ("cpu_cores", "cores"),
        ("memory", "bytes"),
        ("gpu_memory", "bytes"),
    ];
    for (key, unit) in keys {
        let mut used = BigRational::from_integer(0.into());
        for component_name in instances.values() {
            let Some(component) = components.get(component_name) else {
                continue;
            };
            if !component["resources"][key].is_null() {
                if let Some(value) = quantity(&component["resources"][key], unit) {
                    used += value;
                } else {
                    issues.push(issue(
                        "S1250",
                        format!("{key} resource has wrong unit"),
                        component_name,
                    ));
                }
            }
        }
        let capacity = quantity(&product["capacity"][key], unit);
        if !product["capacity"][key].is_null() && capacity.is_none() {
            issues.push(issue(
                "S1250",
                format!("{key} capacity has wrong unit"),
                product_id,
            ));
            continue;
        }
        if used > capacity.unwrap_or_else(|| BigRational::from_integer(0.into())) {
            issues.push(issue(
                "S1250",
                format!("{key} exceeds product capacity"),
                product_id,
            ));
        }
    }
    let mut buses: BTreeMap<String, BigRational> = BTreeMap::new();
    for component_name in instances.values() {
        let Some(component) = components.get(component_name) else {
            continue;
        };
        for port in component["ports"].as_array().into_iter().flatten() {
            let Some(bus) = port["bus"].as_str() else {
                continue;
            };
            if bus.is_empty() {
                issues.push(issue("S1251", "bus name is empty", component_name));
                continue;
            }
            let key = format!("bus.{bus}");
            if product["capacity"].get(&key).is_none() {
                issues.push(issue(
                    "S1251",
                    format!("product does not declare {key}"),
                    component_name,
                ));
                continue;
            }
            let Some(bytes) = port["type"]
                .as_str()
                .and_then(crate::robotics::c_layout_size)
            else {
                issues.push(issue(
                    "S1252",
                    "bus port lacks a checked canonical C layout",
                    component_name,
                ));
                continue;
            };
            let Some(rate) = quantity(&port["rate"], "rate") else {
                issues.push(issue(
                    "S1252",
                    "bus port rate has the wrong unit",
                    component_name,
                ));
                continue;
            };
            let Some(bits) = bytes.checked_mul(8) else {
                issues.push(issue(
                    "S1252",
                    "bus port layout size exceeds the limit",
                    component_name,
                ));
                continue;
            };
            *buses
                .entry(key)
                .or_insert_with(|| BigRational::from_integer(0.into())) +=
                rate * BigRational::from_integer(BigInt::from(bits));
        }
    }
    for (key, used) in buses {
        let Some(capacity) = quantity(&product["capacity"][&key], "bits_per_second") else {
            issues.push(issue(
                "S1251",
                format!("{key} capacity has the wrong unit"),
                product_id,
            ));
            continue;
        };
        if used > capacity {
            issues.push(issue(
                "S1250",
                format!("{key} bandwidth exceeds product capacity"),
                product_id,
            ));
        }
    }
}

fn transform_pairs<'a>(ty: &'a str, pairs: &mut Vec<(&'a str, &'a str, bool)>) {
    let Some((head, args)) = crate::robotics::generic_parts(ty) else {
        return;
    };
    match (head, args.as_slice()) {
        ("Transform", [from, to]) => pairs.push((from, to, true)),
        ("std::robotics::Pose", [_, parent, child]) => pairs.push((child, parent, true)),
        ("std::robotics::PlannerRequest", [_, _, base, tool, object]) => {
            // A requested pose consumes measured frames; it does not publish
            // their current transform.
            pairs.extend([
                (*tool, *base, false),
                (*object, *base, false),
                (*object, *tool, false),
            ]);
        }
        _ => {}
    }
    for arg in args {
        transform_pairs(arg, pairs);
    }
}

fn frame_checks(document: &Value, issues: &mut Vec<Issue>) {
    if document["planner_frame_policy"] != "request-consumer-v1" {
        issues.push(issue(
            "S1200",
            "planner frame policy is not request-consumer-v1",
            document["system"]["id"].as_str().unwrap_or(""),
        ));
        return;
    }
    let parents: BTreeMap<_, _> = document["frames"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|frame| Some((frame["name"].as_str()?, frame["parent"].as_str())))
        .collect();
    if parents.is_empty() {
        return;
    }
    let roots: Vec<_> = parents
        .iter()
        .filter(|(_, parent)| parent.is_none())
        .collect();
    if roots.len() != 1 {
        issues.push(issue(
            "S1200",
            "frame tree must have exactly one root",
            document["system"]["id"].as_str().unwrap_or(""),
        ));
        return;
    }
    let root = roots[0].0;
    for frame in parents.keys() {
        let mut path = BTreeSet::new();
        let mut current = *frame;
        while current != *root {
            if !path.insert(current) {
                issues.push(issue("S1200", "frame tree contains a cycle", frame));
                break;
            }
            let Some(parent) = parents.get(current).and_then(|parent| *parent) else {
                issues.push(issue("S1200", "frame is unreachable from the root", frame));
                break;
            };
            current = parent;
        }
    }
    if issues.iter().any(|issue| issue.code == "S1200") {
        return;
    }

    let components = components(document);
    let instances = instances(document);
    let mut used_edges = BTreeSet::new();
    let mut sources: BTreeMap<(&str, &str), Vec<String>> = BTreeMap::new();
    for (instance, component_name) in &instances {
        let Some(component) = components.get(component_name) else {
            continue;
        };
        for port in component["ports"].as_array().into_iter().flatten() {
            let Some(ty) = port["type"].as_str() else {
                continue;
            };
            let mut pairs = Vec::new();
            transform_pairs(ty, &mut pairs);
            for (from, to, publishes) in pairs {
                if !parents.contains_key(from) || !parents.contains_key(to) {
                    issues.push(issue(
                        "S1200",
                        "transform names an undeclared frame",
                        component_name,
                    ));
                    continue;
                }
                // A transform may cross several edges. Every underlying tree edge needs a source.
                let mut from_path = BTreeSet::new();
                let mut current = from;
                from_path.insert(current);
                while let Some(parent) = parents.get(current).and_then(|parent| *parent) {
                    current = parent;
                    from_path.insert(current);
                }
                let mut common = to;
                while !from_path.contains(common) {
                    let Some(parent) = parents.get(common).and_then(|parent| *parent) else {
                        break;
                    };
                    common = parent;
                }
                let mut current = from;
                while current != common {
                    let Some(parent) = parents.get(current).and_then(|parent| *parent) else {
                        break;
                    };
                    used_edges.insert((current, parent));
                    current = parent;
                }
                current = to;
                while current != common {
                    let Some(parent) = parents.get(current).and_then(|parent| *parent) else {
                        break;
                    };
                    used_edges.insert((current, parent));
                    current = parent;
                }
                if publishes && port["direction"] == "out" {
                    if parents.get(from).and_then(|parent| *parent) == Some(to) {
                        sources.entry((from, to)).or_default().push(format!(
                            "{instance}.{}",
                            port["name"].as_str().unwrap_or("?")
                        ));
                    } else if parents.get(to).and_then(|parent| *parent) == Some(from) {
                        sources.entry((to, from)).or_default().push(format!(
                            "{instance}.{}",
                            port["name"].as_str().unwrap_or("?")
                        ));
                    }
                }
            }
        }
        if let Some(params) = component["params"].as_object() {
            for (name, value) in params {
                if value["kind"] != "identity-transform" {
                    continue;
                }
                let (Some(child), Some(parent)) = (value["from"].as_str(), value["to"].as_str())
                else {
                    issues.push(issue(
                        "S1203",
                        "identity transform is missing its frame types",
                        component_name,
                    ));
                    continue;
                };
                if value["type"] != format!("Transform<{child},{parent}>")
                    || parents.get(child).and_then(|declared| *declared) != Some(parent)
                {
                    issues.push(issue(
                        "S1203",
                        "identity transform must source its direct child-to-parent edge",
                        component_name,
                    ));
                    continue;
                }
                used_edges.insert((child, parent));
                sources
                    .entry((child, parent))
                    .or_default()
                    .push(format!("{instance}.{name}"));
            }
        }
    }
    for (child, parent) in used_edges {
        match sources.get(&(child, parent)) {
            None => issues.push(issue(
                "S1201",
                format!("transform edge {child} -> {parent} has no source"),
                child,
            )),
            Some(paths) if paths.len() == 1 => {}
            Some(paths) => {
                let subject = paths
                    .last()
                    .and_then(|path| path.split_once('.'))
                    .and_then(|(instance, _)| instances.get(instance).copied())
                    .unwrap_or(child);
                issues.push(issue(
                    "S1202",
                    format!("transform edge {child} -> {parent} has multiple sources"),
                    subject,
                ));
            }
        }
    }
}

pub(super) fn license_term(
    term: &spdx::LicenseReq,
    product: &Value,
    component: &Value,
    link: &str,
) -> Result<(), (&'static str, &'static str)> {
    if term.addition.is_some() {
        return Err((
            "S1305",
            "SPDX exception is outside the declared license lattice",
        ));
    }
    let license = &component["license"];
    let proprietary_distribution = product["distribution"] == "proprietary-binary";
    let commercial = product["commercial"] == true;
    let isolation = component["isolation"].as_str().unwrap_or("");
    let name = match &term.license {
        spdx::LicenseItem::Spdx { id, .. } => id.name,
        spdx::LicenseItem::Other(reference) => {
            let name = reference.lic_ref.as_str();
            if commercial && name.to_ascii_lowercase().contains("noncommercial")
                || commercial
                    && name.to_ascii_lowercase().contains("nvidia")
                    && name.to_ascii_lowercase().contains("nc")
            {
                return Err((
                    "S1304",
                    "non-commercial license conflicts with commercial product",
                ));
            }
            if name.starts_with("Proprietary")
                && license["grant"].as_str().is_some_and(|id| !id.is_empty())
            {
                return Ok(());
            }
            return Err((
                "S1305",
                "unknown or proprietary LicenseRef needs a declared grant",
            ));
        }
    };
    if (name.contains("-NC-") || name.contains("NonCommercial")) && commercial {
        return Err((
            "S1304",
            "non-commercial license conflicts with commercial product",
        ));
    }
    if matches!(
        name,
        "Apache-2.0" | "BSD-2-Clause" | "BSD-3-Clause" | "MIT" | "ISC" | "Zlib"
    ) {
        return Ok(());
    }
    if !proprietary_distribution {
        return if name.starts_with("GPL-")
            || name.starts_with("AGPL-")
            || name.starts_with("LGPL-")
            || name == "MPL-2.0"
        {
            Ok(())
        } else {
            Err(("S1305", "license is outside the declared lattice"))
        };
    }
    if name.starts_with("AGPL-") {
        if license["network_exposed"] != false {
            return Err(("S1301", "AGPL component must declare network_exposed false"));
        }
        if isolation != "separate-process" || link != "process" || license["modified"] != false {
            return Err((
                "S1300",
                "AGPL component must be a separate unmodified process",
            ));
        }
        return Ok(());
    }
    if name.starts_with("GPL-") {
        return if isolation == "separate-process"
            && link == "process"
            && license["modified"] == false
        {
            Ok(())
        } else {
            Err((
                "S1300",
                "GPL component must be a separate unmodified process",
            ))
        };
    }
    if name.starts_with("LGPL-") {
        return if (isolation == "in-process" && link == "dynamic" && license["replaceable"] == true)
            || (isolation == "separate-process" && link == "process")
        {
            Ok(())
        } else {
            Err((
                "S1302",
                "LGPL in-process component must be dynamic and replaceable",
            ))
        };
    }
    if name == "MPL-2.0" {
        return if license["modified"] == false {
            Ok(())
        } else {
            Err(("S1303", "MPL files must be declared unmodified"))
        };
    }
    Err(("S1305", "license is outside the declared lattice"))
}

fn license_checks(document: &Value, issues: &mut Vec<Issue>) {
    let components = components(document);
    let instances = instances(document);
    let Some(product_id) = document["system"]["product"].as_str() else {
        return;
    };
    let Some(product) = document["products"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|product| product["id"] == product_id)
    else {
        return;
    };
    for component_name in instances.values() {
        let Some(component) = components.get(component_name) else {
            continue;
        };
        let expression = component["license"]["expression"].as_str().unwrap_or("");
        let Ok(expression) = spdx::Expression::parse(expression) else {
            issues.push(issue("S1305", "invalid SPDX expression", component_name));
            continue;
        };
        let implementation = &component["implementation"];
        let actual_link = if implementation["kind"] == "invar" {
            Some("static")
        } else {
            implementation["target"].as_str().and_then(|target| {
                components
                    .get(target)?
                    .get("license")?
                    .get("link")?
                    .as_str()
            })
        };
        let declared_link = component["license"]["link"].as_str();
        if declared_link.is_some() && actual_link.is_some() && declared_link != actual_link {
            issues.push(issue(
                "S1305",
                "declared link mode differs from implementation",
                component_name,
            ));
            continue;
        }
        let link = declared_link.or(actual_link).unwrap_or("");
        let mut rejected = Vec::new();
        let allowed =
            expression.evaluate(|term| match license_term(term, product, component, link) {
                Ok(()) => true,
                Err(failure) => {
                    rejected.push(failure);
                    false
                }
            });
        if !allowed {
            let (code, message) = rejected
                .first()
                .copied()
                .unwrap_or(("S1305", "license is not allowed"));
            issues.push(issue(code, message, component_name));
        }
    }
}

pub fn check(document: &Value) -> Vec<Issue> {
    let mut issues = Vec::new();
    time_checks(document, &mut issues);
    frame_checks(document, &mut issues);
    resource_checks(document, &mut issues);
    license_checks(document, &mut issues);
    issues
}
