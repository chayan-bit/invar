//! Checked, canonical compile-time system description.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::front::{
    self, FrontError, Mode, Param, Program, Span, SystemClause, SystemItem, SystemItemKind,
    SystemQuantity, Type,
};

mod compiler;
mod components;
pub mod foreign;
mod foreign_layout;
mod generate;
pub mod graph;
mod imports;
mod mount;
pub mod planner_scene;
mod products;
mod quantities;
mod robot;
pub mod ros2;
mod ros2_tendon;
mod run_report;
pub mod sequencer;
mod substep_admission;
pub mod tool_projection;
mod urdf;
mod xml;

pub use compiler::{compile, compile_at, compile_manifest};
use components::*;
use foreign_layout::*;
use imports::*;
use products::*;
use quantities::*;

pub use generate::{generate_mjcf, generate_ros2_mjcf, generate_urdf};

const MAX_ITEMS: usize = 256;
const MAX_IR_BYTES: usize = 1_048_576;
pub const SCHEMA_JSON: &str = include_str!("../schema/invar-system-v12.schema.json");

#[derive(Clone, Debug, PartialEq)]
pub struct SystemIr {
    document: Value,
    warnings: Vec<(String, Span)>,
    robot_warnings: Vec<(String, Span)>,
    origins: BTreeMap<String, Span>,
    checked_source: bool,
    assets: BTreeMap<String, BTreeMap<String, Vec<u8>>>,
}

impl SystemIr {
    pub fn canonical_json(&self) -> String {
        self.document.to_string()
    }
    pub fn warnings(&self) -> &[(String, Span)] {
        &self.warnings
    }
    pub fn robot_warnings(&self) -> &[(String, Span)] {
        &self.robot_warnings
    }

    /// A source-compiled document that passed all graph checks.
    pub fn checked_document(&self) -> Result<&Value, &'static str> {
        if !self.checked_source || !graph::check(&self.document).is_empty() {
            return Err("system IR has not passed the graph checks");
        }
        Ok(&self.document)
    }

    /// True for a source-compiled model, which can be graph checked; this does not mean it passed.
    pub fn graph_ready(&self) -> bool {
        self.checked_source
    }

    pub fn graph_issues(&self) -> Vec<(&'static str, String, Span)> {
        if !self.graph_ready() {
            return Vec::new();
        }
        let mut issues: Vec<_> = graph::check(&self.document)
            .into_iter()
            .map(|issue| {
                let span = self
                    .origins
                    .get(&issue.subject)
                    .copied()
                    .unwrap_or(Span { start: 0, end: 0 });
                (issue.code, issue.message, span)
            })
            .collect();
        issues.sort_by_key(|(code, _, span)| {
            let family = if code.as_bytes().get(1..3) == Some(b"11") {
                0
            } else if *code < "S1250" {
                1
            } else if *code < "S1300" {
                2
            } else {
                3
            };
            (family, span.start)
        });
        issues
    }

    pub fn report(&self) -> Result<String, &'static str> {
        self.checked_document()?;
        let system = self.document["system"]["name"]
            .as_str()
            .unwrap_or("<unnamed>");
        let instances = self.document["system"]["instances"]
            .as_array()
            .map_or(0, Vec::len);
        let connections = self.document["connections"].as_array().map_or(0, Vec::len);
        let flows = self.document["flows"].as_array().map_or(0, Vec::len);
        let schema = self.document["schema"].as_str().unwrap_or("");
        let mut report = format!(
            "# Invar system check: {system}\n\n- IR: {schema}\n- Instances: {instances}\n- Connections: {connections}\n- Flows: {flows}\n- Timing: passed\n- Frame tree: passed\n- Resources: passed\n- License and isolation: passed (SPDX list {spdx_list})\n\nDeadlines and WCET values are declared budgets, not measurements, until R1-06 evidence exists.\n\nBus bandwidth models declared port rate × canonical C layout bytes × 8; it excludes serialization and network overhead.\n\nThe license check is an engineering check, not legal advice.\n",
            spdx_list = spdx::identifiers::VERSION,
        );
        for (message, _) in &self.robot_warnings {
            report.push_str(&format!("- S1604 warning: {message}\n"));
        }
        Ok(report)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() > MAX_IR_BYTES {
            return Err("system IR size limit exceeded");
        }
        let document: Value =
            serde_json::from_slice(bytes).map_err(|_| "invalid system IR JSON")?;
        if document["schema"] != "invar-system-v12" {
            return Err("unsupported system IR version");
        }
        let schema: Value =
            serde_json::from_str(SCHEMA_JSON).map_err(|_| "invalid system IR schema")?;
        if !jsonschema::is_valid(&schema, &document) {
            return Err("invalid system IR model");
        }
        Ok(Self {
            document,
            warnings: Vec::new(),
            robot_warnings: Vec::new(),
            origins: BTreeMap::new(),
            checked_source: false,
            assets: BTreeMap::new(),
        })
    }
}

fn error(code: &'static str, message: impl Into<String>, span: Span) -> FrontError {
    FrontError::source(code, message, span)
}

pub fn import_mjcf(path: &Path) -> Result<(String, Vec<String>), FrontError> {
    xml::import_mjcf(path)
        .map_err(|issue| error(issue.code, issue.message, Span { start: 0, end: 0 }))
}

pub fn import_urdf(path: &Path) -> Result<(String, Vec<String>), FrontError> {
    urdf::import_urdf(path)
        .map_err(|issue| error(issue.code, issue.message, Span { start: 0, end: 0 }))
}

pub fn canonical_mjcf(path: &Path) -> Result<String, FrontError> {
    xml::load_mjcf(path)
        .map(|tree| tree.render())
        .map_err(|issue| error(issue.code, issue.message, Span { start: 0, end: 0 }))
}

pub fn canonical_urdf(path: &Path) -> Result<String, FrontError> {
    urdf::canonical_urdf(path)
        .map_err(|issue| error(issue.code, issue.message, Span { start: 0, end: 0 }))
}
