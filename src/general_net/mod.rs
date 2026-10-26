//! GNET checked executable-net/compiler boundary, independently versioned.
//!
//! Reuses MIR's typed data wires and checked ownership/control facts. Explicit
//! agents and block activation routes expose the sequential operational rules.
//! A proof admits extraction of program-specific control/data MIR for compiled
//! lowering; it never certifies GPU coverage, concurrency, or performance.

use crate::{
    check::CheckedProgram,
    front::{Span, Type},
    mir, mir_net, ownership,
};

pub mod compiled;
mod compiler_child;
pub mod coverage;
pub mod device_host;
pub mod law_collective;
pub mod normal_form;
pub mod numeric_owned_resume_host;
pub mod oracle;
mod owned_numeric_decode;
mod owned_numeric_task;
pub mod path_sum;
pub mod provenance;
pub mod regions;
pub mod task;

pub const CONTRACT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub functions: usize,
    pub blocks: usize,
    pub instructions: usize,
    pub wires: usize,
    pub literal_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            functions: 128,
            blocks: 4096,
            instructions: 100_000,
            wires: 200_000,
            literal_bytes: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockSite {
    pub function: usize,
    pub block: usize,
}

/// A token exists only for the selected block activation in one call frame.
/// Position advances once after completion; call completion precedes its next
/// position. A loop routes a fresh activation to the target block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub block: BlockSite,
    pub position: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallTarget {
    Function(usize),
    Host {
        function: usize,
        reference_oracle: String,
    },
    Intrinsic(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Agent {
    pub token: Token,
    pub next: Token,
    pub rule: coverage::Rule,
    pub observation: coverage::Observation,
    pub inputs: Vec<mir::ValueId>,
    pub output: Option<(mir::ValueId, Type)>,
    pub span: Span,
    pub call: Option<CallTarget>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    Goto {
        target: BlockSite,
        args: Vec<mir::ValueId>,
    },
    /// Only the selected route receives an activation token.
    Branch {
        condition: mir::ValueId,
        yes: BlockSite,
        no: BlockSite,
    },
    Return {
        value: Option<mir::ValueId>,
        cleanup: Vec<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Control {
    pub token: Token,
    pub params: Vec<(mir::ValueId, Type)>,
    pub rule: coverage::Rule,
    pub span: Span,
    pub route: Route,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Version,
    Limit,
    Ownership(crate::front::FrontError),
    Mir(mir::Error),
    Graph(mir_net::Error),
    Annotation,
    StaleProof,
}

/// Editable candidate. Every compilation/oracle boundary checks an opaque
/// proof against current wires and annotations. No mutable verified handle.
#[derive(Clone, Debug)]
pub struct Net {
    pub contract_version: u32,
    pub graph: mir_net::Net,
    pub agents: Vec<Agent>,
    pub control: Vec<Control>,
    limits: Limits,
}

#[derive(Clone, Debug)]
pub struct Proof {
    graph: mir_net::Proof,
    agents: Vec<Agent>,
    control: Vec<Control>,
}

#[cfg(test)]
impl Proof {
    /// Test-only tamper: the same proof with a graph fingerprint that differs
    /// but has exactly the same Debug width (see `mir_net::Proof`).
    pub(crate) fn with_flipped_graph_fingerprint(&self) -> Self {
        Self {
            graph: self.graph.with_flipped_fingerprint(),
            agents: self.agents.clone(),
            control: self.control.clone(),
        }
    }
}

fn check_limits(program: &mir::Program, limits: Limits) -> Result<(), Error> {
    if program.functions.len() > limits.functions {
        return Err(Error::Limit);
    }
    let (mut blocks, mut instructions, mut wires, mut bytes) = (0usize, 0usize, 0usize, 0usize);
    for function in &program.functions {
        blocks = blocks
            .checked_add(function.blocks.len())
            .ok_or(Error::Limit)?;
        if blocks > limits.blocks {
            return Err(Error::Limit);
        }
        for block in &function.blocks {
            instructions = instructions
                .checked_add(block.instructions.len())
                .ok_or(Error::Limit)?;
            if instructions > limits.instructions {
                return Err(Error::Limit);
            }
            for instruction in &block.instructions {
                // Count variable arities before allocating an operand vector.
                use mir::Operation as O;
                let count = match &instruction.kind {
                    O::Array(items) | O::Call { args: items, .. } => items.len(),
                    O::Record { fields, .. } => fields.len(),
                    O::Variant { payload, .. } => usize::from(payload.is_some()),
                    op => coverage::operands(op).len(),
                };
                wires = wires.checked_add(count).ok_or(Error::Limit)?;
                if let O::Blob { bytes: literal, .. } = &instruction.kind {
                    bytes = bytes.checked_add(literal.len()).ok_or(Error::Limit)?;
                }
                if wires > limits.wires || bytes > limits.literal_bytes {
                    return Err(Error::Limit);
                }
            }
            if let Some(term) = &block.end {
                let count = match term {
                    mir::Terminator::Goto { args, .. } => args.len(),
                    mir::Terminator::Branch { .. } => 1,
                    mir::Terminator::Return { value, .. } => usize::from(value.is_some()),
                };
                wires = wires.checked_add(count).ok_or(Error::Limit)?;
                if wires > limits.wires {
                    return Err(Error::Limit);
                }
            }
        }
    }
    Ok(())
}

fn annotations(program: &mir::Program) -> Result<(Vec<Agent>, Vec<Control>), Error> {
    let mut agents = Vec::new();
    let mut control = Vec::new();
    for (f, function) in program.functions.iter().enumerate() {
        for (b, block) in function.blocks.iter().enumerate() {
            let site = BlockSite {
                function: f,
                block: b,
            };
            for (index, instruction) in block.instructions.iter().enumerate() {
                let (rule, mut observation) = coverage::operation(&instruction.kind);
                let call = if let mir::Operation::Call { name, .. } = &instruction.kind {
                    match program.functions.iter().position(|body| body.name == *name) {
                        Some(target) => match &program.functions[target].foreign_oracle {
                            Some(oracle) => {
                                observation = coverage::Observation::HostBoundary;
                                Some(CallTarget::Host {
                                    function: target,
                                    reference_oracle: oracle.clone(),
                                })
                            }
                            None => Some(CallTarget::Function(target)),
                        },
                        None => Some(CallTarget::Intrinsic(name.clone())),
                    }
                } else {
                    None
                };
                agents.push(Agent {
                    token: Token {
                        block: site,
                        position: index,
                    },
                    next: Token {
                        block: site,
                        position: index + 1,
                    },
                    rule,
                    observation,
                    inputs: coverage::operands(&instruction.kind),
                    output: instruction.output.clone(),
                    span: instruction.span,
                    call,
                });
            }
            let term = block.end.as_ref().ok_or_else(|| {
                Error::Mir(mir::Error {
                    code: "MIR_INVALID",
                    message: "missing executable-net terminator".to_owned(),
                })
            })?;
            let (rule, _, span) = coverage::terminator(term);
            let target = |block| BlockSite { function: f, block };
            let route = match term {
                mir::Terminator::Goto {
                    target: block,
                    args,
                    ..
                } => Route::Goto {
                    target: target(*block),
                    args: args.clone(),
                },
                mir::Terminator::Branch {
                    condition, yes, no, ..
                } => Route::Branch {
                    condition: *condition,
                    yes: target(*yes),
                    no: target(*no),
                },
                mir::Terminator::Return { value, cleanup, .. } => Route::Return {
                    value: *value,
                    cleanup: cleanup.clone(),
                },
            };
            control.push(Control {
                token: Token {
                    block: site,
                    position: block.instructions.len(),
                },
                params: block.params.clone(),
                rule,
                span,
                route,
            });
        }
    }
    Ok((agents, control))
}

impl Net {
    /// Full checked-source structural lowering through existing typed MIR.
    /// Ownership is checked here rather than presumed from type checking.
    pub fn lower(source: &CheckedProgram, limits: Limits) -> Result<Self, Error> {
        ownership::check(source).map_err(Error::Ownership)?;
        let program = mir::lower(source).map_err(Error::Mir)?;
        Self::encode(&program, limits)
    }

    /// Check budgets before cloning or building the graph; retain v1 MIR/net
    /// artifacts unchanged. MIR verifies dominance, types, loans and cleanup.
    pub fn encode(program: &mir::Program, limits: Limits) -> Result<Self, Error> {
        check_limits(program, limits)?;
        mir::verify(program).map_err(Error::Mir)?;
        let graph = mir_net::Net::encode(
            program,
            mir_net::Limits {
                functions: limits.functions,
                blocks: limits.blocks,
                instructions: limits.instructions,
                wires: limits.wires,
            },
        )
        .map_err(Error::Graph)?;
        let (agents, control) = annotations(program)?;
        Ok(Self {
            contract_version: CONTRACT_VERSION,
            graph,
            agents,
            control,
            limits,
        })
    }

    pub fn verify(&self) -> Result<Proof, Error> {
        if self.contract_version != CONTRACT_VERSION {
            return Err(Error::Version);
        }
        if self.agents.len() > self.limits.instructions || self.control.len() > self.limits.blocks {
            return Err(Error::Limit);
        }
        let graph = self.graph.verify().map_err(Error::Graph)?;
        let program = self.graph.extract(&graph).map_err(Error::Graph)?;
        check_limits(&program, self.limits)?;
        let (agents, control) = annotations(&program)?;
        if self.agents != agents || self.control != control {
            return Err(Error::Annotation);
        }
        Ok(Proof {
            graph,
            agents,
            control,
        })
    }

    /// Checked net-to-compiled boundary. Returned MIR is input for the existing
    /// native emitter or future program-specific GPU emitters. It is not
    /// execution or evidence that a backend implements every admitted form.
    pub fn compiled_mir(&self, proof: &Proof) -> Result<mir::Program, Error> {
        if self.contract_version != CONTRACT_VERSION
            || self.agents != proof.agents
            || self.control != proof.control
        {
            return Err(Error::StaleProof);
        }
        self.graph
            .extract(&proof.graph)
            .map_err(|error| match error {
                mir_net::Error::StaleProof => Error::StaleProof,
                other => Error::Graph(other),
            })
    }
}
