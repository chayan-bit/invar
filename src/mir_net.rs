//! Typed MIR region connections used by the M1 interaction extraction path.
//!
//! Operand values live in explicit typed wires. The retained MIR skeleton has
//! empty operand ports, so extraction must consume the graph connections.

use std::collections::{BTreeMap, HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use crate::{front::Type, mir};

pub const SCHEMA_VERSION: u32 = 1;
const EMPTY_PORT: usize = usize::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    Instruction {
        function: usize,
        block: usize,
        index: usize,
        port: usize,
    },
    Terminator {
        function: usize,
        block: usize,
        port: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wire {
    pub site: Site,
    pub producer: mir::ValueId,
    pub ty: Type,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub functions: usize,
    pub blocks: usize,
    pub instructions: usize,
    pub wires: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Schema,
    Limit,
    MissingWire(Site),
    DuplicateWire(Site),
    ExtraWire(Site),
    Type(Site),
    StaleProof,
    Mir(mir::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Proof {
    fingerprint: u64,
}

#[cfg(test)]
impl Proof {
    /// Test-only tamper: a proof whose fingerprint differs from this one in
    /// its lowest bit. Powers of ten are even, so `n ^ 1` never crosses one and
    /// the decimal Debug width is preserved exactly.
    pub(crate) fn with_flipped_fingerprint(self) -> Self {
        Self {
            fingerprint: self.fingerprint ^ 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Net {
    pub schema_version: u32,
    skeleton: mir::Program,
    pub wires: Vec<Wire>,
    pub revision: u64,
    pub source_identity: u64,
    limits: Limits,
}

fn fingerprint<T: std::fmt::Debug>(value: &T) -> u64 {
    let mut hash = DefaultHasher::new();
    format!("{value:?}").hash(&mut hash);
    hash.finish()
}

fn operation_ports(op: &mut mir::Operation) -> Vec<&mut mir::ValueId> {
    use mir::Operation;
    match op {
        Operation::Array(items) | Operation::Call { args: items, .. } => items.iter_mut().collect(),
        Operation::Unary { value, .. }
        | Operation::Bind { value, .. }
        | Operation::Store { value, .. }
        | Operation::ForNext(value) => vec![value],
        Operation::Binary { left, right, .. } => vec![left, right],
        Operation::Index { array, index } => vec![array, index],
        Operation::Field { record, .. } => vec![record],
        Operation::VariantTag(value)
        | Operation::VariantPayload { value, .. }
        | Operation::VariantTakePayload { value, .. }
        | Operation::DropValue(value) => vec![value],
        Operation::Variant { payload, .. } => payload.iter_mut().collect(),
        Operation::Record { fields, .. } => fields.iter_mut().map(|(_, value)| value).collect(),
        Operation::StoreIndex { index, value, .. } => vec![index, value],
        Operation::Tick
        | Operation::I32(_)
        | Operation::U32(_)
        | Operation::F32(_)
        | Operation::F64(_)
        | Operation::Bool(_)
        | Operation::Blob { .. }
        | Operation::Read(_)
        | Operation::Move(_)
        | Operation::MoveField { .. }
        | Operation::ArrayRef { .. }
        | Operation::RecordRef { .. }
        | Operation::Length(_)
        | Operation::Drop(_) => Vec::new(),
    }
}

fn terminator_ports(term: &mut mir::Terminator) -> Vec<&mut mir::ValueId> {
    match term {
        mir::Terminator::Goto { args, .. } => args.iter_mut().collect(),
        mir::Terminator::Branch { condition, .. } => vec![condition],
        mir::Terminator::Return { value, .. } => value.iter_mut().collect(),
    }
}

fn visit_ports(
    program: &mut mir::Program,
    mut visit: impl FnMut(Site, &mut mir::ValueId) -> Result<(), Error>,
) -> Result<(), Error> {
    for (function, body) in program.functions.iter_mut().enumerate() {
        for (block, basic) in body.blocks.iter_mut().enumerate() {
            for (index, instruction) in basic.instructions.iter_mut().enumerate() {
                for (port, value) in operation_ports(&mut instruction.kind)
                    .into_iter()
                    .enumerate()
                {
                    visit(
                        Site::Instruction {
                            function,
                            block,
                            index,
                            port,
                        },
                        value,
                    )?;
                }
            }
            if let Some(end) = &mut basic.end {
                for (port, value) in terminator_ports(end).into_iter().enumerate() {
                    visit(
                        Site::Terminator {
                            function,
                            block,
                            port,
                        },
                        value,
                    )?;
                }
            }
        }
    }
    Ok(())
}

impl Net {
    fn check_limits(program: &mir::Program, limits: Limits, wires: usize) -> Result<(), Error> {
        if program.functions.len() > limits.functions || wires > limits.wires {
            return Err(Error::Limit);
        }
        let mut blocks = 0usize;
        let mut instructions = 0usize;
        let mut ports = 0usize;
        for function in &program.functions {
            blocks = blocks.saturating_add(function.blocks.len());
            if blocks > limits.blocks {
                return Err(Error::Limit);
            }
            for block in &function.blocks {
                instructions = instructions.saturating_add(block.instructions.len());
                if instructions > limits.instructions {
                    return Err(Error::Limit);
                }
                for instruction in &block.instructions {
                    use mir::Operation as O;
                    let count = match &instruction.kind {
                        O::Array(items) | O::Call { args: items, .. } => items.len(),
                        O::Unary { .. } | O::Bind { .. } | O::Store { .. } | O::ForNext(_) => 1,
                        O::Binary { .. } | O::Index { .. } | O::StoreIndex { .. } => 2,
                        _ => 0,
                    };
                    ports = ports.saturating_add(count);
                    if ports > limits.wires {
                        return Err(Error::Limit);
                    }
                }
                if let Some(end) = &block.end {
                    let count = match end {
                        mir::Terminator::Goto { args, .. } => args.len(),
                        mir::Terminator::Branch { .. } => 1,
                        mir::Terminator::Return { value, .. } => usize::from(value.is_some()),
                    };
                    ports = ports.saturating_add(count);
                    if ports > limits.wires {
                        return Err(Error::Limit);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn encode(program: &mir::Program, limits: Limits) -> Result<Self, Error> {
        Self::check_limits(program, limits, 0)?;
        mir::verify(program).map_err(Error::Mir)?;
        let mut skeleton = program.clone();
        let mut types = Vec::<HashMap<mir::ValueId, Type>>::new();
        for function in &program.functions {
            let mut values = HashMap::new();
            for block in &function.blocks {
                values.extend(block.params.iter().cloned());
                for instruction in &block.instructions {
                    if let Some((id, ty)) = &instruction.output {
                        values.insert(*id, ty.clone());
                    }
                }
            }
            types.push(values);
        }
        let mut wires = Vec::new();
        visit_ports(&mut skeleton, |site, value| {
            let function = match site {
                Site::Instruction { function, .. } | Site::Terminator { function, .. } => function,
            };
            let ty = types[function]
                .get(value)
                .cloned()
                .ok_or(Error::Type(site))?;
            wires.push(Wire {
                site,
                producer: *value,
                ty,
            });
            if wires.len() > limits.wires {
                return Err(Error::Limit);
            }
            *value = EMPTY_PORT;
            Ok(())
        })?;
        let net = Self {
            schema_version: SCHEMA_VERSION,
            skeleton,
            wires,
            revision: 0,
            source_identity: {
                let digest = program.contract_digest();
                u64::from_le_bytes(digest[..8].try_into().expect("fixed SHA256 digest width"))
            },
            limits,
        };
        net.verify()?;
        Ok(net)
    }

    fn rebuild(&self) -> Result<mir::Program, Error> {
        if self.schema_version != SCHEMA_VERSION
            || self.skeleton.schema_version != mir::SCHEMA_VERSION
        {
            return Err(Error::Schema);
        }
        Self::check_limits(&self.skeleton, self.limits, self.wires.len())?;
        let mut connections = BTreeMap::new();
        for wire in &self.wires {
            if connections.insert(wire.site, wire).is_some() {
                return Err(Error::DuplicateWire(wire.site));
            }
        }
        let mut program = self.skeleton.clone();
        let mut types = Vec::<HashMap<mir::ValueId, Type>>::new();
        for function in &program.functions {
            let mut values = HashMap::new();
            for block in &function.blocks {
                values.extend(block.params.iter().cloned());
                for instruction in &block.instructions {
                    if let Some((id, ty)) = &instruction.output {
                        values.insert(*id, ty.clone());
                    }
                }
            }
            types.push(values);
        }
        visit_ports(&mut program, |site, value| {
            if *value != EMPTY_PORT {
                return Err(Error::Schema);
            }
            let wire = connections.remove(&site).ok_or(Error::MissingWire(site))?;
            let function = match site {
                Site::Instruction { function, .. } | Site::Terminator { function, .. } => function,
            };
            if types[function].get(&wire.producer) != Some(&wire.ty) {
                return Err(Error::Type(site));
            }
            *value = wire.producer;
            Ok(())
        })?;
        if let Some((&site, _)) = connections.first_key_value() {
            return Err(Error::ExtraWire(site));
        }
        mir::verify(&program).map_err(Error::Mir)?;
        Ok(program)
    }

    pub fn verify(&self) -> Result<Proof, Error> {
        self.rebuild()?;
        Ok(Proof {
            fingerprint: fingerprint(self),
        })
    }

    pub fn extract(&self, proof: &Proof) -> Result<mir::Program, Error> {
        if proof.fingerprint != fingerprint(self) {
            return Err(Error::StaleProof);
        }
        self.rebuild()
    }
}
