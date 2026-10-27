//! Versioned typed control-flow boundary for the bootstrap language.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    check::CheckedProgram,
    front::{
        BinaryOp, Block, ElseArm, Expr, ExprKind, Mode, Place, Span, Stmt, StmtKind, Type, UnaryOp,
    },
};

pub const SCHEMA_VERSION: u32 = 1;
/// Structural MIR limits, checked by [`verify`] before any superlinear analysis.
pub const MAX_FUNCTIONS: usize = 128;
pub const MAX_BLOCKS_PER_FUNCTION: usize = 4_096;
pub const MAX_LOCALS_PER_FUNCTION: usize = 8_192;
pub const MAX_INSTRUCTIONS: usize = 100_000;
pub type BlockId = usize;
pub type ValueId = usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: &'static str,
    pub message: String,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error {
        code: "MIR_INVALID",
        message: message.into(),
    }
}

#[derive(Clone, Debug)]
pub struct Program {
    pub schema_version: u32,
    pub numeric_policy: crate::numeric_contract::NumericPolicy,
    pub layout_version: u32,
    pub tensor_contracts: Vec<crate::tensor::Contract>,
    pub lowered_folds: crate::numeric_contract::LoweredFolds,
    pub functions: Vec<Function>,
    pub records: Vec<crate::front::RecordDecl>,
    pub variants: Vec<crate::front::VariantDecl>,
}

impl Program {
    /// The first function named `name`.
    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Stable typed source identity: hash-map iteration never defines cache keys.
    pub fn contract_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        let header = format!(
            "{:?}",
            (
                self.schema_version,
                self.numeric_policy,
                self.layout_version,
                &self.tensor_contracts,
                &self.records,
                &self.variants
            )
        );
        hash.update((header.len() as u64).to_le_bytes());
        hash.update(header.as_bytes());
        for function in &self.functions {
            hash.update(crate::tensor::function_fingerprint(function));
        }
        hash.finalize().into()
    }
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub foreign_oracle: Option<String>,
    pub params: Vec<Parameter>,
    pub result: Type,
    pub return_from: Option<String>,
    pub span: Span,
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
    pub locals: HashMap<String, Local>,
    pub declaration_order: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub ty: Type,
    pub mode: Option<Mode>,
}

#[derive(Clone, Debug)]
pub struct Local {
    pub ty: Type,
    pub mutable: bool,
    pub mode: Option<Mode>,
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub params: Vec<(ValueId, Type)>,
    pub instructions: Vec<Instruction>,
    pub end: Option<Terminator>,
}

#[derive(Clone, Debug)]
pub struct Instruction {
    pub output: Option<(ValueId, Type)>,
    pub kind: Operation,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Operation {
    Tick,
    I32(i32),
    U32(u32),
    F32(crate::numeric_contract::F32Bits),
    F64(crate::numeric_contract::F64Bits),
    Bool(bool),
    Blob {
        bytes: Vec<u8>,
        text: bool,
    },
    Read(String),
    Move(String),
    MoveField {
        name: String,
        index: usize,
    },
    ArrayRef {
        name: String,
        loan: Option<Mode>,
    },
    RecordRef {
        name: String,
        loan: Option<Mode>,
    },
    Array(Vec<ValueId>),
    Record {
        name: String,
        fields: Vec<(usize, ValueId)>,
    },
    Field {
        record: ValueId,
        index: usize,
    },
    Variant {
        name: String,
        tag: usize,
        payload: Option<ValueId>,
    },
    VariantTag(ValueId),
    VariantPayload {
        value: ValueId,
        tag: usize,
    },
    VariantTakePayload {
        value: ValueId,
        tag: usize,
    },
    DropValue(ValueId),
    Unary {
        op: UnaryOp,
        value: ValueId,
        operator: Span,
    },
    Binary {
        op: BinaryOp,
        left: ValueId,
        right: ValueId,
        operator: Span,
    },
    Index {
        array: ValueId,
        index: ValueId,
    },
    Length(String),
    Call {
        name: String,
        args: Vec<ValueId>,
    },
    Bind {
        name: String,
        value: ValueId,
    },
    Store {
        name: String,
        value: ValueId,
    },
    StoreIndex {
        name: String,
        index: ValueId,
        value: ValueId,
    },
    Drop(Vec<String>),
    ForNext(ValueId),
}

#[derive(Clone, Debug)]
pub enum Terminator {
    Goto {
        target: BlockId,
        args: Vec<ValueId>,
        span: Span,
    },
    Branch {
        condition: ValueId,
        yes: BlockId,
        no: BlockId,
        span: Span,
    },
    Return {
        value: Option<ValueId>,
        cleanup: Vec<String>,
        span: Span,
    },
}

struct Builder<'a> {
    source: &'a CheckedProgram,
    result: Type,
    types: HashMap<Span, Type>,
    blocks: Vec<BasicBlock>,
    current: BlockId,
    next_value: ValueId,
    scopes: Vec<Vec<String>>,
    locals: HashMap<String, Local>,
    declaration_order: Vec<String>,
}

include!("mir/builder.rs");

/// Lower checked source to explicit blocks. Construction is followed by verification.
pub fn lower(source: &CheckedProgram) -> Result<Program, Error> {
    let mut functions = Vec::new();
    for function in &source.program.functions {
        let mut builder = Builder {
            source,
            result: function.result.clone(),
            types: source.types_for(&function.name),
            blocks: Vec::new(),
            current: 0,
            next_value: 0,
            scopes: vec![
                function
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect(),
            ],
            locals: function
                .params
                .iter()
                .map(|param| {
                    (
                        param.name.clone(),
                        Local {
                            ty: param.ty.clone(),
                            mutable: false,
                            mode: param.mode,
                        },
                    )
                })
                .collect(),
            declaration_order: function
                .params
                .iter()
                .map(|param| param.name.clone())
                .collect(),
        };
        builder.new_block();
        builder.block(&function.body)?;
        if builder.blocks[builder.current].end.is_none() {
            builder.end(Terminator::Return {
                value: None,
                cleanup: builder.cleanup(),
                span: function.body.span,
            });
        }
        // An if whose both arms return leaves an unreachable join. It still has a terminator.
        for block in &mut builder.blocks {
            if block.end.is_none() {
                block.end = Some(Terminator::Return {
                    value: None,
                    cleanup: Vec::new(),
                    span: function.body.span,
                });
            }
        }
        let mut lowered = Function {
            name: function.name.clone(),
            foreign_oracle: None,
            params: function
                .params
                .iter()
                .map(|param| Parameter {
                    name: param.name.clone(),
                    ty: param.ty.clone(),
                    mode: param.mode,
                })
                .collect(),
            result: function.result.clone(),
            return_from: function.return_from.clone(),
            span: function.span,
            entry: 0,
            blocks: builder.blocks,
            locals: builder.locals,
            declaration_order: builder.declaration_order,
        };
        if crate::check::is_frame_helper(&lowered.name) {
            let origin = Span { start: 0, end: 1 };
            lowered.span = origin;
            for block in &mut lowered.blocks {
                for instruction in &mut block.instructions {
                    instruction.span = origin;
                    match &mut instruction.kind {
                        Operation::Unary { operator, .. } | Operation::Binary { operator, .. } => {
                            *operator = origin;
                        }
                        _ => {}
                    }
                }
                if let Some(end) = &mut block.end {
                    match end {
                        Terminator::Goto { span, .. }
                        | Terminator::Branch { span, .. }
                        | Terminator::Return { span, .. } => *span = origin,
                    }
                }
            }
        }
        functions.push(lowered);
    }
    for foreign in &source.program.foreign_functions {
        functions.push(Function {
            name: foreign.name.clone(),
            foreign_oracle: Some(foreign.oracle.clone()),
            params: foreign
                .params
                .iter()
                .map(|param| Parameter {
                    name: param.name.clone(),
                    ty: param.ty.clone(),
                    mode: param.mode,
                })
                .collect(),
            result: foreign.result.clone(),
            return_from: None,
            span: foreign.span,
            entry: 0,
            blocks: Vec::new(),
            locals: HashMap::new(),
            declaration_order: foreign
                .params
                .iter()
                .map(|param| param.name.clone())
                .collect(),
        });
    }
    let mut program = Program {
        schema_version: SCHEMA_VERSION,
        numeric_policy: crate::numeric_contract::NumericPolicy::STRICT_F32,
        layout_version: crate::layout_contract::VERSION,
        tensor_contracts: Vec::new(),
        lowered_folds: source.lowered_folds.clone(),
        functions,
        records: source.program.records.clone(),
        variants: source.program.variants.clone(),
    };
    program.tensor_contracts = crate::tensor::annotate(source, &program);
    verify(&program)?;
    Ok(program)
}

include!("mir/verify.rs");

include!("mir/dominance.rs");

#[cfg(test)]
mod tests {
    use super::{BlockId, Dominance, lower};
    use crate::{check, front};

    /// Dominator sets by the textbook fixed point, restricted to reachable blocks.
    fn naive(function: &super::Function, predecessors: &[Vec<BlockId>]) -> Vec<Vec<bool>> {
        let count = function.blocks.len();
        let mut reachable = vec![false; count];
        let mut stack = vec![function.entry];
        while let Some(block) = stack.pop() {
            if !std::mem::replace(&mut reachable[block], true) {
                let end = function.blocks[block].end.as_ref().unwrap();
                stack.extend(end.targets().into_iter().map(|(id, _)| id));
            }
        }
        let mut sets = vec![reachable.clone(); count];
        sets[function.entry] = (0..count).map(|id| id == function.entry).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for block in (0..count).filter(|b| reachable[*b] && *b != function.entry) {
                let mut next = reachable.clone();
                for pred in predecessors[block].iter().filter(|p| reachable[**p]) {
                    for (id, keep) in next.iter_mut().enumerate() {
                        *keep &= sets[*pred][id];
                    }
                }
                next[block] = true;
                if next != sets[block] {
                    sets[block] = next;
                    changed = true;
                }
            }
        }
        (0..count)
            .map(|b| (0..count).map(|d| reachable[b] && sets[b][d]).collect())
            .collect()
    }

    #[test]
    fn dominator_tree_matches_naive_fixed_point() {
        let source = "fn main() -> i32 { var x: i32 = 0; while x < 9 { if x < 3 { x = x + 2; } else { if x == 5 { return x; } else {} x = x + 1; } } for i in 0u32 .. 3u32 { if true { return 1; } else { return 2; } } return x; }";
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = lower(&checked).unwrap();
        for function in &program.functions {
            let mut predecessors = vec![Vec::new(); function.blocks.len()];
            for (id, block) in function.blocks.iter().enumerate() {
                for (target, _) in block.end.as_ref().unwrap().targets() {
                    predecessors[target].push(id);
                }
            }
            let expected = naive(function, &predecessors);
            let dominance = Dominance::new(function, &predecessors);
            for (block, row) in expected.iter().enumerate() {
                for (dominator, dominates) in row.iter().enumerate() {
                    assert_eq!(dominance.dominates(dominator, block), *dominates);
                }
            }
        }
    }
}
