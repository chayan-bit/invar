//! Bootstrap lexer and parser. Spans are UTF-8 byte offsets.

use serde_json::{Value, json};

const MAX_SOURCE_BYTES: usize = 1_048_576;
const MAX_NESTING: usize = 32;
/// Bound on the height of the statement/expression tree along any path.
/// Syntactic nesting is capped at [`MAX_NESTING`], but left-associative
/// operator chains and `else if` chains deepen the tree without nesting, and
/// every later pass walks it recursively. This keeps those walkers stack-safe
/// on a 2 MiB thread in Debug builds (see `tests/robustness_limits.rs`).
const MAX_TREE_HEIGHT: usize = 192;

fn tree_height(child_height: usize) -> Result<usize, FrontError> {
    if child_height >= MAX_TREE_HEIGHT {
        return Err(FrontError::tool(
            "SOURCE_NESTING_LIMIT",
            "syntax tree height limit exceeded",
        ));
    }
    Ok(child_height + 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Source,
    Tool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FrontError {
    pub kind: ErrorKind,
    pub code: &'static str,
    pub message: String,
    pub span: Option<Span>,
    unit_conversion_fix: Option<(Span, &'static str)>,
}

impl FrontError {
    pub(crate) fn source(code: &'static str, message: impl Into<String>, span: Span) -> Self {
        Self {
            kind: ErrorKind::Source,
            code,
            message: message.into(),
            span: Some(span),
            unit_conversion_fix: None,
        }
    }

    pub fn tool(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Tool,
            code,
            message: message.into(),
            span: None,
            unit_conversion_fix: None,
        }
    }

    pub(crate) fn with_unit_conversion_fix(mut self, span: Span, unit: &'static str) -> Self {
        self.unit_conversion_fix = Some((span, unit));
        self
    }

    /// Deterministic version-1 diagnostic document for this front-end result.
    pub fn to_json(&self, path: &str) -> Value {
        let span = self
            .span
            .map(|span| json!({"start": span.start, "end": span.end}));
        json!({
            "schema_version": 1,
            "status": match self.kind { ErrorKind::Source => "source_error", ErrorKind::Tool => "tool_error" },
            "diagnostics": [{
                "code": self.code,
                "message": self.message,
                "path": path,
                "span": span,
                "notes": []
            }]
        })
    }

    /// CLI representation with checked byte and line locations.
    pub fn to_diagnostics_json(&self, path: &str, source: &[u8]) -> Value {
        let mut diagnostic =
            crate::diagnostics::diagnostic(self.code, &self.message, path, source, self.span);
        if self.kind == ErrorKind::Source
            && let Some((span, unit)) = self.unit_conversion_fix
        {
            let candidate = crate::fixes::unit_conversion(span, unit);
            let valid = crate::fixes::validated(
                source,
                vec![candidate],
                crate::fixes::Recheck::Clean,
                true,
            );
            let fixes = crate::fixes::to_json(path, source, &valid);
            if !fixes.is_empty() {
                diagnostic["suggested_fixes"] = fixes.into();
            }
        }
        crate::diagnostics::document(
            match self.kind {
                ErrorKind::Source => "source_error",
                ErrorKind::Tool => "tool_error",
            },
            vec![diagnostic],
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Program {
    pub functions: Vec<Function>,
    pub foreign_functions: Vec<ForeignFunction>,
    pub closures: Vec<ClosureDecl>,
    pub frames: Vec<FrameDecl>,
    pub modules: Vec<ModuleDecl>,
    pub imports: Vec<Import>,
    pub records: Vec<RecordDecl>,
    pub variants: Vec<VariantDecl>,
    pub system_items: Vec<SystemItem>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignFunction {
    pub name: String,
    pub params: Vec<Param>,
    pub result: Type,
    pub oracle: String,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemItem {
    pub module: Option<String>,
    pub kind: SystemItemKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SystemItemKind {
    ComponentImport {
        path: String,
        robot: Option<String>,
    },
    Component {
        name: String,
        external: bool,
        clauses: Vec<SystemClause>,
    },
    Connect {
        source: String,
        target: String,
        policy: String,
    },
    Flow {
        name: String,
        ports: Vec<String>,
        within: SystemQuantity,
    },
    Product {
        name: String,
        clauses: Vec<SystemClause>,
    },
    System {
        name: String,
        products: Vec<String>,
        instances: Vec<(String, String)>,
    },
    RobotModel(RobotModelDecl),
    CellModel(CellModelDecl),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotModelDecl {
    pub name: String,
    pub license: String,
    pub asset: Option<RobotAssetDecl>,
    pub links: Vec<RobotLinkDecl>,
    pub joints: Vec<RobotJointDecl>,
    pub actuators: Vec<RobotActuatorDecl>,
    pub sensors: Vec<RobotSensorDecl>,
    pub tool: Option<(String, RobotPoseDecl)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotAssetDecl {
    pub kind: String,
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotPoseDecl {
    pub xyz: Vec<SystemQuantity>,
    pub quat: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotGeometryDecl {
    pub role: String,
    pub kind: String,
    pub dimensions: Vec<String>,
    pub path: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotLinkDecl {
    pub name: String,
    pub parent: String,
    pub origin: RobotPoseDecl,
    pub mass: SystemQuantity,
    pub inertial_origin: Option<RobotPoseDecl>,
    pub inertia: Vec<String>,
    pub geometries: Vec<RobotGeometryDecl>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotJointDecl {
    pub name: String,
    pub kind: String,
    pub parent: String,
    pub child: String,
    pub axis: Vec<String>,
    pub position: Option<(SystemQuantity, SystemQuantity)>,
    pub velocity: Option<SystemQuantity>,
    pub effort: Option<SystemQuantity>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotActuatorDecl {
    pub name: String,
    pub mode: String,
    pub target_kind: String,
    pub target: String,
    pub gear: String,
    pub control: (SystemQuantity, SystemQuantity),
    /// Source-declared home hold command, overriding the vendored `home` key control.
    pub hold: Option<SystemQuantity>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotSensorDecl {
    pub name: String,
    pub kind: String,
    pub source: String,
    pub intrinsics: Option<RobotCameraIntrinsicsDecl>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotCameraIntrinsicsDecl {
    pub width: SystemQuantity,
    pub height: SystemQuantity,
    pub fx: SystemQuantity,
    pub fy: SystemQuantity,
    pub cx: SystemQuantity,
    pub cy: SystemQuantity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellModelDecl {
    pub name: String,
    pub robot: String,
    pub mount: RobotPoseDecl,
    /// `mount ... on FIXTURE exclude (LINK, ...)`: the fixture the robot is mounted
    /// on and the robot links whose contact with that fixture is excluded.
    pub mount_surface: Option<(String, Vec<String>)>,
    pub objects: Vec<CellObjectDecl>,
    pub io: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellObjectDecl {
    pub kind: String,
    pub name: String,
    pub pose: RobotPoseDecl,
    pub geometries: Vec<RobotGeometryDecl>,
    pub dynamic: Option<SystemQuantity>,
    pub grasp: Option<RobotPoseDecl>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemQuantity {
    pub value: String,
    pub unit: String,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SystemClause {
    Port {
        direction: String,
        name: String,
        ty: Type,
        rate: SystemQuantity,
        latency: Option<SystemQuantity>,
        optional: bool,
        layout: Option<String>,
        bus: Option<String>,
        span: Span,
    },
    Param {
        name: String,
        ty: Type,
        value: SystemQuantity,
        range: Option<(SystemQuantity, SystemQuantity)>,
        span: Span,
    },
    IdentityParam {
        name: String,
        ty: Type,
        span: Span,
    },
    Resource(Vec<(String, SystemQuantity)>),
    License {
        expression: String,
        link: Option<String>,
        modified: Option<bool>,
        replaceable: Option<bool>,
        network_exposed: Option<bool>,
        grant: Option<String>,
        span: Span,
    },
    Isolation(String),
    Wcet(SystemQuantity),
    Deadline(SystemQuantity),
    Impl {
        kind: String,
        target: String,
    },
    Abi {
        kind: String,
        prefix: String,
    },
    Lifecycle(Vec<String>),
    Config(String),
    Input(String),
    Golden {
        path: String,
        tolerances: Vec<SystemQuantity>,
    },
    Property(String),
    Shim(String),
    Distribution(String),
    Commercial(bool),
    Robot(String),
    Capacity(Vec<(String, SystemQuantity)>),
}

#[derive(Debug, PartialEq, Eq)]
pub struct FrameDecl {
    pub name: String,
    pub parent: Option<String>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureDecl {
    pub name: String,
    pub captures: Vec<ClosureCapture>,
    pub params: Vec<Param>,
    pub result: Type,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureCapture {
    pub name: String,
    pub ty: Type,
    pub mode: Option<Mode>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ModuleDecl {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Import {
    pub owner: Option<String>,
    pub module: String,
    pub item: String,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordDecl {
    pub name: String,
    pub module: Option<String>,
    pub public: bool,
    pub generic_params: Vec<RecordGenericParam>,
    pub fields: Vec<RecordField>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordGenericParam {
    pub name: String,
    pub kind: RecordGenericKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordGenericKind {
    Value,
    Frame,
    Unit,
    Count,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordField {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantDecl {
    pub name: String,
    pub module: Option<String>,
    pub public: bool,
    pub arms: Vec<VariantArmDecl>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantArmDecl {
    pub name: String,
    pub payload: Option<Type>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub module: Option<String>,
    pub public: bool,
    pub generic_params: Vec<String>,
    pub params: Vec<Param>,
    pub result: Type,
    pub return_from: Option<String>,
    pub body: Block,
    pub span: Span,
    pub fold: crate::numeric_contract::FoldTier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub mode: Option<Mode>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Read,
    Mut,
    Take,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    I32,
    U32,
    F32,
    F64,
    Bool,
    Unit,
    ArrayI32,
    ArrayU32,
    ArrayF32,
    ArrayF64,
    ArrayBool,
    Named(String),
}

pub(crate) fn result_parts(name: &str) -> Option<(Type, Type)> {
    fn component(value: &str, depth: usize) -> Option<Type> {
        match value {
            "i32" => Some(Type::I32),
            "u32" => Some(Type::U32),
            "f32" => Some(Type::F32),
            "f64" => Some(Type::F64),
            "Bool" => Some(Type::Bool),
            "Bytes" | "String" | "Utf8Error" | "ForeignError" => {
                Some(Type::Named(value.to_owned()))
            }
            _ if depth < 4 && parts(value, depth + 1).is_some() => {
                Some(Type::Named(value.to_owned()))
            }
            _ => None,
        }
    }
    fn parts(name: &str, depth: usize) -> Option<(Type, Type)> {
        let inside = name.strip_prefix("Result<")?.strip_suffix('>')?;
        let mut level = 0usize;
        for (index, byte) in inside.bytes().enumerate() {
            match byte {
                b'<' => level = level.checked_add(1)?,
                b'>' => level = level.checked_sub(1)?,
                b',' if level == 0 => {
                    return Some((
                        component(&inside[..index], depth)?,
                        component(&inside[index + 1..], depth)?,
                    ));
                }
                _ => {}
            }
        }
        None
    }
    parts(name, 1)
}

pub(crate) fn view_parts(name: &str) -> Option<(Type, bool)> {
    match name {
        "View<i32>" => Some((Type::I32, false)),
        "View<u32>" => Some((Type::U32, false)),
        "MutView<i32>" => Some((Type::I32, true)),
        "MutView<u32>" => Some((Type::U32, true)),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub statements: Vec<Stmt>,
    pub span: Span,
    height: usize,
}

impl Block {
    pub(crate) fn new(statements: Vec<Stmt>, span: Span) -> Result<Self, FrontError> {
        let child_height = statements.iter().map(|stmt| stmt.height).max();
        Ok(Self {
            height: tree_height(child_height.unwrap_or(0))?,
            statements,
            span,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
    height: usize,
}

impl Stmt {
    pub(crate) fn new(kind: StmtKind, span: Span) -> Result<Self, FrontError> {
        let expr = |expr: &Expr| expr.height;
        let child_height = match &kind {
            StmtKind::Binding { value, .. } | StmtKind::Expr(value) => value.height,
            StmtKind::Assign { place, value } => match place {
                Place::Name(..) => value.height,
                Place::Index { index, .. } => index.height.max(value.height),
            },
            StmtKind::Return(value) => value.as_ref().map_or(0, expr),
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => condition.height.max(then_block.height).max(match else_arm {
                Some(ElseArm::Block(block)) => block.height,
                Some(ElseArm::If(stmt)) => stmt.height,
                None => 0,
            }),
            StmtKind::While { condition, body } => condition.height.max(body.height),
            StmtKind::For {
                lower, upper, body, ..
            } => lower.height.max(upper.height).max(body.height),
        };
        Ok(Self {
            height: tree_height(child_height)?,
            kind,
            span,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElseArm {
    Block(Block),
    If(Box<Stmt>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Place {
    Name(String, Span),
    Index {
        name: String,
        index: Expr,
        span: Span,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StmtKind {
    Binding {
        mutable: bool,
        name: String,
        annotation: Option<Type>,
        value: Expr,
    },
    Assign {
        place: Place,
        value: Expr,
    },
    Return(Option<Expr>),
    If {
        condition: Expr,
        then_block: Block,
        else_arm: Option<ElseArm>,
    },
    While {
        condition: Expr,
        body: Block,
    },
    For {
        name: String,
        lower: Expr,
        upper: Expr,
        body: Block,
    },
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
    height: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInit {
    pub name: String,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchArm {
    pub variant: String,
    pub binding: Option<String>,
    pub value: Expr,
    pub span: Span,
}

impl Expr {
    pub(crate) fn new(kind: ExprKind, span: Span) -> Result<Self, FrontError> {
        let child_height = match &kind {
            ExprKind::Integer(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::Text(_)
            | ExprKind::Bytes(_)
            | ExprKind::Name(_)
            | ExprKind::Move(_) => 0,
            ExprKind::MoveField { base, .. } => base.height,
            ExprKind::Call { args, .. } | ExprKind::Array(args) => {
                args.iter().map(|arg| arg.height).max().unwrap_or(0)
            }
            ExprKind::Record { fields, .. } => fields
                .iter()
                .map(|field| field.value.height)
                .max()
                .unwrap_or(0),
            ExprKind::Field { base, .. } => base.height,
            ExprKind::Match { value, arms } => value
                .height
                .max(arms.iter().map(|arm| arm.value.height).max().unwrap_or(0)),
            ExprKind::Index { base, index }
            | ExprKind::Binary {
                left: base,
                right: index,
                ..
            } => base.height.max(index.height),
            ExprKind::Unary { value, .. } | ExprKind::Group(value) | ExprKind::Try(value) => {
                value.height
            }
        };
        Ok(Self {
            height: tree_height(child_height)?,
            kind,
            span,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprKind {
    Integer(String),
    Float(String),
    Bool(bool),
    Text(String),
    Bytes(Vec<u8>),
    Name(String),
    Move(String),
    MoveField {
        base: Box<Expr>,
        field: String,
    },
    Group(Box<Expr>),
    Try(Box<Expr>),
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Array(Vec<Expr>),
    Record {
        name: String,
        fields: Vec<FieldInit>,
    },
    Match {
        value: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    Field {
        base: Box<Expr>,
        name: String,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        op_span: Span,
        value: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        op_span: Span,
        right: Box<Expr>,
    },
}

mod items;
mod query;
pub use query::{named, ungroup};

include!("front/lexer.rs");

include!("front/parser.rs");

pub fn parse_system(bytes: &[u8]) -> Result<Program, FrontError> {
    parse(bytes).map_err(|failure| {
        if failure.kind == ErrorKind::Source && !failure.code.starts_with('S') {
            FrontError::source(
                "S1000",
                failure.message,
                failure.span.unwrap_or(Span { start: 0, end: 0 }),
            )
        } else {
            failure
        }
    })
}

pub fn contains_system_declaration(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|source| lex(source).ok())
        .is_some_and(|tokens| {
            tokens.iter().enumerate().any(|(index, token)| {
                if matches!(token.kind, TokenKind::Keyword("extern"))
                    && matches!(tokens.get(index + 1).map(|next| &next.kind), Some(TokenKind::Name(name)) if name == "c")
                {
                    return false;
                }
                matches!(
                    token.kind,
                    TokenKind::Keyword(
                        "component" | "extern" | "connect" | "flow" | "product" | "system"
                    )
                ) || matches!(&token.kind, TokenKind::Name(name) if matches!(name.as_str(), "robot" | "cell"))
            })
        })
}
