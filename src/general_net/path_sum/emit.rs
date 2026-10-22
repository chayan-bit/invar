//! One program-specific source for Metal (MSL 3.1), CUDA (NVRTC) and the CPU
//! model (C++17), selected by `__METAL_VERSION__` / `__CUDACC__`.
//!
//! Every operator is emitted unchecked: the narrow licence proved it stays in
//! its type. Signed `+ - *` go through `uint` so no target sees signed
//! overflow. Calls and loops still raise lane flags (depth, loop budget) and
//! return at once, so nothing runs on a value the proof did not cover.

use std::{collections::BTreeMap, fmt::Write};

use super::{
    Combine, Enumerator, FLAG_BUDGET, FLAG_DEPTH, FLAG_GEOMETRY, FLAG_GUARD, FLAG_RAW, GROUP,
    Lowering, Mutation, range::Combines,
};
use crate::{
    front::{BinaryOp, Type, UnaryOp},
    general_net::regions::tree::{self, Expr, FunctionTree, LoopKind, Stmt},
};

type Fail = &'static str;

/// Iterations one loop instance may run before its lane falls back.
const LOOP_BUDGET: u32 = 1 << 20;
/// Nodes the serial enumerator visits before it falls back.
const SERIAL_BUDGET: u32 = 1 << 24;
/// Root parameters the fixed `Args` block carries.
pub const MAX_PARAMS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Plain,
    Nested { level: usize, levels: usize },
    Root,
}

struct Emitter<'a> {
    trees: &'a [Option<&'a FunctionTree>],
    combines: &'a Combines,
    combine: Combine,
    holes: BTreeMap<String, usize>,
    out: String,
    temps: usize,
    /// Hole of each child slot filled on the path being emitted: slots are
    /// static, so no target indexes the node's child arrays at run time.
    slots: Vec<usize>,
    /// The interval-low parameter: a lane evaluates a leaf only when it owns
    /// it (the leaf interval starts at its key).
    owner: Option<String>,
    /// `MAXC`: child slots per node.
    children: usize,
}

pub(super) fn source(
    trees: &[Option<&FunctionTree>],
    lowering: &Lowering,
    combines: &Combines,
) -> Result<String, Fail> {
    let root = trees[lowering.function].ok_or("root")?;
    if root.params.len() > MAX_PARAMS {
        return Err("more than eight root parameters");
    }
    let names = super::holes(root);
    let holes: BTreeMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let mut e = Emitter {
        trees,
        combines,
        combine: lowering.combine,
        holes,
        out: String::new(),
        temps: 0,
        slots: Vec::new(),
        children: 1,
        owner: match lowering.enumerator {
            Enumerator::Interval { lo, .. } => Some(root.params[lo].name.clone()),
            Enumerator::Serial => None,
        },
    };
    let lets = self_lets(root)?;
    e.prelude(lowering, root, lets.max(1));
    e.children = lets.max(1);
    let order = reachable(trees, lowering.function)?;
    let levels = |i: usize| lowering.facts.levels.get(&i).copied().unwrap_or(1).max(1);
    for &i in &order {
        let tree = trees[i].ok_or("unstructured callee")?;
        if calls_self(tree) {
            for level in 0..levels(i) {
                e.signature(
                    tree,
                    Kind::Nested {
                        level,
                        levels: levels(i),
                    },
                )?;
                e.out.push_str(";\n");
            }
        } else {
            e.signature(tree, Kind::Plain)?;
            e.out.push_str(";\n");
        }
    }
    for &i in &order {
        let tree = trees[i].ok_or("unstructured callee")?;
        if calls_self(tree) {
            for level in 0..levels(i) {
                e.function(
                    tree,
                    Kind::Nested {
                        level,
                        levels: levels(i),
                    },
                )?;
            }
        } else {
            e.function(tree, Kind::Plain)?;
        }
    }
    e.function(root, Kind::Root)?;
    e.lane(lowering, root);
    if lowering.mutation == Mutation::DropLaneFlags {
        e.out.push_str("#define DROP_LANE_FLAGS\n");
    }
    e.out.push_str(KERNELS);
    Ok(e.out)
}

/// Callees reachable from `root` (excluding it), callees first.
fn reachable(trees: &[Option<&FunctionTree>], root: usize) -> Result<Vec<usize>, Fail> {
    fn visit(
        trees: &[Option<&FunctionTree>],
        i: usize,
        seen: &mut Vec<usize>,
        out: &mut Vec<usize>,
    ) -> Result<(), Fail> {
        if seen.contains(&i) {
            return Ok(());
        }
        seen.push(i);
        let tree = trees
            .get(i)
            .copied()
            .flatten()
            .ok_or("unstructured callee")?;
        for name in callees(tree) {
            if name == tree.name || matches!(name.as_str(), "wrapping_add" | "wrapping_mul") {
                continue;
            }
            let j = trees
                .iter()
                .position(|t| t.is_some_and(|t| t.name == name))
                .ok_or("unknown callee")?;
            visit(trees, j, seen, out)?;
        }
        out.push(i);
        Ok(())
    }
    let mut seen = Vec::new();
    let mut out = Vec::new();
    visit(trees, root, &mut seen, &mut out)?;
    out.retain(|i| *i != root);
    Ok(out)
}

fn calls_self(tree: &FunctionTree) -> bool {
    super::range::recursive(tree)
}

/// Names of the functions `tree` calls, in walk order.
fn callees(tree: &FunctionTree) -> Vec<String> {
    let mut out = Vec::new();
    tree::walk(&tree.body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        for expr in tree::stmt_exprs(stmt) {
            let mut calls = Vec::new();
            expr.calls(&mut calls);
            for call in calls {
                if let Expr::Call { name, .. } = call
                    && !out.contains(name)
                {
                    out.push(name.clone());
                }
            }
        }
    });
    out
}

/// Number of `let h = root(..)` sites; refused inside loops.
fn self_lets(tree: &FunctionTree) -> Result<usize, Fail> {
    let mut count = 0;
    let mut looped = false;
    fn visit(name: &str, list: &[Stmt], in_loop: bool, count: &mut usize, looped: &mut bool) {
        for stmt in list {
            match stmt {
                Stmt::Let {
                    value: Expr::Call { name: n, .. },
                    ..
                }
                | Stmt::Assign {
                    value: Expr::Call { name: n, .. },
                    ..
                } if n == name => {
                    *count += 1;
                    *looped |= in_loop;
                }
                Stmt::If { yes, no, .. } => {
                    visit(name, yes, in_loop, count, looped);
                    visit(name, no, in_loop, count, looped);
                }
                Stmt::Loop(inner) => visit(name, &inner.body, true, count, looped),
                _ => {}
            }
        }
    }
    visit(&tree.name, &tree.body, false, &mut count, &mut looped);
    if looped {
        return Err("self call inside a loop");
    }
    Ok(count)
}

/// Whether control never falls off the end of `body`.
fn ends(body: &[Stmt]) -> bool {
    match body.last() {
        Some(Stmt::Return { .. }) => true,
        Some(Stmt::If { yes, no, .. }) => ends(yes) && ends(no),
        _ => false,
    }
}

fn c_type(ty: &Type) -> Result<&'static str, Fail> {
    Ok(match ty {
        Type::I32 => "int",
        Type::U32 => "uint",
        Type::Bool => "bool",
        _ => return Err("non-scalar type"),
    })
}

fn function_name(tree: &FunctionTree, kind: Kind) -> String {
    match kind {
        Kind::Plain => format!("f{}", tree.index),
        Kind::Nested { level, .. } => format!("f{}_{level}", tree.index),
        Kind::Root => "node".to_owned(),
    }
}

impl Emitter<'_> {
    fn line(&mut self, text: &str) {
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn temp(&mut self) -> String {
        self.temps += 1;
        format!("t{}", self.temps)
    }

    fn prelude(&mut self, lowering: &Lowering, root: &FunctionTree, children: usize) {
        let mut s = String::from(PRELUDE);
        let _ = writeln!(s, "#define GROUP {GROUP}u");
        let _ = writeln!(s, "#define NP {}", root.params.len().max(1));
        let _ = writeln!(s, "#define MAXC {children}");
        let _ = writeln!(s, "#define MAXD {}u", lowering.max_depth);
        let _ = writeln!(s, "#define LOOP_BUDGET {LOOP_BUDGET}u");
        let _ = writeln!(s, "#define SERIAL_BUDGET {SERIAL_BUDGET}u");
        let _ = writeln!(
            s,
            "#define FLAG_GEOMETRY {FLAG_GEOMETRY}u\n#define FLAG_DEPTH {FLAG_DEPTH}u\n#define FLAG_BUDGET {FLAG_BUDGET}u\n#define FLAG_RAW {FLAG_RAW}u\n#define FLAG_GUARD {FLAG_GUARD}u"
        );
        let _ = writeln!(
            s,
            "struct Args {{ int root[{MAX_PARAMS}]; uint lanes; uint groups; }};"
        );
        let _ = writeln!(
            s,
            "struct Node {{ int kind; int value; int k; uint count; int ca[MAXC][NP]; uint cc[MAXC]; }};"
        );
        match lowering.combine {
            Combine::Residue(p) => {
                let p = u64::from(p);
                // Leaves under a combine and reduced constants are in [0, P)
                // at run time; the analysis may bound them lower still.
                let bound = |iv: Option<super::range::Iv>| {
                    iv.map_or(0, |iv| (iv.hi.max(0) as u64).min(p - 1))
                };
                let (leaf_hi, const_hi) = (
                    bound(lowering.facts.leaves),
                    bound(lowering.facts.constants),
                );
                let coef_hi = self
                    .combines
                    .values()
                    .flat_map(|(cs, _)| cs.iter().map(|(_, c)| c.rem_euclid(p as i64) as u64))
                    .max()
                    .unwrap_or(0);
                let narrow = |x: u64| (p - 1).saturating_mul(x) <= u64::from(u32::MAX);
                let mul = |ok: bool| {
                    if ok {
                        // MULU re-checks the bound in the CPU model.
                        "(MULU((uint)(a), (uint)(b)) % PMOD)"
                    } else {
                        "((uint)(((u64)(a) * (u64)(b)) % (u64)PMOD))"
                    }
                };
                let _ = writeln!(s, "#define PMOD {p}u");
                let _ = writeln!(
                    s,
                    "#define MULV(a, b) {}",
                    mul(narrow(leaf_hi.max(const_hi)))
                );
                let _ = writeln!(s, "#define MULW(a, b) {}", mul(narrow(coef_hi)));
                s.push_str(RESIDUE);
            }
            Combine::Sum => s.push_str(SUM),
        }
        self.out.push_str(&s);
    }

    fn signature(&mut self, tree: &FunctionTree, kind: Kind) -> Result<(), Fail> {
        let mut params = Vec::new();
        for p in &tree.params {
            params.push(format!("{} v_{}", c_type(&p.ty)?, p.name));
        }
        if kind == Kind::Root {
            if self.owner.is_some() {
                params.push("int key".to_owned());
            }
            params.push("REF Node &n".to_owned());
        }
        params.push("REF uint &fl".to_owned());
        let result = if kind == Kind::Root {
            "void"
        } else {
            c_type(&tree.result)?
        };
        let _ = write!(
            self.out,
            "DEV {result} {}({})",
            function_name(tree, kind),
            params.join(", ")
        );
        Ok(())
    }

    fn function(&mut self, tree: &FunctionTree, kind: Kind) -> Result<(), Fail> {
        self.signature(tree, kind)?;
        self.line(" {");
        for (name, ty) in &tree.locals {
            if tree.params.iter().all(|p| p.name != *name) {
                let ty = c_type(ty)?;
                self.line(&format!("{ty} v_{name} = 0;"));
            }
        }
        let fx = Fx {
            trees: self.trees,
            tree,
            kind,
        };
        self.block(&fx, &tree.body)?;
        self.line("fl |= FLAG_GEOMETRY;");
        self.line(if kind == Kind::Root {
            "return;"
        } else {
            "return 0;"
        });
        self.line("}");
        Ok(())
    }

    fn block(&mut self, fx: &Fx<'_>, body: &[Stmt]) -> Result<(), Fail> {
        for stmt in body {
            self.stmt(fx, stmt)?;
        }
        Ok(())
    }

    fn stmt(&mut self, fx: &Fx<'_>, stmt: &Stmt) -> Result<(), Fail> {
        match stmt {
            Stmt::Let { name, value } | Stmt::Assign { name, value } => {
                if fx.kind == Kind::Root
                    && let Some(args) = fx.self_args(value)
                {
                    let hole = *self.holes.get(name).ok_or("self call outside a let")?;
                    let args = self.args(fx, args)?;
                    let slot = self.slots.len();
                    for (j, a) in args.iter().enumerate() {
                        self.line(&format!("n.ca[{slot}][{j}] = (int)({a});"));
                    }
                    self.line(&format!("n.count = {}u;", slot + 1));
                    self.slots.push(hole);
                    return Ok(());
                }
                let v = self.expr(fx, value)?;
                self.line(&format!("v_{name} = {v};"));
            }
            Stmt::Eval(value) => {
                let v = self.expr(fx, value)?;
                self.line(&format!("(void)({v});"));
            }
            Stmt::Store { .. } => return Err("array store"),
            Stmt::If { cond, yes, no } => {
                let c = self.expr(fx, cond)?;
                let before = self.slots.clone();
                self.line(&format!("if ({c}) {{"));
                self.block(fx, yes)?;
                let after_yes = std::mem::replace(&mut self.slots, before);
                self.line("} else {");
                self.block(fx, no)?;
                self.line("}");
                // A branch that returns leaves no slots behind; two that fall
                // through must agree, or the slot would depend on the path.
                if ends(no) {
                    self.slots = after_yes;
                } else if !ends(yes) && after_yes != self.slots {
                    return Err("self calls differ across branches");
                }
            }
            Stmt::Loop(inner) => {
                let guard = self.temp();
                let stop = fx.stop();
                match &inner.kind {
                    LoopKind::For { var, lo, hi } => {
                        let lo = self.expr(fx, lo)?;
                        let hi = self.expr(fx, hi)?;
                        let end = self.temp();
                        self.line(&format!("uint {guard} = 0u; uint {end} = {hi};"));
                        self.line(&format!(
                            "for (v_{var} = {lo}; v_{var} < {end}; v_{var}++) {{"
                        ));
                        self.line(&format!(
                            "if (++{guard} > LOOP_BUDGET) {{ fl |= FLAG_BUDGET; {stop} }}"
                        ));
                    }
                    LoopKind::While { cond } => {
                        self.line(&format!("uint {guard} = 0u;"));
                        self.line("for (;;) {");
                        let c = self.expr(fx, cond)?;
                        self.line(&format!("if (!({c})) break;"));
                        self.line(&format!(
                            "if (++{guard} > LOOP_BUDGET) {{ fl |= FLAG_BUDGET; {stop} }}"
                        ));
                    }
                }
                self.block(fx, &inner.body)?;
                self.line("}");
            }
            Stmt::Return { value, span } => {
                let value = value.as_ref().ok_or("return without value")?;
                if fx.kind != Kind::Root {
                    let v = self.expr(fx, value)?;
                    self.line(&format!("return {v};"));
                    return Ok(());
                }
                if let Some(args) = fx.self_args(value) {
                    let args = self.args(fx, args)?;
                    self.line("if (n.count != 0u) { fl |= FLAG_GEOMETRY; return; }");
                    for (j, a) in args.iter().enumerate() {
                        self.line(&format!("n.ca[0][{j}] = (int)({a});"));
                    }
                    self.line("n.kind = 1; n.count = 1u; return;");
                } else if let Some((coefficients, constant)) =
                    self.combines.get(&(span.start, span.end))
                {
                    let k = self.expr(fx, constant)?;
                    if coefficients
                        .iter()
                        .any(|(name, _)| !self.holes.contains_key(name))
                    {
                        return Err("combine hole");
                    }
                    self.line(&format!("n.kind = 2; n.k = (int)({k});"));
                    if let Combine::Residue(p) = self.combine {
                        // NonNegativeConstants and NoOverflowCombine at run time.
                        let worst: i64 = coefficients
                            .iter()
                            .map(|(_, c)| c * (i64::from(p) - 1))
                            .sum();
                        let limit = i64::from(i32::MAX) - worst;
                        if coefficients.iter().any(|(_, c)| *c < 0) || limit < 0 {
                            self.line("fl |= FLAG_GUARD; return;");
                        } else {
                            self.line(&format!(
                                "if (n.k < 0 || n.k > {limit}) {{ fl |= FLAG_GUARD; return; }}"
                            ));
                        }
                    }
                    for (slot, hole) in self.slots.iter().enumerate() {
                        let c = coefficients
                            .iter()
                            .find(|(name, _)| self.holes.get(name) == Some(hole))
                            .map_or(0, |(_, c)| match self.combine {
                                Combine::Residue(p) => c.rem_euclid(i64::from(p)),
                                Combine::Sum => *c,
                            });
                        let _ = writeln!(self.out, "n.cc[{slot}] = {c}u;");
                    }
                    self.line("return;");
                } else {
                    if let Some(lo) = &self.owner {
                        // Only the owner publishes a leaf; the others skip it.
                        let _ = writeln!(self.out, "if (v_{lo} != key) {{ n.kind = 0; return; }}");
                    }
                    let v = self.expr(fx, value)?;
                    self.line(&format!("n.kind = 0; n.value = (int)({v}); return;"));
                }
            }
        }
        Ok(())
    }

    fn args(&mut self, fx: &Fx<'_>, args: &[Expr]) -> Result<Vec<String>, Fail> {
        args.iter().map(|a| self.expr(fx, a)).collect()
    }

    fn expr(&mut self, fx: &Fx<'_>, expr: &Expr) -> Result<String, Fail> {
        Ok(match expr {
            Expr::I32(x) if *x == i32::MIN => "(-2147483647 - 1)".to_owned(),
            Expr::I32(x) => format!("({x})"),
            Expr::U32(x) => format!("{x}u"),
            Expr::Bool(b) => b.to_string(),
            Expr::Local(name) => format!("v_{name}"),
            Expr::Unary { op, value, .. } => {
                let v = self.expr(fx, value)?;
                match (op, fx.ty(value)?) {
                    (UnaryOp::Not, _) => format!("(!{v})"),
                    (UnaryOp::Neg, Type::I32) => format!("NEGI({v})"),
                    _ => return Err("negation of a non-i32"),
                }
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                let ty = fx.ty(left)?;
                let a = self.expr(fx, left)?;
                let b = self.expr(fx, right)?;
                let sym = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Rem => "%",
                    BinaryOp::Eq => "==",
                    BinaryOp::Ne => "!=",
                    BinaryOp::Lt => "<",
                    BinaryOp::Le => "<=",
                    BinaryOp::Gt => ">",
                    BinaryOp::Ge => ">=",
                    BinaryOp::And => "&",
                    BinaryOp::Or => "|",
                };
                // Narrow-operator macros of the prelude, i32 then u32.
                let [signed, unsigned] = match op {
                    BinaryOp::Add => ["ADDI(", "ADDU("],
                    BinaryOp::Sub => ["SUBI(", "SUBU("],
                    BinaryOp::Mul => ["MULI(", "MULU("],
                    BinaryOp::Div => ["DIVI(", "DIVU("],
                    BinaryOp::Rem => ["REMI(", "REMU("],
                    BinaryOp::And | BinaryOp::Or => return Ok(format!("((bool)({a} {sym} {b}))")),
                    _ => return Ok(format!("({a} {sym} {b})")),
                };
                match ty {
                    Type::I32 => format!("{signed}{a}, {b})"),
                    Type::U32 => format!("{unsigned}{a}, {b})"),
                    _ => return Err("arithmetic on a non-integer"),
                }
            }
            Expr::Logic { and, left, right } => {
                let t = self.temp();
                let a = self.expr(fx, left)?;
                self.line(&format!("bool {t} = {a};"));
                self.line(&format!("if ({}{t}) {{", if *and { "" } else { "!" }));
                let b = self.expr(fx, right)?;
                self.line(&format!("{t} = {b};"));
                self.line("}");
                t
            }
            Expr::Call { name, args, .. } => {
                let args = self.args(fx, args)?;
                if let [a, b] = &args[..] {
                    match name.as_str() {
                        "wrapping_add" => return Ok(format!("((uint)({a}) + (uint)({b}))")),
                        "wrapping_mul" => return Ok(format!("((uint)({a}) * (uint)({b}))")),
                        _ => {}
                    }
                }
                let callee = self
                    .trees
                    .iter()
                    .flatten()
                    .find(|t| t.name == *name)
                    .ok_or("unknown callee")?;
                let stop = fx.stop();
                let t = self.temp();
                let ty = c_type(&callee.result)?;
                let target = match fx.kind {
                    Kind::Nested { level, levels } if callee.name == fx.tree.name => {
                        if level + 1 == levels {
                            // The analysis proved this level makes no self call.
                            self.line(&format!("fl |= FLAG_DEPTH; {stop}"));
                            return Ok("0".to_owned());
                        }
                        format!("f{}_{}", callee.index, level + 1)
                    }
                    Kind::Root if callee.name == fx.tree.name => {
                        return Err("self call outside a let or return");
                    }
                    _ if calls_self(callee) => format!("f{}_0", callee.index),
                    _ => format!("f{}", callee.index),
                };
                let mut all = args;
                all.push("fl".to_owned());
                self.line(&format!("{ty} {t} = {target}({});", all.join(", ")));
                self.line(&format!("if (fl) {{ {stop} }}"));
                t
            }
            _ => return Err("non-scalar expression"),
        })
    }

    fn lane(&mut self, lowering: &Lowering, root: &FunctionTree) {
        let call: Vec<String> = root
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| match p.ty {
                Type::U32 => format!("(uint)p[{i}]"),
                _ => format!("p[{i}]"),
            })
            .chain(self.owner.iter().map(|_| "key".to_owned()))
            .chain(["n".to_owned(), "fl".to_owned()])
            .collect();
        let call = format!("node({});", call.join(", "));
        let np = root.params.len();
        let m = lowering.mutation;
        let table = match (lowering.combine, m) {
            (Combine::Residue(_), Mutation::DropTableGuard) | (Combine::Sum, _) => "",
            (Combine::Residue(_), _) => {
                "if (n.value < 0 || (uint)n.value >= PMOD) { o.flags |= FLAG_GUARD; return; }\n"
            }
        };
        let leaf = if m == Mutation::SkipRaw {
            // The raw root is reduced like any leaf, with no guard to stop it.
            "if (under) {\n".to_owned() + table + "}\nadd_value(o, w, n.value);"
        } else {
            format!(
                "if (under) {{\n{table}add_value(o, w, n.value);\n}} else {{ o.flags |= FLAG_RAW; o.raw = (uint)n.value; }}"
            )
        };
        let text = match lowering.enumerator {
            Enumerator::Interval { lo, hi } => {
                let owner = if m == Mutation::ConstantEveryLane {
                    String::new()
                } else {
                    format!(" && p[{lo}] == key")
                };
                let children = self.children;
                // Every combine constant a literal zero: nothing to add.
                let constant = if self.combines.values().all(|(_, k)| k.literal() == Some(0)) {
                    String::new()
                } else {
                    format!("if (n.kind == 2{owner}) add_constant(o, w, n.k);")
                };
                let clear: String = (0..children).map(|j| format!(" n.cc[{j}] = 0u;")).collect();
                // Child slots are unrolled with constant indices so every
                // target keeps the node in registers.
                let mut select = String::new();
                let geometry = m != Mutation::DropGeometry;
                // A statically proven partition is only re-checked by the
                // CPU model; otherwise a broken one falls back.
                let check = |cond: String| {
                    if lowering.geometry {
                        format!("PROVEN(!({cond}));")
                    } else {
                        format!("if ({cond}) {{ o.flags |= FLAG_GEOMETRY; return; }}")
                    }
                };
                if geometry {
                    for j in 0..children {
                        let next = if j + 1 < children {
                            format!(
                                " || ({}u < c && n.ca[{j}][{hi}] != n.ca[{}][{lo}])",
                                j + 1,
                                j + 1
                            )
                        } else {
                            String::new()
                        };
                        let cond = format!(
                            "{j}u < c && (n.ca[{j}][{lo}] >= n.ca[{j}][{hi}]{next} || ({}u == c && n.ca[{j}][{hi}] != p[{hi}]))",
                            j + 1
                        );
                        let _ = writeln!(select, "{}", check(cond));
                    }
                }
                // With the partition validated and the key inside the parent,
                // the owning child is the first whose high exceeds the key:
                // test from the last slot down so the first one wins.
                for j in (0..children).rev() {
                    let copy: String = (0..np)
                        .map(|i| format!(" q[{i}] = n.ca[{j}][{i}];"))
                        .collect();
                    let low = if geometry {
                        String::new()
                    } else {
                        format!("n.ca[{j}][{lo}] <= key && ")
                    };
                    // A proven partition's last child ends at the parent's
                    // `hi`, beyond every key: the widest slot needs no test.
                    let inside = if geometry && lowering.geometry && j + 1 == children {
                        String::new()
                    } else {
                        format!(" && {low}key < n.ca[{j}][{hi}]")
                    };
                    let _ = writeln!(
                        select,
                        "if ({j}u < c{inside}) {{ pick = {j}u; cw = n.cc[{j}];{copy} }}"
                    );
                }
                let first = if geometry {
                    check(format!("c == 0u || n.ca[0][{lo}] != p[{lo}]")) + "\n"
                } else {
                    String::new()
                };
                let found = check("pick == c".to_owned());
                let weight = if m == Mutation::WrongWeight {
                    "under = true;"
                } else {
                    "w = MULW(w, cw); under = true;"
                };
                format!(
                    r"DEV void lane(Args a, uint gid, REF Part &o) {{
int p[NP]; for (uint i = 0u; i < {np}u; i++) p[i] = a.root[i];
int key = (int)((uint)a.root[{lo}] + gid);
uint w = 1u; bool under = false; uint fl = 0u;
for (uint depth = 0u; ; depth++) {{
if (depth > MAXD) {{ o.flags |= FLAG_DEPTH; return; }}
Node n; n.count = 0u; n.kind = -1;{clear}
{call}
if (fl) {{ o.flags |= fl; return; }}
if (n.kind == 0) {{
if (p[{lo}] != key) return;
{leaf}
return;
}}
{constant}
uint c = n.count;
{first}uint pick = c; uint cw = 0u; int q[NP];
{select}{found}
if (n.kind == 2) {{ {weight} }}
for (uint i = 0u; i < {np}u; i++) p[i] = q[i];
}}
}}
"
                )
            }
            Enumerator::Serial => {
                let push = if m == Mutation::WrongWeight {
                    "w"
                } else {
                    "n.kind == 2 ? MULW(w, n.cc[j - 1u]) : w"
                };
                format!(
                    r"#define STACK ((MAXD + 2u) * MAXC)
DEV void lane(Args a, uint gid, REF Part &o) {{
if (gid != 0u) return;
int sp[STACK][NP]; uint sw[STACK]; uint sd[STACK]; bool su[STACK];
for (uint i = 0u; i < {np}u; i++) sp[0][i] = a.root[i];
sw[0] = 1u; sd[0] = 0u; su[0] = false;
uint top = 1u; uint work = 0u; uint fl = 0u;
while (top != 0u) {{
top--;
int p[NP]; for (uint i = 0u; i < {np}u; i++) p[i] = sp[top][i];
uint w = sw[top]; uint depth = sd[top]; bool under = su[top];
if (depth > MAXD) {{ o.flags |= FLAG_DEPTH; return; }}
if (++work > SERIAL_BUDGET) {{ o.flags |= FLAG_BUDGET; return; }}
Node n; n.count = 0u; n.kind = -1;
{call}
if (fl) {{ o.flags |= fl; return; }}
if (n.kind == 0) {{
{leaf}
continue;
}}
if (n.kind == 2) add_constant(o, w, n.k);
for (uint j = n.count; j > 0u; j--) {{
if (top == STACK) {{ o.flags |= FLAG_DEPTH; return; }}
for (uint i = 0u; i < {np}u; i++) sp[top][i] = n.ca[j - 1u][i];
sw[top] = {push};
sd[top] = depth + 1u; su[top] = under || n.kind == 2;
top++;
}}
}}
}}
"
                )
            }
        };
        self.out.push_str(&text);
    }
}

struct Fx<'a> {
    trees: &'a [Option<&'a FunctionTree>],
    tree: &'a FunctionTree,
    kind: Kind,
}

impl Fx<'_> {
    fn stop(&self) -> &'static str {
        if self.kind == Kind::Root {
            "return;"
        } else {
            "return 0;"
        }
    }

    fn self_args<'e>(&self, value: &'e Expr) -> Option<&'e [Expr]> {
        match value {
            Expr::Call { name, args, .. } if *name == self.tree.name => Some(args),
            _ => None,
        }
    }

    fn ty(&self, expr: &Expr) -> Result<Type, Fail> {
        Ok(match expr {
            Expr::I32(_) => Type::I32,
            Expr::U32(_) => Type::U32,
            Expr::Bool(_) | Expr::Logic { .. } => Type::Bool,
            Expr::Local(name) => self.tree.locals.get(name).cloned().ok_or("unknown local")?,
            Expr::Unary {
                op: UnaryOp::Not, ..
            } => Type::Bool,
            Expr::Unary { value, .. } => self.ty(value)?,
            Expr::Binary { op, left, .. } => match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                    self.ty(left)?
                }
                _ => Type::Bool,
            },
            Expr::Call { name, .. } => match name.as_str() {
                "wrapping_add" | "wrapping_mul" => Type::U32,
                _ => self
                    .trees
                    .iter()
                    .flatten()
                    .find(|t| t.name == *name)
                    .ok_or("unknown callee")?
                    .result
                    .clone(),
            },
            _ => return Err("non-scalar expression"),
        })
    }
}

// The emitted call graph is acyclic (nested levels are cloned), so every
// device function is forced inline as in a handwritten kernel.
const PRELUDE: &str = r"// Generated by invar P4a path-sum lowering. Do not edit.
#if defined(__METAL_VERSION__)
#include <metal_stdlib>
using namespace metal;
#define DEV __attribute__((always_inline))
#define REF thread
typedef long i64;
typedef ulong u64;
#elif defined(__CUDACC__)
#define DEV __device__ __forceinline__
#define REF
typedef unsigned int uint;
typedef long long i64;
typedef unsigned long long u64;
#else
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define DEV static inline
#define REF
typedef unsigned int uint;
typedef long long i64;
typedef unsigned long long u64;
#endif
#if defined(PROOF_CHECK) && !defined(__METAL_VERSION__) && !defined(__CUDACC__)
// CPU model only: every narrow operator re-checks the licence proof.
static int chk_i(i64 v) { if (v < -2147483647LL - 1 || v > 2147483647LL) std::abort(); return (int)v; }
static uint chk_u(i64 v) { if (v < 0 || v > 4294967295LL) std::abort(); return (uint)v; }
static int chk_div(int a, int b, bool rem) { if (b == 0 || (a == -2147483647 - 1 && b == -1)) std::abort(); return rem ? a % b : a / b; }
static uint chk_divu(uint a, uint b, bool rem) { if (b == 0u) std::abort(); return rem ? a % b : a / b; }
// A statically proven fact (the partition geometry) re-checked.
#define PROVEN(c) do { if (!(c)) std::abort(); } while (0)
#define ADDI(a, b) chk_i((i64)(a) + (i64)(b))
#define SUBI(a, b) chk_i((i64)(a) - (i64)(b))
#define MULI(a, b) chk_i((i64)(a) * (i64)(b))
#define NEGI(a) chk_i(-(i64)(a))
#define DIVI(a, b) chk_div((a), (b), false)
#define REMI(a, b) chk_div((a), (b), true)
#define ADDU(a, b) chk_u((i64)(a) + (i64)(b))
#define SUBU(a, b) chk_u((i64)(a) - (i64)(b))
#define MULU(a, b) chk_u((i64)(a) * (i64)(b))
#define DIVU(a, b) chk_divu((a), (b), false)
#define REMU(a, b) chk_divu((a), (b), true)
#else
// Proven in range: signed arithmetic wraps through uint, so no target sees UB.
#define PROVEN(c) do { } while (0)
#define ADDI(a, b) ((int)((uint)(a) + (uint)(b)))
#define SUBI(a, b) ((int)((uint)(a) - (uint)(b)))
#define MULI(a, b) ((int)((uint)(a) * (uint)(b)))
#define NEGI(a) ((int)(0u - (uint)(a)))
#define DIVI(a, b) ((a) / (b))
#define REMI(a, b) ((a) % (b))
#define ADDU(a, b) ((a) + (b))
#define SUBU(a, b) ((a) - (b))
#define MULU(a, b) ((a) * (b))
#define DIVU(a, b) ((a) / (b))
#define REMU(a, b) ((a) % (b))
#endif
";

const RESIDUE: &str = r"typedef uint Val;
DEV Val vzero() { return 0u; }
DEV Val vmerge(Val a, Val b) { uint s = a + b; return s >= PMOD ? s - PMOD : s; }
struct Part { uint flags; uint raw; Val v; };
DEV void add_value(REF Part &o, uint w, int x) { o.v = vmerge(o.v, MULV(w, (uint)x)); }
DEV void add_constant(REF Part &o, uint w, int k) { o.v = vmerge(o.v, MULV(w, (uint)k % PMOD)); }
";

const SUM: &str = r"#define MULW(a, b) (a)
struct Val { i64 s; u64 m; };
DEV Val vzero() { Val v; v.s = 0; v.m = 0u; return v; }
DEV Val vmerge(Val a, Val b) { Val v; v.s = a.s + b.s; v.m = a.m + b.m; return v; }
struct Part { uint flags; uint raw; Val v; };
DEV void add_value(REF Part &o, uint w, int x) { (void)w; o.v.s += (i64)x; o.v.m += (u64)(x < 0 ? -(i64)x : (i64)x); }
DEV void add_constant(REF Part &o, uint w, int k) { add_value(o, w, k); }
";

const KERNELS: &str = r#"#if defined(DROP_LANE_FLAGS)
#define LANE_ANY(o) false
#else
#define LANE_ANY(o) ((o.flags | o.raw) != 0u)
#endif
#if defined(__METAL_VERSION__)
#define KERNEL(name) kernel void name(constant Args &arg [[buffer(0)]], device Part *part [[buffer(1)]], device Part *out [[buffer(2)]], uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]])
#define IDS
#define SHARED threadgroup Val v[GROUP]; threadgroup atomic_uint sf[2];
#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define FLAGS_INIT() if (tid == 0u) { atomic_store_explicit(&sf[0], 0u, memory_order_relaxed); atomic_store_explicit(&sf[1], 0u, memory_order_relaxed); } BARRIER();
#define FLAGS_OR(o) if (LANE_ANY(o)) { atomic_fetch_or_explicit(&sf[0], o.flags, memory_order_relaxed); atomic_fetch_or_explicit(&sf[1], o.raw, memory_order_relaxed); }
#define FLAGS_READ(r) r.flags = atomic_load_explicit(&sf[0], memory_order_relaxed); r.raw = atomic_load_explicit(&sf[1], memory_order_relaxed);
// Values reduce in shared memory; lane flags and raw words (rare) are ORed
// into two shared words, ordered before the read by the reduction barriers.
#define REDUCE() for (uint s = GROUP / 2u; s != 0u; s >>= 1) { BARRIER(); if (tid < s) v[tid] = vmerge(v[tid], v[tid + s]); }
#define PUBLISH(dst, o) FLAGS_OR(o); v[tid] = o.v; REDUCE(); if (tid == 0u) { Part r; r.v = v[0]; FLAGS_READ(r); dst = r; }
#elif defined(__CUDACC__)
#define KERNEL(name) extern "C" __global__ void name(const Args arg, Part *part, Part *out)
#define IDS uint tid = threadIdx.x; uint group = blockIdx.x;
#define SHARED __shared__ Val v[GROUP]; __shared__ uint sf[2];
#define BARRIER() __syncthreads()
#define FLAGS_INIT()
// The first reduction barrier also votes on whether any lane has a flag or
// raw word; only then (rare) are they ORed through two shared words, so the
// common case has no flag barrier or shared flag traffic.
#define PUBLISH(dst, o) v[tid] = o.v; { bool any = __syncthreads_or(LANE_ANY(o)) != 0; \
if (any) { if (tid == 0u) { sf[0] = 0u; sf[1] = 0u; } BARRIER(); if (LANE_ANY(o)) { atomicOr(&sf[0], o.flags); atomicOr(&sf[1], o.raw); } BARRIER(); } \
for (uint s = GROUP / 2u; s != 0u; s >>= 1) { if (tid < s) v[tid] = vmerge(v[tid], v[tid + s]); if (s > 1u) BARRIER(); } \
if (tid == 0u) { Part r; r.v = v[0]; r.flags = any ? sf[0] : 0u; r.raw = any ? sf[1] : 0u; dst = r; } }
#endif
DEV Part zero() { Part o; o.flags = 0u; o.raw = 0u; o.v = vzero(); return o; }
#if defined(__METAL_VERSION__) || defined(__CUDACC__)
KERNEL(pathsum_root) {
IDS
if (tid == 0u && group == 0u) { Args a = arg; Part o = zero(); lane(a, 0u, o); out[0] = o; }
}
KERNEL(pathsum_fused) {
IDS SHARED FLAGS_INIT()
Args a = arg; Part o = zero();
for (uint g = tid; g < a.lanes; g += GROUP) lane(a, g, o);
PUBLISH(out[0], o)
}
KERNEL(pathsum_map) {
IDS SHARED FLAGS_INIT()
Args a = arg; Part o = zero(); uint g = group * GROUP + tid;
if (g < a.lanes) lane(a, g, o);
PUBLISH(part[group], o)
}
KERNEL(pathsum_reduce) {
IDS SHARED FLAGS_INIT()
Part o = zero();
for (uint i = tid; i < arg.groups; i += GROUP) { Part x = part[i]; o.flags |= x.flags; o.raw |= x.raw; o.v = vmerge(o.v, x.v); }
PUBLISH(out[0], o)
}
#else
int main() {
Args a; uint shape;
while (std::scanf("%u %u", &a.lanes, &shape) == 2) {
for (uint i = 0u; i < NP; i++) if (std::scanf("%d", &a.root[i]) != 1) return 2;
a.groups = 0u; Part o = zero();
if (shape == 0u) lane(a, 0u, o); else for (uint g = 0u; g < a.lanes; g++) lane(a, g, o);
uint w[sizeof(Part) / 4]; std::memcpy(w, &o, sizeof o);
for (uint i = 0u; i < sizeof(Part) / 4; i++) std::printf("%u%c", w[i], i + 1 == sizeof(Part) / 4 ? '\n' : ' ');
std::fflush(stdout);
}
return 0;
}
#endif
"#;
