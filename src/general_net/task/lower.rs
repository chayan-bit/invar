//! Statement trees to the C dialect shared by the CPU model, CUDA and Metal.
//!
//! Machine functions become `gt_step_<i>`: a resumable state machine over a
//! `GtFrame` (one 32-bit slot per local) that returns at every machine call
//! (`GT_R_CALL`), at a fork site in fork mode (`GT_R_FORK`, all sibling
//! arguments evaluated in source order, the first failing argument recorded
//! as pending), after a fork join has bound its results (`GT_R_YIELD`, a
//! commit point), at a return or at a failure. Every other function becomes a
//! plain checked function `gt_fn_<i>` with register locals.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use super::{Function, Mutation, Program, Refusal};
use crate::{
    front::{BinaryOp, Span, Type, UnaryOp},
    general_net::regions::tree::{Expr, LoopKind, Stmt},
};

/// Fork sites (from the certified normal form) whose first sibling sits
/// outside every loop of `body`.
pub(super) fn fork_points(
    body: &[Stmt],
    forks: &[super::super::normal_form::ForkSite],
) -> Vec<(Span, usize)> {
    let mut out = Vec::new();
    fn visit(
        list: &[Stmt],
        forks: &[super::super::normal_form::ForkSite],
        out: &mut Vec<(Span, usize)>,
    ) {
        for stmt in list {
            match stmt {
                Stmt::Let {
                    value: Expr::Call { span, .. },
                    ..
                } => {
                    if let Some(site) = forks.iter().find(|site| site.start == *span) {
                        out.push((site.start, site.siblings));
                    }
                }
                Stmt::If { yes, no, .. } => {
                    visit(yes, forks, out);
                    visit(no, forks, out);
                }
                _ => {}
            }
        }
    }
    visit(body, forks, &mut out);
    out
}

fn machine_call<'e>(expr: &'e Expr, machine: &BTreeSet<String>) -> Option<(&'e str, &'e [Expr])> {
    match expr {
        Expr::Call { name, args, .. } if machine.contains(name) => Some((name, args)),
        _ => None,
    }
}

fn mentions_machine(expr: &Expr, machine: &BTreeSet<String>) -> Option<Span> {
    let mut calls = Vec::new();
    expr.calls(&mut calls);
    calls.into_iter().find_map(|call| match call {
        Expr::Call { name, span, .. } if machine.contains(name) => Some(*span),
        _ => None,
    })
}

/// Machine calls only as whole statement values, never in loops or nested.
pub(super) fn check_machine(
    tree: &crate::general_net::regions::tree::FunctionTree,
    machine: &BTreeSet<String>,
) -> Result<(), Refusal> {
    fn list(
        stmts: &[Stmt],
        in_loop: bool,
        name: &str,
        machine: &BTreeSet<String>,
    ) -> Result<(), Refusal> {
        for stmt in stmts {
            let top = match stmt {
                Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::Eval(value) => {
                    Some(value)
                }
                Stmt::Return {
                    value: Some(value), ..
                } => Some(value),
                _ => None,
            };
            let nested = |expr: &Expr| mentions_machine(expr, machine);
            if let Some(value) = top
                && let Some((_, args)) = machine_call(value, machine)
            {
                if in_loop {
                    return Err(Refusal::TaskCallInLoop {
                        function: name.to_owned(),
                    });
                }
                if let Some(span) = args.iter().find_map(nested) {
                    return Err(Refusal::NestedTaskCall {
                        function: name.to_owned(),
                        span,
                    });
                }
                continue;
            }
            match stmt {
                Stmt::If { cond, yes, no } => {
                    if let Some(span) = nested(cond) {
                        return Err(Refusal::NestedTaskCall {
                            function: name.to_owned(),
                            span,
                        });
                    }
                    list(yes, in_loop, name, machine)?;
                    list(no, in_loop, name, machine)?;
                }
                Stmt::Loop(inner) => {
                    let heads = match &inner.kind {
                        LoopKind::For { lo, hi, .. } => vec![lo, hi],
                        LoopKind::While { cond } => vec![cond],
                    };
                    if let Some(span) = heads.into_iter().find_map(nested) {
                        return Err(Refusal::NestedTaskCall {
                            function: name.to_owned(),
                            span,
                        });
                    }
                    list(&inner.body, true, name, machine)?;
                }
                other => {
                    for expr in crate::general_net::regions::tree::stmt_exprs(other) {
                        if let Some(span) = nested(expr) {
                            if in_loop {
                                return Err(Refusal::TaskCallInLoop {
                                    function: name.to_owned(),
                                });
                            }
                            return Err(Refusal::NestedTaskCall {
                                function: name.to_owned(),
                                span,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
    list(&tree.body, false, &tree.name, machine)
}

fn is_call_stmt(stmt: &Stmt, machine: &BTreeSet<String>) -> bool {
    match stmt {
        Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::Eval(value) => {
            machine_call(value, machine).is_some()
        }
        Stmt::Return {
            value: Some(value), ..
        } => machine_call(value, machine).is_some(),
        _ => false,
    }
}

/// Whether some path performs a machine call after an earlier call statement
/// that is not its immediate sibling (a continuation call).
pub(super) fn calls_after_call(body: &[Stmt], machine: &BTreeSet<String>) -> bool {
    fn walk(list: &[Stmt], seen: bool, machine: &BTreeSet<String>) -> (bool, bool) {
        let (mut seen, mut previous_call) = (seen, false);
        for stmt in list {
            if is_call_stmt(stmt, machine) {
                if seen && !previous_call {
                    return (true, true);
                }
                seen = true;
                previous_call = true;
                continue;
            }
            previous_call = false;
            if let Stmt::If { yes, no, .. } = stmt {
                let (a, sa) = walk(yes, seen, machine);
                let (b, sb) = walk(no, seen, machine);
                if a || b {
                    return (true, true);
                }
                seen |= sa || sb;
            }
        }
        (false, seen)
    }
    walk(body, false, machine).0
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ty {
    I32,
    U32,
    Bool,
}

fn ty_of(ty: &Type) -> Ty {
    match ty {
        Type::U32 => Ty::U32,
        Type::Bool => Ty::Bool,
        _ => Ty::I32,
    }
}

fn c_type(ty: Ty) -> &'static str {
    match ty {
        Ty::I32 => "int",
        Ty::U32 => "uint",
        Ty::Bool => "bool",
    }
}

/// Where the code being emitted runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Plain function: locals are C variables, failure returns 0.
    Direct,
    /// State machine: locals are frame slots, failure returns `GT_R_FAIL`.
    Machine,
}

struct Emitter<'p> {
    program: &'p Program,
    function: &'p Function,
    machine: BTreeSet<String>,
    mode: Mode,
    out: String,
    temp: usize,
    state: usize,
    /// Failure exit: `None` uses the mode's return, `Some` breaks out of a
    /// fork-argument block.
    in_fork_args: bool,
    mutation: Mutation,
}

impl Emitter<'_> {
    fn w(&mut self, text: &str) {
        self.out.push_str(text);
    }

    fn fail_exit(&self) -> &'static str {
        if self.in_fork_args {
            "break;"
        } else if self.mode == Mode::Machine {
            "return GT_R_FAIL;"
        } else {
            "return 0;"
        }
    }

    fn trap(&mut self, status: u32, span: Span) {
        let exit = self.fail_exit();
        let _ = write!(
            self.out,
            "{{st.status={status}u;st.lo={}u;st.hi={}u;{exit}}}",
            span.start, span.end
        );
    }

    fn fresh(&mut self) -> String {
        self.temp += 1;
        format!("t{}", self.temp)
    }

    fn local_ty(&self, name: &str) -> Ty {
        ty_of(self.function.tree.locals.get(name).unwrap_or(&Type::I32))
    }

    fn read(&self, name: &str) -> String {
        let slot = self.function.slots.get(name).copied().unwrap_or(0);
        match (self.mode, self.local_ty(name)) {
            (Mode::Direct, _) => format!("l{slot}"),
            (Mode::Machine, Ty::I32) => format!("fr.v[{slot}]"),
            (Mode::Machine, Ty::U32) => format!("((uint)fr.v[{slot}])"),
            (Mode::Machine, Ty::Bool) => format!("(fr.v[{slot}]!=0)"),
        }
    }

    fn set(&mut self, name: &str, value: &str) {
        let slot = self.function.slots.get(name).copied().unwrap_or(0);
        let line = match self.mode {
            Mode::Direct => format!("l{slot}=({});", value),
            Mode::Machine => format!("fr.v[{slot}]=(int)({value});"),
        };
        self.w(&line);
    }

    fn ty(&self, expr: &Expr) -> Ty {
        match expr {
            Expr::I32(_) => Ty::I32,
            Expr::U32(_) => Ty::U32,
            Expr::Bool(_) | Expr::Logic { .. } => Ty::Bool,
            Expr::Local(name) => self.local_ty(name),
            Expr::Unary {
                op: UnaryOp::Not, ..
            } => Ty::Bool,
            Expr::Unary { value, .. } => self.ty(value),
            Expr::Binary { op, left, .. } => match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                    self.ty(left)
                }
                _ => Ty::Bool,
            },
            Expr::Call { name, .. } => match self.program.names.get(name) {
                Some(index) => ty_of(&self.program.functions[index].tree.result),
                None => Ty::U32,
            },
            _ => Ty::I32,
        }
    }

    /// Emit statements computing `expr`; return a C expression for its value.
    fn expr(&mut self, expr: &Expr) -> Result<String, Refusal> {
        Ok(match expr {
            Expr::I32(i32::MIN) => "(-2147483647-1)".to_owned(),
            Expr::I32(x) => format!("({x})"),
            Expr::U32(x) => format!("{x}u"),
            Expr::Bool(x) => x.to_string(),
            Expr::Local(name) => self.read(name),
            Expr::Unary { op, value, span } => {
                let v = self.expr(value)?;
                let t = self.fresh();
                match op {
                    UnaryOp::Not => {
                        let _ = write!(self.out, "bool {t}=!({v});");
                    }
                    UnaryOp::Neg => {
                        let _ = write!(self.out, "int {t}=({v});if({t}==(-2147483647-1))");
                        self.trap(super::status::OVERFLOW, *span);
                        let _ = write!(self.out, "{t}=-{t};");
                    }
                }
                t
            }
            Expr::Binary {
                op,
                left,
                right,
                span,
            } => self.binary(*op, left, right, *span)?,
            Expr::Logic { and, left, right } => {
                let l = self.expr(left)?;
                let t = self.fresh();
                let _ = write!(
                    self.out,
                    "bool {t}=({l});if({}{t}){{",
                    if *and { "" } else { "!" }
                );
                let r = self.expr(right)?;
                let _ = write!(self.out, "{t}=({r});}}");
                t
            }
            Expr::Call { name, args, span } => self.direct_call(name, args, *span)?,
            _ => {
                return Err(Refusal::NonScalar {
                    function: self.function.tree.name.clone(),
                });
            }
        })
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        left: &Expr,
        right: &Expr,
        span: Span,
    ) -> Result<String, Refusal> {
        let ty = self.ty(left);
        let l = self.expr(left)?;
        let r = self.expr(right)?;
        let t = self.fresh();
        let cmp = |sym: &str| format!("bool {t}=(({l}){sym}({r}));");
        let line = match op {
            BinaryOp::Eq => cmp("=="),
            BinaryOp::Ne => cmp("!="),
            BinaryOp::Lt => cmp("<"),
            BinaryOp::Le => cmp("<="),
            BinaryOp::Gt => cmp(">"),
            BinaryOp::Ge => cmp(">="),
            BinaryOp::And => cmp("&&"),
            BinaryOp::Or => cmp("||"),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                let name = match (op, ty) {
                    (BinaryOp::Add, Ty::U32) => "gt_add_u",
                    (BinaryOp::Sub, Ty::U32) => "gt_sub_u",
                    (BinaryOp::Mul, Ty::U32) => "gt_mul_u",
                    (BinaryOp::Add, _) => "gt_add_i",
                    (BinaryOp::Sub, _) => "gt_sub_i",
                    _ => "gt_mul_i",
                };
                let _ = write!(self.out, "{} {t};if(!{name}({l},{r},{t}))", c_type(ty));
                self.trap(super::status::OVERFLOW, span);
                return Ok(t);
            }
            BinaryOp::Div | BinaryOp::Rem => {
                let rhs = self.fresh();
                let _ = write!(self.out, "{} {rhs}=({r});if({rhs}==0)", c_type(ty));
                self.trap(super::status::DIVISION_BY_ZERO, span);
                if ty == Ty::U32 {
                    let sym = if op == BinaryOp::Div { "/" } else { "%" };
                    format!("uint {t}=({l}){sym}{rhs};")
                } else if op == BinaryOp::Div {
                    let lhs = self.fresh();
                    let _ = write!(
                        self.out,
                        "int {lhs}=({l});if({lhs}==(-2147483647-1)&&{rhs}==-1)"
                    );
                    self.trap(super::status::OVERFLOW, span);
                    format!("int {t}={lhs}/{rhs};")
                } else {
                    format!("int {t}=({rhs}==-1)?0:(({l})%{rhs});")
                }
            }
        };
        self.w(&line);
        Ok(t)
    }

    fn direct_call(&mut self, name: &str, args: &[Expr], _span: Span) -> Result<String, Refusal> {
        let values = args
            .iter()
            .map(|arg| self.expr(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let t = self.fresh();
        let Some(&index) = self.program.names.get(name) else {
            let op = if name == "wrapping_add" { "+" } else { "*" };
            let _ = write!(
                self.out,
                "uint {t}=(uint)({}){op}(uint)({});",
                values[0], values[1]
            );
            return Ok(t);
        };
        let callee = &self.program.functions[&index];
        let _ = write!(self.out, "if(depth>={}u)", super::MAX_CALL_DEPTH);
        self.trap(super::status::CALL_DEPTH, Span { start: 0, end: 0 });
        let ty = ty_of(&callee.tree.result);
        let mut call = format!("gt_fn_{index}(");
        for v in &values {
            let _ = write!(call, "{v},");
        }
        let exit = self.fail_exit();
        let _ = write!(
            self.out,
            "{} {t}={call}depth+1u,st);if(st.status!=0u){{{exit}}}",
            c_type(ty)
        );
        Ok(t)
    }

    fn new_state(&mut self) -> usize {
        self.state += 1;
        self.state
    }

    /// Statements (fork runs detected in machine mode).
    fn list(&mut self, stmts: &[Stmt], in_loop: bool) -> Result<(), Refusal> {
        let mut i = 0;
        while i < stmts.len() {
            if self.mode == Mode::Machine
                && !in_loop
                && let Stmt::Let {
                    value: Expr::Call { span, .. },
                    ..
                } = &stmts[i]
                && let Some(&(_, n)) = self.function.forks.iter().find(|(start, _)| start == span)
                && i + n <= stmts.len()
            {
                self.fork(&stmts[i..i + n])?;
                i += n;
                continue;
            }
            self.stmt(&stmts[i], in_loop)?;
            i += 1;
        }
        Ok(())
    }

    fn suspends(&self, stmts: &[Stmt]) -> bool {
        stmts.iter().any(|stmt| match stmt {
            Stmt::If { yes, no, .. } => self.suspends(yes) || self.suspends(no),
            Stmt::Let {
                value: Expr::Call { span, .. },
                ..
            } if self.function.forks.iter().any(|(start, _)| start == span) => true,
            other => is_call_stmt(other, &self.machine),
        })
    }

    /// A machine call as a statement: suspend, resume reading `fr.ret`.
    fn call(
        &mut self,
        dest: Option<&str>,
        tail: bool,
        name: &str,
        args: &[Expr],
    ) -> Result<(), Refusal> {
        let values = args
            .iter()
            .map(|arg| self.expr(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let index = self.program.names[name];
        let _ = write!(self.out, "io.k=1u;io.callee[0]={index}u;");
        for (j, v) in values.iter().enumerate() {
            let _ = write!(self.out, "io.args[{j}]=(int)({v});");
        }
        let s = self.new_state();
        let _ = write!(self.out, "fr.pc={s}u;return GT_R_CALL;}}case {s}u:{{");
        if tail {
            self.w("io.value=fr.ret;return GT_R_RETURN;");
        } else if let Some(name) = dest {
            let value = match self.local_ty(name) {
                Ty::U32 => "(uint)fr.ret",
                Ty::Bool => "(fr.ret!=0)",
                Ty::I32 => "fr.ret",
            };
            self.set(name, value);
        }
        Ok(())
    }

    /// An L20 fork run: all siblings at once in fork mode, one by one otherwise.
    fn fork(&mut self, run: &[Stmt]) -> Result<(), Refusal> {
        let p = self.program.params;
        self.w("if(fork){io.k=0u;do{");
        self.in_fork_args = true;
        for (j, stmt) in run.iter().enumerate() {
            let Stmt::Let {
                value: Expr::Call { name, args, .. },
                ..
            } = stmt
            else {
                return Err(Refusal::Limit);
            };
            let values = args
                .iter()
                .map(|arg| self.expr(arg))
                .collect::<Result<Vec<_>, _>>()?;
            let index = self.program.names[name];
            let _ = write!(self.out, "io.callee[{j}]={index}u;");
            for (q, v) in values.iter().enumerate() {
                let _ = write!(self.out, "io.args[{}]=(int)({v});", j * p + q);
            }
            let _ = write!(self.out, "io.k={}u;", j + 1);
        }
        self.in_fork_args = false;
        let join = self.new_state();
        let after = self.new_state();
        let _ = write!(self.out, "}}while(false);fr.pc={join}u;return GT_R_FORK;}}");
        for stmt in run {
            self.stmt(stmt, false)?;
        }
        let _ = write!(self.out, "fr.pc={after}u;continue;}}case {join}u:{{");
        let order: Vec<usize> = if self.mutation == Mutation::SwapBindings {
            (0..run.len()).rev().collect()
        } else {
            (0..run.len()).collect()
        };
        for (j, stmt) in run.iter().enumerate() {
            let Stmt::Let { name, .. } = stmt else {
                continue;
            };
            let from = order[j];
            let value = match self.local_ty(name) {
                Ty::U32 => format!("(uint)io.res[{from}]"),
                Ty::Bool => format!("(io.res[{from}]!=0)"),
                Ty::I32 => format!("io.res[{from}]"),
            };
            self.set(name, &value);
        }
        let _ = write!(
            self.out,
            "fr.pc={after}u;return GT_R_YIELD;}}case {after}u:{{"
        );
        Ok(())
    }

    fn ret(&mut self, value: Option<&Expr>) -> Result<(), Refusal> {
        let v = match value {
            Some(expr) => self.expr(expr)?,
            None => "0".to_owned(),
        };
        let line = match self.mode {
            Mode::Machine => format!("io.value=(int)({v});return GT_R_RETURN;"),
            Mode::Direct => format!("return ({v});"),
        };
        self.w(&line);
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt, in_loop: bool) -> Result<(), Refusal> {
        if self.mode == Mode::Machine {
            match stmt {
                Stmt::Let { name, value } | Stmt::Assign { name, value } => {
                    if let Some((callee, args)) = machine_call(value, &self.machine) {
                        return self.call(Some(name), false, callee, args);
                    }
                }
                Stmt::Eval(value) => {
                    if let Some((callee, args)) = machine_call(value, &self.machine) {
                        return self.call(None, false, callee, args);
                    }
                }
                Stmt::Return {
                    value: Some(value), ..
                } => {
                    if let Some((callee, args)) = machine_call(value, &self.machine) {
                        return self.call(None, true, callee, args);
                    }
                }
                Stmt::If { cond, yes, no } if self.suspends(yes) || self.suspends(no) => {
                    let c = self.expr(cond)?;
                    let (sy, sn, sj) = (self.new_state(), self.new_state(), self.new_state());
                    let _ = write!(self.out, "fr.pc=({c})?{sy}u:{sn}u;continue;}}case {sy}u:{{");
                    self.list(yes, in_loop)?;
                    let _ = write!(self.out, "fr.pc={sj}u;continue;}}case {sn}u:{{");
                    self.list(no, in_loop)?;
                    let _ = write!(self.out, "fr.pc={sj}u;continue;}}case {sj}u:{{");
                    return Ok(());
                }
                _ => {}
            }
        }
        match stmt {
            Stmt::Let { name, value } | Stmt::Assign { name, value } => {
                let v = self.expr(value)?;
                self.set(name, &v);
            }
            Stmt::Eval(value) => {
                self.expr(value)?;
            }
            Stmt::Return { value, .. } => self.ret(value.as_ref())?,
            Stmt::If { cond, yes, no } => {
                let c = self.expr(cond)?;
                let _ = write!(self.out, "if({c}){{");
                self.list(yes, in_loop)?;
                self.w("}else{");
                self.list(no, in_loop)?;
                self.w("}");
            }
            Stmt::Loop(inner) => {
                let exit = self.fail_exit();
                let budget = format!(
                    "if(++st.steps>GT_BUDGET){{st.status={}u;st.lo=0u;st.hi=0u;{exit}}}",
                    super::status::STEP_BUDGET
                );
                match &inner.kind {
                    LoopKind::For { var, lo, hi } => {
                        let lo = self.expr(lo)?;
                        let hi = self.expr(hi)?;
                        let x = self.fresh();
                        let _ = write!(self.out, "for(uint {x}=({lo});{x}<({hi});++{x}){{{budget}");
                        self.set(var, &x);
                        self.list(&inner.body, true)?;
                        self.w("}");
                    }
                    LoopKind::While { cond } => {
                        let _ = write!(self.out, "for(;;){{{budget}");
                        let c = self.expr(cond)?;
                        let _ = write!(self.out, "if(!({c}))break;");
                        self.list(&inner.body, true)?;
                        self.w("}");
                    }
                }
            }
            Stmt::Store { .. } => {
                return Err(Refusal::NonScalar {
                    function: self.function.tree.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// C source of every function of `program`: prototypes, plain functions,
/// state machines and the `gt_step` dispatcher.
pub(super) fn functions(program: &Program, mutation: Mutation) -> Result<String, Refusal> {
    let machine = program.machine_names();
    let mut out = String::new();
    for (index, function) in program.functions.iter().filter(|(_, f)| !f.machine) {
        let params: Vec<String> = function
            .tree
            .params
            .iter()
            .map(|p| format!("{} l{}", c_type(ty_of(&p.ty)), function.slots[&p.name]))
            .collect();
        let _ = writeln!(
            out,
            "GT_FN {} gt_fn_{index}({}uint depth,GT_TH GtSt& st);",
            c_type(ty_of(&function.tree.result)),
            params.iter().map(|p| format!("{p},")).collect::<String>()
        );
    }
    let mut steps = BTreeMap::new();
    for (index, function) in &program.functions {
        let mut e = Emitter {
            program,
            function,
            machine: machine.clone(),
            mode: if function.machine {
                Mode::Machine
            } else {
                Mode::Direct
            },
            out: String::new(),
            temp: 0,
            state: 0,
            in_fork_args: false,
            mutation,
        };
        let result = c_type(ty_of(&function.tree.result));
        if function.machine {
            e.w("for(;;){switch(fr.pc){case 0u:{");
            e.list(&function.tree.body, false)?;
            e.w("io.value=0;return GT_R_RETURN;}default:{st.status=7u;return GT_R_FAIL;}}}");
            let _ = writeln!(
                out,
                "GT_FN uint gt_step_{index}(GT_TH GtFrame& fr,GT_TH GtIo& io,GT_TH GtSt& st,bool fork,uint depth){{{}}}",
                e.out
            );
        } else {
            let declared: Vec<String> = function
                .tree
                .locals
                .iter()
                .filter(|(name, _)| !function.tree.params.iter().any(|p| &p.name == *name))
                .map(|(name, ty)| (function.slots.get(name).copied(), ty))
                .filter_map(|(slot, ty)| slot.map(|s| (s, ty)))
                .map(|(slot, ty)| format!("{} l{slot}=0;", c_type(ty_of(ty))))
                .collect();
            e.w(&declared.concat());
            e.list(&function.tree.body, false)?;
            e.w("return 0;");
            let params: String = function
                .tree
                .params
                .iter()
                .map(|p| format!("{} l{},", c_type(ty_of(&p.ty)), function.slots[&p.name]))
                .collect();
            let _ = writeln!(
                out,
                "GT_FN {result} gt_fn_{index}({params}uint depth,GT_TH GtSt& st){{{}}}",
                e.out
            );
            // Step wrapper so a fork may spawn a plain function as a child.
            let args: String = function
                .tree
                .params
                .iter()
                .enumerate()
                .map(|(q, p)| match ty_of(&p.ty) {
                    Ty::I32 => format!("fr.v[{q}],"),
                    Ty::U32 => format!("(uint)fr.v[{q}],"),
                    Ty::Bool => format!("fr.v[{q}]!=0,"),
                })
                .collect();
            let _ = writeln!(
                out,
                "GT_FN uint gt_step_{index}(GT_TH GtFrame& fr,GT_TH GtIo& io,GT_TH GtSt& st,bool fork,uint depth){{{result} r=gt_fn_{index}({args}depth,st);if(st.status!=0u)return GT_R_FAIL;io.value=(int)r;return GT_R_RETURN;}}"
            );
        }
        steps.insert(*index, ());
    }
    out.push_str("GT_FN uint gt_step(GT_TH GtFrame& fr,GT_TH GtIo& io,GT_TH GtSt& st,bool fork,uint depth){switch(fr.fn){");
    for index in steps.keys() {
        let _ = write!(
            out,
            "case {index}u:return gt_step_{index}(fr,io,st,fork,depth);"
        );
    }
    out.push_str("default:st.status=7u;return GT_R_FAIL;}}\n");
    Ok(out)
}
