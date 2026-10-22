//! Loop-region law extraction (normal-form design section 6.2): counted regions, inlined
//! body-local definitions, per-variable linear forms joined in a class lattice,
//! carried strongly connected sets, argmin companions and array footprint
//! classes. Each applied law produces a certificate.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    ArrayLaw, Carried, Certificate, Fact, Guard, IntTy, LawId, LoopForm, LoopNode, Node, Order,
    Rep, Ring, ScalarTy, Subject, VERSION,
    linear::{
        const_val, is_one, linform, mentions_any, names_of, outer_modulus, uses_wrapping,
        wrapping_state, zero,
    },
};
use crate::{
    front::{BinaryOp, Mode, Type},
    general_net::regions::{
        self, Counting, Partition, RegionId,
        tree::{self, Expr, FunctionTree, Loop, LoopKind, Stmt, same},
    },
    laws::Law,
};

pub(super) struct Context<'a> {
    pub partition: &'a Partition,
    pub fingerprint: [u8; 32],
    pub tree: &'a FunctionTree,
}

/// Matrix groups larger than this stay serial; above 4 they are cost-gated.
pub(super) const MAX_MATRIX: usize = 16;

impl Context<'_> {
    pub fn pure_callee(&self, name: &str) -> Option<bool> {
        self.partition
            .functions
            .iter()
            .find(|function| function.name == name)
            .map(|function| function.pure)
    }

    fn certificate(
        &self,
        region: RegionId,
        law: LawId,
        subject: Subject,
        rep: Option<Rep>,
        guards: Vec<Guard>,
        facts: Vec<Fact>,
    ) -> Certificate {
        Certificate {
            version: VERSION,
            fingerprint: self.fingerprint,
            region,
            law,
            subject,
            rep,
            guards,
            facts,
            annotations: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Const,
    WAdd,
    WMul,
    WAff,
    Add,
    Mul,
    AffZ,
    AffMod(u32),
    MatMod(u32),
    MatZ,
    Min,
    Max,
    MaxPlus,
    MinPlus,
    FAdd,
    FAff,
    FMat,
    Fsm,
    Last,
    Arg,
}

struct Update {
    value: Expr,
    guards: Vec<(Expr, bool)>,
}

struct StoreSite<'a> {
    index: &'a Expr,
    value: &'a Expr,
    guards: Vec<(&'a Expr, bool)>,
    depth: usize,
}

struct Body<'a> {
    induction: &'a str,
    hi: Option<&'a Expr>,
    lo_zero: bool,
    body: &'a [Stmt],
    declared: BTreeSet<String>,
    inner: BTreeSet<String>,
}

/// Classify every loop of the function, in source pre-order, and apply L19
/// fusion between adjacent maps.
pub(super) fn extract_loops(cx: &Context<'_>) -> Vec<LoopForm> {
    let mut forms = Vec::new();
    visit_lists(&cx.tree.body, &mut |list| {
        let mut previous: Option<usize> = None;
        let mut between: Vec<&Stmt> = Vec::new();
        for stmt in list {
            if let Stmt::Loop(inner) = stmt {
                let index = forms.len();
                forms.push(classify(cx, inner));
                if let Some(first) = previous {
                    fuse(cx, list, first, index, &between, &mut forms);
                }
                previous = Some(index);
                between.clear();
            } else if previous.is_some() {
                between.push(stmt);
            }
        }
    });
    forms.sort_by_key(|form| order_of(&cx.tree.body, form.region));
    forms
}

fn order_of(body: &[Stmt], region: RegionId) -> usize {
    let mut position = 0;
    let mut found = usize::MAX;
    tree::for_each_loop(body, &mut |inner| {
        if Some(inner.head) == region.head && found == usize::MAX {
            found = position;
        }
        position += 1;
    });
    found
}

fn visit_lists<'a>(body: &'a [Stmt], visit: &mut dyn FnMut(&'a [Stmt])) {
    visit(body);
    for stmt in body {
        match stmt {
            Stmt::If { yes, no, .. } => {
                visit_lists(yes, visit);
                visit_lists(no, visit);
            }
            Stmt::Loop(inner) => visit_lists(&inner.body, visit),
            _ => {}
        }
    }
}

fn serial(region: RegionId, reason: impl Into<String>) -> LoopForm {
    LoopForm {
        region,
        node: Node::Serial {
            reason: reason.into(),
        },
        certificates: Vec::new(),
    }
}

fn region_of(cx: &Context<'_>, inner: &Loop) -> RegionId {
    RegionId {
        function: cx.tree.index,
        head: Some(inner.head),
    }
}

fn shape<'a>(inner: &'a Loop) -> Option<Body<'a>> {
    let (induction, hi, lo_zero, body) = match (&inner.kind, regions::counting(inner)) {
        (LoopKind::For { var, lo, hi }, _) => (
            var.as_str(),
            Some(hi),
            lo.literal() == Some(0),
            inner.body.as_slice(),
        ),
        (LoopKind::While { cond }, Counting::CountedWhile { .. }) => {
            let Expr::Binary { left, right, .. } = cond else {
                return None;
            };
            let Expr::Local(induction) = left.as_ref() else {
                return None;
            };
            let (_, rest) = inner.body.split_last()?;
            (induction.as_str(), Some(right.as_ref()), false, rest)
        }
        _ => return None,
    };
    let mut declared = BTreeSet::new();
    let mut inner_vars = BTreeSet::new();
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| match stmt {
        Stmt::Let { name, .. } => {
            declared.insert(name.clone());
        }
        Stmt::Loop(nested) => {
            if let LoopKind::For { var, .. } = &nested.kind {
                inner_vars.insert(var.clone());
                declared.insert(var.clone());
            }
        }
        _ => {}
    });
    Some(Body {
        induction,
        hi,
        lo_zero,
        body,
        declared,
        inner: inner_vars,
    })
}

fn local_type(cx: &Context<'_>, name: &str) -> Option<Type> {
    cx.tree.locals.get(name).cloned()
}

fn classify(cx: &Context<'_>, inner: &Loop) -> LoopForm {
    let region = region_of(cx, inner);
    let Some(body) = shape(inner) else {
        return serial(region, "uncounted while (data-dependent trip count)");
    };
    let mut updates: BTreeMap<String, Vec<Update>> = BTreeMap::new();
    let mut stores: BTreeMap<String, Vec<StoreSite<'_>>> = BTreeMap::new();
    let mut early_return = false;
    let mut impure: Option<&'static str> = None;
    tree::walk(body.body, &mut Vec::new(), 0, &mut |stmt, guards, depth| {
        for expr in tree::stmt_exprs(stmt) {
            effects(cx, expr, &body, &mut impure);
        }
        match stmt {
            Stmt::Assign { name, value }
                if !body.declared.contains(name) && name != body.induction =>
            {
                updates.entry(name.clone()).or_default().push(Update {
                    value: value.clone(),
                    guards: guards.iter().map(|(g, p)| ((*g).clone(), *p)).collect(),
                });
            }
            Stmt::Assign { name, .. } if name == body.induction => {
                impure.get_or_insert("induction assigned in body");
            }
            Stmt::Store {
                array,
                index,
                value,
                ..
            } if !body.declared.contains(array) => {
                stores.entry(array.clone()).or_default().push(StoreSite {
                    index,
                    value,
                    guards: guards.to_vec(),
                    depth,
                });
            }
            Stmt::Return { .. } => early_return = true,
            _ => {}
        }
    });
    if let Some(reason) = impure {
        return serial(region, reason);
    }
    let carried: BTreeSet<String> = updates.keys().cloned().collect();
    let expanded = inline_locals(body.body, &updates, &body.declared, &carried);
    let mut classes: BTreeMap<String, Option<Class>> = BTreeMap::new();
    for (var, sites) in &expanded {
        let class = sites
            .as_ref()
            .and_then(|sites| scalar_class(cx, var, sites));
        classes.insert(var.clone(), class);
    }
    let deps = dependencies(&expanded, &carried);
    let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for var in &carried {
        let scc: BTreeSet<String> = carried
            .iter()
            .filter(|w| reach(&deps, var, w) && reach(&deps, w, var))
            .cloned()
            .collect();
        if scc.contains(var) {
            let class = joint_linear(cx, &scc, &expanded);
            classes.insert(var.clone(), class);
            groups.insert(var.clone(), scc);
        } else if classes.get(var) == Some(&Some(Class::Last))
            && deps[var]
                .iter()
                .any(|w| matches!(classes.get(w), Some(Some(Class::Min | Class::Max))))
        {
            classes.insert(var.clone(), Some(Class::Arg));
        }
    }
    let mut node = LoopNode {
        counting: regions::counting(inner),
        carried: Vec::new(),
        arrays: Vec::new(),
        first_exit: early_return,
        fused_with: None,
    };
    let mut certificates = Vec::new();
    if let Err(reason) = carried_laws(
        cx,
        region,
        &expanded,
        &classes,
        &groups,
        &deps,
        body.body,
        &mut node,
        &mut certificates,
    ) {
        return serial(region, reason);
    }
    if let Err(reason) = array_laws(
        cx,
        region,
        &body,
        &stores,
        &classes,
        &carried,
        &mut node,
        &mut certificates,
    ) {
        return serial(region, reason);
    }
    if early_return {
        certificates.push(cx.certificate(
            region,
            LawId::L15FirstExit,
            Subject::Exit,
            None,
            Vec::new(),
            Vec::new(),
        ));
    }
    certificates.insert(
        0,
        cx.certificate(
            region,
            LawId::L1Map,
            Subject::Iterations,
            None,
            Vec::new(),
            Vec::new(),
        ),
    );
    LoopForm {
        region,
        node: Node::Parallel(node),
        certificates,
    }
}

/// Calls must reach only pure functions and intrinsics; no `mut` loans, and no
/// owner of the enclosing frame moves inside the loop.
fn effects(cx: &Context<'_>, expr: &Expr, body: &Body<'_>, impure: &mut Option<&'static str>) {
    let mut calls = Vec::new();
    expr.calls(&mut calls);
    for call in calls {
        if let Expr::Call { name, args, .. } = call {
            if cx.pure_callee(name) == Some(false) {
                impure.get_or_insert("call with host effect or mut parameter");
            }
            if args.iter().any(|arg| {
                matches!(
                    arg,
                    Expr::Owner {
                        loan: Some(Mode::Mut),
                        ..
                    }
                )
            }) {
                impure.get_or_insert("mut loan inside loop");
            }
        }
    }
    let mut names = Vec::new();
    collect_moves(expr, &mut names);
    if names.iter().any(|name| !body.declared.contains(name)) {
        impure.get_or_insert("owner moved inside loop");
    }
}

fn collect_moves(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Move(name) => out.push(name.clone()),
        Expr::ArrayLit { items, .. } => items.iter().for_each(|item| collect_moves(item, out)),
        Expr::Index { index, .. } => collect_moves(index, out),
        Expr::Unary { value, .. } => collect_moves(value, out),
        Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => {
            collect_moves(left, out);
            collect_moves(right, out);
        }
        Expr::Call { args, .. } => args.iter().for_each(|arg| collect_moves(arg, out)),
        _ => {}
    }
}

/// Substitute single-definition body-local lets into the updates; an update
/// reading a reassigned local that depends on carried state becomes `None`.
fn inline_locals(
    body: &[Stmt],
    updates: &BTreeMap<String, Vec<Update>>,
    declared: &BTreeSet<String>,
    carried: &BTreeSet<String>,
) -> BTreeMap<String, Option<Vec<Update>>> {
    let mut defs: BTreeMap<String, Option<Expr>> = BTreeMap::new();
    let mut reassigned = BTreeSet::new();
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| match stmt {
        Stmt::Let { name, value } => {
            let entry = if defs.contains_key(name) {
                None
            } else {
                Some(value.clone())
            };
            defs.insert(name.clone(), entry);
        }
        Stmt::Assign { name, .. } if declared.contains(name) => {
            reassigned.insert(name.clone());
        }
        _ => {}
    });
    let mut tainted: BTreeSet<String> = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        tree::walk(body, &mut Vec::new(), 0, &mut |stmt, guards, _| {
            let (name, mut sources) = match stmt {
                Stmt::Let { name, value } => (name, names_of(value)),
                Stmt::Assign { name, value } => {
                    let mut sources = names_of(value);
                    for (guard, _) in guards {
                        sources.extend(names_of(guard));
                    }
                    (name, sources)
                }
                _ => return,
            };
            if !declared.contains(name) || tainted.contains(name) {
                return;
            }
            sources.retain(|source| carried.contains(source) || tainted.contains(source));
            if !sources.is_empty() {
                tainted.insert(name.clone());
                changed = true;
            }
        });
    }
    let bad: BTreeSet<String> = tainted.intersection(&reassigned).cloned().collect();
    updates
        .iter()
        .map(|(var, sites)| {
            let expanded: Vec<Update> = sites
                .iter()
                .map(|site| Update {
                    value: expand(&site.value, &defs, &reassigned, 0),
                    guards: site
                        .guards
                        .iter()
                        .map(|(g, p)| (expand(g, &defs, &reassigned, 0), *p))
                        .collect(),
                })
                .collect();
            let poisoned = expanded.iter().any(|site| {
                mentions_any(&site.value, &bad)
                    || site.guards.iter().any(|(g, _)| mentions_any(g, &bad))
            });
            (var.clone(), (!poisoned).then_some(expanded))
        })
        .collect()
}

fn expand(
    expr: &Expr,
    defs: &BTreeMap<String, Option<Expr>>,
    reassigned: &BTreeSet<String>,
    depth: usize,
) -> Expr {
    if depth > 30 {
        return expr.clone();
    }
    let recurse = |e: &Expr| Box::new(expand(e, defs, reassigned, depth));
    match expr {
        Expr::Local(name) => match defs.get(name) {
            Some(Some(definition)) if !reassigned.contains(name) => {
                expand(definition, defs, reassigned, depth + 1)
            }
            _ => expr.clone(),
        },
        Expr::Index { array, index, span } => Expr::Index {
            array: array.clone(),
            index: recurse(index),
            span: *span,
        },
        Expr::Unary { op, value, span } => Expr::Unary {
            op: *op,
            value: recurse(value),
            span: *span,
        },
        Expr::Binary {
            op,
            left,
            right,
            span,
        } => Expr::Binary {
            op: *op,
            left: recurse(left),
            right: recurse(right),
            span: *span,
        },
        Expr::Logic { and, left, right } => Expr::Logic {
            and: *and,
            left: recurse(left),
            right: recurse(right),
        },
        Expr::Call { name, args, span } => Expr::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| expand(arg, defs, reassigned, depth))
                .collect(),
            span: *span,
        },
        _ => expr.clone(),
    }
}

fn guard_names(guards: &[(Expr, bool)]) -> BTreeSet<String> {
    guards.iter().flat_map(|(g, _)| names_of(g)).collect()
}

/// `min`/`max` select: the innermost guard compares the assigned value with
/// the variable, e.g. `if x < v { v = x }`. Returns the order and strictness.
fn select_kind(var: &str, value: &Expr, guards: &[(Expr, bool)]) -> Option<(Order, bool)> {
    let (
        Expr::Binary {
            op, left, right, ..
        },
        true,
    ) = guards.last()?
    else {
        return None;
    };
    let strict = matches!(op, BinaryOp::Lt | BinaryOp::Gt);
    let less = matches!(op, BinaryOp::Lt | BinaryOp::Le);
    if !matches!(
        op,
        BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge
    ) {
        return None;
    }
    let local = Expr::Local(var.to_owned());
    if same(left, value) && **right == local {
        return Some((if less { Order::Min } else { Order::Max }, strict));
    }
    if same(right, value) && **left == local {
        return Some((if less { Order::Max } else { Order::Min }, strict));
    }
    None
}

fn scalar_class(cx: &Context<'_>, var: &str, sites: &[Update]) -> Option<Class> {
    let ty = local_type(cx, var)?;
    let own: BTreeSet<String> = BTreeSet::from([var.to_owned()]);
    let mut kinds: Vec<Class> = Vec::new();
    let mut moduli = BTreeSet::new();
    for site in sites {
        let mention = guard_names(&site.guards).contains(var);
        if let Some((order, _)) = select_kind(var, &site.value, &site.guards) {
            kinds.push(if order == Order::Min {
                Class::Min
            } else {
                Class::Max
            });
            continue;
        }
        if mention {
            return None;
        }
        if !mentions_any(&site.value, &own) {
            kinds.push(Class::Const);
            continue;
        }
        if uses_wrapping(&site.value) && !wrapping_state(&site.value, &own) {
            return None;
        }
        if let Some((inner, modulus)) = outer_modulus(&site.value) {
            let form = linform(inner, &own)?;
            if form.keys().flatten().all(|key| key == var)
                && form.contains_key(&Some(var.to_owned()))
            {
                moduli.insert(modulus);
                kinds.push(Class::AffMod(modulus));
                continue;
            }
            return None;
        }
        if let Expr::Index { index, .. } = &site.value
            && names_of(index).contains(var)
        {
            kinds.push(Class::Fsm);
            continue;
        }
        let form = linform(&site.value, &own)?;
        let keys: BTreeSet<&String> = form.keys().flatten().collect();
        if keys.len() != 1 || !keys.contains(&var.to_owned()) {
            return None;
        }
        let coefficient = &form[&Some(var.to_owned())];
        let wrap = uses_wrapping(&site.value);
        let constant = form.contains_key(&None);
        kinds.push(
            match (is_one(coefficient), ty == Type::F32, wrap, constant) {
                (true, true, _, _) => Class::FAdd,
                (true, false, true, _) => Class::WAdd,
                (true, false, false, _) => Class::Add,
                (false, false, false, false) => Class::Mul,
                (false, false, true, false) => Class::WMul,
                (false, true, _, _) => Class::FAff,
                (false, false, true, true) => Class::WAff,
                (false, false, false, true) => Class::AffZ,
            },
        );
    }
    let all = |allowed: &[Class]| kinds.iter().all(|kind| allowed.contains(kind));
    let has = |kind: Class| kinds.contains(&kind);
    if kinds.is_empty() {
        return None;
    }
    if all(&[Class::Const]) {
        return Some(Class::Last);
    }
    if all(&[Class::WAdd]) {
        return Some(Class::WAdd);
    }
    if all(&[Class::WMul]) {
        return Some(Class::WMul);
    }
    if all(&[Class::WAdd, Class::WMul, Class::WAff, Class::Const]) {
        return Some(Class::WAff);
    }
    if all(&[Class::Add]) {
        return Some(Class::Add);
    }
    if all(&[Class::Mul]) {
        return Some(Class::Mul);
    }
    if all(&[Class::Add, Class::AffZ, Class::Const, Class::Mul]) {
        return Some(Class::AffZ);
    }
    if moduli.len() == 1
        && kinds
            .iter()
            .all(|kind| matches!(kind, Class::AffMod(_) | Class::Const))
    {
        return moduli.first().map(|&p| Class::AffMod(p));
    }
    if all(&[Class::Min]) {
        return Some(Class::Min);
    }
    if all(&[Class::Max]) {
        return Some(Class::Max);
    }
    if all(&[Class::Max, Class::Add, Class::Const]) && has(Class::Max) {
        return Some(Class::MaxPlus);
    }
    if all(&[Class::Min, Class::Add, Class::Const]) && has(Class::Min) {
        return Some(Class::MinPlus);
    }
    if all(&[Class::FAdd]) {
        return Some(Class::FAdd);
    }
    if all(&[Class::FAdd, Class::FAff, Class::Const]) {
        return Some(Class::FAff);
    }
    if all(&[Class::Fsm]) {
        return Some(Class::Fsm);
    }
    None
}

fn dependencies(
    expanded: &BTreeMap<String, Option<Vec<Update>>>,
    carried: &BTreeSet<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    expanded
        .iter()
        .map(|(var, sites)| {
            let mut deps = BTreeSet::new();
            for site in sites.iter().flatten() {
                deps.extend(names_of(&site.value));
                deps.extend(guard_names(&site.guards));
            }
            deps.retain(|name| carried.contains(name) && name != var);
            (var.clone(), deps)
        })
        .collect()
}

fn reach(deps: &BTreeMap<String, BTreeSet<String>>, from: &str, to: &str) -> bool {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from.to_owned()];
    while let Some(current) = stack.pop() {
        for next in deps.get(&current).into_iter().flatten() {
            if next == to {
                return true;
            }
            if seen.insert(next.clone()) {
                stack.push(next.clone());
            }
        }
    }
    false
}

fn joint_linear(
    cx: &Context<'_>,
    scc: &BTreeSet<String>,
    expanded: &BTreeMap<String, Option<Vec<Update>>>,
) -> Option<Class> {
    let mut moduli = BTreeSet::new();
    let mut f32 = false;
    for var in scc {
        f32 |= local_type(cx, var) == Some(Type::F32);
        for site in expanded.get(var)?.as_ref()? {
            if uses_wrapping(&site.value) && !wrapping_state(&site.value, scc) {
                return None;
            }
            if guard_names(&site.guards)
                .iter()
                .any(|name| scc.contains(name))
            {
                return None;
            }
            let inner = match outer_modulus(&site.value) {
                Some((inner, modulus)) => {
                    moduli.insert(Some(modulus));
                    inner
                }
                None => {
                    moduli.insert(None);
                    &site.value
                }
            };
            let form = linform(inner, scc)?;
            if form.values().any(|c| mentions_any(c, scc)) {
                return None;
            }
        }
    }
    if moduli.len() > 1 {
        return None;
    }
    if f32 {
        return Some(Class::FMat);
    }
    match moduli.first() {
        Some(Some(p)) => Some(Class::MatMod(*p)),
        _ => Some(Class::MatZ),
    }
}

fn int_ty(cx: &Context<'_>, var: &str) -> Option<IntTy> {
    match local_type(cx, var)? {
        Type::I32 => Some(IntTy::I32),
        Type::U32 => Some(IntTy::U32),
        _ => None,
    }
}

fn scalar_ty(cx: &Context<'_>, var: &str) -> Option<ScalarTy> {
    match local_type(cx, var)? {
        Type::I32 => Some(ScalarTy::I32),
        Type::U32 => Some(ScalarTy::U32),
        Type::F32 => Some(ScalarTy::F32),
        _ => None,
    }
}

/// Facts for one variable's updates: affine `(a, b)` per update, or resets and
/// selects, in source order.
fn update_facts(var: &str, sites: &[Update]) -> Vec<Fact> {
    let own: BTreeSet<String> = BTreeSet::from([var.to_owned()]);
    sites
        .iter()
        .map(|site| {
            if select_kind(var, &site.value, &site.guards).is_some() {
                return Fact::Select(site.value.clone());
            }
            if !mentions_any(&site.value, &own) {
                return Fact::Reset(site.value.clone());
            }
            let inner = outer_modulus(&site.value).map_or(&site.value, |(inner, _)| inner);
            match linform(inner, &own) {
                Some(form) => Fact::Affine {
                    coefficient: form
                        .get(&Some(var.to_owned()))
                        .cloned()
                        .unwrap_or_else(zero),
                    offset: form.get(&None).cloned().unwrap_or_else(zero),
                },
                None => Fact::Transition(site.value.clone()),
            }
        })
        .collect()
}

fn ring_for(cx: &Context<'_>, var: &str, sites: &[Update], modulus: Option<u32>) -> Option<Ring> {
    let ty = int_ty(cx, var)?;
    if let Some(modulus) = modulus {
        return Some(Ring::Residue { modulus, ty });
    }
    let wrapping = sites.iter().filter(|s| uses_wrapping(&s.value)).count();
    match wrapping {
        0 => Some(Ring::Checked64(ty)),
        n if n == sites.len() && ty == IntTy::U32 => Some(Ring::Wrapping32),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn carried_laws(
    cx: &Context<'_>,
    region: RegionId,
    expanded: &BTreeMap<String, Option<Vec<Update>>>,
    classes: &BTreeMap<String, Option<Class>>,
    groups: &BTreeMap<String, BTreeSet<String>>,
    deps: &BTreeMap<String, BTreeSet<String>>,
    body: &[Stmt],
    node: &mut LoopNode,
    certificates: &mut Vec<Certificate>,
) -> Result<(), String> {
    let bad: Vec<&String> = classes
        .iter()
        .filter(|(_, class)| class.is_none())
        .map(|(var, _)| var)
        .collect();
    if !bad.is_empty() {
        return Err(format!("carried {bad:?} not lawful"));
    }
    let scan = scanned(body, classes);
    // Successive scans: emit variables whose dependencies are already resolved.
    let mut resolved: BTreeSet<String> = BTreeSet::new();
    let mut pending: Vec<&String> = classes.keys().collect();
    while !pending.is_empty() {
        let before = pending.len();
        let mut next = Vec::new();
        for var in pending {
            let Some(Some(class)) = classes.get(var) else {
                continue;
            };
            if *class == Class::Arg {
                continue;
            }
            let group = groups.get(var);
            let members: BTreeSet<String> = group
                .cloned()
                .unwrap_or_else(|| BTreeSet::from([var.clone()]));
            if resolved.contains(var) {
                continue;
            }
            let waiting = members.iter().any(|member| {
                deps[member].iter().any(|d| {
                    !members.contains(d)
                        && !resolved.contains(d)
                        && classes.get(d) != Some(&Some(Class::Arg))
                })
            });
            if waiting {
                next.push(var);
                continue;
            }
            let sites = expanded
                .get(var)
                .and_then(Option::as_ref)
                .ok_or_else(|| format!("carried {var} not lawful"))?;
            let (law, rep, guards) =
                law_for(cx, var, *class, sites, &members, classes, expanded, deps)?;
            let facts = if members.len() > 1 {
                Vec::new()
            } else {
                update_facts(var, sites)
            };
            certificates.push(cx.certificate(
                region,
                law,
                Subject::Var(var.clone()),
                Some(rep.clone()),
                guards,
                facts,
            ));
            node.carried.push(Carried {
                var: var.clone(),
                law,
                rep,
                scan: members.iter().any(|m| scan.contains(m)),
            });
            resolved.extend(members);
        }
        pending = next;
        if pending.len() == before {
            return Err("cyclic carried dependencies".to_owned());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn law_for(
    cx: &Context<'_>,
    var: &str,
    class: Class,
    sites: &[Update],
    members: &BTreeSet<String>,
    classes: &BTreeMap<String, Option<Class>>,
    expanded: &BTreeMap<String, Option<Vec<Update>>>,
    deps: &BTreeMap<String, BTreeSet<String>>,
) -> Result<(LawId, Rep, Vec<Guard>), String> {
    let int = || int_ty(cx, var).ok_or_else(|| format!("carried {var} has no integer type"));
    let residue_guards = |var: &str| {
        vec![
            Guard::ReducedSeed(var.to_owned()),
            Guard::NonNegativeCoefficients(var.to_owned()),
        ]
    };
    Ok(match class {
        Class::WAdd => (
            LawId::L2SealedFold,
            Rep::Sealed(Law::WrappingSum),
            Vec::new(),
        ),
        Class::WMul => (
            LawId::L2SealedFold,
            Rep::Sealed(Law::WrappingProduct),
            Vec::new(),
        ),
        Class::WAff => (LawId::L4Affine, Rep::Affine(Ring::Wrapping32), Vec::new()),
        Class::Add => (LawId::L3TrapLiftedFold, Rep::LiftedSum(int()?), Vec::new()),
        Class::Mul => (
            LawId::L3TrapLiftedFold,
            Rep::SaturatedProduct(int()?),
            Vec::new(),
        ),
        Class::AffZ => {
            let ring =
                ring_for(cx, var, sites, None).ok_or("mixed wrapping and checked updates")?;
            (LawId::L4Affine, Rep::Affine(ring), Vec::new())
        }
        Class::AffMod(modulus) => {
            let ring = ring_for(cx, var, sites, Some(modulus)).ok_or("residue on non-integer")?;
            (LawId::L4Affine, Rep::Affine(ring), residue_guards(var))
        }
        Class::MatMod(_) | Class::MatZ => {
            if members.len() > MAX_MATRIX {
                return Err(format!("matrix group of {} variables", members.len()));
            }
            let modulus = match class {
                Class::MatMod(p) => Some(p),
                _ => None,
            };
            let mut all_sites = Vec::new();
            let mut tys = BTreeSet::new();
            for member in members {
                tys.insert(int_ty(cx, member).ok_or("matrix over non-integer")?);
                if let Some(Some(member_sites)) = expanded.get(member) {
                    all_sites.extend(member_sites.iter().map(|s| Update {
                        value: s.value.clone(),
                        guards: s.guards.clone(),
                    }));
                }
            }
            if tys.len() != 1 {
                return Err("matrix over mixed integer types".to_owned());
            }
            let ring = ring_for(cx, var, &all_sites, modulus)
                .ok_or("mixed wrapping and checked updates")?;
            let guards = if modulus.is_some() {
                members.iter().flat_map(|m| residue_guards(m)).collect()
            } else {
                Vec::new()
            };
            (
                LawId::L5Matrix,
                Rep::Matrix {
                    ring,
                    vars: members.iter().cloned().collect(),
                },
                guards,
            )
        }
        Class::MaxPlus | Class::MinPlus => (
            LawId::L6Tropical,
            Rep::Tropical {
                order: if class == Class::MaxPlus {
                    Order::Max
                } else {
                    Order::Min
                },
                ty: int()?,
            },
            Vec::new(),
        ),
        Class::Min | Class::Max => {
            let order = if class == Class::Min {
                Order::Min
            } else {
                Order::Max
            };
            let mut strict = BTreeSet::new();
            for site in sites {
                let (_, s) = select_kind(var, &site.value, &site.guards).ok_or("select shape")?;
                strict.insert(s);
            }
            let [strict] = strict.into_iter().collect::<Vec<_>>()[..] else {
                return Err("mixed strict and non-strict selects".to_owned());
            };
            let companions = companions_of(var, sites, classes, expanded, deps)?;
            (
                LawId::L7Select,
                Rep::Select {
                    order,
                    strict,
                    elem: scalar_ty(cx, var).ok_or("select on non-scalar")?,
                    companions,
                },
                Vec::new(),
            )
        }
        Class::Last => (LawId::L12OrderedWrite, Rep::Last, Vec::new()),
        Class::FAdd | Class::FAff | Class::FMat => {
            return Err("needs f32 reassociation (strict f32 order)".to_owned());
        }
        Class::Fsm => super::fsm::law(cx, var, sites.iter().map(|s| &s.value))?,
        Class::Const | Class::Arg => return Err(format!("carried {var} not lawful")),
    })
}

/// Argmin companions: `Last` variables assigned under the same guards as the
/// select variable's updates.
fn companions_of(
    var: &str,
    sites: &[Update],
    classes: &BTreeMap<String, Option<Class>>,
    expanded: &BTreeMap<String, Option<Vec<Update>>>,
    deps: &BTreeMap<String, BTreeSet<String>>,
) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for (other, class) in classes {
        if *class != Some(Class::Arg) || !deps[other].contains(var) {
            continue;
        }
        let owners: Vec<&String> = deps[other]
            .iter()
            .filter(|d| matches!(classes.get(*d), Some(Some(Class::Min | Class::Max))))
            .collect();
        if owners.len() != 1 {
            return Err(format!("companion {other} of several selects"));
        }
        let other_sites = expanded
            .get(other)
            .and_then(Option::as_ref)
            .ok_or("companion not lawful")?;
        let aligned = other_sites.iter().all(|site| {
            sites.iter().any(|own| {
                own.guards.len() == site.guards.len()
                    && own
                        .guards
                        .iter()
                        .zip(&site.guards)
                        .all(|((a, p), (b, q))| p == q && same(a, b))
            })
        });
        if !aligned {
            return Err(format!("companion {other} outside its select"));
        }
        out.push(other.clone());
    }
    Ok(out)
}

fn scanned(body: &[Stmt], classes: &BTreeMap<String, Option<Class>>) -> BTreeSet<String> {
    let carried: BTreeSet<String> = classes.keys().cloned().collect();
    let mut used = BTreeSet::new();
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, guards, _| {
        let own = match stmt {
            Stmt::Assign { name, .. } => Some(name),
            _ => None,
        };
        for expr in tree::stmt_exprs(stmt) {
            for name in names_of(expr) {
                if carried.contains(&name) && Some(&name) != own {
                    used.insert(name);
                }
            }
        }
        let arg_owner = own.is_some_and(|o| classes.get(o) == Some(&Some(Class::Arg)));
        for (guard, _) in guards {
            for name in names_of(guard) {
                if carried.contains(&name) && !arg_owner {
                    used.insert(name);
                }
            }
        }
    });
    used.retain(|name| !matches!(classes.get(name), Some(None | Some(Class::Arg))));
    used
}

fn index_reads(body: &[Stmt]) -> Vec<(&str, &Expr)> {
    fn collect<'a>(expr: &'a Expr, out: &mut Vec<(&'a str, &'a Expr)>) {
        match expr {
            Expr::Index { array, index, .. } => {
                out.push((array, index));
                collect(index, out);
            }
            Expr::ArrayLit { items, .. } => items.iter().for_each(|item| collect(item, out)),
            Expr::Unary { value, .. } => collect(value, out),
            Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => {
                collect(left, out);
                collect(right, out);
            }
            Expr::Call { args, .. } => args.iter().for_each(|arg| collect(arg, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        for expr in tree::stmt_exprs(stmt) {
            collect(expr, &mut out);
        }
    });
    out
}

/// Owner references (call arguments, moves) of `array` inside the body.
fn owner_used(body: &[Stmt], array: &str) -> bool {
    let mut used = false;
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        for expr in tree::stmt_exprs(stmt) {
            let mut stack = vec![expr];
            while let Some(e) = stack.pop() {
                match e {
                    Expr::Owner { name, .. } | Expr::Move(name) if name == array => used = true,
                    Expr::ArrayLit { items, .. } | Expr::Call { args: items, .. } => {
                        stack.extend(items)
                    }
                    Expr::Index { index, .. } => stack.push(index),
                    Expr::Unary { value, .. } => stack.push(value),
                    Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => {
                        stack.push(left);
                        stack.push(right);
                    }
                    _ => {}
                }
            }
        }
    });
    used
}

pub(super) enum Injective {
    Yes,
    /// Injective when the named loop-invariant stride is not zero.
    Stride(String),
    No,
}

/// Iteration-injective store index: the induction, `a*i + r` with
/// loop-invariant `a` and `r`, or mixed radix `w*K + i` / `i*K + w`.
fn injective(ix: &Expr, body: &Body<'_>, varying: &BTreeSet<String>) -> Injective {
    let ind = body.induction;
    if *ix == Expr::Local(ind.to_owned()) {
        return Injective::Yes;
    }
    let own: BTreeSet<String> = BTreeSet::from([ind.to_owned()]);
    if let Some(form) = linform(ix, &own)
        && form.contains_key(&Some(ind.to_owned()))
        && form.keys().flatten().all(|key| key == ind)
    {
        let stride = &form[&Some(ind.to_owned())];
        let mut rest = names_of(stride);
        if let Some(constant) = form.get(&None) {
            rest.extend(names_of(constant));
        }
        if !rest
            .iter()
            .any(|name| varying.contains(name) || name == ind)
        {
            return match (const_val(stride), stride) {
                (Some(0), _) => Injective::No,
                (Some(_), _) => Injective::Yes,
                (None, Expr::Local(name)) => Injective::Stride(name.clone()),
                _ => Injective::No,
            };
        }
    }
    if !body.lo_zero {
        return Injective::No;
    }
    if let Expr::Binary {
        op: BinaryOp::Add,
        left,
        right,
        ..
    } = ix
    {
        for (term, other) in [(left, right), (right, left)] {
            if **term == Expr::Local(ind.to_owned())
                && let Expr::Binary {
                    op: BinaryOp::Mul,
                    left: a,
                    right: b,
                    ..
                } = other.as_ref()
                && let Some(hi) = body.hi
                && (same(a, hi) || same(b, hi))
            {
                return Injective::Yes;
            }
            if let Expr::Binary {
                op: BinaryOp::Mul,
                left: a,
                right: b,
                ..
            } = term.as_ref()
                && let Expr::Local(w) = other.as_ref()
                && body.inner.contains(w)
            {
                let k = if **a == Expr::Local(ind.to_owned()) {
                    b
                } else if **b == Expr::Local(ind.to_owned()) {
                    a
                } else {
                    continue;
                };
                if inner_extent(body.body, w).is_some_and(|extent| same(extent, k)) {
                    return Injective::Yes;
                }
            }
        }
    }
    Injective::No
}

/// `hi` of the inner `for w in 0 .. hi` loop.
fn inner_extent<'a>(body: &'a [Stmt], var: &str) -> Option<&'a Expr> {
    let mut found = None;
    tree::for_each_loop(body, &mut |inner| {
        if let LoopKind::For { var: v, lo, hi } = &inner.kind
            && v == var
            && lo.literal() == Some(0)
        {
            found = Some(hi);
        }
    });
    found
}

/// `K` when `ix = i*K + r` with literal `K` and `0 <= r < K` for every inner
/// loop value.
fn block_of(ix: &Expr, body: &Body<'_>) -> Option<i64> {
    if !body.lo_zero {
        return None;
    }
    let ind = body.induction;
    let own: BTreeSet<String> = BTreeSet::from([ind.to_owned()]);
    let form = linform(ix, &own)?;
    if !form.contains_key(&Some(ind.to_owned())) || form.keys().flatten().any(|key| key != ind) {
        return None;
    }
    let k = const_val(&form[&Some(ind.to_owned())]).filter(|k| *k > 0)?;
    let mut extents = BTreeMap::new();
    tree::for_each_loop(body.body, &mut |inner| {
        if let LoopKind::For { var, lo, hi } = &inner.kind
            && lo.literal() == Some(0)
            && let Some(h) = const_val(hi)
        {
            extents.insert(var.clone(), h - 1);
        }
    });
    let r = match form.get(&None) {
        Some(rest) => upper(rest, &extents)?,
        None => 0,
    };
    (r < k).then_some(k)
}

fn upper(expr: &Expr, extents: &BTreeMap<String, i64>) -> Option<i64> {
    if let Some(v) = const_val(expr) {
        return (v >= 0).then_some(v);
    }
    match expr {
        Expr::Local(name) => extents.get(name).copied(),
        Expr::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Mul),
            left,
            right,
            ..
        } => {
            let (a, b) = (upper(left, extents)?, upper(right, extents)?);
            if *op == BinaryOp::Add {
                a.checked_add(b)
            } else {
                a.checked_mul(b)
            }
        }
        _ => None,
    }
}

/// `a[k] op e` with `a[k]` the stored element: `Some(wrapping)`.
fn combine_update(array: &str, index: &Expr, value: &Expr) -> Option<bool> {
    let target = |e: &Expr| matches!(e, Expr::Index { array: a, index: i, .. } if a == array && same(i, index));
    let reads_target = |e: &Expr| e.mentions(array);
    match value {
        Expr::Binary {
            op: BinaryOp::Add,
            left,
            right,
            ..
        } => {
            let (t, e) = if target(left) {
                (left, right)
            } else {
                (right, left)
            };
            (target(t) && !reads_target(e)).then_some(false)
        }
        Expr::Call { name, args, .. } if name == "wrapping_add" && args.len() == 2 => {
            let (t, e) = if target(&args[0]) {
                (&args[0], &args[1])
            } else {
                (&args[1], &args[0])
            };
            (target(t) && !reads_target(e)).then_some(true)
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn array_laws(
    cx: &Context<'_>,
    region: RegionId,
    body: &Body<'_>,
    stores: &BTreeMap<String, Vec<StoreSite<'_>>>,
    classes: &BTreeMap<String, Option<Class>>,
    carried: &BTreeSet<String>,
    node: &mut LoopNode,
    certificates: &mut Vec<Certificate>,
) -> Result<(), String> {
    let reads = index_reads(body.body);
    let mut varying: BTreeSet<String> = body.declared.clone();
    varying.extend(carried.iter().cloned());
    varying.extend(body.inner.iter().cloned());
    let mut bad = Vec::new();
    for (array, sites) in stores {
        if owner_used(body.body, array) {
            bad.push(array.clone());
            continue;
        }
        let read_ix: Vec<&Expr> = reads
            .iter()
            .filter(|(a, _)| a == array)
            .map(|(_, ix)| *ix)
            .collect();
        let at_store = |ix: &Expr| sites.iter().any(|site| same(site.index, ix));
        let facts: Vec<Fact> = sites
            .iter()
            .map(|site| Fact::Store {
                index: site.index.clone(),
                value: site.value.clone(),
            })
            .collect();
        let mut strides = Vec::new();
        let all_injective = sites
            .iter()
            .all(|site| match injective(site.index, body, &varying) {
                Injective::Yes => true,
                Injective::Stride(name) => {
                    strides.push(name);
                    true
                }
                Injective::No => false,
            });
        let blocks: BTreeSet<Option<i64>> = sites
            .iter()
            .map(|site| site.index)
            .chain(read_ix.iter().copied())
            .map(|ix| block_of(ix, body))
            .collect();
        let literal = |e: &Expr| e.literal().is_some() || matches!(e, Expr::Bool(_) | Expr::F32(_));
        let law = if all_injective && read_ix.iter().all(|ix| at_store(ix)) {
            let guards = strides
                .into_iter()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(Guard::StrideNonZero)
                .collect();
            Some((LawId::L1Map, ArrayLaw::Own, guards))
        } else if blocks.len() == 1 && !blocks.contains(&None) {
            Some((LawId::L1Map, ArrayLaw::Own, Vec::new()))
        } else if sites.iter().all(|site| literal(site.value))
            && sites.iter().all(|site| same(site.value, sites[0].value))
        {
            if read_ix.is_empty() {
                Some((
                    LawId::L14MonotoneFrontier,
                    ArrayLaw::Monotone { own_writer: false },
                    Vec::new(),
                ))
            } else if own_writer(array, body, sites, stores, carried) {
                Some((
                    LawId::L14MonotoneFrontier,
                    ArrayLaw::Monotone { own_writer: true },
                    Vec::new(),
                ))
            } else {
                None
            }
        } else if let Some(wrapping) = combine_all(array, sites)
            && read_ix.len() == sites.len()
            && read_ix.iter().all(|ix| at_store(ix))
            && sites.iter().all(|site| site.depth == 0 || wrapping)
        {
            let int = matches!(local_type(cx, array), Some(Type::ArrayU32 | Type::ArrayI32));
            int.then_some((
                LawId::L11ScatterCombine,
                ArrayLaw::Combine { wrapping },
                Vec::new(),
            ))
        } else if read_ix.is_empty()
            && let Some(counter) = compaction_counter(sites, classes)
        {
            Some((
                LawId::L13Compaction,
                ArrayLaw::Compaction { counter },
                Vec::new(),
            ))
        } else if read_ix.is_empty() {
            Some((LawId::L12OrderedWrite, ArrayLaw::Ordered, Vec::new()))
        } else {
            None
        };
        match law {
            Some((law, array_law, guards)) => {
                certificates.push(cx.certificate(
                    region,
                    law,
                    Subject::Array(array.clone()),
                    None,
                    guards,
                    facts,
                ));
                node.arrays.push((array.clone(), array_law));
            }
            None => bad.push(array.clone()),
        }
    }
    if !bad.is_empty() {
        return Err(format!("arrays {bad:?} not lawful"));
    }
    // A counter used as a compaction index is certified as L13.
    for (_, law) in &node.arrays {
        if let ArrayLaw::Compaction { counter } = law {
            for carried in &mut node.carried {
                if &carried.var == counter
                    && let Rep::LiftedSum(ty) = carried.rep
                {
                    carried.law = LawId::L13Compaction;
                    carried.rep = Rep::Counter(ty);
                }
            }
            for certificate in certificates.iter_mut() {
                if certificate.subject == Subject::Var(counter.clone())
                    && let Some(Rep::LiftedSum(ty)) = certificate.rep
                {
                    certificate.law = LawId::L13Compaction;
                    certificate.rep = Some(Rep::Counter(ty));
                }
            }
        }
    }
    Ok(())
}

fn combine_all(array: &str, sites: &[StoreSite<'_>]) -> Option<bool> {
    let kinds: BTreeSet<bool> = sites
        .iter()
        .map(|site| combine_update(array, site.index, site.value))
        .collect::<Option<_>>()?;
    match kinds.into_iter().collect::<Vec<_>>()[..] {
        [wrapping] => Some(wrapping),
        _ => None,
    }
}

fn compaction_counter(
    sites: &[StoreSite<'_>],
    classes: &BTreeMap<String, Option<Class>>,
) -> Option<String> {
    let names: BTreeSet<&String> = sites
        .iter()
        .map(|site| match site.index {
            Expr::Local(name) if classes.get(name) == Some(&Some(Class::Add)) => Some(name),
            _ => None,
        })
        .collect::<Option<_>>()?;
    match names.into_iter().collect::<Vec<_>>()[..] {
        [name] => Some(name.clone()),
        _ => None,
    }
}

/// L14 with reads: one top-level store `a[i] = c` whose guards read only
/// loop-invariant state, read-only arrays and `a[i]` itself.
fn own_writer(
    array: &str,
    body: &Body<'_>,
    sites: &[StoreSite<'_>],
    stores: &BTreeMap<String, Vec<StoreSite<'_>>>,
    carried: &BTreeSet<String>,
) -> bool {
    let [site] = sites else {
        return false;
    };
    let ind = Expr::Local(body.induction.to_owned());
    if site.depth != 0 || *site.index != ind {
        return false;
    }
    site.guards.iter().all(|(guard, _)| {
        let names = names_of(guard);
        let scalar_ok = names.iter().all(|name| {
            name == body.induction
                || name == array
                || stores.contains_key(name)
                || !(carried.contains(name) || body.declared.contains(name))
        });
        let mut reads_ok = true;
        let mut stack = vec![*guard];
        while let Some(e) = stack.pop() {
            match e {
                Expr::Index {
                    array: a, index, ..
                } => {
                    if a == array {
                        reads_ok &= **index == ind;
                    } else if stores.contains_key(a) {
                        reads_ok = false;
                    }
                    stack.push(index);
                }
                Expr::Unary { value, .. } => stack.push(value),
                Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => {
                    stack.push(left);
                    stack.push(right);
                }
                Expr::Call { args, .. } => stack.extend(args),
                _ => {}
            }
        }
        scalar_ok && reads_ok
    })
}

/// L19: fuse loop `second` into the adjacent map `first` of the same list.
fn fuse(
    cx: &Context<'_>,
    list: &[Stmt],
    first: usize,
    second: usize,
    between: &[&Stmt],
    forms: &mut [LoopForm],
) {
    let loops: Vec<&Loop> = list
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Loop(inner) => Some(inner.as_ref()),
            _ => None,
        })
        .collect();
    let find = |form: &LoopForm| {
        loops
            .iter()
            .copied()
            .find(|l| Some(l.head) == form.region.head)
    };
    let (Some(a), Some(b)) = (find(&forms[first]), find(&forms[second])) else {
        return;
    };
    let is_map = |form: &LoopForm| {
        matches!(&form.node, Node::Parallel(node) if node.carried.is_empty() && !node.first_exit
            && node.fused_with.is_none()
            && node.arrays.iter().all(|(_, law)| *law == ArrayLaw::Own))
    };
    if !is_map(&forms[first]) || !is_map(&forms[second]) {
        return;
    }
    let (
        LoopKind::For {
            var: va,
            lo: la,
            hi: ha,
        },
        LoopKind::For {
            var: vb,
            lo: lb,
            hi: hb,
        },
    ) = (&a.kind, &b.kind)
    else {
        return;
    };
    let equal_upper = same(ha, hb)
        || fusion_bound(cx.tree, ha, 0)
            .zip(fusion_bound(cx.tree, hb, 0))
            .is_some_and(|(a, b)| same(&a, &b));
    if !same(la, lb) || !equal_upper {
        return;
    }
    let Some(aliases) = fusion_aliases(between, a) else {
        return;
    };
    if !fusion_footprints(a, b, va, vb, &aliases) {
        return;
    }
    let second_region = forms[second].region;
    if let Node::Parallel(node) = &mut forms[first].node {
        node.fused_with = Some(second_region);
    }
    let certificate = cx.certificate(
        forms[first].region,
        LawId::L19MapFusion,
        Subject::Fused(second_region),
        None,
        Vec::new(),
        Vec::new(),
    );
    forms[first].certificates.push(certificate);
}

/// Immutable owner lengths and literal extents are propagated for at most the checker's 16 hops
/// (an array literal is one); other definitions and reassignment keep the loops separate.
fn fusion_bound(tree: &FunctionTree, bound: &Expr, depth: usize) -> Option<Expr> {
    if depth > 16 {
        return None;
    }
    let Expr::Len(owner) = bound else {
        return matches!(bound, Expr::U32(_)).then(|| bound.clone());
    };
    let mut definitions = Vec::new();
    let mut assigned = false;
    tree::walk(
        &tree.body,
        &mut Vec::new(),
        0,
        &mut |stmt, _, _| match stmt {
            Stmt::Let { name, value } if name == owner => definitions.push(value.clone()),
            Stmt::Assign { name, .. } if name == owner => assigned = true,
            _ => {}
        },
    );
    if assigned {
        return None;
    }
    match definitions.as_slice() {
        [] if tree.params.iter().any(|p| p.name == *owner) => Some(bound.clone()),
        [Expr::Move(from)] => fusion_bound(tree, &Expr::Len(from.clone()), depth + 1),
        [Expr::Call { name, args, .. }]
            if matches!(name.as_str(), "filled_u32" | "filled_i32" | "filled_f32") =>
        {
            fusion_bound(tree, args.first()?, depth + 1)
        }
        [Expr::ArrayLit { items, .. }] if depth < 16 => items.len().try_into().ok().map(Expr::U32),
        _ => None,
    }
}

/// Statements between fused loops: allocations, literals and moves (aliases)
/// that do not read anything the first loop writes. Returns `alias -> owner`.
pub(super) fn fusion_aliases(between: &[&Stmt], first: &Loop) -> Option<BTreeMap<String, String>> {
    let writes = written_arrays(&first.body);
    let mut aliases = BTreeMap::new();
    for stmt in between {
        let Stmt::Let { name, value } = stmt else {
            return None;
        };
        match value {
            Expr::Move(owner) => {
                aliases.insert(name.clone(), owner.clone());
            }
            Expr::Call {
                name: callee, args, ..
            } if callee.starts_with("filled_") => {
                let ok = args.iter().all(|arg| {
                    let mut stack = vec![arg];
                    let mut clean = true;
                    while let Some(e) = stack.pop() {
                        match e {
                            Expr::Index { array, .. } if writes.contains(array) => clean = false,
                            Expr::Local(_)
                            | Expr::Len(_)
                            | Expr::I32(_)
                            | Expr::U32(_)
                            | Expr::F32(_)
                            | Expr::Bool(_) => {}
                            Expr::Binary { left, right, .. } => {
                                stack.push(left);
                                stack.push(right);
                            }
                            _ => clean = false,
                        }
                    }
                    clean
                });
                if !ok {
                    return None;
                }
            }
            e if e.literal().is_some() => {}
            _ => return None,
        }
    }
    Some(aliases)
}

pub(super) fn written_arrays(body: &[Stmt]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        if let Stmt::Store { array, .. } = stmt {
            out.insert(array.clone());
        }
    });
    out
}

/// The second loop reads the first loop's outputs only at its own index, the
/// first stores them only at its own index, and the second writes nothing the
/// first touches.
pub(super) fn fusion_footprints(
    a: &Loop,
    b: &Loop,
    va: &str,
    vb: &str,
    aliases: &BTreeMap<String, String>,
) -> bool {
    let resolve = |name: &str| {
        aliases
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    };
    let writes_a = written_arrays(&a.body);
    let writes_b: BTreeSet<String> = written_arrays(&b.body).iter().map(|n| resolve(n)).collect();
    let reads_a: BTreeSet<String> = index_reads(&a.body)
        .iter()
        .map(|(n, _)| (*n).to_owned())
        .collect();
    if writes_b
        .iter()
        .any(|n| writes_a.contains(n) || reads_a.contains(n))
    {
        return false;
    }
    let mut a_own = true;
    tree::walk(&a.body, &mut Vec::new(), 0, &mut |stmt, _, depth| {
        if let Stmt::Store { index, .. } = stmt
            && (depth != 0 || *index != Expr::Local(va.to_owned()))
        {
            a_own = false;
        }
    });
    let b_reads_ok = index_reads(&b.body)
        .iter()
        .filter(|(n, _)| writes_a.contains(&resolve(n)))
        .all(|(_, ix)| **ix == Expr::Local(vb.to_owned()));
    let b_owner_ok = !writes_a.iter().any(|owner| {
        owner_used(&b.body, owner)
            || aliases
                .iter()
                .any(|(alias, o)| o == owner && owner_used(&b.body, alias))
    });
    a_own && b_reads_ok && b_owner_ok
}
