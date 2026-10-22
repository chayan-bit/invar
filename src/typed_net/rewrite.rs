//! Proof-carrying graph rewrites.

use std::collections::BTreeSet;

use super::*;

impl Net {
    pub fn rewrite_beta(&mut self, limits: Limits) -> Result<Option<Rewrite>, Error> {
        self.verify(limits)?;
        // The M1 rule admits a whole closed term. Rewriting beneath a returned
        // closure would change the oracle's observable closure body.
        for id in [self.entry] {
            let Some(node) = self.nodes[id].as_ref() else {
                continue;
            };
            if !matches!(node.op, Op::Apply) || node.effect != Effect::Total {
                continue;
            }
            let [function_id, argument_id] = node.inputs.as_slice() else {
                continue;
            };
            let function = self.node(*function_id)?;
            let Op::Lambda { binder } = function.op else {
                continue;
            };
            let mut captures = BTreeSet::new();
            self.free_variables(
                function.inputs[0],
                &mut BTreeSet::from([binder]),
                &mut captures,
            );
            if !captures.is_empty() {
                continue;
            }
            let body_id = function.inputs[0];
            let mut candidate = self.clone();
            candidate.node_mut(id)?.op = Op::Let { binder };
            candidate.node_mut(id)?.inputs = vec![*argument_id, body_id];
            candidate.nodes[*function_id] = None;
            let (op, inputs) = {
                let node = candidate.node(id)?;
                (node.op.clone(), node.inputs.clone())
            };
            candidate.node_mut(id)?.effect = effect_for(&candidate, &op, &inputs);
            let before = self.revision;
            candidate.revision = candidate
                .revision
                .checked_add(1)
                .ok_or(Error::RuleRejected)?;
            candidate.rule_history.push(BETA_RULE);
            candidate.verify(limits)?;
            *self = candidate;
            return Ok(Some(Rewrite {
                rule: BETA_RULE,
                before,
                after: self.revision,
            }));
        }
        Ok(None)
    }

    pub fn rewrite_map_fusion(&mut self, limits: Limits) -> Result<Option<Rewrite>, Error> {
        self.verify(limits)?;
        if let Some(rewrite) = self.fuse_zip_operand(limits)? {
            return Ok(Some(rewrite));
        }
        for id in 0..self.nodes.len() {
            let Some(node) = self.nodes[id].as_ref() else {
                continue;
            };
            let outer = match &node.op {
                Op::Map(op) => vec![*op],
                Op::FusedMap(ops) => ops.clone(),
                _ => continue,
            };
            let inner_id = node.inputs[0];
            let inner = self.node(inner_id)?;
            if node.effect != Effect::Total || inner.effect != Effect::Total {
                continue;
            }
            if let Op::ZipMap { op, len } = &inner.op {
                if outer.len() > limits.map_ops {
                    return Ok(None);
                }
                let mut candidate = self.clone();
                candidate.node_mut(id)?.op = Op::FusedZipMap {
                    op: *op,
                    len: *len,
                    left: Vec::new(),
                    right: Vec::new(),
                    suffix: outer,
                };
                candidate.node_mut(id)?.inputs = inner.inputs.clone();
                candidate.nodes[inner_id] = None;
                let (op, inputs) = {
                    let node = candidate.node(id)?;
                    (node.op.clone(), node.inputs.clone())
                };
                candidate.node_mut(id)?.effect = effect_for(&candidate, &op, &inputs);
                let before = self.revision;
                candidate.revision = candidate
                    .revision
                    .checked_add(1)
                    .ok_or(Error::RuleRejected)?;
                candidate.rule_history.push(MAP_FUSION_RULE);
                candidate.verify(limits)?;
                *self = candidate;
                return Ok(Some(Rewrite {
                    rule: MAP_FUSION_RULE,
                    before,
                    after: self.revision,
                }));
            }
            let mut operations = match &inner.op {
                Op::Map(op) => vec![*op],
                Op::FusedMap(ops) => ops.clone(),
                _ => continue,
            };
            operations.extend(outer);
            if operations.len() > limits.map_ops {
                return Ok(None);
            }
            let input = inner.inputs[0];
            let mut candidate = self.clone();
            candidate.node_mut(id)?.op = Op::FusedMap(operations);
            candidate.node_mut(id)?.inputs = vec![input];
            candidate.nodes[inner_id] = None;
            let (op, inputs) = {
                let node = candidate.node(id)?;
                (node.op.clone(), node.inputs.clone())
            };
            candidate.node_mut(id)?.effect = effect_for(&candidate, &op, &inputs);
            let before = self.revision;
            candidate.revision = candidate
                .revision
                .checked_add(1)
                .ok_or(Error::RuleRejected)?;
            candidate.rule_history.push(MAP_FUSION_RULE);
            candidate.verify(limits)?;
            *self = candidate;
            return Ok(Some(Rewrite {
                rule: MAP_FUSION_RULE,
                before,
                after: self.revision,
            }));
        }
        Ok(None)
    }

    /// Fold a total map over a live input into the zip that consumes it:
    /// `zip(map(x), y)` becomes one element pass with `map` applied to `x` first.
    fn fuse_zip_operand(&mut self, limits: Limits) -> Result<Option<Rewrite>, Error> {
        for id in 0..self.nodes.len() {
            let Some(node) = self.nodes[id].as_ref() else {
                continue;
            };
            let (op, len, mut pre, suffix) = match &node.op {
                Op::ZipMap { op, len } => (*op, *len, [Vec::new(), Vec::new()], Vec::new()),
                Op::FusedZipMap {
                    op,
                    len,
                    left,
                    right,
                    suffix,
                } => (*op, *len, [left.clone(), right.clone()], suffix.clone()),
                _ => continue,
            };
            let inputs = node.inputs.clone();
            for (side, &child_id) in inputs.iter().enumerate() {
                let child = self.node(child_id)?;
                let ops = match &child.op {
                    Op::Map(op) => vec![*op],
                    Op::FusedMap(ops) => ops.clone(),
                    _ => continue,
                };
                let grandchild = child.inputs[0];
                if child.effect != Effect::Total
                    || !matches!(self.node(grandchild)?.op, Op::InputU32(_))
                {
                    continue;
                }
                let mut combined = ops;
                combined.extend(pre[side].iter().copied());
                pre[side] = combined;
                if pre[0].len() + pre[1].len() + suffix.len() > limits.map_ops {
                    return Ok(None);
                }
                let mut candidate = self.clone();
                let [left, right] = pre;
                let node = candidate.node_mut(id)?;
                node.op = Op::FusedZipMap {
                    op,
                    len,
                    left,
                    right,
                    suffix,
                };
                node.inputs[side] = grandchild;
                candidate.nodes[child_id] = None;
                let before = self.revision;
                candidate.revision = candidate
                    .revision
                    .checked_add(1)
                    .ok_or(Error::RuleRejected)?;
                candidate.rule_history.push(MAP_FUSION_RULE);
                candidate.verify(limits)?;
                *self = candidate;
                return Ok(Some(Rewrite {
                    rule: MAP_FUSION_RULE,
                    before,
                    after: self.revision,
                }));
            }
        }
        Ok(None)
    }
}
