//! Graph verification: topology, types, affine uses, effects and limits.

use std::collections::{BTreeMap, BTreeSet};

use super::*;

impl Net {
    /// Verify port degree, rooted topology, types, affine uses, effect facts and limits.
    pub fn verify(&self, limits: Limits) -> Result<Proof, Error> {
        if self.schema_version != SCHEMA_VERSION || self.source_identity == 0 {
            return Err(Error::Schema);
        }
        if self.revision != self.rule_history.len() as u64
            || self
                .rule_history
                .iter()
                .any(|rule| !matches!(*rule, BETA_RULE | MAP_FUSION_RULE))
        {
            return Err(Error::RuleRejected);
        }
        if self.nodes.len() > limits.nodes {
            return Err(Error::NodeLimit);
        }
        let mut seen_ids = BTreeSet::new();
        let mut port_count = 0usize;
        let mut parent_count = vec![0usize; self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            let Some(node) = node else { continue };
            if !seen_ids.insert(node.source_id) {
                return Err(Error::DuplicateSource(node.source_id));
            }
            if node
                .op
                .arity()
                .is_some_and(|arity| arity != node.inputs.len())
            {
                return Err(Error::Arity(id));
            }
            if let Op::ArrayU32(values) = &node.op
                && values.len() > limits.array_len
            {
                return Err(Error::ArrayLimit);
            }
            if let Op::FusedMap(ops) = &node.op
                && (ops.is_empty() || ops.len() > limits.map_ops)
            {
                return Err(Error::MapLimit);
            }
            if let Op::FusedZipMap {
                left,
                right,
                suffix,
                ..
            } = &node.op
                && (left.len() + right.len() + suffix.len() == 0
                    || left.len() + right.len() + suffix.len() > limits.map_ops)
            {
                return Err(Error::MapLimit);
            }
            port_count = port_count
                .checked_add(node.inputs.len() + 1)
                .ok_or(Error::PortLimit)?;
            if port_count > limits.ports {
                return Err(Error::PortLimit);
            }
            for &input in &node.inputs {
                self.node(input)?;
                parent_count[input] += 1;
            }
        }
        for (id, count) in parent_count.into_iter().enumerate() {
            if self.nodes[id].is_none() {
                continue;
            }
            let root = id == self.entry || self.functions.values().any(|f| f.body == id);
            if count != usize::from(!root) {
                return Err(Error::SharedOrOrphan(id));
            }
        }
        let mut stack = BTreeSet::new();
        let mut visited = BTreeSet::new();
        self.walk(self.entry, 1, limits.depth, &mut stack, &mut visited)?;
        for function in self.functions.values() {
            self.walk(function.body, 1, limits.depth, &mut stack, &mut visited)?;
        }
        if visited.len() != self.nodes.iter().flatten().count() {
            return Err(Error::SharedOrOrphan(0));
        }
        let signatures = self
            .functions
            .iter()
            .map(|(name, f)| (name.clone(), f.signature.clone()))
            .collect::<BTreeMap<_, _>>();
        for function in self.functions.values() {
            let env = function.signature.params.iter().cloned().collect();
            let (ty, _, uses) = self.check(function.body, &env, &signatures)?;
            if ty != function.signature.result {
                return Err(Error::Type(function.body));
            }
            for (binder, ty) in &function.signature.params {
                if !ty.copyable() && uses.get(binder).copied().unwrap_or(0) > 1 {
                    return Err(Error::DuplicatedOwner(*binder));
                }
            }
        }
        self.check(self.entry, &BTreeMap::new(), &signatures)?;
        Ok(Proof {
            revision: self.revision,
            fingerprint: self.fingerprint(),
        })
    }

    fn walk(
        &self,
        id: NodeId,
        depth: usize,
        max_depth: usize,
        stack: &mut BTreeSet<NodeId>,
        visited: &mut BTreeSet<NodeId>,
    ) -> Result<(), Error> {
        if depth > max_depth {
            return Err(Error::DepthLimit);
        }
        if !stack.insert(id) {
            return Err(Error::Cycle(id));
        }
        if !visited.insert(id) {
            return Err(Error::SharedOrOrphan(id));
        }
        for &child in &self.node(id)?.inputs {
            self.walk(child, depth + 1, max_depth, stack, visited)?;
        }
        stack.remove(&id);
        Ok(())
    }

    fn check(
        &self,
        id: NodeId,
        env: &BTreeMap<term::Binder, Type>,
        signatures: &BTreeMap<String, Signature>,
    ) -> Result<(Type, Effect, BTreeMap<term::Binder, usize>), Error> {
        let node = self.node(id)?;
        let mut uses = BTreeMap::new();
        let (ty, effect) = match &node.op {
            Op::Unit => (Type::Unit, Effect::Total),
            Op::Bool(_) => (Type::Bool, Effect::Total),
            Op::Nat(_) => (Type::Nat, Effect::Total),
            Op::ArrayU32(_) | Op::InputU32(_) => (Type::ArrayU32, Effect::Total),
            Op::InputScalar(_) => (Type::Nat, Effect::Total),
            Op::Fail(_) => (node.ty.clone(), Effect::MayFail),
            Op::Var(binder) => {
                uses.insert(*binder, 1);
                (
                    env.get(binder)
                        .cloned()
                        .ok_or(Error::MissingBinder(id, *binder))?,
                    Effect::Total,
                )
            }
            Op::Lambda { binder } => {
                let Type::Function(input, output) = &node.ty else {
                    return Err(Error::Type(id));
                };
                if env.contains_key(binder) {
                    return Err(Error::DuplicatedOwner(*binder));
                }
                let mut nested = env.clone();
                nested.insert(*binder, *input.clone());
                let (body_ty, _, mut body_uses) =
                    self.check(node.inputs[0], &nested, signatures)?;
                if &body_ty != output.as_ref() {
                    return Err(Error::Type(id));
                }
                if !input.copyable() && body_uses.remove(binder).unwrap_or(0) > 1 {
                    return Err(Error::DuplicatedOwner(*binder));
                }
                body_uses.remove(binder);
                uses = body_uses;
                (node.ty.clone(), Effect::Total)
            }
            Op::Pair
            | Op::Apply
            | Op::Let { .. }
            | Op::Select
            | Op::Call { .. }
            | Op::Map(_)
            | Op::FusedMap(_)
            | Op::ZipMap { .. }
            | Op::FusedZipMap { .. }
            | Op::PairReduce { .. }
            | Op::Reduce { .. }
            | Op::AffineScan { .. }
            | Op::BoundedScan { .. } => {
                let mut child = Vec::new();
                for (index, &input) in node.inputs.iter().enumerate() {
                    let child_env = match &node.op {
                        Op::Let { binder } if index == 1 => {
                            let mut next = env.clone();
                            next.insert(*binder, self.node(node.inputs[0])?.ty.clone());
                            next
                        }
                        _ => env.clone(),
                    };
                    child.push(self.check(input, &child_env, signatures)?);
                }
                let mut effect = match &node.op {
                    Op::Call { .. } => Effect::Unknown,
                    Op::BoundedScan { .. } => Effect::MayFail,
                    _ => child
                        .iter()
                        .fold(Effect::Total, |e, (_, next, _)| e.join(*next)),
                };
                if matches!(node.op, Op::Apply) {
                    effect = effect.join(match &self.node(node.inputs[0])?.op {
                        Op::Lambda { .. } => {
                            self.node(self.node(node.inputs[0])?.inputs[0])?.effect
                        }
                        _ => Effect::Unknown,
                    });
                }
                let ty = match &node.op {
                    Op::Pair => {
                        Type::Pair(Box::new(child[0].0.clone()), Box::new(child[1].0.clone()))
                    }
                    Op::Apply => {
                        let Type::Function(input, output) = &child[0].0 else {
                            return Err(Error::Type(id));
                        };
                        if input.as_ref() != &child[1].0 {
                            return Err(Error::Type(id));
                        }
                        *output.clone()
                    }
                    Op::Let { binder } => {
                        if env.contains_key(binder) {
                            return Err(Error::DuplicatedOwner(*binder));
                        }
                        let count = child[1].2.remove(binder).unwrap_or(0);
                        if !child[0].0.copyable() && count > 1 {
                            return Err(Error::DuplicatedOwner(*binder));
                        }
                        child[1].0.clone()
                    }
                    Op::Select => {
                        if child[0].0 != Type::Bool || child[1].0 != child[2].0 {
                            return Err(Error::Type(id));
                        }
                        child[1].0.clone()
                    }
                    Op::Call { name } => {
                        let signature = signatures.get(name).ok_or(Error::UnknownFunction(id))?;
                        if child.len() != signature.params.len()
                            || child
                                .iter()
                                .zip(&signature.params)
                                .any(|(arg, (_, ty))| &arg.0 != ty)
                        {
                            return Err(Error::Type(id));
                        }
                        signature.result.clone()
                    }
                    Op::Map(_) | Op::FusedMap(_) => {
                        if child[0].0 != Type::ArrayU32 {
                            return Err(Error::Type(id));
                        }
                        Type::ArrayU32
                    }
                    Op::ZipMap { len, .. } | Op::FusedZipMap { len, .. } => {
                        if child.len() != 2 || child.iter().any(|(ty, _, _)| *ty != Type::ArrayU32)
                            || node.inputs.iter().any(|input| matches!(self.node(*input), Ok(Node { op: Op::ArrayU32(values), .. }) if *len > values.len()))
                        {
                            return Err(Error::Type(id));
                        }
                        Type::ArrayU32
                    }
                    Op::PairReduce { rule } => {
                        if *rule != crate::pair_reduction::RULE_REVISION {
                            return Err(Error::RuleRejected);
                        }
                        if child.len() != 4
                            || child[..2].iter().any(|(ty, _, _)| *ty != Type::Nat)
                            || child[2..].iter().any(|(ty, _, _)| *ty != Type::ArrayU32)
                            || child.iter().any(|(_, effect, _)| *effect != Effect::Total)
                        {
                            return Err(Error::Type(id));
                        }
                        Type::Pair(Box::new(Type::Nat), Box::new(Type::Nat))
                    }
                    Op::Reduce { rule, .. } => {
                        if *rule != crate::laws::RULE_REVISION {
                            return Err(Error::RuleRejected);
                        }
                        if child.len() != 2
                            || child[0].0 != Type::Nat
                            || child[1].0 != Type::ArrayU32
                        {
                            return Err(Error::Type(id));
                        }
                        Type::Nat
                    }
                    Op::BoundedScan {
                        family,
                        prefix,
                        rule,
                    } => {
                        if !family.valid() || *rule != family.rule() || *prefix == Prefix::Exclusive
                        {
                            return Err(Error::RuleRejected);
                        }
                        if child.iter().any(|(_, effect, _)| *effect != Effect::Total) {
                            return Err(Error::Effect(id));
                        }
                        let seed_type = match family {
                            BoundedFamily::Table(_) => Type::Nat,
                            BoundedFamily::Matrix(_) => Type::ArrayU32,
                        };
                        if child[0].0 != seed_type
                            || child[1].0 != Type::ArrayU32
                            || child[2].0 != Type::Nat
                        {
                            return Err(Error::Type(id));
                        }
                        let seed = match family {
                            BoundedFamily::Table(_) => self.scalar_slot(node.inputs[0])?,
                            BoundedFamily::Matrix(_) => self.array_slot(node.inputs[0])?,
                        };
                        let input = self.array_slot(node.inputs[1])?;
                        self.scalar_slot(node.inputs[2])?;
                        if matches!(family, BoundedFamily::Matrix(_)) && seed == input {
                            return Err(Error::Type(id));
                        }
                        if matches!(family, BoundedFamily::Table(_)) && *prefix == Prefix::None {
                            Type::Nat
                        } else {
                            Type::ArrayU32
                        }
                    }
                    Op::AffineScan { prefix, rule, .. } => {
                        if *rule != crate::recurrence::RULE_REVISION {
                            return Err(Error::RuleRejected);
                        }
                        if child.first().is_none_or(|(ty, _, _)| *ty != Type::Nat)
                            || child.iter().skip(1).any(|(ty, _, _)| *ty != Type::ArrayU32)
                        {
                            return Err(Error::Type(id));
                        }
                        if *prefix == Prefix::None {
                            Type::Nat
                        } else {
                            Type::ArrayU32
                        }
                    }
                    _ => unreachable!(),
                };
                if matches!(node.op, Op::Select) {
                    let branch = child.pop().ok_or(Error::Arity(id))?.2;
                    let yes = child.pop().ok_or(Error::Arity(id))?.2;
                    max_counts(&mut uses, yes);
                    max_counts(&mut uses, branch);
                    add_counts(&mut uses, child.remove(0).2);
                } else {
                    for (_, _, counts) in child {
                        add_counts(&mut uses, counts);
                    }
                }
                (ty, effect)
            }
        };
        if node.ty != ty {
            return Err(Error::Type(id));
        }
        if node.effect != effect {
            return Err(Error::Effect(id));
        }
        Ok((ty, effect, uses))
    }

    pub fn check_proof(&self, proof: &Proof) -> Result<(), Error> {
        if proof.revision != self.revision || proof.fingerprint != self.fingerprint() {
            return Err(Error::StaleProof);
        }
        Ok(())
    }
}
