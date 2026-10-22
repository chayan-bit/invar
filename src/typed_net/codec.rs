//! Encoding internal source terms into the graph and decoding them back.

use std::collections::{BTreeMap, BTreeSet};

use super::*;

fn encode_term(
    net: &mut Net,
    term: &term::Term,
    types: &BTreeMap<u32, Type>,
) -> Result<NodeId, Error> {
    use term::Kind;
    let mut inputs = Vec::new();
    let op = match &term.kind {
        Kind::Unit => Op::Unit,
        Kind::Bool(value) => Op::Bool(*value),
        Kind::Nat(value) => Op::Nat(*value),
        Kind::Var(binder) => Op::Var(*binder),
        Kind::Lambda { binder, body } => {
            inputs.push(encode_term(net, body, types)?);
            Op::Lambda { binder: *binder }
        }
        Kind::Pair(a, b) => {
            inputs.push(encode_term(net, a, types)?);
            inputs.push(encode_term(net, b, types)?);
            Op::Pair
        }
        Kind::Apply(a, b) => {
            inputs.push(encode_term(net, a, types)?);
            inputs.push(encode_term(net, b, types)?);
            Op::Apply
        }
        Kind::Let {
            binder,
            value,
            body,
        } => {
            inputs.push(encode_term(net, value, types)?);
            inputs.push(encode_term(net, body, types)?);
            Op::Let { binder: *binder }
        }
        Kind::If { condition, yes, no } => {
            inputs.push(encode_term(net, condition, types)?);
            inputs.push(encode_term(net, yes, types)?);
            inputs.push(encode_term(net, no, types)?);
            Op::Select
        }
        Kind::Call { name, args } => {
            for arg in args {
                inputs.push(encode_term(net, arg, types)?);
            }
            Op::Call { name: name.clone() }
        }
        Kind::Fail(label) => Op::Fail(label),
    };
    let ty = types
        .get(&term.id)
        .cloned()
        .ok_or(Error::Type(net.nodes.len()))?;
    let effect = effect_for(net, &op, &inputs);
    Ok(net.push(Node {
        source_id: term.id,
        ty,
        effect,
        inputs,
        op,
    }))
}

/// Encode a typed internal source fragment without evaluating it.
pub fn encode(
    program: &term::Program,
    entry: &term::Term,
    types: &BTreeMap<u32, Type>,
    signatures: &BTreeMap<String, Signature>,
    limits: Limits,
) -> Result<Net, Error> {
    if program.functions.len() != signatures.len() {
        return Err(Error::Schema);
    }
    let mut pending = vec![(entry, 1usize)];
    pending.extend(program.functions.values().map(|f| (&f.body, 1)));
    let mut count = 0usize;
    while let Some((term, depth)) = pending.pop() {
        if depth > limits.depth {
            return Err(Error::DepthLimit);
        }
        count = count.checked_add(1).ok_or(Error::NodeLimit)?;
        if count > limits.nodes {
            return Err(Error::NodeLimit);
        }
        let children: Vec<&term::Term> = match &term.kind {
            term::Kind::Pair(a, b) | term::Kind::Apply(a, b) => vec![a, b],
            term::Kind::Lambda { body, .. } => vec![body],
            term::Kind::Let { value, body, .. } => vec![value, body],
            term::Kind::If { condition, yes, no } => vec![condition, yes, no],
            term::Kind::Call { args, .. } => args.iter().collect(),
            _ => Vec::new(),
        };
        pending.extend(children.into_iter().map(|child| (child, depth + 1)));
    }
    let mut net = Net::empty(0);
    net.entry = encode_term(&mut net, entry, types)?;
    for (name, function) in &program.functions {
        let signature = signatures.get(name).ok_or(Error::Schema)?.clone();
        if function.params
            != signature
                .params
                .iter()
                .map(|(binder, _)| *binder)
                .collect::<Vec<_>>()
        {
            return Err(Error::Schema);
        }
        let body = encode_term(&mut net, &function.body, types)?;
        net.functions
            .insert(name.clone(), Function { signature, body });
    }
    let encoded_ids = net
        .nodes
        .iter()
        .flatten()
        .map(|node| node.source_id)
        .collect::<BTreeSet<_>>();
    if encoded_ids != types.keys().copied().collect() {
        return Err(Error::Schema);
    }
    net.seal_source();
    net.verify(limits)?;
    Ok(net)
}

impl Net {
    pub(super) fn decode_term(&self, id: NodeId) -> Result<term::Term, Error> {
        let node = self.node(id)?;
        let child = |index| self.decode_term(node.inputs[index]).map(Box::new);
        let kind = match &node.op {
            Op::Unit => term::Kind::Unit,
            Op::Bool(value) => term::Kind::Bool(*value),
            Op::Nat(value) => term::Kind::Nat(*value),
            Op::Var(binder) => term::Kind::Var(*binder),
            Op::Pair => term::Kind::Pair(child(0)?, child(1)?),
            Op::Lambda { binder } => term::Kind::Lambda {
                binder: *binder,
                body: child(0)?,
            },
            Op::Apply => term::Kind::Apply(child(0)?, child(1)?),
            Op::Let { binder } => term::Kind::Let {
                binder: *binder,
                value: child(0)?,
                body: child(1)?,
            },
            Op::Select => term::Kind::If {
                condition: child(0)?,
                yes: child(1)?,
                no: child(2)?,
            },
            Op::Call { name } => term::Kind::Call {
                name: name.clone(),
                args: node
                    .inputs
                    .iter()
                    .map(|&id| self.decode_term(id))
                    .collect::<Result<_, _>>()?,
            },
            Op::Fail(label) => term::Kind::Fail(label),
            Op::ArrayU32(_)
            | Op::InputU32(_)
            | Op::InputScalar(_)
            | Op::PairReduce { .. }
            | Op::Reduce { .. }
            | Op::BoundedScan { .. }
            | Op::AffineScan { .. }
            | Op::Map(_)
            | Op::FusedMap(_)
            | Op::ZipMap { .. }
            | Op::FusedZipMap { .. } => return Err(Error::Type(id)),
        };
        Ok(term::Term {
            id: node.source_id,
            kind,
        })
    }

    /// Reconstruct source terms solely from typed graph connectivity.
    pub fn decode(&self, proof: &Proof) -> Result<(term::Program, term::Term), Error> {
        self.check_proof(proof)?;
        let mut program = term::Program::default();
        for (name, function) in &self.functions {
            program.functions.insert(
                name.clone(),
                term::Function {
                    params: function
                        .signature
                        .params
                        .iter()
                        .map(|(binder, _)| *binder)
                        .collect(),
                    body: self.decode_term(function.body)?,
                },
            );
        }
        Ok((program, self.decode_term(self.entry)?))
    }
}
