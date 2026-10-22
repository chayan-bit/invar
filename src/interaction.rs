//! Bounded, untyped Lafont interaction combinators (rule revision 1).
//!
//! This reference calculus carries no Invar types, ownership, or demand.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Gamma,
    Delta,
    Epsilon,
}

impl Kind {
    fn arity(self) -> u8 {
        match self {
            Self::Gamma | Self::Delta => 3,
            Self::Epsilon => 1,
        }
    }
}

impl TryFrom<&str> for Kind {
    type Error = String;

    fn try_from(name: &str) -> Result<Self, Self::Error> {
        match name {
            "gamma" => Ok(Self::Gamma),
            "delta" => Ok(Self::Delta),
            "epsilon" => Ok(Self::Epsilon),
            _ => Err(format!("unsupported interaction agent: {name}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Port {
    Agent(usize, u8),
    Boundary(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Maximum total agent IDs allocated, including consumed agents.
    pub agents: usize,
    /// Maximum live agent ports plus boundary ports.
    pub ports: usize,
    pub steps: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidPort(Port),
    DuplicatePort(Port),
    DanglingPort(Port),
    SelfWire(Port),
    AgentLimit,
    PortLimit,
    StepLimit,
    Storage,
    InvalidRewrite,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Net {
    agents: Vec<Option<Kind>>,
    boundaries: usize,
    wires: Vec<(Port, Port)>,
    cyclic_wires: usize,
}

impl Net {
    pub fn new(boundaries: usize) -> Self {
        Self {
            agents: Vec::new(),
            boundaries,
            wires: Vec::new(),
            cyclic_wires: 0,
        }
    }

    /// Number of closed wires with no ports, preserved across rewrites.
    pub fn cyclic_wires(&self) -> usize {
        self.cyclic_wires
    }

    pub fn add_agent(&mut self, kind: Kind, limits: Limits) -> Result<usize, Error> {
        self.check_capacity(kind, limits)?;
        self.agents.try_reserve(1).map_err(|_| Error::Storage)?;
        let id = self.agents.len();
        self.agents.push(Some(kind));
        Ok(id)
    }

    pub fn connect(&mut self, left: Port, right: Port) -> Result<(), Error> {
        self.check_port(left)?;
        self.check_port(right)?;
        if left == right {
            return Err(Error::SelfWire(left));
        }
        for endpoint in [left, right] {
            if self
                .wires
                .iter()
                .any(|&(a, b)| a == endpoint || b == endpoint)
            {
                return Err(Error::DuplicatePort(endpoint));
            }
        }
        self.wires.try_reserve(1).map_err(|_| Error::Storage)?;
        self.wires.push((left, right));
        Ok(())
    }

    fn check_port(&self, port: Port) -> Result<(), Error> {
        match port {
            Port::Boundary(id) if id < self.boundaries => Ok(()),
            Port::Agent(id, index)
                if self
                    .agents
                    .get(id)
                    .and_then(|kind| *kind)
                    .is_some_and(|kind| index < kind.arity()) =>
            {
                Ok(())
            }
            _ => Err(Error::InvalidPort(port)),
        }
    }

    fn check_capacity(&self, kind: Kind, limits: Limits) -> Result<(), Error> {
        if self.agents.len() >= limits.agents {
            return Err(Error::AgentLimit);
        }
        let used_ports = self
            .agents
            .iter()
            .flatten()
            .try_fold(self.boundaries, |n, kind| {
                n.checked_add(usize::from(kind.arity()))
            });
        if used_ports
            .and_then(|n| n.checked_add(usize::from(kind.arity())))
            .is_none_or(|total| total > limits.ports)
        {
            return Err(Error::PortLimit);
        }
        Ok(())
    }

    pub fn validate(&self, limits: Limits) -> Result<(), Error> {
        if self.agents.len() > limits.agents {
            return Err(Error::AgentLimit);
        }
        if self.boundaries > limits.ports {
            return Err(Error::PortLimit);
        }
        let mut expected = BTreeSet::new();
        for id in 0..self.boundaries {
            expected.insert(Port::Boundary(id));
        }
        for (id, kind) in self.agents.iter().enumerate() {
            if let Some(kind) = kind {
                for port in 0..kind.arity() {
                    expected.insert(Port::Agent(id, port));
                }
            }
        }
        if expected.len() > limits.ports {
            return Err(Error::PortLimit);
        }
        let mut seen = BTreeSet::new();
        for &(left, right) in &self.wires {
            for port in [left, right] {
                self.check_port(port)?;
                if !seen.insert(port) {
                    return Err(Error::DuplicatePort(port));
                }
            }
            if left == right {
                return Err(Error::SelfWire(left));
            }
        }
        if let Some(port) = expected.difference(&seen).next() {
            return Err(Error::DanglingPort(*port));
        }
        Ok(())
    }

    pub fn boundary_pairs(&self, limits: Limits) -> Result<Vec<(usize, usize)>, Error> {
        self.validate(limits)?;
        if self.agents.iter().any(Option::is_some) {
            return Err(Error::InvalidRewrite);
        }
        let mut pairs = self
            .wires
            .iter()
            .map(|&(a, b)| match (a, b) {
                (Port::Boundary(x), Port::Boundary(y)) => Ok((x.min(y), x.max(y))),
                _ => Err(Error::InvalidRewrite),
            })
            .collect::<Result<Vec<_>, _>>()?;
        pairs.sort_unstable();
        Ok(pairs)
    }

    fn redex(&self) -> Option<(usize, usize)> {
        self.wires
            .iter()
            .filter_map(|&(a, b)| match (a, b) {
                (Port::Agent(x, 0), Port::Agent(y, 0)) if x != y => Some((x.min(y), x.max(y))),
                _ => None,
            })
            .min()
    }

    /// Reduces to normal form or reports an explicit resource limit. The net remains valid on error.
    pub fn reduce(&mut self, limits: Limits) -> Result<usize, Error> {
        self.validate(limits)?;
        let mut steps = 0;
        while let Some((left, right)) = self.redex() {
            if steps == limits.steps {
                return Err(Error::StepLimit);
            }
            // ponytail: clone per rewrite keeps limit failures atomic; use a transactional
            // edge arena only if bounded reference graphs prove too slow.
            let mut next = self.clone();
            next.rewrite(left, right, limits)?;
            next.validate(limits)?;
            *self = next;
            steps += 1;
        }
        Ok(steps)
    }

    fn rewrite(&mut self, left: usize, right: usize, limits: Limits) -> Result<(), Error> {
        let a = self.agents[left].ok_or(Error::InvalidRewrite)?;
        let b = self.agents[right].ok_or(Error::InvalidRewrite)?;
        let old_aux: BTreeSet<_> = [(left, a), (right, b)]
            .into_iter()
            .flat_map(|(id, kind)| (1..kind.arity()).map(move |p| Port::Agent(id, p)))
            .collect();
        let mut transient = Vec::new();
        self.wires.retain(|&(x, y)| {
            let touches = [x, y]
                .into_iter()
                .any(|port| matches!(port, Port::Agent(id, _) if id == left || id == right));
            if touches
                && !(x == Port::Agent(left, 0) && y == Port::Agent(right, 0)
                    || x == Port::Agent(right, 0) && y == Port::Agent(left, 0))
            {
                transient.push((x, y));
            }
            !touches
        });
        self.agents[left] = None;
        self.agents[right] = None;
        let aux = |id, p| Port::Agent(id, p);
        match (a, b) {
            (Kind::Gamma, Kind::Gamma) => {
                transient.extend([(aux(left, 1), aux(right, 2)), (aux(left, 2), aux(right, 1))]);
            }
            (Kind::Delta, Kind::Delta) => {
                transient.extend([(aux(left, 1), aux(right, 1)), (aux(left, 2), aux(right, 2))]);
            }
            (Kind::Epsilon, Kind::Epsilon) => {}
            (Kind::Epsilon, Kind::Gamma | Kind::Delta)
            | (Kind::Gamma | Kind::Delta, Kind::Epsilon) => {
                let other = if a == Kind::Epsilon { right } else { left };
                for port in 1..=2 {
                    let fresh = self.add_agent(Kind::Epsilon, limits)?;
                    transient.push((aux(other, port), aux(fresh, 0)));
                }
            }
            (Kind::Gamma, Kind::Delta) | (Kind::Delta, Kind::Gamma) => {
                let gamma = if a == Kind::Gamma { left } else { right };
                let delta = if a == Kind::Delta { left } else { right };
                let d0 = self.add_agent(Kind::Delta, limits)?;
                let d1 = self.add_agent(Kind::Delta, limits)?;
                let c0 = self.add_agent(Kind::Gamma, limits)?;
                let c1 = self.add_agent(Kind::Gamma, limits)?;
                transient.extend([
                    (aux(d0, 0), aux(gamma, 1)),
                    (aux(d1, 0), aux(gamma, 2)),
                    (aux(c0, 0), aux(delta, 1)),
                    (aux(c1, 0), aux(delta, 2)),
                    (aux(d0, 1), aux(c0, 1)),
                    (aux(d0, 2), aux(c1, 1)),
                    (aux(d1, 1), aux(c0, 2)),
                    (aux(d1, 2), aux(c1, 2)),
                ]);
            }
        }

        // Glue the left and right rule interfaces through the old wires. A
        // component with no surviving endpoint is a closed wire loop.
        let mut neighbors: BTreeMap<Port, Vec<Port>> = BTreeMap::new();
        for (x, y) in transient {
            neighbors.entry(x).or_default().push(y);
            neighbors.entry(y).or_default().push(x);
        }
        let mut visited = BTreeSet::new();
        for start in neighbors.keys().copied() {
            if !visited.insert(start) {
                continue;
            }
            let mut stack = vec![start];
            let mut ends = Vec::new();
            while let Some(port) = stack.pop() {
                let degree = neighbors.get(&port).ok_or(Error::InvalidRewrite)?.len();
                if old_aux.contains(&port) {
                    if degree != 2 {
                        return Err(Error::InvalidRewrite);
                    }
                } else {
                    if degree != 1 {
                        return Err(Error::InvalidRewrite);
                    }
                    ends.push(port);
                }
                for &next in neighbors.get(&port).ok_or(Error::InvalidRewrite)? {
                    if visited.insert(next) {
                        stack.push(next);
                    }
                }
            }
            match ends.as_slice() {
                [] => {
                    self.cyclic_wires = self.cyclic_wires.checked_add(1).ok_or(Error::PortLimit)?;
                }
                [x, y] => self.connect(*x, *y)?,
                _ => return Err(Error::InvalidRewrite),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, Kind, Limits, Net, Port};
    use std::collections::BTreeSet;

    const LIMITS: Limits = Limits {
        agents: 64,
        ports: 200,
        steps: 16,
    };

    fn a(id: usize, port: u8) -> Port {
        Port::Agent(id, port)
    }

    fn b(id: usize) -> Port {
        Port::Boundary(id)
    }

    fn pair(kind: Kind) -> Net {
        let mut net = Net::new(4);
        let x = net.add_agent(kind, LIMITS).unwrap();
        let y = net.add_agent(kind, LIMITS).unwrap();
        net.connect(a(x, 0), a(y, 0)).unwrap();
        for (port, boundary) in [(a(x, 1), 0), (a(x, 2), 1), (a(y, 1), 2), (a(y, 2), 3)] {
            net.connect(port, b(boundary)).unwrap();
        }
        net
    }

    fn normalized_wires(net: &Net) -> BTreeSet<(Port, Port)> {
        net.wires
            .iter()
            .map(|&(x, y)| if x < y { (x, y) } else { (y, x) })
            .collect()
    }

    #[test]
    fn issue_m1_03_minimal_reducer() {
        // Fixture GG-1: Lafont's crossed rule, unlike symmetric annihilation.
        let mut gamma = pair(Kind::Gamma);
        assert_eq!(gamma.reduce(LIMITS), Ok(1));
        assert_eq!(gamma.boundary_pairs(LIMITS), Ok(vec![(0, 3), (1, 2)]));

        // Fixture DD-1: corresponding auxiliaries remain aligned.
        let mut delta = pair(Kind::Delta);
        assert_eq!(delta.reduce(LIMITS), Ok(1));
        assert_eq!(delta.boundary_pairs(LIMITS), Ok(vec![(0, 2), (1, 3)]));

        // Fixture EE-1: empty normal form, no boundary.
        let mut empty = Net::new(0);
        let x = empty.add_agent(Kind::Epsilon, LIMITS).unwrap();
        let y = empty.add_agent(Kind::Epsilon, LIMITS).unwrap();
        empty.connect(a(x, 0), a(y, 0)).unwrap();
        assert_eq!(empty.reduce(LIMITS), Ok(1));
        assert_eq!(empty.boundary_pairs(LIMITS), Ok(vec![]));

        // Fixtures EG-1 and ED-1: each auxiliary receives a fresh eraser.
        for kind in [Kind::Gamma, Kind::Delta] {
            for eraser_first in [true, false] {
                let mut net = Net::new(2);
                let first = net
                    .add_agent(if eraser_first { Kind::Epsilon } else { kind }, LIMITS)
                    .unwrap();
                let second = net
                    .add_agent(if eraser_first { kind } else { Kind::Epsilon }, LIMITS)
                    .unwrap();
                let (e, other) = if eraser_first {
                    (first, second)
                } else {
                    (second, first)
                };
                net.connect(a(e, 0), a(other, 0)).unwrap();
                net.connect(a(other, 1), b(0)).unwrap();
                net.connect(a(other, 2), b(1)).unwrap();
                assert_eq!(net.reduce(LIMITS), Ok(1));
                assert_eq!(
                    normalized_wires(&net),
                    BTreeSet::from([(a(2, 0), b(0)), (a(3, 0), b(1))])
                );
            }
        }

        // Fixture GD-1: all four fresh agents and their exact eight wires.
        let mut mixed = Net::new(4);
        let g = mixed.add_agent(Kind::Gamma, LIMITS).unwrap();
        let d = mixed.add_agent(Kind::Delta, LIMITS).unwrap();
        mixed.connect(a(g, 0), a(d, 0)).unwrap();
        for (port, boundary) in [(a(g, 1), 0), (a(g, 2), 1), (a(d, 1), 2), (a(d, 2), 3)] {
            mixed.connect(port, b(boundary)).unwrap();
        }
        assert_eq!(mixed.reduce(LIMITS), Ok(1));
        assert_eq!(
            normalized_wires(&mixed),
            BTreeSet::from([
                (a(2, 0), b(0)),
                (a(3, 0), b(1)),
                (a(4, 0), b(2)),
                (a(5, 0), b(3)),
                (a(2, 1), a(4, 1)),
                (a(2, 2), a(5, 1)),
                (a(3, 1), a(4, 2)),
                (a(3, 2), a(5, 2)),
            ])
        );

        // Fixture GG-2: two active pairs composed through their auxiliaries.
        let mut composed = Net::new(4);
        for _ in 0..4 {
            composed.add_agent(Kind::Gamma, LIMITS).unwrap();
        }
        for (x, y) in [
            (a(0, 0), a(1, 0)),
            (a(2, 0), a(3, 0)),
            (a(0, 1), b(0)),
            (a(0, 2), b(1)),
            (a(1, 1), a(2, 1)),
            (a(1, 2), a(2, 2)),
            (a(3, 1), b(2)),
            (a(3, 2), b(3)),
        ] {
            composed.connect(x, y).unwrap();
        }
        assert_eq!(composed.reduce(LIMITS), Ok(2));
        assert_eq!(composed.boundary_pairs(LIMITS), Ok(vec![(0, 2), (1, 3)]));

        // Fixture GG-CLOSED: internal auxiliary wires produce a cyclic wire.
        let mut closed = Net::new(0);
        let x = closed.add_agent(Kind::Gamma, LIMITS).unwrap();
        let y = closed.add_agent(Kind::Gamma, LIMITS).unwrap();
        closed.connect(a(x, 0), a(y, 0)).unwrap();
        closed.connect(a(x, 1), a(y, 1)).unwrap();
        closed.connect(a(x, 2), a(y, 2)).unwrap();
        assert_eq!(closed.reduce(LIMITS), Ok(1));
        assert_eq!(closed.boundary_pairs(LIMITS), Ok(vec![]));
        assert_eq!(closed.cyclic_wires(), 1);

        // Wrong-port mutation changes the observed boundary contract.
        let mut wrong_port = pair(Kind::Gamma);
        let wire = wrong_port
            .wires
            .iter_mut()
            .find(|(x, _)| *x == a(1, 1))
            .unwrap();
        wire.1 = b(3);
        let wire = wrong_port
            .wires
            .iter_mut()
            .find(|(x, _)| *x == a(1, 2))
            .unwrap();
        wire.1 = b(2);
        assert_eq!(wrong_port.reduce(LIMITS), Ok(1));
        assert_ne!(wrong_port.boundary_pairs(LIMITS), Ok(vec![(0, 3), (1, 2)]));
    }

    #[test]
    fn issue_m1_03_minimal_reducer_all_boundary_permutations() {
        for kind in [Kind::Gamma, Kind::Delta] {
            for x in 0..4 {
                for y in 0..4 {
                    for z in 0..4 {
                        for w in 0..4 {
                            let ports = [x, y, z, w];
                            if ports.into_iter().collect::<BTreeSet<_>>().len() != 4 {
                                continue;
                            }
                            let mut net = Net::new(4);
                            let left = net.add_agent(kind, LIMITS).unwrap();
                            let right = net.add_agent(kind, LIMITS).unwrap();
                            net.connect(a(left, 0), a(right, 0)).unwrap();
                            for (port, boundary) in [
                                (a(left, 1), x),
                                (a(left, 2), y),
                                (a(right, 1), z),
                                (a(right, 2), w),
                            ] {
                                net.connect(port, b(boundary)).unwrap();
                            }
                            assert_eq!(net.reduce(LIMITS), Ok(1));
                            let expected = match kind {
                                Kind::Gamma => [(x, w), (y, z)],
                                Kind::Delta => [(x, z), (y, w)],
                                Kind::Epsilon => unreachable!(),
                            };
                            let mut expected = expected
                                .map(|(a, b)| (a.min(b), a.max(b)))
                                .into_iter()
                                .collect::<Vec<_>>();
                            expected.sort_unstable();
                            assert_eq!(net.boundary_pairs(LIMITS), Ok(expected));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn issue_m1_03_minimal_reducer_rejects_invalid_graphs_and_limits() {
        let mut dangling = Net::new(2);
        assert_eq!(dangling.validate(LIMITS), Err(Error::DanglingPort(b(0))));
        let e = dangling.add_agent(Kind::Epsilon, LIMITS).unwrap();
        assert_eq!(
            dangling.connect(a(e, 1), b(0)),
            Err(Error::InvalidPort(a(e, 1)))
        );
        assert_eq!(
            dangling.connect(a(e, 0), b(2)),
            Err(Error::InvalidPort(b(2)))
        );
        dangling.connect(a(e, 0), b(0)).unwrap();
        assert_eq!(
            dangling.connect(b(0), b(1)),
            Err(Error::DuplicatePort(b(0)))
        );
        assert_eq!(
            Kind::try_from("omega").unwrap_err(),
            "unsupported interaction agent: omega"
        );

        let mut malformed = pair(Kind::Gamma);
        malformed.wires.push((a(0, 1), b(0)));
        assert_eq!(
            malformed.validate(LIMITS),
            Err(Error::DuplicatePort(a(0, 1)))
        );

        let mut no_steps = pair(Kind::Gamma);
        let before = no_steps.clone();
        assert_eq!(
            no_steps.reduce(Limits { steps: 0, ..LIMITS }),
            Err(Error::StepLimit)
        );
        assert_eq!(no_steps, before);

        // Fixture GD-LOOP: corresponding auxiliary wires reproduce two
        // gamma/delta active pairs. The finite limit is not a divergence proof.
        let mut growth = Net::new(0);
        let g = growth.add_agent(Kind::Gamma, LIMITS).unwrap();
        let d = growth.add_agent(Kind::Delta, LIMITS).unwrap();
        growth.connect(a(g, 0), a(d, 0)).unwrap();
        growth.connect(a(g, 1), a(d, 1)).unwrap();
        growth.connect(a(g, 2), a(d, 2)).unwrap();
        assert_eq!(
            growth.reduce(Limits { steps: 2, ..LIMITS }),
            Err(Error::StepLimit)
        );
        growth.validate(LIMITS).unwrap();

        let node_budget = pair(Kind::Gamma);
        assert_eq!(
            node_budget.validate(Limits {
                agents: 1,
                ..LIMITS
            }),
            Err(Error::AgentLimit)
        );
        let mut erasure = Net::new(2);
        let e = erasure.add_agent(Kind::Epsilon, LIMITS).unwrap();
        let g = erasure.add_agent(Kind::Gamma, LIMITS).unwrap();
        erasure.connect(a(e, 0), a(g, 0)).unwrap();
        erasure.connect(a(g, 1), b(0)).unwrap();
        erasure.connect(a(g, 2), b(1)).unwrap();
        let before = erasure.clone();
        assert_eq!(
            erasure.reduce(Limits {
                agents: 3,
                ..LIMITS
            }),
            Err(Error::AgentLimit)
        );
        assert_eq!(erasure, before);
        assert_eq!(
            erasure.reduce(Limits { ports: 5, ..LIMITS }),
            Err(Error::PortLimit)
        );
        assert_eq!(erasure, before);
    }
}
