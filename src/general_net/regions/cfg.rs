//! Control-flow facts of one net-derived MIR function: successors,
//! predecessors, dominators (Cooper-Harvey-Kennedy over reverse postorder) and
//! natural loops. Only blocks reachable from the entry take part.

use crate::mir::{BlockId, Function, Terminator};

pub(crate) struct Cfg {
    pub successors: Vec<Vec<BlockId>>,
    pub predecessors: Vec<Vec<BlockId>>,
    /// Immediate dominator of each reachable block; the entry maps to itself.
    idom: Vec<Option<BlockId>>,
    /// Reverse-postorder rank, used by the dominator intersection.
    rank: Vec<Option<usize>>,
}

fn targets(end: Option<&Terminator>) -> Vec<BlockId> {
    match end {
        Some(Terminator::Goto { target, .. }) => vec![*target],
        Some(Terminator::Branch { yes, no, .. }) => vec![*yes, *no],
        Some(Terminator::Return { .. }) | None => Vec::new(),
    }
}

impl Cfg {
    /// `None` when a terminator names a block outside the function (MIR
    /// verification rejects that, so it signals a stale or forged program).
    pub fn new(function: &Function) -> Option<Self> {
        let count = function.blocks.len();
        let successors: Vec<Vec<BlockId>> = function
            .blocks
            .iter()
            .map(|block| targets(block.end.as_ref()))
            .collect();
        if successors.iter().flatten().any(|&target| target >= count) || function.entry >= count {
            return None;
        }
        let mut predecessors = vec![Vec::new(); count];
        for (block, outs) in successors.iter().enumerate() {
            for &target in outs {
                predecessors[target].push(block);
            }
        }
        let mut postorder = Vec::with_capacity(count);
        let mut visited = vec![false; count];
        let mut stack = vec![(function.entry, 0usize)];
        visited[function.entry] = true;
        while let Some((block, next)) = stack.last_mut() {
            let block = *block;
            if let Some(&successor) = successors[block].get(*next) {
                *next += 1;
                if !visited[successor] {
                    visited[successor] = true;
                    stack.push((successor, 0));
                }
            } else {
                postorder.push(block);
                stack.pop();
            }
        }
        let mut rank = vec![None; count];
        for (index, &block) in postorder.iter().enumerate() {
            rank[block] = Some(index);
        }
        let order: Vec<BlockId> = postorder.iter().rev().copied().collect();
        let mut idom: Vec<Option<BlockId>> = vec![None; count];
        idom[function.entry] = Some(function.entry);
        let mut changed = true;
        while changed {
            changed = false;
            for &block in order.iter().skip(1) {
                let mut candidate = None;
                for &pred in &predecessors[block] {
                    if idom[pred].is_none() {
                        continue;
                    }
                    candidate = Some(match candidate {
                        None => pred,
                        Some(current) => intersect(pred, current, &idom, &rank)?,
                    });
                }
                if candidate.is_some() && idom[block] != candidate {
                    idom[block] = candidate;
                    changed = true;
                }
            }
        }
        Some(Self {
            successors,
            predecessors,
            idom,
            rank,
        })
    }

    pub fn reachable(&self, block: BlockId) -> bool {
        self.rank.get(block).is_some_and(Option::is_some)
    }

    /// Whether `a` dominates `b` (reflexive). Unreachable blocks dominate nothing.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if !self.reachable(a) || !self.reachable(b) {
            return false;
        }
        let mut current = b;
        loop {
            if current == a {
                return true;
            }
            match self.idom[current] {
                Some(parent) if parent != current => current = parent,
                _ => return false,
            }
        }
    }

    /// Loop heads with their latches: `latch -> head` edges where the head
    /// dominates the latch. Every verified builder CFG is reducible, so these
    /// are exactly the DFS back edges.
    pub fn loop_heads(&self) -> Vec<(BlockId, Vec<BlockId>)> {
        let mut heads: Vec<(BlockId, Vec<BlockId>)> = Vec::new();
        for (block, outs) in self.successors.iter().enumerate() {
            if !self.reachable(block) {
                continue;
            }
            for &target in outs {
                if self.dominates(target, block) {
                    match heads.iter_mut().find(|(head, _)| *head == target) {
                        Some((_, latches)) => latches.push(block),
                        None => heads.push((target, vec![block])),
                    }
                }
            }
        }
        heads.sort_by_key(|(head, _)| *head);
        heads
    }

    /// Natural loop body of `head`: the head plus every block that reaches a
    /// latch without passing through the head. Sorted ascending.
    pub fn natural_loop(&self, head: BlockId, latches: &[BlockId]) -> Vec<BlockId> {
        let mut inside = vec![false; self.successors.len()];
        inside[head] = true;
        let mut stack: Vec<BlockId> = latches.to_vec();
        while let Some(block) = stack.pop() {
            if std::mem::replace(&mut inside[block], true) {
                continue;
            }
            stack.extend(self.predecessors[block].iter().copied());
        }
        inside
            .iter()
            .enumerate()
            .filter_map(|(block, &flag)| flag.then_some(block))
            .collect()
    }
}

fn intersect(
    mut a: BlockId,
    mut b: BlockId,
    idom: &[Option<BlockId>],
    rank: &[Option<usize>],
) -> Option<BlockId> {
    while a != b {
        while rank[a]? < rank[b]? {
            a = idom[a]?;
        }
        while rank[b]? < rank[a]? {
            b = idom[b]?;
        }
    }
    Some(a)
}
