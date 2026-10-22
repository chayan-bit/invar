/// Dominator tree of the blocks reachable from the entry, computed with the
/// Cooper-Harvey-Kennedy iterative algorithm over reverse postorder. Queries
/// are O(1) interval checks on a preorder numbering of the tree.
struct Dominance {
    /// Reachable blocks in reverse postorder; the entry comes first.
    order: Vec<BlockId>,
    /// Preorder entry/exit numbers in the dominator tree; `None` if unreachable.
    interval: Vec<Option<(usize, usize)>>,
}

impl Dominance {
    fn new(function: &Function, predecessors: &[Vec<BlockId>]) -> Self {
        let count = function.blocks.len();
        let successors: Vec<Vec<BlockId>> = function
            .blocks
            .iter()
            .map(|block| {
                block.end.as_ref().map_or_else(Vec::new, |end| {
                    end.targets().into_iter().map(|(id, _)| id).collect()
                })
            })
            .collect();
        let mut visited = vec![false; count];
        let mut postorder = Vec::new();
        let mut stack = vec![(function.entry, 0)];
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
        for (index, block) in postorder.iter().enumerate() {
            rank[*block] = Some(index);
        }
        let order: Vec<BlockId> = postorder.into_iter().rev().collect();
        let mut idom = vec![None; count];
        idom[function.entry] = Some(function.entry);
        let mut changed = true;
        while changed {
            changed = false;
            for &block in order.iter().skip(1) {
                let mut new_idom = None;
                for &predecessor in &predecessors[block] {
                    if idom[predecessor].is_none() {
                        continue;
                    }
                    new_idom = Some(match new_idom {
                        None => predecessor,
                        Some(current) => intersect(predecessor, current, &idom, &rank),
                    });
                }
                if new_idom.is_some() && idom[block] != new_idom {
                    idom[block] = new_idom;
                    changed = true;
                }
            }
        }
        let mut children = vec![Vec::new(); count];
        for &block in order.iter().skip(1) {
            if let Some(parent) = idom[block] {
                children[parent].push(block);
            }
        }
        let mut interval = vec![None; count];
        let mut clock = 0;
        let mut stack = vec![(function.entry, 0)];
        while let Some((block, next)) = stack.last_mut() {
            let block = *block;
            if *next == 0 {
                interval[block] = Some((clock, clock));
                clock += 1;
            }
            if let Some(&child) = children[block].get(*next) {
                *next += 1;
                stack.push((child, 0));
            } else {
                if let Some((enter, _)) = interval[block] {
                    interval[block] = Some((enter, clock));
                }
                stack.pop();
            }
        }
        Self { order, interval }
    }

    /// Whether `dominator` dominates the reachable block `block`.
    fn dominates(&self, dominator: BlockId, block: BlockId) -> bool {
        match (self.interval.get(dominator), self.interval.get(block)) {
            (Some(Some((outer_in, outer_out))), Some(Some((inner_in, inner_out)))) => {
                outer_in <= inner_in && inner_out <= outer_out
            }
            _ => false,
        }
    }
}

/// Nearest common dominator of two processed blocks; `rank` is postorder.
fn intersect(
    mut left: BlockId,
    mut right: BlockId,
    idom: &[Option<BlockId>],
    rank: &[Option<usize>],
) -> BlockId {
    while left != right {
        while rank[left] < rank[right] {
            left = idom[left].unwrap_or(left);
        }
        while rank[right] < rank[left] {
            right = idom[right].unwrap_or(right);
        }
    }
    left
}

fn check_use(
    id: ValueId,
    block: BlockId,
    position: usize,
    defs: &HashMap<ValueId, Definition>,
    dominance: &Dominance,
) -> Result<(), Error> {
    let definition = defs
        .get(&id)
        .ok_or_else(|| invalid("undefined MIR value"))?;
    if definition.block == block {
        if definition
            .position
            .is_some_and(|defined_at| defined_at >= position)
        {
            return Err(invalid("MIR use before definition"));
        }
    } else if !dominance.dominates(definition.block, block) {
        return Err(invalid("MIR definition does not dominate use"));
    }
    Ok(())
}
