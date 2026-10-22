//! Independent Rust references for the family baseline fixtures. A fixture's
//! full output is compared with these before any of its timings are reported,
//! so a wrong library or handwritten kernel fails instead of producing a time.

const LCG_MUL: u32 = 1_664_525;
const LCG_ADD: u32 = 1_013_904_223;
const GATHER_MUL: u32 = 2_654_435_761;

/// The manifest input recipe `lcg1664525_1013904223_plus_slot17_v1`.
pub fn inputs(n: usize, count: usize) -> Vec<Vec<u32>> {
    (0..count)
        .map(|slot| (0..n).map(|i| lcg(i, slot)).collect())
        .collect()
}

fn lcg(i: usize, slot: usize) -> u32 {
    (i as u32)
        .wrapping_mul(LCG_MUL)
        .wrapping_add(LCG_ADD)
        .wrapping_add(17 * slot as u32)
}

/// Frontier work is pinned to this many level-synchronous rounds (M4-05 caps).
pub const FRONTIER_ROUNDS: usize = 16;

/// Rows and columns of a near-square grid with `n` cells.
pub fn grid(n: usize) -> (usize, usize) {
    let mut rows = (n as f64).sqrt() as usize;
    while !n.is_multiple_of(rows) {
        rows -= 1;
    }
    (rows, n / rows)
}

pub fn gather_indices(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| (u64::from(i) * u64::from(GATHER_MUL) % n as u64) as u32)
        .collect()
}

pub fn ref_transpose(input: &[u32], rows: usize, cols: usize) -> Vec<u32> {
    let mut out = vec![0; input.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = input[r * cols + c];
        }
    }
    out
}

/// Radius-1, zero-boundary, wrapping u32 five-point stencil, one step.
pub fn ref_stencil_u32(input: &[u32], rows: usize, cols: usize) -> Vec<u32> {
    let at = |r: usize, c: usize| input[r * cols + c];
    let mut out = vec![0u32; input.len()];
    for r in 0..rows {
        for c in 0..cols {
            let mut v = at(r, c);
            if r > 0 {
                v = v.wrapping_add(at(r - 1, c));
            }
            if r + 1 < rows {
                v = v.wrapping_add(at(r + 1, c));
            }
            if c > 0 {
                v = v.wrapping_add(at(r, c - 1));
            }
            if c + 1 < cols {
                v = v.wrapping_add(at(r, c + 1));
            }
            out[r * cols + c] = v;
        }
    }
    out
}

pub fn ref_segment_sums(input: &[u32], offsets: &[u32]) -> Vec<u32> {
    offsets
        .windows(2)
        .map(|w| {
            input[w[0] as usize..w[1] as usize]
                .iter()
                .fold(0u32, |a, x| a.wrapping_add(*x))
        })
        .collect()
}

pub fn ref_compact(input: &[u32]) -> Vec<u32> {
    input.iter().copied().filter(|x| *x < 1 << 31).collect()
}

/// Dense adjacency of the calibration graph: u -> u+1 and u -> 5u+1 (mod v).
pub fn frontier_edges(v: usize) -> Vec<u32> {
    let mut edges = vec![0u32; v * v];
    for u in 0..v {
        edges[u * v + (u + 1) % v] = 1;
        edges[u * v + (u * 5 + 1) % v] = 1;
    }
    edges
}

/// Level-synchronous BFS from vertex 0 over `rounds` rounds: the seen mask
/// and each round's next-frontier mask, with the M4-05 device law (a vertex
/// joins `next` only while it is unseen at the time its edge is scanned).
pub fn ref_frontier(edges: &[u32], v: usize, rounds: usize) -> (Vec<u32>, Vec<Vec<u32>>) {
    let (mut seen, mut frontier) = (vec![0u32; v], vec![0u32; v]);
    let mut masks = Vec::with_capacity(rounds);
    frontier[0] = 1;
    for _ in 0..rounds {
        let mut next = vec![0u32; v];
        for u in 0..v {
            if frontier[u] == 1 && seen[u] == 0 {
                seen[u] = 1;
                for (w, bit) in next.iter_mut().enumerate() {
                    if edges[u * v + w] == 1 && seen[w] == 0 {
                        *bit = 1;
                    }
                }
            }
        }
        masks.push(next.clone());
        frontier = next;
    }
    (seen, masks)
}
