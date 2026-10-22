#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) mod device;

use crate::{
    eval::{self, EvalError, TrapCategory, Value},
    front::Span,
    numeric_contract::{Comparison, F32Bits as F},
    sorting::{self, KeyKind},
};
use std::{
    cell::{Cell, RefCell},
    ops::Range,
    rc::Rc,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Point {
    coords: [F; 3],
    cell: [i32; 3],
    index: u32,
}

/// Opaque snapshot of finite points, sorted by full signed cell coordinates.
/// Original indices are retained; no source array storage is borrowed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    points: Vec<Point>,
    cells: Vec<Range<usize>>,
    excluded: u32,
}

impl Grid {
    pub(crate) fn storage_elements(&self) -> usize {
        // Three coordinates, three cell coordinates, original index; cell ranges
        // use two host words. Charge them conservatively as four u32 elements.
        7 * self.points.len() + 4 * self.cells.len() + 1
    }
}

fn invalid() -> EvalError {
    EvalError::Tool {
        code: "INTERNAL_TYPE",
        message: "invalid checked spatial operands".into(),
    }
}
fn resource() -> EvalError {
    EvalError::Tool {
        code: "RESOURCE_LIMIT",
        message: "spatial allocation failed".into(),
    }
}
fn reserve<T>(length: usize) -> Result<Vec<T>, EvalError> {
    crate::sort_cpu::reserve(length).map_err(|_| resource())
}
fn trap(category: TrapCategory, span: Span) -> EvalError {
    EvalError::Trap { category, span }
}
fn finite(point: [F; 3]) -> bool {
    point.iter().all(|v| !v.is_nan() && !v.is_infinite())
}

// The oracle uses rational division; the runtime uses exact integer decomposition.
fn cell_index(coord: F, cell: F, span: Span) -> Result<i32, EvalError> {
    crate::spatial_cell::index(coord.to_bits(), cell.to_bits())
        .ok_or_else(|| trap(TrapCategory::IntegerOverflow, span))
}

fn arrays(args: &[Value]) -> Result<[std::cell::Ref<'_, Vec<F>>; 3], EvalError> {
    let [
        Value::ArrayF32(x),
        Value::ArrayF32(y),
        Value::ArrayF32(z),
        ..,
    ] = args
    else {
        return Err(invalid());
    };
    Ok([
        x.try_borrow().map_err(|_| invalid())?,
        y.try_borrow().map_err(|_| invalid())?,
        z.try_borrow().map_err(|_| invalid())?,
    ])
}
fn shape(coords: &[std::cell::Ref<'_, Vec<F>>; 3], span: Span) -> Result<usize, EvalError> {
    let n = coords[0].len();
    if coords[1].len() != n || coords[2].len() != n {
        return Err(trap(TrapCategory::IndexOutOfBounds, span));
    }
    Ok(n)
}
fn build(args: &[Value], span: Span, live: &mut usize) -> Result<Value, EvalError> {
    let coords = arrays(args)?;
    let n = shape(&coords, span)?;
    let Some(Value::F32(cell)) = args.get(3) else {
        return Err(invalid());
    };
    if cell.is_nan() || cell.is_infinite() || !cell.compare(Comparison::Gt, F::POS_ZERO) {
        return Err(trap(TrapCategory::IntegerOverflow, span));
    }
    let mut points = reserve(n)?;
    let mut excluded = 0;
    for (i, ((&x, &y), &z)) in coords[0]
        .iter()
        .zip(coords[1].iter())
        .zip(coords[2].iter())
        .enumerate()
    {
        let point = [x, y, z];
        if !finite(point) {
            excluded += 1;
            continue;
        }
        // Validate in original point/axis order, before sorting or publication.
        let key = [
            cell_index(point[0], *cell, span)?,
            cell_index(point[1], *cell, span)?,
            cell_index(point[2], *cell, span)?,
        ];
        points.push(Point {
            coords: point,
            cell: key,
            index: u32::try_from(i).map_err(|_| trap(TrapCategory::IntegerOverflow, span))?,
        });
    }
    // Three stable LSD coordinate passes reuse the merged Invar sort intrinsic's
    // word-buffer runtime. Full signed coordinates avoid hash/Morton collisions.
    for axis in (0..3).rev() {
        let mut keys = reserve(points.len())?;
        keys.extend(points.iter().map(|p| p.cell[axis] as u32));
        let order = sorting::argsort_words(&keys, KeyKind::I32, 1).map_err(|_| resource())?;
        let mut sorted = reserve(points.len())?;
        sorted.extend(order.iter().map(|&i| points[i as usize].clone()));
        points = sorted;
    }
    let mut cells: Vec<Range<usize>> = reserve(points.len())?;
    for i in 0..points.len() {
        if i == 0 || points[i].cell != points[i - 1].cell {
            cells.push(i..i + 1);
        } else if let Some(last) = cells.last_mut() {
            last.end = i + 1;
        }
    }
    let grid = Grid {
        points,
        cells,
        excluded,
    };
    eval::admit_live(live, grid.storage_elements())?;
    Ok(Value::VoxelGrid(Rc::new(grid)))
}

fn distance(point: [F; 3], query: [F; 3]) -> F {
    let dx = point[0].sub(query[0]);
    let dy = point[1].sub(query[1]);
    let dz = point[2].sub(query[2]);
    dx.mul(dx).add(dy.mul(dy)).add(dz.mul(dz))
}
fn u32s(values: Vec<u32>) -> Value {
    Value::ArrayU32(Rc::new(RefCell::new(values)))
}

pub(crate) fn invoke(
    name: &str,
    args: &[Value],
    span: Span,
    live: &mut usize,
) -> Result<Value, EvalError> {
    invoke_counted(
        name,
        args,
        span,
        live,
        1,
        &Cell::new(0),
        &crate::schedule::Hooks::default(),
    )
}

pub(super) fn invoke_counted(
    name: &str,
    args: &[Value],
    span: Span,
    live: &mut usize,
    workers: usize,
    joined: &Cell<usize>,
    hooks: &crate::schedule::Hooks,
) -> Result<Value, EvalError> {
    if name == "voxel_grid_build" {
        return build(args, span, live);
    }
    let Some(Value::VoxelGrid(grid)) = args.first() else {
        return Err(invalid());
    };
    if name == "grid_excluded" {
        return Ok(Value::U32(grid.excluded));
    }
    let coords = arrays(args.get(1..).ok_or_else(invalid)?)?;
    let n = shape(&coords, span)?;
    let Some(Value::U32(stride)) = args.last() else {
        return Err(invalid());
    };
    let slots = u32::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(*stride))
        .ok_or_else(|| trap(TrapCategory::IntegerOverflow, span))? as usize;
    if slots > eval::MAX_ARRAY_LEN {
        return Err(EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            message: "spatial result array limit exceeded".into(),
        });
    }
    let radius = if name == "radius_neighbors" {
        let Some(Value::F32(radius)) = args.get(4) else {
            return Err(invalid());
        };
        Some(*radius)
    } else if name == "knn" {
        None
    } else {
        return Err(invalid());
    };
    let allowed = radius.is_none_or(|r| !r.is_nan() && !r.compare(Comparison::Lt, F::POS_ZERO));
    let threshold = radius.map(|r| r.mul(r));
    let allocation = if radius.is_some() {
        n + slots + 2
    } else {
        2 * slots + 2
    };
    eval::admit_live(live, allocation)?;
    let mut counts = reserve(n)?;
    let mut indices = reserve(slots)?;
    let mut distances = reserve(if radius.is_none() { slots } else { 0 })?;
    let slices = [
        coords[0].as_slice(),
        coords[1].as_slice(),
        coords[2].as_slice(),
    ];
    let grid = grid.as_ref();
    let before = hooks.joined.load(std::sync::atomic::Ordering::SeqCst);
    let scheduled = crate::schedule::parallel_chunks(n, workers, hooks, |range| {
        query_rows(grid, slices, range, *stride as usize, allowed, threshold)
            .map_err(|_| crate::schedule::Error::Allocation)
    });
    joined.set(joined.get() + hooks.joined.load(std::sync::atomic::Ordering::SeqCst) - before);
    let (parts, _, _) = scheduled.map_err(|error| EvalError::Tool {
        code: error.code(),
        message: format!("spatial workers failed: {error:?}"),
    })?;
    // parallel_chunks joins in source-range order; rows never depend on worker
    // completion order, and no result owner is published until every join succeeds.
    for (part_counts, part_indices, part_distances) in parts {
        counts.extend(part_counts);
        indices.extend(part_indices);
        distances.extend(part_distances);
    }
    let fields = if radius.is_some() {
        vec![u32s(counts), u32s(indices)]
    } else {
        vec![
            u32s(indices),
            Value::ArrayF32(Rc::new(RefCell::new(distances))),
        ]
    };
    Ok(Value::Record(Rc::new(RefCell::new(fields))))
}

// Keep each query independent, with a distance/index total key. The reference
// calls this with one worker; the CPU scheduler supplies bounded scoped workers.
type Rows = (Vec<u32>, Vec<u32>, Vec<F>);
fn query_rows(
    grid: &Grid,
    coords: [&[F]; 3],
    range: Range<usize>,
    stride: usize,
    allowed: bool,
    threshold: Option<F>,
) -> Result<Rows, EvalError> {
    let mut counts = reserve(range.len())?;
    let slots = range.len().checked_mul(stride).ok_or_else(resource)?;
    let mut indices = reserve(slots)?;
    let mut distances = reserve(if threshold.is_none() { slots } else { 0 })?;
    let mut row = reserve(grid.points.len())?;
    for q in range {
        row.clear();
        let query = [coords[0][q], coords[1][q], coords[2][q]];
        if allowed && finite(query) && (threshold.is_some() || stride != 0) {
            // ponytail: exhaustive occupied-cell traversal; add conservative
            // strict-f32 pruning when qualifying production query performance.
            // Empty shells are never enumerated, and underflow/ties stay exact.
            for cell in &grid.cells {
                for point in &grid.points[cell.clone()] {
                    let d2 = distance(point.coords, query);
                    if threshold.is_none_or(|r2| d2.compare(Comparison::Le, r2)) {
                        row.push((d2, point.index));
                    }
                }
            }
        }
        row.sort_unstable_by_key(|(d2, index)| (d2.to_bits(), *index));
        counts.push(row.len() as u32); // Build admits at most u32::MAX original indices.
        for slot in 0..stride {
            let (d2, index) = row.get(slot).copied().unwrap_or((F::INFINITY, u32::MAX));
            indices.push(index);
            if threshold.is_none() {
                distances.push(d2);
            }
        }
    }
    Ok((counts, indices, distances))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_cell_floor_matches_independent_rationals() {
        let span = Span { start: 0, end: 1 };
        let mut seed = 0x43454c4cu32;
        for _ in 0..2048 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let coord = F::from_bits(seed);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let cell = F::from_bits(seed & 0x7fffffff);
            if !finite([coord, cell, cell]) || cell.is_zero() {
                continue;
            }
            let a = num_rational::BigRational::from_float(f32::from_bits(coord.to_bits()) as f64)
                .unwrap();
            let b = num_rational::BigRational::from_float(f32::from_bits(cell.to_bits()) as f64)
                .unwrap();
            let expected = i32::try_from((a / b).floor().to_integer());
            match expected {
                Ok(value) => assert_eq!(cell_index(coord, cell, span), Ok(value)),
                Err(_) => assert_eq!(
                    cell_index(coord, cell, span),
                    Err(trap(TrapCategory::IntegerOverflow, span))
                ),
            }
        }
    }
    #[test]
    fn exact_cell_floor_boundaries() {
        let span = Span { start: 0, end: 1 };
        let f = |v: f32| F::from_bits(v.to_bits());
        assert_eq!(cell_index(f(-1.0), f(3.0), span), Ok(-1));
        assert_eq!(cell_index(f(-2147483648.0), f(1.0), span), Ok(i32::MIN));
        assert!(cell_index(f(2147483648.0), f(1.0), span).is_err());
        assert_eq!(cell_index(F::from_bits(1), f(1.0), span), Ok(0));
        assert_eq!(cell_index(F::from_bits(0x80000001), f(1.0), span), Ok(-1));
        assert_eq!(cell_index(f(1.0), f(0.1), span), Ok(9));
    }
}
