//! Source ownership and publication around the raw device spatial ABI.
use super::*;
use invar_device::spatial::Plan;

pub(crate) type Output = [Vec<u32>; 3];
fn plan_error() -> EvalError {
    EvalError::Tool {
        code: "RESOURCE_LIMIT",
        message: "spatial device plan exceeds its bounds".into(),
    }
}
fn packed(coords: &[std::cell::Ref<'_, Vec<F>>; 3]) -> Result<Vec<u32>, EvalError> {
    let mut out = reserve(3 * coords[0].len())?;
    for axis in coords {
        out.extend(axis.iter().map(|f| f.to_bits()));
    }
    Ok(out)
}

pub(crate) fn invoke(
    name: &str,
    args: &[Value],
    span: Span,
    live: &mut usize,
    launch: &mut impl FnMut(&Plan, [&[u32]; 4]) -> Result<Output, EvalError>,
    sort: &mut impl FnMut(&[u32], &[u32]) -> Result<(Vec<u32>, Vec<u32>), EvalError>,
) -> Result<Value, EvalError> {
    if name == "voxel_grid_build" {
        let coords = arrays(args)?;
        let n = shape(&coords, span)?;
        let Some(Value::F32(cell)) = args.get(3) else {
            return Err(invalid());
        };
        if cell.is_nan() || cell.is_infinite() || !cell.compare(Comparison::Gt, F::POS_ZERO) {
            return Err(trap(TrapCategory::IntegerOverflow, span));
        }
        let input = packed(&coords)?;
        let plan = Plan::mapping(n as u32, cell.to_bits()).map_err(|_| plan_error())?;
        let [status, keys, _] = launch(&plan, [&input, &[], &[], &[]])?;
        if status.len() != n || keys.len() != 3 * n {
            return Err(invalid());
        }
        let mut ids = reserve(n)?;
        for (i, &s) in status.iter().enumerate() {
            let is_finite = (0..3).all(|a| finite([F::from_bits(input[a * n + i]); 3]));
            if s > 2 || (s == 1) == is_finite {
                return Err(invalid());
            }
            if s == 2 {
                return Err(trap(TrapCategory::IntegerOverflow, span));
            }
            if s == 0 {
                ids.push(i as u32);
            }
        }
        let mut seen = reserve(n)?;
        seen.resize(n, false);
        for axis in (0..3).rev() {
            let mut axis_keys = reserve(ids.len())?;
            axis_keys.extend(ids.iter().map(|&i| keys[axis * n + i as usize]));
            let (out_keys, out_ids) = sort(&axis_keys, &ids)?;
            if out_keys.len() != ids.len() || out_ids.len() != ids.len() {
                return Err(invalid());
            }
            seen.fill(false);
            let mut previous = None;
            for (&key, &id) in out_keys.iter().zip(&out_ids) {
                let i = id as usize;
                if i >= n
                    || status[i] != 0
                    || seen[i]
                    || key != keys[axis * n + i]
                    || previous.is_some_and(|p| p > key as i32)
                {
                    return Err(invalid());
                }
                seen[i] = true;
                previous = Some(key as i32);
            }
            // Verify stability, including the original order inherited from
            // lower coordinate passes; no host sort substitutes for device work.
            let mut ranks = reserve(n)?;
            ranks.resize(n, 0usize);
            for (rank, &id) in ids.iter().enumerate() {
                ranks[id as usize] = rank;
            }
            if out_keys
                .windows(2)
                .zip(out_ids.windows(2))
                .any(|(k, i)| k[0] == k[1] && ranks[i[0] as usize] >= ranks[i[1] as usize])
            {
                return Err(invalid());
            }
            ids = out_ids;
        }
        let m = ids.len();
        let mut sorted_keys = reserve(3 * m)?;
        for axis in 0..3 {
            sorted_keys.extend(ids.iter().map(|&i| keys[axis * n + i as usize]));
        }
        let plan = Plan::boundaries(m as u32).map_err(|_| plan_error())?;
        let [starts, _, _] = launch(&plan, [&sorted_keys, &[], &[], &[]])?;
        if starts.len() != m {
            return Err(invalid());
        }
        let mut points: Vec<Point> = reserve(m)?;
        let mut cells: Vec<Range<usize>> = reserve(m)?;
        for (i, &id) in ids.iter().enumerate() {
            let key = std::array::from_fn(|a| keys[a * n + id as usize] as i32);
            let boundary = i == 0 || key != points[i - 1].cell;
            if starts[i] != u32::from(boundary) {
                return Err(invalid());
            }
            if boundary {
                cells.push(i..i + 1);
            } else if let Some(c) = cells.last_mut() {
                c.end = i + 1;
            }
            points.push(Point {
                coords: std::array::from_fn(|a| F::from_bits(input[a * n + id as usize])),
                cell: key,
                index: id,
            });
        }
        let grid = Grid {
            points,
            cells,
            excluded: (n - m) as u32,
        };
        eval::admit_live(live, grid.storage_elements())?;
        return Ok(Value::VoxelGrid(Rc::new(grid)));
    }
    let Some(Value::VoxelGrid(grid)) = args.first() else {
        return Err(invalid());
    };
    if name == "grid_excluded" {
        return Ok(Value::U32(grid.excluded));
    }
    let coords = arrays(args.get(1..).ok_or_else(invalid)?)?;
    let q = shape(&coords, span)?;
    let Some(Value::U32(stride)) = args.last() else {
        return Err(invalid());
    };
    let slots = u32::try_from(q)
        .ok()
        .and_then(|q| q.checked_mul(*stride))
        .ok_or_else(|| trap(TrapCategory::IntegerOverflow, span))? as usize;
    if slots > eval::MAX_ARRAY_LEN {
        return Err(EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            message: "spatial result array limit exceeded".into(),
        });
    }
    let radius = if name == "radius_neighbors" {
        let Some(Value::F32(r)) = args.get(4) else {
            return Err(invalid());
        };
        Some(r.to_bits())
    } else if name == "knn" {
        None
    } else {
        return Err(invalid());
    };
    eval::admit_live(
        live,
        if radius.is_some() {
            q + slots + 2
        } else {
            2 * slots + 2
        },
    )?;
    let n = grid.points.len();
    let mut points = reserve(3 * n)?;
    for axis in 0..3 {
        points.extend(grid.points.iter().map(|p| p.coords[axis].to_bits()));
    }
    let queries = packed(&coords)?;
    let mut ids = reserve(n)?;
    ids.extend(grid.points.iter().map(|p| p.index));
    let mut ranges = reserve(2 * grid.cells.len())?;
    for c in &grid.cells {
        ranges.extend([c.start as u32, c.end as u32]);
    }
    let plan = Plan::query(n as u32, q as u32, grid.cells.len() as u32, *stride, radius)
        .map_err(|_| plan_error())?;
    let [counts, indices, distances] = launch(&plan, [&points, &queries, &ids, &ranges])?;
    if counts.len() != q
        || indices.len() != slots
        || distances.len() != slots
        || counts.iter().any(|&c| c as usize > n)
    {
        return Err(invalid());
    }
    // Reject malformed device data before an owned source record is published.
    let mut valid_ids = reserve(n)?;
    valid_ids.extend_from_slice(&ids);
    valid_ids.sort_unstable();
    for row in 0..q {
        let mut previous = None;
        let mut padded = false;
        for slot in row * (*stride as usize)..(row + 1) * (*stride as usize) {
            let id = indices[slot];
            let d2 = distances[slot];
            if id == u32::MAX {
                if d2 != 0x7f800000 {
                    return Err(invalid());
                }
                padded = true;
            } else {
                if padded
                    || valid_ids.binary_search(&id).is_err()
                    || d2 > 0x7f800000
                    || previous.is_some_and(|p| p >= (d2, id))
                {
                    return Err(invalid());
                }
                previous = Some((d2, id));
            }
        }
        let retained = indices[row * (*stride as usize)..(row + 1) * (*stride as usize)]
            .iter()
            .take_while(|&&id| id != u32::MAX)
            .count();
        if retained != (*stride as usize).min(counts[row] as usize) {
            return Err(invalid());
        }
    }
    let fields = if radius.is_some() {
        vec![u32s(counts), u32s(indices)]
    } else {
        vec![
            u32s(indices),
            Value::ArrayF32(Rc::new(RefCell::new(
                distances.into_iter().map(F::from_bits).collect(),
            ))),
        ]
    };
    Ok(Value::Record(Rc::new(RefCell::new(fields))))
}
