// Standalone spatial ABI; operands are admitted by verified MIR signatures.
fn spatial_failure(error: invar_spatial::Error, start: u32, end: u32) -> ! {
    match error {
        invar_spatial::Error::Shape => trap(3, start, end),
        invar_spatial::Error::Overflow => trap(1, start, end),
        invar_spatial::Error::Allocation => tool(3),
        invar_spatial::Error::Internal => tool(1),
    }
}
fn spatial_coords(
    runtime: &Runtime,
    handles: [u64; 3],
) -> Result<[&[u32]; 3], invar_spatial::Error> {
    let get = |handle| match array(runtime, handle) {
        Some(Array::U32(v)) => Ok(v.as_slice()),
        _ => Err(invar_spatial::Error::Internal),
    };
    Ok([get(handles[0])?, get(handles[1])?, get(handles[2])?])
}
fn spatial_words(words: Vec<u32>) -> u64 {
    let handle = new_array(2, words.len(), 0);
    let committed = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        if let Some(Array::U32(target)) = array_mut(&mut runtime, handle) {
            *target = words;
            true
        } else {
            false
        }
    });
    if !committed {
        tool(1);
    }
    handle
}
#[unsafe(no_mangle)]
pub extern "C" fn invar_spatial_v1(
    op: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    start: u32,
    end: u32,
) -> u64 {
    use invar_spatial::Error;
    if op == 0 {
        let grid = RUNTIME
            .with(|cell| {
                let runtime = cell.try_borrow().map_err(|_| Error::Internal)?;
                let coords = spatial_coords(&runtime, [a0, a1, a2])?;
                invar_spatial::Grid::build(coords, u32::try_from(a3).map_err(|_| Error::Internal)?)
            })
            .unwrap_or_else(|e| spatial_failure(e, start, end));
        let inserted = RUNTIME.with(|cell| {
            let mut runtime = cell.borrow_mut();
            let length = grid.storage_elements();
            let total = runtime
                .live_elements
                .checked_add(length)
                .filter(|n| *n <= MAX_LIVE_ELEMENTS)
                .ok_or(2u32)?;
            runtime.arrays.try_reserve(1).map_err(|_| 3u32)?;
            runtime.live_elements = total;
            runtime.arrays.push(Some(Array::Grid(grid)));
            Ok(runtime.arrays.len() as u64)
        });
        return inserted.unwrap_or_else(|e| tool(e));
    }
    if op == 3 {
        let excluded = RUNTIME.with(|cell| match array(&cell.borrow(), a0) {
            Some(Array::Grid(grid)) => Some(grid.excluded),
            _ => None,
        });
        return u64::from(excluded.unwrap_or_else(|| tool(1)));
    }
    if op != 1 && op != 2 {
        tool(1);
    }
    let stride = u32::try_from(if op == 1 { a5 } else { a4 }).unwrap_or_else(|_| tool(1));
    // Shapes and u32 multiplication precede tool budgets, including empty or
    // non-finite query rows and radii that would otherwise produce no matches.
    let sizes = RUNTIME
        .with(|cell| {
            let runtime = cell.try_borrow().map_err(|_| Error::Internal)?;
            let n = invar_spatial::shape(spatial_coords(&runtime, [a1, a2, a3])?)?;
            let slots = u32::try_from(n)
                .ok()
                .and_then(|n| n.checked_mul(stride))
                .ok_or(Error::Overflow)? as usize;
            Ok((n, slots, runtime.live_elements))
        })
        .unwrap_or_else(|e| spatial_failure(e, start, end));
    let (n, slots, live) = sizes;
    if slots > MAX_ARRAY_LEN {
        tool(2);
    }
    let charge = if op == 1 {
        n + slots + 2
    } else {
        2 * slots + 2
    };
    if live
        .checked_add(charge)
        .is_none_or(|n| n > MAX_LIVE_ELEMENTS)
    {
        tool(2);
    }
    let rows = RUNTIME
        .with(|cell| {
            let runtime = cell.try_borrow().map_err(|_| Error::Internal)?;
            let Some(Array::Grid(grid)) = array(&runtime, a0) else {
                return Err(Error::Internal);
            };
            let radius = if op == 1 {
                Some(u32::try_from(a4).map_err(|_| Error::Internal)?)
            } else {
                None
            };
            grid.query(spatial_coords(&runtime, [a1, a2, a3])?, radius, stride)
        })
        .unwrap_or_else(|e| spatial_failure(e, start, end));
    let (first, second) = if op == 1 {
        (spatial_words(rows.counts), spatial_words(rows.indices))
    } else {
        (spatial_words(rows.indices), spatial_words(rows.distances))
    };
    let record = new_array(3, 2, 0);
    invar_object_set_v1(record, 0, first as i64, 1);
    invar_object_set_v1(record, 1, second as i64, 1);
    record
}
