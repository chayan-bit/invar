// Embedded in the standalone Rust host, with the existing numeric contract and
// radix/cell helpers. No source evaluator, model, or compiler runs at execution.
use crate::{
    numeric_contract::{Comparison, F32Bits as F},
    sort_cpu, spatial_cell,
};
use std::ops::Range;

#[derive(Debug)]
pub enum Error {
    Shape,
    Overflow,
    Allocation,
    Internal,
}
fn reserve<T>(n: usize) -> Result<Vec<T>, Error> {
    sort_cpu::reserve(n).map_err(|_| Error::Allocation)
}
fn finite(p: [F; 3]) -> bool {
    p.iter().all(|v| !v.is_nan() && !v.is_infinite())
}
#[derive(Clone)]
struct Point {
    coords: [F; 3],
    key: [i32; 3],
    index: u32,
}
pub struct Grid {
    points: Vec<Point>,
    cells: Vec<Range<usize>>,
    pub excluded: u32,
}
impl Grid {
    pub fn storage_elements(&self) -> usize {
        7 * self.points.len() + 4 * self.cells.len() + 1
    }
    pub fn build(coords: [&[u32]; 3], cell: u32) -> Result<Self, Error> {
        let n = shape(coords)?;
        if !(1..0x7f800000).contains(&cell) {
            return Err(Error::Overflow);
        }
        let mut points = reserve(n)?;
        let mut excluded = 0;
        for i in 0..n {
            let p = std::array::from_fn(|a| F::from_bits(coords[a][i]));
            if !finite(p) {
                excluded += 1;
                continue;
            }
            let mut key = [0; 3];
            for axis in 0..3 {
                key[axis] = spatial_cell::index(p[axis].to_bits(), cell).ok_or(Error::Overflow)?;
            }
            points.push(Point {
                coords: p,
                key,
                index: u32::try_from(i).map_err(|_| Error::Overflow)?,
            });
        }
        for axis in (0..3).rev() {
            let mut keys = reserve(points.len())?;
            keys.extend(points.iter().map(|p| p.key[axis] as u32));
            let order =
                sort_cpu::order_with(&keys, sort_cpu::KeyKind::I32, None, sort_cpu::serial_pairs)
                    .map_err(|_| Error::Allocation)?;
            let mut sorted = reserve(points.len())?;
            sorted.extend(order.iter().map(|&i| points[i as usize].clone()));
            points = sorted;
        }
        let mut cells: Vec<Range<usize>> = reserve(points.len())?;
        for i in 0..points.len() {
            if i == 0 || points[i].key != points[i - 1].key {
                cells.push(i..i + 1);
            } else if let Some(last) = cells.last_mut() {
                last.end = i + 1;
            }
        }
        Ok(Self {
            points,
            cells,
            excluded,
        })
    }
    pub fn query(
        &self,
        coords: [&[u32]; 3],
        radius: Option<u32>,
        stride: u32,
    ) -> Result<Rows, Error> {
        let n = shape(coords)?;
        let slots = u32::try_from(n)
            .ok()
            .and_then(|n| n.checked_mul(stride))
            .ok_or(Error::Overflow)? as usize;
        let mut counts = reserve(n)?;
        let mut indices = reserve(slots)?;
        let mut distances = reserve(if radius.is_none() { slots } else { 0 })?;
        let radius = radius.map(F::from_bits);
        let allowed = radius.is_none_or(|r| !r.is_nan() && !r.compare(Comparison::Lt, F::POS_ZERO));
        let threshold = radius.map(|r| r.mul(r));
        let mut row = reserve(self.points.len())?;
        for q in 0..n {
            row.clear();
            let query = std::array::from_fn(|a| F::from_bits(coords[a][q]));
            if allowed && finite(query) && (radius.is_some() || stride != 0) {
                // ponytail: exact occupied-cell scan, replace with conservative
                // strict-f32 bounds before qualifying production performance.
                for cell in &self.cells {
                    for point in &self.points[cell.clone()] {
                        let dx = point.coords[0].sub(query[0]);
                        let dy = point.coords[1].sub(query[1]);
                        let dz = point.coords[2].sub(query[2]);
                        let d2 = dx.mul(dx).add(dy.mul(dy)).add(dz.mul(dz));
                        if threshold.is_none_or(|r| d2.compare(Comparison::Le, r)) {
                            row.push((d2, point.index));
                        }
                    }
                }
            }
            row.sort_unstable_by_key(|(d2, index)| (d2.to_bits(), *index));
            counts.push(u32::try_from(row.len()).map_err(|_| Error::Overflow)?);
            for slot in 0..stride as usize {
                let (d2, index) = row.get(slot).copied().unwrap_or((F::INFINITY, u32::MAX));
                indices.push(index);
                if radius.is_none() {
                    distances.push(d2.to_bits());
                }
            }
        }
        Ok(Rows {
            counts,
            indices,
            distances,
        })
    }
}
pub struct Rows {
    pub counts: Vec<u32>,
    pub indices: Vec<u32>,
    pub distances: Vec<u32>,
}
pub fn shape(coords: [&[u32]; 3]) -> Result<usize, Error> {
    let n = coords[0].len();
    if coords[1].len() != n || coords[2].len() != n {
        Err(Error::Shape)
    } else {
        Ok(n)
    }
}
