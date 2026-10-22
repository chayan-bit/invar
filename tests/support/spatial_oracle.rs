//! Independent bounded all-pairs oracle, not a grid/runtime implementation.
//! All distance arithmetic reuses the strict software binary32 oracle.
//! Error variants model ruled trap categories; Limit is only a fixture budget.

use invar::numeric_contract::{Comparison, F32Bits as F};
use num_bigint::BigInt;
use num_rational::BigRational;

pub const MAX_POINTS: usize = 1024;
const MAX_PAIRS: usize = 65_536;
const MAX_OUTPUT: usize = 65_536;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    IndexOutOfBounds,
    IntegerOverflow,
    Limit,
}

#[derive(Clone, Copy)]
pub struct Cloud<'a> {
    xs: &'a [F],
    ys: &'a [F],
    zs: &'a [F],
}

impl<'a> Cloud<'a> {
    pub fn new(xs: &'a [F], ys: &'a [F], zs: &'a [F]) -> Result<Self, Error> {
        if xs.len() != ys.len() || xs.len() != zs.len() {
            return Err(Error::IndexOutOfBounds);
        }
        if xs.len() > MAX_POINTS {
            return Err(Error::Limit);
        }
        Ok(Self { xs, ys, zs })
    }

    fn point(self, i: usize) -> [F; 3] {
        [self.xs[i], self.ys[i], self.zs[i]]
    }
}

pub fn squared_distance(point: [F; 3], query: [F; 3]) -> F {
    let dx = point[0].sub(query[0]);
    let dy = point[1].sub(query[1]);
    let dz = point[2].sub(query[2]);
    dx.mul(dx).add(dy.mul(dy)).add(dz.mul(dz))
}

pub fn output_slots(queries: u32, stride: u32) -> Result<u32, Error> {
    queries.checked_mul(stride).ok_or(Error::IntegerOverflow)
}

fn output_len(points: Cloud<'_>, queries: Cloud<'_>, stride: u32) -> Result<usize, Error> {
    let queries_len = u32::try_from(queries.xs.len()).map_err(|_| Error::IntegerOverflow)?;
    let length = output_slots(queries_len, stride)? as usize;
    let pairs = points
        .xs
        .len()
        .checked_mul(queries.xs.len())
        .ok_or(Error::Limit)?;
    if pairs > MAX_PAIRS || length > MAX_OUTPUT {
        return Err(Error::Limit);
    }
    Ok(length)
}

fn finite(point: [F; 3]) -> bool {
    point.iter().all(|v| !v.is_nan() && !v.is_infinite())
}

pub fn grid_excluded(points: Cloud<'_>) -> u32 {
    (0..points.xs.len())
        .filter(|&i| !finite(points.point(i)))
        .count() as u32
}

// Exact binary32 decomposition is independent of any rounded division kernel.
fn exact(value: F) -> BigRational {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    let significand = (bits & 0x7f_ffff) | if exponent == 0 { 0 } else { 1 << 23 };
    let power = if exponent == 0 {
        -149
    } else {
        exponent as i32 - 150
    };
    let numerator = BigInt::from(significand) * if bits >> 31 == 0 { 1 } else { -1 };
    if power >= 0 {
        BigRational::from_integer(numerator << power as usize)
    } else {
        BigRational::new(numerator, BigInt::from(1) << (-power) as usize)
    }
}

pub fn cell_index(coord: F, cell: F) -> Result<i32, Error> {
    if !finite([coord, cell, cell]) || !cell.compare(Comparison::Gt, F::POS_ZERO) {
        return Err(Error::IntegerOverflow);
    }
    i32::try_from((exact(coord) / exact(cell)).floor().to_integer())
        .map_err(|_| Error::IntegerOverflow)
}

pub fn validate_grid(points: Cloud<'_>, cell: F) -> Result<(), Error> {
    cell_index(F::POS_ZERO, cell)?;
    for i in 0..points.xs.len() {
        let point = points.point(i);
        if finite(point) {
            for coord in point {
                cell_index(coord, cell)?;
            }
        }
    }
    Ok(())
}

fn ranked(points: Cloud<'_>, query: [F; 3]) -> Vec<(u32, F)> {
    // ponytail: exhaustive distance scan plus row sort for small oracle fixtures;
    // future production paths must use the grid, never disguise this as a device path.
    if !finite(query) {
        return Vec::new();
    }
    let mut row: Vec<_> = (0..points.xs.len())
        .filter(|&i| finite(points.point(i)))
        .map(|i| (i as u32, squared_distance(points.point(i), query)))
        .collect();
    // Finite coordinates produce nonnegative squared distances, including +inf;
    // their raw bits have numerical order, and index makes the key unique.
    row.sort_unstable_by_key(|(index, d2)| (d2.to_bits(), *index));
    row
}

pub fn radius_neighbors(
    points: Cloud<'_>,
    queries: Cloud<'_>,
    radius: F,
    max_k: u32,
) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let length = output_len(points, queries, max_k)?;
    let radius2 = radius.mul(radius);
    let allowed = !radius.is_nan() && !radius.compare(Comparison::Lt, F::POS_ZERO);
    let mut counts = Vec::with_capacity(queries.xs.len());
    let mut indices = Vec::with_capacity(length);
    for q in 0..queries.xs.len() {
        let row = if allowed {
            ranked(points, queries.point(q))
        } else {
            Vec::new()
        };
        let count = row
            .iter()
            .filter(|(_, d2)| d2.compare(Comparison::Le, radius2))
            .count();
        counts.push(count as u32);
        indices.extend(
            row.into_iter()
                .take(count)
                .map(|(index, _)| index)
                .chain(std::iter::repeat(u32::MAX))
                .take(max_k as usize),
        );
    }
    Ok((counts, indices))
}

pub fn knn(points: Cloud<'_>, queries: Cloud<'_>, k: u32) -> Result<(Vec<u32>, Vec<F>), Error> {
    let length = output_len(points, queries, k)?;
    let mut indices = Vec::with_capacity(length);
    let mut distances = Vec::with_capacity(length);
    for q in 0..queries.xs.len() {
        let row = ranked(points, queries.point(q));
        for slot in 0..k as usize {
            let (index, d2) = row.get(slot).copied().unwrap_or((u32::MAX, F::INFINITY));
            indices.push(index);
            distances.push(d2);
        }
    }
    Ok((indices, distances))
}
