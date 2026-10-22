//! Immutable version 1 finite-state and wrapping-u32 matrix summaries.
//!
//! Composition means first self, then next. These exact finite models do not
//! prove source-state independence, effects, typed extraction or launch legality.
#![forbid(unsafe_code)]

use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Data {
    Table([u8; 16]),
    Matrix([u32; 16]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SummaryFamily {
    dimension: u8,
    data: Data,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    TableSize,
    TableEntry { index: usize, value: u8 },
    MatrixDimension,
    MatrixCount,
    FamilyMismatch,
    ShapeMismatch,
    StateShape,
    StateValue,
    StepLimit,
    ByteLimit,
    CompositionLimit,
    Allocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Admission limits for the ordered scan, with deterministic logical costs.
/// These limits are not wall-clock or process-residency measurements.
pub struct Budget {
    pub max_steps: usize,
    pub max_bytes: usize,
    pub max_composition_ops: u64,
}

impl SummaryFamily {
    pub fn table(entries: &[u8]) -> Result<Self, Error> {
        if entries.is_empty() || entries.len() > 16 {
            return Err(Error::TableSize);
        }
        for (index, &value) in entries.iter().enumerate() {
            if usize::from(value) >= entries.len() {
                return Err(Error::TableEntry { index, value });
            }
        }
        let mut table = [0; 16];
        table[..entries.len()].copy_from_slice(entries);
        Ok(Self {
            dimension: entries.len() as u8,
            data: Data::Table(table),
        })
    }

    /// Dense row-major matrix acting on a column vector, modulo 2^32.
    pub fn matrix(dimension: u8, coefficients: &[u32]) -> Result<Self, Error> {
        if !(1..=4).contains(&dimension) {
            return Err(Error::MatrixDimension);
        }
        let d = usize::from(dimension);
        if coefficients.len() != d * d {
            return Err(Error::MatrixCount);
        }
        let mut matrix = [0; 16];
        matrix[..coefficients.len()].copy_from_slice(coefficients);
        Ok(Self {
            dimension,
            data: Data::Matrix(matrix),
        })
    }

    pub fn dimension(&self) -> u8 {
        self.dimension
    }
    pub fn version(&self) -> u32 {
        1
    }
    pub fn family_id(&self) -> &'static str {
        match self.data {
            Data::Table(_) => "finite-state-table-v1",
            Data::Matrix(_) => "u32-matrix-mod2p32-v1",
        }
    }

    pub fn identity(&self) -> Self {
        let d = usize::from(self.dimension);
        let data = match self.data {
            Data::Table(_) => {
                let mut t = [0; 16];
                for (i, v) in t[..d].iter_mut().enumerate() {
                    *v = i as u8;
                }
                Data::Table(t)
            }
            Data::Matrix(_) => {
                let mut m = [0; 16];
                for i in 0..d {
                    m[i * d + i] = 1;
                }
                Data::Matrix(m)
            }
        };
        Self {
            dimension: self.dimension,
            data,
        }
    }

    fn compatible(&self, next: &Self) -> Result<(), Error> {
        if std::mem::discriminant(&self.data) != std::mem::discriminant(&next.data) {
            return Err(Error::FamilyMismatch);
        }
        if self.dimension != next.dimension {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    pub fn then(&self, next: &Self) -> Result<Self, Error> {
        self.compatible(next)?;
        let d = usize::from(self.dimension);
        let data = match (&self.data, &next.data) {
            (Data::Table(first), Data::Table(second)) => {
                let mut result = [0; 16];
                for i in 0..d {
                    result[i] = second[usize::from(first[i])];
                }
                Data::Table(result)
            }
            (Data::Matrix(first), Data::Matrix(second)) => {
                let mut result = [0u32; 16];
                for row in 0..d {
                    for col in 0..d {
                        let mut value = 0u32;
                        for k in 0..d {
                            value = value
                                .wrapping_add(second[row * d + k].wrapping_mul(first[k * d + col]));
                        }
                        result[row * d + col] = value;
                    }
                }
                Data::Matrix(result)
            }
            _ => return Err(Error::FamilyMismatch),
        };
        Ok(Self {
            dimension: self.dimension,
            data,
        })
    }

    fn check_state(&self, state: &[u32]) -> Result<(), Error> {
        match self.data {
            Data::Table(_) => {
                if state.len() != 1 {
                    return Err(Error::StateShape);
                }
                if state[0] >= u32::from(self.dimension) {
                    return Err(Error::StateValue);
                }
            }
            Data::Matrix(_) if state.len() != usize::from(self.dimension) => {
                return Err(Error::StateShape);
            }
            Data::Matrix(_) => {}
        }
        Ok(())
    }

    pub fn apply(&self, state: &[u32]) -> Result<Vec<u32>, Error> {
        self.check_state(state)?;
        let d = usize::from(self.dimension);
        let width = match self.data {
            Data::Table(_) => 1,
            Data::Matrix(_) => d,
        };
        let mut output = Vec::new();
        output
            .try_reserve_exact(width)
            .map_err(|_| Error::Allocation)?;
        match &self.data {
            Data::Table(table) => output.push(u32::from(table[state[0] as usize])),
            Data::Matrix(matrix) => {
                for row in 0..d {
                    let mut value = 0u32;
                    for k in 0..d {
                        value = value.wrapping_add(matrix[row * d + k].wrapping_mul(state[k]));
                    }
                    output.push(value);
                }
            }
        }
        Ok(output)
    }

    fn work(&self) -> (u64, u64, usize) {
        let d = u64::from(self.dimension);
        // Active element reads/writes plus wrapping arithmetic: table composition
        // reads two entries and writes one; each matrix dot-product term reads
        // two coefficients, multiplies and adds, then writes its result once.
        // This conservatively counts these primitives even if optimized away.
        // Zero initialization, copies, loop/index bookkeeping and allocations
        // are excluded; this is not a machine-instruction or runtime estimate.
        match self.data {
            Data::Table(_) => (3 * d, 2, 1),
            Data::Matrix(_) => (
                4 * d * d * d + d * d,
                4 * d * d + d,
                usize::from(self.dimension),
            ),
        }
    }
}

/// Ordered Blelloch up/down scan with inclusive actions on the original seed.
/// For n>0, P=next_power_of_two(n), combines=2*(P-1)+n. Work budget charges
/// combines*family_composition_work + n*family_action_work. For tables these
/// costs are 3*K and 2; for matrices they are 4*d^3+d^2 and 4*d^2+d.
/// Byte admission charges (P+2)*size_of(SummaryFamily) plus
/// n*(size_of(Vec<u32>)+state_width*size_of(u32)): requested tree/output heap
/// payload and two logical summary scratch values. Allocator overhead, other
/// stack storage and caller-owned inputs are excluded.
/// All admission precedes allocation/scan; failures publish no observations.
pub fn prefixes(
    transitions: &[SummaryFamily],
    seed: &[u32],
    budget: Budget,
) -> Result<Vec<Vec<u32>>, Error> {
    let n = transitions.len();
    if n > budget.max_steps {
        return Err(Error::StepLimit);
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let first = &transitions[0];
    first.check_state(seed)?;
    let p = n.checked_next_power_of_two().ok_or(Error::ByteLimit)?;
    let (composition, action, width) = first.work();
    let combines = p
        .checked_sub(1)
        .and_then(|x| x.checked_mul(2))
        .and_then(|x| x.checked_add(n))
        .ok_or(Error::CompositionLimit)?;
    let work = u64::try_from(combines)
        .ok()
        .and_then(|x| x.checked_mul(composition))
        .and_then(|x| {
            u64::try_from(n)
                .ok()
                .and_then(|n| n.checked_mul(action))
                .and_then(|a| x.checked_add(a))
        })
        .ok_or(Error::CompositionLimit)?;
    if work > budget.max_composition_ops {
        return Err(Error::CompositionLimit);
    }
    let tree_bytes = p
        .checked_add(2)
        .and_then(|x| x.checked_mul(size_of::<SummaryFamily>()));
    let output_bytes = width
        .checked_mul(size_of::<u32>())
        .and_then(|x| x.checked_add(size_of::<Vec<u32>>()))
        .and_then(|x| x.checked_mul(n));
    let bytes = tree_bytes
        .and_then(|t| output_bytes.and_then(|o| t.checked_add(o)))
        .ok_or(Error::ByteLimit)?;
    if bytes > budget.max_bytes {
        return Err(Error::ByteLimit);
    }
    for transition in transitions {
        first.compatible(transition)?;
    }
    let identity = first.identity();
    let mut tree = Vec::new();
    tree.try_reserve_exact(p).map_err(|_| Error::Allocation)?;
    tree.resize(p, identity);
    tree[..n].copy_from_slice(transitions);
    let mut stride = 1;
    while stride < p {
        // p is a representable power of two and stride<=p/2, so 2*stride fits.
        for base in (0..p).step_by(2 * stride) {
            let left = base + stride - 1;
            let right = base + 2 * stride - 1;
            tree[right] = tree[left].then(&tree[right])?;
        }
        stride *= 2;
    }
    tree[p - 1] = identity;
    stride = p / 2;
    while stride > 0 {
        for base in (0..p).step_by(2 * stride) {
            let left = base + stride - 1;
            let right = base + 2 * stride - 1;
            let total = tree[left];
            tree[left] = tree[right];
            tree[right] = tree[right].then(&total)?;
        }
        stride /= 2;
    }
    let mut output = Vec::new();
    output.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    for (prefix, transition) in tree.iter().zip(transitions) {
        output.push(prefix.then(transition)?.apply(seed)?);
    }
    Ok(output)
}
