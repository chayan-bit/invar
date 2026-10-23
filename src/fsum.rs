//! Deterministic float reductions (`sum_tree`, `dot_tree`, `sum_exact`, `dot_exact`).
//!
//! This file is the fast engine implementation shared by MIR, `cpu-parallel`,
//! device finishing and, verbatim, the generated native host (`native.rs`
//! wraps it in `mod fsum`). It therefore uses only `core`/`std` and no crate
//! paths. The independent reference semantics live in `fsum_reference.rs`.
//!
//! Tree shape (normative, spec NB.1.6): leaves are left-to-right folds
//! of consecutive blocks of [`LEAF`] elements; node `(L, j)` exists iff
//! `j * 2^L < m` (leaf count `m`) and is the sum of its two children, or its
//! left child alone when the right one does not exist. Any aligned chunk of
//! `2^c` leaves can therefore be reduced independently, which is how every
//! engine parallelizes without changing a single bit.
//!
//! Exact sums: binary32 inputs are deposited, by exponent, into 64-bit integer
//! bins (Malcolm 1971), normalized to a two's-complement integer and rounded
//! once to nearest-even with integer arithmetic only.

/// Elements per leaf, summed left to right.
pub const LEAF: usize = 8;
/// The canonical NaN every intrinsic returns for a NaN result.
pub const CANONICAL_NAN: u32 = 0x7fc0_0000;
/// Exact-sum scale: a binary32 value is `significand * 2^(position - 149)`.
const SUM_SCALE: i32 = -149;
/// Exact-dot scale: a product is `significand * 2^(position - 298)`.
const DOT_SCALE: i32 = -298;
/// Bit positions of the dot accumulator: 506 + 24 split bits + headroom.
pub const DOT_BINS: usize = 532;
/// 32-bit digits needed for any u32-length sum or dot (608 bits).
pub const DIGITS: usize = 19;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    SumTree,
    DotTree,
    SumExact,
    DotExact,
}

impl Kind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "sum_tree" => Some(Self::SumTree),
            "dot_tree" => Some(Self::DotTree),
            "sum_exact" => Some(Self::SumExact),
            "dot_exact" => Some(Self::DotExact),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::SumTree => "sum_tree",
            Self::DotTree => "dot_tree",
            Self::SumExact => "sum_exact",
            Self::DotExact => "dot_exact",
        }
    }

    /// Stable code used by the native host ABI and device launches.
    pub fn code(self) -> u32 {
        match self {
            Self::SumTree => 1,
            Self::DotTree => 2,
            Self::SumExact => 3,
            Self::DotExact => 4,
        }
    }

    pub fn from_code(code: u32) -> Option<Self> {
        [Self::SumTree, Self::DotTree, Self::SumExact, Self::DotExact]
            .into_iter()
            .find(|kind| kind.code() == code)
    }

    pub fn is_dot(self) -> bool {
        matches!(self, Self::DotTree | Self::DotExact)
    }

    pub fn is_exact(self) -> bool {
        matches!(self, Self::SumExact | Self::DotExact)
    }
}

/// The result word of a finished reduction, with NaN canonicalized.
pub fn canonical(value: f32) -> u32 {
    if value.is_nan() {
        CANONICAL_NAN
    } else {
        value.to_bits()
    }
}

/// Number of leaves for `n` elements.
pub fn leaf_count(n: usize) -> usize {
    n.div_ceil(LEAF)
}

/// Root of the aligned pairwise tree over `count` items (items may be leaves
/// or the nodes of any common level). Empty input is `+0`.
pub fn fold_aligned(count: usize, mut item: impl FnMut(usize) -> f32) -> f32 {
    // (level, value); levels strictly decrease from bottom to top.
    let mut stack: [(u32, f32); 66] = [(0, 0.0); 66];
    let mut depth = 0;
    for index in 0..count {
        let mut level = 0;
        let mut value = item(index);
        while depth > 0 && stack[depth - 1].0 == level {
            depth -= 1;
            value += stack[depth].1;
            level += 1;
        }
        stack[depth] = (level, value);
        depth += 1;
    }
    // A missing right subtree carries its left sibling unchanged.
    let mut root = 0.0;
    for (position, &(_, value)) in stack[..depth].iter().enumerate().rev() {
        root = if position + 1 == depth {
            value
        } else {
            value + root
        };
    }
    root
}

/// One leaf: the left-to-right fold of its block, seeded by its first term.
pub fn leaf(n: usize, index: usize, term: &impl Fn(usize) -> f32) -> f32 {
    let start = index * LEAF;
    let end = (start + LEAF).min(n);
    let mut acc = term(start);
    for position in start + 1..end {
        acc += term(position);
    }
    acc
}

/// Level-`level` node `index` of the tree over `n` terms.
pub fn tree_node(n: usize, level: u32, index: usize, term: &impl Fn(usize) -> f32) -> f32 {
    let leaves = leaf_count(n);
    let first = index << level;
    let count = (leaves - first).min(1 << level);
    let mut block = [0.0f32; BLOCK];
    fold_aligned(count, |offset| {
        if offset % BLOCK == 0 {
            block = leaf_block(n, first + offset, term);
        }
        block[offset % BLOCK]
    })
}

/// Leaves formed together as independent chains, so they pipeline.
const BLOCK: usize = 8;

/// Leaves `first..first + BLOCK`; each still adds its terms in source order
/// from the first. Leaves past the end are `0` and never read.
fn leaf_block(n: usize, first: usize, term: &impl Fn(usize) -> f32) -> [f32; BLOCK] {
    let base = first * LEAF;
    if base + BLOCK * LEAF > n {
        let leaves = leaf_count(n);
        return std::array::from_fn(|j| {
            if first + j < leaves {
                leaf(n, first + j, term)
            } else {
                0.0
            }
        });
    }
    let mut acc: [f32; BLOCK] = std::array::from_fn(|j| term(base + j * LEAF));
    for k in 1..LEAF {
        for (j, value) in acc.iter_mut().enumerate() {
            *value += term(base + j * LEAF + k);
        }
    }
    acc
}

/// `sum_tree`/`dot_tree` over `n` terms, serially: the root node.
pub fn tree(n: usize, term: impl Fn(usize) -> f32) -> u32 {
    canonical(tree_node(n, chunk_level(leaf_count(n), 1), 0, &term))
}

pub fn sum_tree(xs: &[u32]) -> u32 {
    tree(xs.len(), |i| f32::from_bits(xs[i]))
}

/// Products round before the tree; there is no fused multiply-add.
pub fn dot_tree(a: &[u32], b: &[u32]) -> u32 {
    tree(a.len().min(b.len()), |i| {
        f32::from_bits(a[i]) * f32::from_bits(b[i])
    })
}

/// Special-value and signed-zero state of an exact reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Flags {
    pub nan: bool,
    pub pos_inf: bool,
    pub neg_inf: bool,
    /// A term other than `-0` (finite nonzero, `+0`, or nonfinite) was seen.
    pub not_neg_zero: bool,
}

impl Flags {
    /// Device word: bit 0 NaN, 1 +inf, 2 -inf, 3 not-negative-zero.
    pub fn from_word(word: u32) -> Self {
        Self {
            nan: word & 1 != 0,
            pos_inf: word & 2 != 0,
            neg_inf: word & 4 != 0,
            not_neg_zero: word & 8 != 0,
        }
    }

    pub fn merge(self, other: Self) -> Self {
        Self {
            nan: self.nan || other.nan,
            pos_inf: self.pos_inf || other.pos_inf,
            neg_inf: self.neg_inf || other.neg_inf,
            not_neg_zero: self.not_neg_zero || other.not_neg_zero,
        }
    }
}

/// Exponent-indexed exact accumulator. Each deposit adds less than `2^24` to
/// one bin, so `2^39` deposits cannot overflow; u32 lengths stay far below.
#[derive(Clone, Debug)]
pub struct Exact {
    pub bins: Vec<i64>,
    pub flags: Flags,
    pub terms: u64,
    dot: bool,
}

/// Binary32 fields: (negative, position, significand) of a finite word.
fn fields(word: u32) -> (bool, usize, u64) {
    let exponent = (word >> 23) & 0xff;
    let fraction = u64::from(word & 0x007f_ffff);
    if exponent == 0 {
        (word >> 31 == 1, 0, fraction)
    } else {
        (word >> 31 == 1, exponent as usize - 1, fraction | 1 << 23)
    }
}

fn is_nan(word: u32) -> bool {
    word & 0x7fff_ffff > 0x7f80_0000
}

fn is_inf(word: u32) -> bool {
    word & 0x7fff_ffff == 0x7f80_0000
}

fn is_zero(word: u32) -> bool {
    word & 0x7fff_ffff == 0
}

impl Exact {
    pub fn new(dot: bool) -> Self {
        Self {
            bins: vec![0; if dot { DOT_BINS } else { 256 }],
            flags: Flags::default(),
            terms: 0,
            dot,
        }
    }

    pub fn add(&mut self, word: u32) {
        self.terms += 1;
        if word != 0x8000_0000 {
            self.flags.not_neg_zero = true;
        }
        if is_nan(word) {
            self.flags.nan = true;
        } else if is_inf(word) {
            self.mark_inf(word >> 31 == 1);
        } else {
            let (negative, position, significand) = fields(word);
            let value = significand as i64;
            self.bins[position] += if negative { -value } else { value };
        }
    }

    pub fn add_product(&mut self, a: u32, b: u32) {
        self.terms += 1;
        let negative = (a ^ b) >> 31 == 1;
        let zero = is_zero(a) || is_zero(b);
        if !(zero && negative) {
            self.flags.not_neg_zero = true;
        }
        if is_nan(a) || is_nan(b) || (zero && (is_inf(a) || is_inf(b))) {
            self.flags.nan = true;
        } else if is_inf(a) || is_inf(b) {
            self.mark_inf(negative);
        } else if !zero {
            let (_, pa, sa) = fields(a);
            let (_, pb, sb) = fields(b);
            let product = sa * sb;
            let (high, low) = ((product >> 24) as i64, (product & 0xff_ffff) as i64);
            let position = pa + pb;
            if negative {
                self.bins[position + 24] -= high;
                self.bins[position] -= low;
            } else {
                self.bins[position + 24] += high;
                self.bins[position] += low;
            }
        }
    }

    fn mark_inf(&mut self, negative: bool) {
        if negative {
            self.flags.neg_inf = true;
        } else {
            self.flags.pos_inf = true;
        }
    }

    /// Order-independent merge of two partial accumulators.
    pub fn merge(&mut self, other: &Self) {
        for (bin, value) in self.bins.iter_mut().zip(&other.bins) {
            *bin += value;
        }
        self.flags = self.flags.merge(other.flags);
        self.terms += other.terms;
    }

    /// The correctly rounded binary32 result word.
    pub fn finish(&self) -> u32 {
        let scale = if self.dot { DOT_SCALE } else { SUM_SCALE };
        finish_exact(self.terms, self.flags, &self.bins, scale)
    }
}

pub fn sum_exact(xs: &[u32]) -> u32 {
    let mut exact = Exact::new(false);
    xs.iter().for_each(|&word| exact.add(word));
    exact.finish()
}

pub fn dot_exact(a: &[u32], b: &[u32]) -> u32 {
    let mut exact = Exact::new(true);
    a.iter().zip(b).for_each(|(&x, &y)| exact.add_product(x, y));
    exact.finish()
}

/// Special values first, then signed zero, then one rounding of the exact
/// integer `sum(bins[p] * 2^p) * 2^scale`.
pub fn finish_exact(terms: u64, flags: Flags, bins: &[i64], scale: i32) -> u32 {
    finish(terms, flags, scale, || normalize(bins))
}

/// Finish device exact partials: rows of `DIGITS` digit sums (digit `k`
/// weighs `2^(32k)`), each followed by its flag word.
pub fn finish_rows(rows: &[i64], terms: u64, dot: bool) -> u32 {
    let mut sums = [0i128; DIGITS];
    let mut flags = Flags::default();
    for row in rows.chunks_exact(DIGITS + 1) {
        for (sum, &digit) in sums.iter_mut().zip(row) {
            *sum += i128::from(digit);
        }
        flags = flags.merge(Flags::from_word(row[DIGITS] as u32));
    }
    let scale = if dot { DOT_SCALE } else { SUM_SCALE };
    finish(terms, flags, scale, || signed_digits(|index| sums[index]))
}

fn finish(
    terms: u64,
    flags: Flags,
    scale: i32,
    magnitude: impl FnOnce() -> (bool, [u32; DIGITS]),
) -> u32 {
    if terms == 0 {
        return 0;
    }
    if flags.nan || (flags.pos_inf && flags.neg_inf) {
        return CANONICAL_NAN;
    }
    if flags.pos_inf {
        return 0x7f80_0000;
    }
    if flags.neg_inf {
        return 0xff80_0000;
    }
    let (negative, magnitude) = magnitude();
    if magnitude.iter().all(|&digit| digit == 0) {
        return if flags.not_neg_zero { 0 } else { 0x8000_0000 };
    }
    round(&magnitude, negative, scale)
}

/// Two's-complement normalization of positional bins into sign and 32-bit
/// little-endian magnitude digits. Bin `p` weighs `2^p`.
pub fn normalize(bins: &[i64]) -> (bool, [u32; DIGITS]) {
    signed_digits(|index| {
        bins.iter()
            .skip(index * 32)
            .take(32)
            .enumerate()
            .map(|(offset, &value)| i128::from(value) << offset)
            .sum()
    })
}

/// Carry-propagate digit contributions (`digit(k)` weighs `2^(32k)`) into a
/// sign and magnitude. The total must fit in `DIGITS` digits.
fn signed_digits(digit: impl Fn(usize) -> i128) -> (bool, [u32; DIGITS]) {
    let mut digits = [0u32; DIGITS];
    let mut carry: i128 = 0;
    for (index, out) in digits.iter_mut().enumerate() {
        carry += digit(index);
        *out = (carry & 0xffff_ffff) as u32;
        carry >>= 32;
    }
    let negative = carry < 0;
    if negative {
        let mut borrow = true;
        for out in &mut digits {
            let (value, overflow) = (!*out).overflowing_add(u32::from(borrow));
            *out = value;
            borrow = borrow && overflow;
        }
    }
    (negative, digits)
}

fn bit(digits: &[u32], position: i64) -> bool {
    position >= 0
        && digits
            .get((position / 32) as usize)
            .is_some_and(|digit| digit >> (position % 32) & 1 == 1)
}

fn any_below(digits: &[u32], position: i64) -> bool {
    (0..position.max(0)).any(|index| bit(digits, index))
}

/// Round `magnitude * 2^scale` to nearest-even binary32 (nonzero magnitude).
pub fn round(magnitude: &[u32], negative: bool, scale: i32) -> u32 {
    let high = (0..magnitude.len() as i64 * 32)
        .rev()
        .find(|&position| bit(magnitude, position))
        .unwrap_or(0);
    let sign = if negative { 0x8000_0000 } else { 0 };
    // Drop `shift` low bits: 24 significant bits, or the subnormal quantum.
    let shift = (high - 23).max(i64::from(-149 - scale));
    let mut quotient: u64 = 0;
    for position in (shift.max(0)..=high).rev() {
        quotient = quotient << 1 | u64::from(bit(magnitude, position));
    }
    if shift < 0 {
        quotient <<= -shift;
    }
    let half = bit(magnitude, shift - 1);
    if half && (any_below(magnitude, shift - 1) || quotient & 1 == 1) {
        quotient += 1;
    }
    let mut exponent = shift + i64::from(scale);
    if quotient == 1 << 24 {
        quotient >>= 1;
        exponent += 1;
    }
    if quotient < 1 << 23 {
        return sign | quotient as u32;
    }
    let field = exponent + 150;
    if field >= 255 {
        return sign | 0x7f80_0000;
    }
    sign | (field as u32) << 23 | (quotient as u32 & 0x007f_ffff)
}

/// Level of the aligned chunk a parallel engine reduces independently, so
/// that at most `parts` chunks cover `leaves` leaves.
pub fn chunk_level(leaves: usize, parts: usize) -> u32 {
    let per_part = leaves.div_ceil(parts.max(1)).max(1);
    usize::BITS - (per_part - 1).leading_zeros()
}
