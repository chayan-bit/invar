//! Metered data production and receipt proof; no MIR hook or publication.
use super::{
    Access, Budget, Candidate, Contract, Cost, Error, Law, MAX_STAGING_BYTES, MAX_VISITS, reserve,
};
use crate::front::Span;
use crate::{check::CheckedProgram, general_net};
use sha2::{Digest, Sha256};
use std::fmt::{self, Write};

const METADATA_BYTES: usize = 4096;
// Two fixed receipt headers, including digest/law/extent/identity comparisons.
const HEADER_VISITS: usize = 512;

/// Captured only after the ordinary compiler verifies actual checked source
/// and net. Rechecks stream the current structures, not saved law descriptors.
#[derive(Debug)]
pub(super) struct Seal {
    source: [u8; 32],
    source_contract: [u8; 32],
    checked: [u8; 32],
    net: [u8; 32],
    proof: [u8; 32],
    /// Recognition reserve for `source_law` + `role_bytes`, sized at seal time
    /// (see `recognition`); charged at run time only after the checked hash
    /// matches the sealed program.
    recognition: usize,
    pub(super) visits: usize,
}
struct HashWriter<'a> {
    hash: Sha256,
    meter: &'a mut Meter,
}
impl Write for HashWriter<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.meter.visit(text.len()).map_err(|_| fmt::Error)?;
        self.hash.update(text.as_bytes());
        Ok(())
    }
}
fn fingerprint(value: &impl fmt::Debug, meter: &mut Meter) -> Result<[u8; 32], Error> {
    let mut writer = HashWriter {
        hash: Sha256::new(),
        meter,
    };
    write!(&mut writer, "{value:?}").map_err(|_| Error::Budget)?;
    Ok(writer.hash.finalize().into())
}
/// Name-lookup bound: comparing `entry` against every function name costs at
/// most each name's length plus one terminating comparison.
fn name_lookup(checked: &CheckedProgram) -> Result<usize, Error> {
    checked
        .program
        .functions
        .iter()
        .try_fold(0usize, |n, f| n.checked_add(f.name.len())?.checked_add(1))
        .ok_or(Error::Budget)
}
/// Recognition reserve: `lookups` name lookups plus the entry function's own
/// Debug stream, never more than the whole checked stream it replaces.
fn recognition_bound(
    lookups: usize,
    entry_stream: usize,
    checked_stream: usize,
) -> Result<usize, Error> {
    Ok(lookups
        .checked_add(entry_stream)
        .ok_or(Error::Budget)?
        .min(checked_stream))
}
/// Recognition (`source_law`, then `role_bytes`) performs two linear name
/// lookups over the functions, then inspects only the entry function's
/// parameters, statements, expressions and names; every inspected node and
/// string appears in that function's Debug stream. It never touches other
/// functions' bodies or the expression-type maps. Computed at seal time
/// (unmetered compiler work, like the rest of `Seal::new`).
fn recognition(
    checked: &CheckedProgram,
    entry: &str,
    checked_stream: usize,
) -> Result<usize, Error> {
    let lookups = name_lookup(checked)?.checked_mul(2).ok_or(Error::Budget)?;
    let f = checked
        .program
        .functions
        .iter()
        .find(|f| f.name == entry)
        .ok_or(Error::Binding)?;
    let mut entry_stream = Meter {
        used: 0,
        limit: usize::MAX,
    };
    fingerprint(f, &mut entry_stream)?;
    recognition_bound(lookups, entry_stream.used, checked_stream)
}
impl Seal {
    pub(super) fn new(
        source: &[u8],
        checked: &CheckedProgram,
        net: &general_net::Net,
        proof: &general_net::Proof,
        source_contract: [u8; 32],
        entry: &str,
    ) -> Result<Self, Error> {
        let mut meter = Meter {
            used: source.len(),
            limit: usize::MAX,
        };
        let before = meter.used;
        let checked_hash = fingerprint(checked, &mut meter)?;
        let checked_stream = meter.used.checked_sub(before).ok_or(Error::Budget)?;
        // Reserve role/name/shape recognition: two name lookups plus the entry
        // function's stream, bounded by the checked stream.
        let recognition = recognition(checked, entry, checked_stream)?;
        meter.visit(recognition)?;
        let net_hash = fingerprint(net, &mut meter)?;
        let proof_hash = fingerprint(proof, &mut meter)?;
        Ok(Self {
            source: Sha256::digest(source).into(),
            source_contract,
            checked: checked_hash,
            net: net_hash,
            proof: proof_hash,
            recognition,
            visits: meter.used,
        })
    }
}

impl Contract {
    fn verify_bounded(&self, meter: &mut Meter) -> Result<Law, Error> {
        meter.visit(self.source.len())?;
        let source_hash: [u8; 32] = Sha256::digest(&self.source).into();
        let checked_hash = fingerprint(&self.checked, meter)?;
        // The sealed recognition reserve is valid only for the sealed checked
        // program: refuse a changed program before charging it.
        if checked_hash != self.seal.checked {
            return Err(Error::Binding);
        }
        meter.visit(self.seal.recognition)?;
        let (span, law) = super::source_law(&self.checked, &self.entry)?;
        let net_hash = fingerprint(&self.net, meter)?;
        let proof_hash = fingerprint(&self.proof, meter)?;
        if source_hash != self.source_hash
            || source_hash != self.seal.source
            || self.source_contract != self.seal.source_contract
            || checked_hash != self.seal.checked
            || net_hash != self.seal.net
            || proof_hash != self.seal.proof
            || span != self.loop_span
            || law != self.law
            || law != Law::WrappingSum
            || self.runtime_names != role_bytes(&self.checked, &self.entry)?
        {
            return Err(Error::Binding);
        }
        Ok(law)
    }
}

#[derive(Debug)]
pub(super) struct Binding {
    source_hash: [u8; 32],
    source_contract: [u8; 32],
    span: Span,
    law: Law,
    revision: &'static str,
    values_identity: usize,
    indices_identity: usize,
    bins: u32,
    values: Vec<u32>,
    indices: Vec<u32>,
}

pub(crate) struct Meter {
    pub(crate) used: usize,
    pub(crate) limit: usize,
}
impl Meter {
    pub(crate) fn visit(&mut self, count: usize) -> Result<(), Error> {
        let next = self.used.checked_add(count).ok_or(Error::Budget)?;
        if next > self.limit {
            return Err(Error::Budget);
        }
        self.used = next;
        Ok(())
    }
}

impl Cost {
    /// Data-only planning reserve. This excludes source/net/proof streams and
    /// is not an admission receipt; an admitted candidate's cost includes them.
    pub fn bounded(a: usize, i: usize, n: usize, budget: Budget) -> Result<Self, Error> {
        let h = if i <= 1 {
            0
        } else {
            (usize::BITS - (i - 1).leading_zeros()) as usize
        };
        let bytes = a
            .checked_add(i)
            .and_then(|v| v.checked_add(n))
            .and_then(|v| v.checked_mul(size_of::<u32>()))
            .and_then(|v| v.checked_add(i.checked_mul(2 * size_of::<Access>())?))
            .and_then(|v| v.checked_add(n.checked_mul(size_of::<u64>())?))
            .and_then(|v| v.checked_add(i))
            .and_then(|v| v.checked_add(METADATA_BYTES))
            .ok_or(Error::Budget)?;
        let visits = a
            .checked_mul(2)
            .and_then(|v| v.checked_add(n.checked_mul(8)?))
            .and_then(|v| v.checked_add(i.checked_mul(28)?))
            .and_then(|v| v.checked_add(i.checked_mul(h)?.checked_mul(16)?))
            .and_then(|v| v.checked_add(HEADER_VISITS))
            .ok_or(Error::Budget)?;
        if bytes > budget.bytes.min(MAX_STAGING_BYTES) || visits > budget.visits.min(MAX_VISITS) {
            return Err(Error::Budget);
        }
        Ok(Self { bytes, visits })
    }
}

fn copy_words(words: &[u32], meter: &mut Meter) -> Result<Vec<u32>, Error> {
    let mut result = Vec::new();
    reserve(&mut result, words.len())?;
    for &word in words {
        meter.visit(1)?;
        result.push(word);
    }
    Ok(result)
}
fn greater(a: &[Access], left: usize, right: usize, meter: &mut Meter) -> Result<bool, Error> {
    meter.visit(2)?;
    Ok(a.get(left).ok_or(Error::Receipt)?.bin > a.get(right).ok_or(Error::Receipt)?.bin)
}
fn swap(a: &mut [Access], left: usize, right: usize, meter: &mut Meter) -> Result<(), Error> {
    meter.visit(4)?;
    let l = *a.get(left).ok_or(Error::Receipt)?;
    let r = *a.get(right).ok_or(Error::Receipt)?;
    *a.get_mut(left).ok_or(Error::Receipt)? = r;
    *a.get_mut(right).ok_or(Error::Receipt)? = l;
    Ok(())
}
fn sift(a: &mut [Access], mut root: usize, end: usize, meter: &mut Meter) -> Result<(), Error> {
    if end > a.len() {
        return Err(Error::Receipt);
    }
    loop {
        let left = root
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or(Error::Budget)?;
        if left >= end {
            return Ok(());
        }
        let right = left.checked_add(1).ok_or(Error::Budget)?;
        let child = if right < end && greater(a, right, left, meter)? {
            right
        } else {
            left
        };
        if !greater(a, child, root, meter)? {
            return Ok(());
        }
        swap(a, root, child, meter)?;
        root = child;
    }
}
fn heap_sort(a: &mut [Access], meter: &mut Meter) -> Result<(), Error> {
    for root in (0..a.len() / 2).rev() {
        sift(a, root, a.len(), meter)?;
    }
    for end in (1..a.len()).rev() {
        swap(a, 0, end, meter)?;
        sift(a, 0, end, meter)?;
    }
    Ok(())
}

// The historical larger-domain source oracle shares this checked sort, but
// remains outside the smaller optimized producer's payload/receipt budget.
pub(super) fn sort_observation(a: &mut [Access]) -> Result<usize, Error> {
    let h = if a.len() <= 1 {
        0
    } else {
        (usize::BITS - (a.len() - 1).leading_zeros()) as usize
    };
    let limit = a
        .len()
        .checked_mul(h)
        .and_then(|v| v.checked_mul(16))
        .and_then(|v| v.checked_add(a.len().checked_mul(4)?))
        .ok_or(Error::Budget)?;
    let mut meter = Meter { used: 0, limit };
    heap_sort(a, &mut meter)?;
    Ok(meter.used)
}

pub(super) fn role_bytes(checked: &CheckedProgram, entry: &str) -> Result<usize, Error> {
    let f = checked
        .program
        .functions
        .iter()
        .find(|f| f.name == entry)
        .ok_or(Error::Binding)?;
    let Some(first) = f.body.statements.first() else {
        return Err(Error::Binding);
    };
    let crate::front::StmtKind::Binding { name, .. } = &first.kind else {
        return Err(Error::Binding);
    };
    f.params.iter().try_fold(
        f.name.len().checked_add(name.len()).ok_or(Error::Budget)?,
        |n, p| n.checked_add(p.name.len()).ok_or(Error::Budget),
    )
}

/// Family-private snapshots, guarded before entry. No public constructor or
/// publication authority; runtime pins the verified immutable source context.
pub(crate) struct RuntimePrepared {
    binding: Binding,
}
pub(crate) struct RuntimeData {
    binding: Binding,
    pub(crate) candidate: Candidate,
    pub(crate) proof_visits: usize,
    pub(crate) produced_visits: usize,
}
impl Contract {
    pub(crate) fn runtime_header(&self) -> Result<usize, Error> {
        self.runtime_names
            .checked_mul(8)
            .and_then(|v| v.checked_add(256))
            .ok_or(Error::Budget)
    }
    pub(crate) fn runtime_publish_header(&self) -> Result<usize, Error> {
        self.runtime_names
            .checked_mul(4)
            .and_then(|v| v.checked_add(256))
            .ok_or(Error::Budget)
    }
    pub(crate) fn runtime_cost(
        &self,
        a: usize,
        i: usize,
        n: usize,
        budget: Budget,
    ) -> Result<Cost, Error> {
        if size_of::<super::host::Payload>()
            + size_of::<super::host::Ledger>()
            + size_of::<super::host::Completion>()
            + size_of::<Meter>()
            + 64 * size_of::<&str>()
            + 512
            > METADATA_BYTES
        {
            return Err(Error::Budget);
        }
        let base = Cost::bounded(a, i, n, Budget::default())?;
        let h = if i <= 1 {
            0
        } else {
            (usize::BITS - (i - 1).leading_zeros()) as usize
        };
        // All proof/preparation, generation, independent receipt, current input,
        // fresh output checks and full N-word publication. 1024 covers fixed
        // headers, identity/loan/shape/fuel/condition checks, metadata and
        // allocation attempts. 12*runtime_names covers all byte-dependent role
        // lookups (8 in preparation, 4 in publication). The 32769 minimum makes retained attempt count
        // at most one under the stronger 65536-visit run-wide ledger.
        let visits = self
            .seal
            .visits
            .checked_add(1024)
            .and_then(|v| v.checked_add(self.runtime_names.checked_mul(12)?))
            .and_then(|v| v.checked_add(a.checked_mul(2)?))
            .and_then(|v| v.checked_add(n.checked_mul(7)?))
            .and_then(|v| v.checked_add(i.checked_mul(34)?))
            .and_then(|v| v.checked_add(i.checked_mul(h)?.checked_mul(8)?))
            .ok_or(Error::Budget)?
            .max(32769);
        if base.bytes > budget.bytes.min(MAX_STAGING_BYTES)
            || visits > budget.visits.min(MAX_VISITS)
        {
            return Err(Error::Budget);
        }
        Ok(Cost {
            bytes: base.bytes,
            visits,
        })
    }
    pub(crate) fn runtime_prepare(
        &self,
        values: &[u32],
        indices: &[u32],
        bins: u32,
        meter: &mut Meter,
    ) -> Result<RuntimePrepared, Error> {
        let before = meter.used;
        let law = self.verify_bounded(meter)?;
        if meter.used.checked_sub(before) != Some(self.seal.visits) {
            return Err(Error::Binding);
        }
        let n = usize::try_from(bins).map_err(|_| Error::Budget)?;
        for (iteration, &bin) in indices.iter().enumerate() {
            meter.visit(1)?;
            if iteration >= values.len() || usize::try_from(bin).map_err(|_| Error::Budget)? >= n {
                return Err(Error::Bounds { iteration });
            }
        }
        Ok(RuntimePrepared {
            binding: Binding {
                source_hash: self.source_hash,
                source_contract: self.source_contract,
                span: self.loop_span,
                law,
                revision: law.revision(),
                values_identity: values.as_ptr() as usize,
                indices_identity: indices.as_ptr() as usize,
                bins,
                values: copy_words(values, meter)?,
                indices: copy_words(indices, meter)?,
            },
        })
    }
    pub(crate) fn runtime_produce(
        &self,
        prepared: RuntimePrepared,
        meter: &mut Meter,
    ) -> Result<RuntimeData, Error> {
        meter.visit(128)?;
        let binding = prepared.binding;
        if binding.law != Law::WrappingSum
            || binding.source_hash != self.source_hash
            || binding.source_contract != self.source_contract
        {
            return Err(Error::Binding);
        }
        let n = usize::try_from(binding.bins).map_err(|_| Error::Budget)?;
        let mut source_accesses = Vec::new();
        let mut grouped_accesses = Vec::new();
        reserve(&mut source_accesses, binding.indices.len())?;
        reserve(&mut grouped_accesses, binding.indices.len())?;
        for (iteration, &bin) in binding.indices.iter().enumerate() {
            meter.visit(2)?;
            source_accesses.push(Access {
                iteration,
                bin: usize::try_from(bin).map_err(|_| Error::Receipt)?,
            });
        }
        for &access in &source_accesses {
            meter.visit(2)?;
            grouped_accesses.push(access);
        }
        let before = meter.used;
        heap_sort(&mut grouped_accesses, meter)?;
        let sort_visits = meter.used.checked_sub(before).ok_or(Error::Budget)?;
        let mut output = Vec::new();
        reserve(&mut output, n)?;
        for _ in 0..n {
            meter.visit(1)?;
            output.push(binding.law.identity());
        }
        for access in &grouped_accesses {
            meter.visit(4)?;
            let weight = *binding.values.get(access.iteration).ok_or(Error::Receipt)?;
            let slot = output.get_mut(access.bin).ok_or(Error::Receipt)?;
            *slot = binding.law.combine(*slot, weight);
        }
        Ok(RuntimeData {
            binding,
            candidate: Candidate {
                output,
                source_accesses,
                grouped_accesses,
                sort_visits,
            },
            proof_visits: self.seal.visits,
            produced_visits: meter.used,
        })
    }
    pub(crate) fn runtime_validate(
        &self,
        data: &RuntimeData,
        values: &[u32],
        indices: &[u32],
        bins: u32,
        meter: &mut Meter,
    ) -> Result<(), Error> {
        meter.visit(256)?;
        let b = &data.binding;
        let c = &data.candidate;
        let n = usize::try_from(bins).map_err(|_| Error::Receipt)?;
        if b.source_hash != self.source_hash
            || b.source_contract != self.source_contract
            || b.span != self.loop_span
            || b.law != self.law
            || self.law != Law::WrappingSum
            || b.revision != self.law.revision()
            || b.bins != bins
            || b.values_identity != values.as_ptr() as usize
            || b.indices_identity != indices.as_ptr() as usize
            || values.len() != b.values.len()
            || indices.len() != b.indices.len()
            || c.output.len() != n
            || c.source_accesses.len() != indices.len()
            || c.grouped_accesses.len() != indices.len()
            || data.proof_visits != self.seal.visits
            || data.produced_visits
                != meter
                    .used
                    .checked_sub(
                        512usize
                            .checked_add(self.runtime_names.checked_mul(4).ok_or(Error::Receipt)?)
                            .ok_or(Error::Receipt)?,
                    )
                    .ok_or(Error::Receipt)?
        {
            return Err(Error::Receipt);
        }
        let h = if indices.len() <= 1 {
            0
        } else {
            (usize::BITS - (indices.len() - 1).leading_zeros()) as usize
        };
        let sort_bound = indices
            .len()
            .checked_mul(h)
            .and_then(|v| v.checked_mul(8))
            .and_then(|v| v.checked_add(indices.len().checked_mul(12)?))
            .ok_or(Error::Receipt)?;
        let produced = 384usize
            .checked_add(self.runtime_names.checked_mul(8).ok_or(Error::Receipt)?)
            .and_then(|v| v.checked_add(self.seal.visits))
            .and_then(|v| v.checked_add(values.len()))
            .and_then(|v| v.checked_add(indices.len().checked_mul(10)?))
            .and_then(|v| v.checked_add(n.checked_mul(2)?))
            .and_then(|v| v.checked_add(c.sort_visits))
            .ok_or(Error::Receipt)?;
        if c.sort_visits > sort_bound || data.produced_visits != produced {
            return Err(Error::Receipt);
        }
        for (current, saved) in values.iter().zip(&b.values) {
            meter.visit(1)?;
            if current != saved {
                return Err(Error::Receipt);
            }
        }
        for (current, saved) in indices.iter().zip(&b.indices) {
            meter.visit(1)?;
            if current != saved {
                return Err(Error::Receipt);
            }
        }
        let mut sums = Vec::<u64>::new();
        let mut seen = Vec::<u8>::new();
        reserve(&mut sums, n)?;
        reserve(&mut seen, indices.len())?;
        for _ in 0..n {
            meter.visit(1)?;
            sums.push(0);
        }
        for _ in 0..indices.len() {
            meter.visit(1)?;
            seen.push(0);
        }
        for (ordinal, access) in c.source_accesses.iter().enumerate() {
            meter.visit(5)?;
            let bin = usize::try_from(*b.indices.get(ordinal).ok_or(Error::Receipt)?)
                .map_err(|_| Error::Receipt)?;
            if access.iteration != ordinal || access.bin != bin {
                return Err(Error::Receipt);
            }
            let weight = *b.values.get(ordinal).ok_or(Error::Receipt)?;
            let sum = sums.get_mut(bin).ok_or(Error::Receipt)?;
            *sum = sum.checked_add(u64::from(weight)).ok_or(Error::Receipt)?;
        }
        let mut previous = None;
        for access in &c.grouped_accesses {
            meter.visit(4)?;
            if previous.is_some_and(|bin| bin > access.bin) {
                return Err(Error::Receipt);
            }
            previous = Some(access.bin);
            let bin = usize::try_from(*b.indices.get(access.iteration).ok_or(Error::Receipt)?)
                .map_err(|_| Error::Receipt)?;
            let mark = seen.get_mut(access.iteration).ok_or(Error::Receipt)?;
            if *mark != 0 || access.bin != bin || bin >= n {
                return Err(Error::Receipt);
            }
            *mark = 1;
        }
        for mark in seen {
            meter.visit(1)?;
            if mark != 1 {
                return Err(Error::Receipt);
            }
        }
        for (&actual, sum) in c.output.iter().zip(sums) {
            meter.visit(2)?;
            if u64::from(actual) != sum % (1u64 << 32) {
                return Err(Error::Receipt);
            }
        }
        Ok(())
    }
}
#[cfg(test)]
impl RuntimeData {
    pub(crate) fn corrupt(&mut self, fault: usize) {
        match fault {
            0 => self.binding.law = Law::WrappingProduct,
            1 => self.binding.revision = "wrong",
            2 => self.binding.source_hash[0] ^= 1,
            3 => self.binding.source_contract[0] ^= 1,
            4 => self.binding.bins += 1,
            5 => self.binding.span.end -= 1,
            6 => self.binding.values_identity ^= 1,
            7 => self.binding.indices_identity ^= 1,
            8 => self.binding.values[4] ^= 1,
            9 => self.binding.indices[0] ^= 1,
            10 => self.candidate.source_accesses[0].iteration = 1,
            11 => self.candidate.source_accesses[0].bin ^= 1,
            12 => self.candidate.grouped_accesses[0] = self.candidate.grouped_accesses[1],
            13 => {
                self.candidate.grouped_accesses.pop();
            }
            14 => self.candidate.grouped_accesses.swap(0, 3),
            15 => self.candidate.output[0] ^= 1,
            16 => self.candidate.output[6] = 1,
            17 => {
                self.candidate.output.pop();
            }
            18 => self.candidate.sort_visits += 1,
            19 => self.proof_visits += 1,
            _ => self.produced_visits += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, front, general_net, laws};
    use sha2::Digest;
    const SOURCE: &str = "fn hist(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = wrapping_add(out[ix[i]],a[i]); } return out; }";
    fn contract() -> Contract {
        let checked = check::check(front::parse(SOURCE.as_bytes()).unwrap(), false).unwrap();
        Contract::from_checked(SOURCE.as_bytes(), &checked, "hist").unwrap()
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_bounded_stale_law_source_net() {
        for fault in 0..8 {
            let mut c = contract();
            match fault {
                0 => c.law = Law::WrappingProduct,
                1 => c.source_hash[0] ^= 1,
                2 => c.source_contract[0] ^= 1,
                3 => c.loop_span.end -= 1,
                4 => c.entry = "missing".into(),
                5 => c.source.push(b' '),
                6 => {
                    c.source = SOURCE.replace("wrapping_add", "wrapping_mul").into_bytes();
                    c.source_hash = sha2::Sha256::digest(&c.source).into();
                }
                _ => {
                    let changed = SOURCE.replace("wrapping_add", "wrapping_mul");
                    let p = check::check(front::parse(changed.as_bytes()).unwrap(), false).unwrap();
                    c.net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
                    c.proof = c.net.verify().unwrap();
                }
            }
            assert!(c.verified_mir().is_err(), "fault={fault}");
            assert!(
                c.runtime_prepare(&[1], &[0], 1, &mut unmetered()).is_err(),
                "fault={fault}"
            );
        }
        assert_eq!(
            laws::Law::WrappingProduct.revision(),
            laws::Law::WrappingSum.revision()
        );
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_bounded_current_checked_net_proof() {
        for fault in 0..6 {
            let mut c = contract();
            match fault {
                0 => {
                    // Equal debug width preserves every non-hash cost guard.
                    let (_, ty) = c
                        .checked
                        .expression_types
                        .iter_mut()
                        .find(|(_, ty)| *ty == front::Type::U32)
                        .unwrap();
                    *ty = front::Type::I32;
                }
                1 => c.net.agents[0].next.position += 1,
                2 => {
                    // A different proof with exactly the sealed Debug width, so
                    // every non-hash guard holds and only the proof hash differs.
                    // (Searching revisions of another program's net for an equal
                    // decimal width of its random-order fingerprint could miss.)
                    let forged = c.proof.with_flipped_graph_fingerprint();
                    assert_eq!(stream(&forged), stream(&c.proof));
                    assert_ne!(stream(&forged), 0);
                    assert_ne!(format!("{forged:?}"), format!("{:?}", c.proof));
                    c.proof = forged;
                    let mut meter = Meter {
                        used: 0,
                        limit: usize::MAX,
                    };
                    assert_eq!(c.verify_bounded(&mut meter), Err(Error::Binding));
                    assert_eq!(
                        meter.used, c.seal.visits,
                        "must satisfy every non-hash guard"
                    );
                }
                5 => {
                    // A valid proof of a different program, whatever its width.
                    let changed = SOURCE.replace("wrapping_add", "wrapping_mul");
                    let p = check::check(front::parse(changed.as_bytes()).unwrap(), false).unwrap();
                    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
                    c.proof = net.verify().unwrap();
                }
                3 => {
                    let pos = c.source.iter().position(|&byte| byte == b' ').unwrap();
                    c.source[pos] = b'\n';
                    c.source_hash = Sha256::digest(&c.source).into();
                }
                _ => c.seal.visits -= 1,
            }
            assert!(
                c.runtime_prepare(&[1], &[0], 1, &mut unmetered()).is_err(),
                "fault={fault}"
            );
        }
        let c = contract();
        let mut meter = Meter {
            used: 0,
            limit: c.seal.visits - 1,
        };
        assert_eq!(c.verify_bounded(&mut meter), Err(Error::Budget));
    }
    fn stream(value: &impl fmt::Debug) -> usize {
        format!("{value:?}").len()
    }
    fn unmetered() -> Meter {
        Meter {
            used: 0,
            limit: usize::MAX,
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_bounded_out_of_range_refuses() {
        let c = contract();
        for (values, indices, n, iteration) in [
            (&[][..], &[0][..], 0, 0),
            (&[0][..], &[1][..], 1, 0),
            (&[0][..], &[0, 0][..], 1, 1),
        ] {
            assert_eq!(
                c.runtime_prepare(values, indices, n, &mut unmetered())
                    .err(),
                Some(Error::Bounds { iteration })
            );
        }
    }
    /// Independent recomputation of the seal formula: source bytes + checked
    /// stream + recognition reserve + Net stream + Proof stream, where the
    /// reserve is two name lookups plus the entry function's stream, bounded
    /// by the checked stream.
    fn expected_recognition(c: &Contract) -> usize {
        let names: usize = c
            .checked
            .program
            .functions
            .iter()
            .map(|f| f.name.len() + 1)
            .sum();
        let entry = c
            .checked
            .program
            .functions
            .iter()
            .find(|f| f.name == c.entry)
            .unwrap();
        (2 * names + stream(entry)).min(stream(&c.checked))
    }
    fn contract_with_main(main: &str) -> Contract {
        let s = format!("{SOURCE} {main}");
        let checked = check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap();
        Contract::from_checked(s.as_bytes(), &checked, "hist").unwrap()
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_bounded_recognition_reserve_bound() {
        for c in [
            contract(),
            contract_with_main(
                "fn main() -> i32 { let a: Array<u32> = [0u32]; let o = hist(a,a,1u32); return 7; }",
            ),
        ] {
            let recognition = expected_recognition(&c);
            assert!(recognition > 0);
            assert_eq!(c.seal.recognition, recognition);
            assert!(c.seal.recognition < stream(&c.checked));
            assert_eq!(
                c.seal.visits,
                c.source.len()
                    + stream(&c.checked)
                    + recognition
                    + stream(&c.net)
                    + stream(&c.proof)
            );
            let mut meter = Meter {
                used: 0,
                limit: usize::MAX,
            };
            c.verify_bounded(&mut meter).unwrap();
            assert_eq!(meter.used, c.seal.visits);
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_bounded_recognition_independent_of_main_growth()
     {
        let small = contract_with_main(
            "fn main() -> i32 { let a: Array<u32> = [0u32]; let o = hist(a,a,1u32); return 7; }",
        );
        let large = contract_with_main(
            "fn main() -> i32 { let a: Array<u32> = [0u32]; let o = hist(a,a,1u32); let b: Array<u32> = [1u32,2u32,3u32]; let p = hist(b,b,4u32); let q = hist(a,b,2u32); return 7; }",
        );
        assert!(stream(&large.checked) > stream(&small.checked));
        assert_eq!(small.seal.recognition, large.seal.recognition);
        assert_eq!(small.seal.recognition, expected_recognition(&large));
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_bounded_recognition_bound_is_capped() {
        assert_eq!(recognition_bound(10, 20, 100), Ok(30));
        assert_eq!(recognition_bound(10, 20, 30), Ok(30));
        assert_eq!(recognition_bound(10, 20, 25), Ok(25));
        assert_eq!(recognition_bound(usize::MAX, 1, 5), Err(Error::Budget));
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_bounded_recognition_tamper() {
        for delta in [1isize, -1] {
            let mut c = contract();
            c.seal.recognition = c.seal.recognition.checked_add_signed(delta).unwrap();
            let mut meter = Meter {
                used: 0,
                limit: usize::MAX,
            };
            c.verify_bounded(&mut meter).unwrap();
            assert_ne!(meter.used, c.seal.visits, "delta={delta}");
            assert!(
                c.runtime_prepare(&[1], &[0], 1, &mut unmetered()).is_err(),
                "delta={delta}"
            );
        }
        // A changed checked program is refused before the recognition charge.
        let mut c = contract();
        let (_, ty) = c
            .checked
            .expression_types
            .iter_mut()
            .find(|(_, ty)| *ty == front::Type::U32)
            .unwrap();
        *ty = front::Type::I32;
        let mut meter = Meter {
            used: 0,
            limit: usize::MAX,
        };
        assert_eq!(c.verify_bounded(&mut meter), Err(Error::Binding));
        assert_eq!(meter.used, c.source.len() + stream(&c.checked));
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_bounded_sort_meter_and_limits() {
        let mut seed = 0x4d05u32;
        // 128 contributions was the producer's old shape cap; the sort meter is exercised there.
        for count in 0..=128usize {
            for pattern in 0..4 {
                let mut a: Vec<_> = (0..count)
                    .map(|iteration| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        let bin = match pattern {
                            0 => iteration,
                            1 => count - iteration,
                            2 => 0,
                            _ => seed as usize,
                        };
                        Access { iteration, bin }
                    })
                    .collect();
                let h = if count <= 1 {
                    0
                } else {
                    (usize::BITS - (count - 1).leading_zeros()) as usize
                };
                let bound = 8 * count * h + 12 * count;
                let mut meter = Meter {
                    used: 0,
                    limit: bound,
                };
                heap_sort(&mut a, &mut meter).unwrap();
                assert!(a.windows(2).all(|w| w[0].bin <= w[1].bin));
                assert!(meter.used <= bound);
                for ordinal in 0..count {
                    assert_eq!(a.iter().filter(|v| v.iteration == ordinal).count(), 1);
                }
            }
        }
        let mut meter = Meter { used: 0, limit: 0 };
        assert_eq!(
            heap_sort(
                &mut [
                    Access {
                        iteration: 0,
                        bin: 1
                    },
                    Access {
                        iteration: 1,
                        bin: 0
                    }
                ],
                &mut meter
            ),
            Err(Error::Budget)
        );
        assert_eq!(sift(&mut [], 0, 1, &mut meter), Err(Error::Receipt));
    }
}
