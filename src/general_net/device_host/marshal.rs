//! Safe host leases, canonical contiguous layouts, and transactional writeback.
use super::*;
use crate::{front::Mode, numeric_contract::F32Bits};
use std::cell::{Ref, RefMut};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Oversize,
    Alias,
    WrongLayout,
    UnsupportedType,
    Borrowed,
}

pub(super) enum Lease<'a> {
    I32(Ref<'a, Vec<i32>>),
    U32(Ref<'a, Vec<u32>>),
    F32(Ref<'a, Vec<F32Bits>>),
    MutI32(RefMut<'a, Vec<i32>>),
    MutU32(RefMut<'a, Vec<u32>>),
    MutF32(RefMut<'a, Vec<F32Bits>>),
    Record(Ref<'a, Vec<Value>>),
}
impl Lease<'_> {
    fn len(&self) -> usize {
        match self {
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::MutI32(v) => v.len(),
            Self::MutU32(v) => v.len(),
            Self::MutF32(v) => v.len(),
            Self::Record(v) => v.len(),
        }
    }
    fn words(&self, types: &[Type]) -> Result<Vec<u64>, Error> {
        let mut words = Vec::new();
        words
            .try_reserve_exact(self.len())
            .map_err(|_| Error::Limit)?;
        match self {
            Self::I32(v) => words.extend(v.iter().map(|v| i64::from(*v) as u64)),
            Self::U32(v) => words.extend(v.iter().map(|v| u64::from(*v))),
            Self::F32(v) => words.extend(v.iter().map(|v| u64::from(v.to_bits()))),
            Self::MutI32(v) => words.extend(v.iter().map(|v| i64::from(*v) as u64)),
            Self::MutU32(v) => words.extend(v.iter().map(|v| u64::from(*v))),
            Self::MutF32(v) => words.extend(v.iter().map(|v| u64::from(v.to_bits()))),
            Self::Record(v) => {
                if v.len() != types.len() {
                    return Err(Error::Entry(Refusal::WrongLayout));
                }
                for (t, v) in types.iter().zip(v.iter()) {
                    words.push(scalar(t, v).map_err(|_| Error::Entry(Refusal::WrongLayout))?);
                }
            }
        }
        Ok(words)
    }
}
pub(super) struct Argument<'a> {
    handle: usize,
    kind: u64,
    lease: Lease<'a>,
}
pub(super) struct Packet<'a> {
    pub words: Vec<u64>,
    pub leases: Vec<Argument<'a>>,
}
fn scalar(ty: &Type, value: &Value) -> Result<u64, Error> {
    match (ty, value) {
        (Type::I32, Value::I32(x)) => Ok(i64::from(*x) as u64),
        (Type::U32, Value::U32(x)) => Ok(u64::from(*x)),
        (Type::F32, Value::F32(x)) => Ok(u64::from(x.to_bits())),
        (Type::Bool, Value::Bool(x)) => Ok(u64::from(*x)),
        (Type::Unit, Value::Unit) => Ok(0),
        _ => Err(Error::Input),
    }
}
fn identity(v: &Value) -> Option<usize> {
    match v {
        Value::ArrayI32(v) => Some(v.as_ptr().addr()),
        Value::ArrayU32(v) => Some(v.as_ptr().addr()),
        Value::ArrayF32(v) => Some(v.as_ptr().addr()),
        Value::Record(v) => Some(v.as_ptr().addr()),
        _ => None,
    }
}
pub(super) fn encode<'a>(
    artifact: &compiled::Artifact,
    args: &'a [Value],
) -> Result<Packet<'a>, Error> {
    if args.len() != artifact.params().len() {
        return Err(Error::Input);
    }
    let Some(entry) = artifact.entry_layout() else {
        return Ok(Packet {
            words: super::encode(artifact.params(), args)?,
            leases: Vec::new(),
        });
    };
    // Refuse aliases before acquiring any lease or allocating staging. Multiple
    // reads share the same stable owner ID; an empty owner still has identity.
    for (i, arg) in args.iter().enumerate() {
        if let Some(id) = identity(arg) {
            for (j, other) in args[..i].iter().enumerate() {
                if identity(other) == Some(id)
                    && (entry.modes()[i] == Some(Mode::Mut) || entry.modes()[j] == Some(Mode::Mut))
                {
                    return Err(Error::Entry(Refusal::Alias));
                }
            }
        }
    }
    let owned = artifact.owned_layout().ok_or(Error::Malformed)?;
    let start = args.len() + 2;
    let pool = start + owned.allocations() * 4;
    let mut words = Vec::new();
    words
        .try_reserve_exact(entry.input_words())
        .map_err(|_| Error::Limit)?;
    words.resize(entry.input_words(), 0);
    let mut leases = Vec::new();
    let mut ids = Vec::new();
    let mut types = Vec::new();
    let mut live = 0usize;
    for (i, (ty, value)) in artifact.params().iter().zip(args).enumerate() {
        if identity(value).is_none() {
            words[i] = scalar(ty, value).map_err(|e| {
                if matches!(value, Value::ViewI32(..) | Value::ViewU32(..)) {
                    Error::Entry(Refusal::WrongLayout)
                } else {
                    e
                }
            })?;
            continue;
        }
        if let Some(j) = ids.iter().position(|id| Some(*id) == identity(value)) {
            if types[j] != *ty {
                return Err(Error::Entry(Refusal::WrongLayout));
            }
            words[i] = (j + 1) as u64;
            continue;
        }
        let mode = entry.modes()[i];
        let mutable = mode == Some(Mode::Mut);
        let borrow = || Error::Entry(Refusal::Borrowed);
        let mut fields = &[][..];
        let (kind, lease) = match (ty, value, mode) {
            (Type::ArrayI32, Value::ArrayI32(v), Some(Mode::Read | Mode::Mut)) => (
                0,
                if mutable {
                    Lease::MutI32(v.try_borrow_mut().map_err(|_| borrow())?)
                } else {
                    Lease::I32(v.try_borrow().map_err(|_| borrow())?)
                },
            ),
            (Type::ArrayU32, Value::ArrayU32(v), Some(Mode::Read | Mode::Mut)) => (
                1,
                if mutable {
                    Lease::MutU32(v.try_borrow_mut().map_err(|_| borrow())?)
                } else {
                    Lease::U32(v.try_borrow().map_err(|_| borrow())?)
                },
            ),
            (Type::ArrayF32, Value::ArrayF32(v), Some(Mode::Read | Mode::Mut)) => (
                258,
                if mutable {
                    Lease::MutF32(v.try_borrow_mut().map_err(|_| borrow())?)
                } else {
                    Lease::F32(v.try_borrow().map_err(|_| borrow())?)
                },
            ),
            (Type::Named(name), Value::Record(v), Some(Mode::Read)) => {
                let kind = artifact
                    .aggregate_layout()
                    .and_then(|a| a.kinds().iter().find(|k| k.name() == name))
                    .ok_or(Error::Entry(Refusal::WrongLayout))?;
                let compiled::AggregateForm::Record(f) = kind.form() else {
                    return Err(Error::Entry(Refusal::UnsupportedType));
                };
                fields = f;
                (
                    u64::from(kind.id()),
                    Lease::Record(v.try_borrow().map_err(|_| borrow())?),
                )
            }
            _ => return Err(Error::Entry(Refusal::UnsupportedType)),
        };
        let n = lease.len();
        if n > crate::eval::MAX_ARRAY_LEN || leases.len() >= owned.allocations() {
            return Err(Error::Entry(Refusal::Oversize));
        }
        let end = live
            .checked_add(n)
            .filter(|n| *n <= owned.elements())
            .ok_or(Error::Entry(Refusal::Oversize))?;
        let data = lease.words(fields)?;
        let handle = leases.len() + 1;
        let descriptor = start + (handle - 1) * 4;
        words[descriptor..descriptor + 4].copy_from_slice(&[
            live as u64,
            n as u64,
            kind,
            if fields.is_empty() {
                1
            } else {
                (1u64 << (n + 1)) - 1
            },
        ]);
        words[pool + live..pool + end].copy_from_slice(&data);
        words[i] = handle as u64;
        live = end;
        ids.push(identity(value).ok_or(Error::Input)?);
        types.push(ty.clone());
        leases.push(Argument {
            handle,
            kind,
            lease,
        });
    }
    words[args.len()] = leases.len() as u64;
    words[args.len() + 1] = live as u64;
    Ok(Packet { words, leases })
}
fn array<'a>(
    arg: &Argument<'_>,
    owned: &compiled::OwnedLayout,
    heap: &'a [u64],
) -> Result<&'a [u64], Error> {
    let d = (arg.handle - 1).checked_mul(4).ok_or(Error::Malformed)?;
    let descriptor = heap.get(d..d + 4).ok_or(Error::Malformed)?;
    if descriptor[1] != arg.lease.len() as u64
        || descriptor[2] != arg.kind
        || descriptor[3] & 1 != 1
    {
        return Err(Error::Entry(Refusal::WrongLayout));
    }
    let offset = usize::try_from(descriptor[0]).map_err(|_| Error::Malformed)?;
    let end = offset
        .checked_add(arg.lease.len())
        .filter(|n| *n <= owned.elements())
        .ok_or(Error::Malformed)?;
    let pool = owned.allocations() * 4;
    heap.get(pool + offset..pool + end).ok_or(Error::Malformed)
}
pub(super) fn publish(
    args: &mut [Argument<'_>],
    owned: &compiled::OwnedLayout,
    heap: &[u64],
) -> Result<(), Error> {
    // Validate every destination before the first write. RefMut leases already
    // exclude all caller access for the entire in-flight operation.
    for arg in args.iter() {
        let data = array(arg, owned, heap)?;
        let valid = match &arg.lease {
            Lease::MutI32(_) => data.iter().all(|v| i32::try_from(*v as i64).is_ok()),
            Lease::MutU32(_) | Lease::MutF32(_) => data.iter().all(|v| u32::try_from(*v).is_ok()),
            _ => true,
        };
        if !valid {
            return Err(Error::Malformed);
        }
    }
    for arg in args {
        let data = array(arg, owned, heap)?;
        match &mut arg.lease {
            Lease::MutI32(v) => {
                for (x, w) in v.iter_mut().zip(data) {
                    *x = *w as i32;
                }
            }
            Lease::MutU32(v) => {
                for (x, w) in v.iter_mut().zip(data) {
                    *x = *w as u32;
                }
            }
            Lease::MutF32(v) => {
                for (x, w) in v.iter_mut().zip(data) {
                    *x = F32Bits::from_bits(*w as u32);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn result(
    artifact: &compiled::Artifact,
    word: u64,
    heap: &[u64],
    allocated: usize,
) -> Result<Value, Error> {
    fn scalar(ty: &Type, w: u64) -> Result<Value, Error> {
        Ok(match ty {
            Type::I32 => Value::I32(i32::try_from(w as i64).map_err(|_| Error::Malformed)?),
            Type::U32 => Value::U32(u32::try_from(w).map_err(|_| Error::Malformed)?),
            Type::F32 => Value::F32(F32Bits::from_bits(
                u32::try_from(w).map_err(|_| Error::Malformed)?,
            )),
            Type::Bool if w <= 1 => Value::Bool(w != 0),
            Type::Unit if w == 0 => Value::Unit,
            _ => return Err(Error::Malformed),
        })
    }
    if let Some(kind) = match artifact.result() {
        Type::ArrayI32 => Some(0),
        Type::ArrayU32 => Some(1),
        Type::ArrayF32 => Some(258),
        _ => None,
    } {
        let owned = artifact.owned_layout().ok_or(Error::Malformed)?;
        let handle = usize::try_from(word)
            .ok()
            .filter(|h| *h > 0 && *h <= allocated && allocated <= owned.allocations())
            .ok_or(Error::Malformed)?;
        let descriptor = heap
            .get((handle - 1) * 4..handle * 4)
            .ok_or(Error::Malformed)?;
        if descriptor[2] != kind || descriptor[3] != 1 {
            return Err(Error::Entry(Refusal::WrongLayout));
        }
        let offset = usize::try_from(descriptor[0]).map_err(|_| Error::Malformed)?;
        let len = usize::try_from(descriptor[1]).map_err(|_| Error::Malformed)?;
        if len > crate::array_limits::MAX_ARRAY_LEN {
            return Err(Error::Malformed);
        }
        let end = offset
            .checked_add(len)
            .filter(|e| *e <= owned.elements())
            .ok_or(Error::Malformed)?;
        let pool = owned.allocations() * 4;
        let data = heap
            .get(pool + offset..pool + end)
            .ok_or(Error::Malformed)?;
        fn decode_array<T>(
            data: &[u64],
            decode: impl Fn(u64) -> Result<T, Error>,
        ) -> Result<std::rc::Rc<std::cell::RefCell<Vec<T>>>, Error> {
            let mut values = Vec::new();
            values
                .try_reserve_exact(data.len())
                .map_err(|_| Error::Limit)?;
            for word in data {
                values.push(decode(*word)?);
            }
            Ok(std::rc::Rc::new(std::cell::RefCell::new(values)))
        }
        return Ok(match artifact.result() {
            Type::ArrayI32 => Value::ArrayI32(decode_array(data, |w| {
                i32::try_from(w as i64).map_err(|_| Error::Malformed)
            })?),
            Type::ArrayU32 => Value::ArrayU32(decode_array(data, |w| {
                u32::try_from(w).map_err(|_| Error::Malformed)
            })?),
            _ => Value::ArrayF32(decode_array(data, |w| {
                Ok(F32Bits::from_bits(
                    u32::try_from(w).map_err(|_| Error::Malformed)?,
                ))
            })?),
        });
    }
    let Type::Named(name) = artifact.result() else {
        return scalar(artifact.result(), word);
    };
    let kind = artifact
        .aggregate_layout()
        .and_then(|a| a.kinds().iter().find(|k| k.name() == name))
        .ok_or(Error::Malformed)?;
    let compiled::AggregateForm::Record(fields) = kind.form() else {
        return Err(Error::Malformed);
    };
    let owned = artifact.owned_layout().ok_or(Error::Malformed)?;
    let handle = usize::try_from(word)
        .ok()
        .filter(|h| *h > 0 && *h <= allocated && allocated <= owned.allocations())
        .ok_or(Error::Malformed)?;
    let descriptor = heap
        .get((handle - 1) * 4..handle * 4)
        .ok_or(Error::Malformed)?;
    if descriptor[1] != fields.len() as u64
        || descriptor[2] != u64::from(kind.id())
        || descriptor[3] != (1u64 << (fields.len() + 1)) - 1
    {
        return Err(Error::Entry(Refusal::WrongLayout));
    }
    let offset = usize::try_from(descriptor[0]).map_err(|_| Error::Malformed)?;
    let end = offset
        .checked_add(fields.len())
        .filter(|e| *e <= owned.elements())
        .ok_or(Error::Malformed)?;
    let pool = owned.allocations() * 4;
    let values = heap
        .get(pool + offset..pool + end)
        .ok_or(Error::Malformed)?
        .iter()
        .zip(fields)
        .map(|(w, t)| scalar(t, *w))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::Record(std::rc::Rc::new(std::cell::RefCell::new(
        values,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};
    #[test]
    fn returned_array_rejects_forged_descriptors_and_noncanonical_words() {
        let plan = Plan::from_source(
            b"fn run()->Array<f32>{return [-0.0];}",
            "run",
            Target::Metal,
            Limits::default(),
        )
        .unwrap();
        let artifact = plan.artifact();
        let owned = artifact.owned_layout().unwrap();
        let mut heap = vec![0; owned.heap_words()];
        heap[..4].copy_from_slice(&[0, 1, 258, 1]);
        heap[owned.allocations() * 4] = 0x80000000;
        assert!(matches!(
            result(artifact, 1, &heap, 1),
            Ok(Value::ArrayF32(_))
        ));
        assert!(result(artifact, 0, &heap, 1).is_err());
        assert!(result(artifact, 1, &heap, 0).is_err());
        assert!(result(artifact, 1, &heap[..3], 1).is_err());
        for (index, word) in [
            (0, u64::MAX),
            (1, u64::MAX),
            (2, 1),
            (3, 0),
            (3, 3),
            (owned.allocations() * 4, 1u64 << 32),
        ] {
            let mut bad = heap.clone();
            bad[index] = word;
            assert!(
                result(artifact, 1, &bad, 1).is_err(),
                "field {index}, word {word}"
            );
        }
    }
    #[test]
    fn malformed_second_destination_publishes_neither_array() {
        let plan=Plan::from_source(b"fn run(a:mut Array<i32>,b:mut Array<i32>)->i32{a[0u32]=7;b[0u32]=8;return 0;} fn main()->i32{return 0;}","run",Target::Cuda,Limits::default()).unwrap();
        let a = Rc::new(RefCell::new(vec![1]));
        let b = Rc::new(RefCell::new(vec![2]));
        let args = [
            Value::ArrayI32(Rc::clone(&a)),
            Value::ArrayI32(Rc::clone(&b)),
        ];
        let mut packet = encode(plan.artifact(), &args).unwrap();
        let owned = plan.artifact().owned_layout().unwrap();
        let mut heap = packet.words[args.len() + 2..].to_vec();
        heap[owned.allocations() * 4] = 7;
        heap[owned.allocations() * 4 + 1] = 8;
        heap[5] = 2; // second output's claimed length differs from its host lease
        assert!(matches!(
            publish(&mut packet.leases, owned, &heap),
            Err(Error::Entry(Refusal::WrongLayout))
        ));
        drop(packet);
        assert_eq!(*a.borrow(), vec![1]);
        assert_eq!(*b.borrow(), vec![2]);
    }
}
