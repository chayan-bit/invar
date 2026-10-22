//! Independent owned array arena for the CPU net oracle. Copy handles are
//! internal references; affine transfer/drop comes from verified MIR and slots.

use super::{Frame, OracleError, Scalar, ToolFailure, internal, semantic_tool, tool};
use crate::{
    eval::{EvalError, TrapCategory, Value},
    front::{Span, Type},
    mir,
    numeric_contract::F32Bits,
};
use std::{cell::RefCell, rc::Rc};

enum Elements {
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<F32Bits>),
}
impl Elements {
    fn len(&self) -> usize {
        match self {
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }
}

pub(super) struct Arena {
    slots: Vec<Option<Elements>>,
    live: usize,
    cap: usize,
}

impl Arena {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            slots: Vec::new(),
            live: 0,
            cap: cap.min(crate::array_limits::MAX_LIVE_ELEMENTS),
        }
    }
    fn get(&self, id: usize) -> Result<&Elements, OracleError> {
        self.slots
            .get(id)
            .and_then(Option::as_ref)
            .ok_or_else(internal)
    }
    pub(super) fn admit_units(&mut self, count: usize) -> Result<(), OracleError> {
        let live = self.live.checked_add(count).ok_or_else(internal)?;
        if live > crate::array_limits::MAX_LIVE_ELEMENTS {
            return Err(semantic_tool(
                "ARRAY_SIZE_LIMIT",
                "live array element limit exceeded",
            ));
        }
        if live > self.cap {
            return Err(tool(
                ToolFailure::Allocation,
                "live owner unit budget exhausted",
            ));
        }
        self.live = live;
        Ok(())
    }
    pub(super) fn release_units(&mut self, count: usize) -> Result<(), OracleError> {
        self.live = self.live.checked_sub(count).ok_or_else(internal)?;
        Ok(())
    }
    pub(super) fn release_owner(&mut self, id: usize) -> Result<usize, OracleError> {
        let data = self
            .slots
            .get_mut(id)
            .and_then(Option::take)
            .ok_or_else(internal)?;
        let length = data.len();
        self.release_units(length)?;
        Ok(length)
    }
    pub(super) fn import(&mut self, value: &Value) -> Result<Scalar, OracleError> {
        fn copy<T: Copy>(values: &[T]) -> Result<Vec<T>, OracleError> {
            let mut data = Vec::new();
            data.try_reserve_exact(values.len())
                .map_err(|_| tool(ToolFailure::Allocation, "host oracle staging failed"))?;
            data.extend_from_slice(values);
            Ok(data)
        }
        let data = match value {
            Value::ArrayI32(v) => {
                let v = v.try_borrow().map_err(|_| internal())?;
                self.size(v.len())?;
                Elements::I32(copy(&v)?)
            }
            Value::ArrayU32(v) => {
                let v = v.try_borrow().map_err(|_| internal())?;
                self.size(v.len())?;
                Elements::U32(copy(&v)?)
            }
            Value::ArrayF32(v) => {
                let v = v.try_borrow().map_err(|_| internal())?;
                self.size(v.len())?;
                Elements::F32(copy(&v)?)
            }
            _ => return Err(internal()),
        };
        self.alloc(data)
    }
    fn alloc(&mut self, data: Elements) -> Result<Scalar, OracleError> {
        self.slots
            .try_reserve(1)
            .map_err(|_| tool(ToolFailure::Allocation, "array arena allocation failed"))?;
        self.admit_units(data.len())?;
        let id = self.slots.len();
        self.slots.push(Some(data));
        Ok(Scalar::Array(id))
    }
    pub(super) fn value(&self, id: usize) -> Result<Value, OracleError> {
        Ok(match self.get(id)? {
            Elements::I32(v) => Value::ArrayI32(Rc::new(RefCell::new(v.clone()))),
            Elements::U32(v) => Value::ArrayU32(Rc::new(RefCell::new(v.clone()))),
            Elements::F32(v) => Value::ArrayF32(Rc::new(RefCell::new(v.clone()))),
        })
    }
    fn words(&self, id: usize) -> Result<Vec<u32>, OracleError> {
        let data = self.get(id)?;
        let mut words = reserve(data.len())?;
        match data {
            Elements::I32(v) => words.extend(v.iter().map(|&x| x as u32)),
            Elements::U32(v) => words.extend_from_slice(v),
            Elements::F32(v) => words.extend(v.iter().map(|x| x.to_bits())),
        }
        Ok(words)
    }
    fn commit_words(&mut self, id: usize, words: &[u32]) -> Result<(), OracleError> {
        let data = self
            .slots
            .get_mut(id)
            .and_then(Option::as_mut)
            .ok_or_else(internal)?;
        if data.len() != words.len() {
            return Err(internal());
        }
        match data {
            Elements::I32(v) => v.iter_mut().zip(words).for_each(|(x, &w)| *x = w as i32),
            Elements::U32(v) => v.copy_from_slice(words),
            Elements::F32(v) => v
                .iter_mut()
                .zip(words)
                .for_each(|(x, &w)| *x = F32Bits::from_bits(w)),
        }
        Ok(())
    }
    /// Only arithmetic is shared with D5's frozen oracle. Net activations,
    /// source fuel, traps and cleanup have already run independently.
    // ponytail: repeats products; retain an exact accumulator if oracle throughput matters.
    pub(super) fn finish_tc(
        &mut self,
        inputs: [usize; 2],
        dims: [u32; 3],
        value: Scalar,
    ) -> Result<(), OracleError> {
        let Scalar::Array(output) = value else {
            return Err(internal());
        };
        if inputs.contains(&output) {
            return Err(internal());
        }
        if dims.contains(&0) {
            return Ok(());
        }
        let words = |id, count| {
            let Elements::F32(values) = self.get(id)? else {
                return Err(internal());
            };
            let values = usize::try_from(count)
                .ok()
                .and_then(|count| values.get(..count))
                .ok_or_else(internal)?;
            Ok(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        };
        let [r, k, n] = dims.map(u64::from);
        let words =
            invar_device::tc::reference(dims, &words(inputs[0], r * k)?, &words(inputs[1], k * n)?)
                .map_err(|_| semantic_tool("TC_INPUT_RANGE", "invalid TC reference inputs"))?;
        let Some(Elements::F32(values)) = self.slots.get_mut(output).and_then(Option::as_mut)
        else {
            return Err(internal());
        };
        if values.len() != words.len() {
            return Err(internal());
        }
        for (value, word) in values.iter_mut().zip(words) {
            *value = F32Bits::from_bits(word);
        }
        Ok(())
    }
    fn size(&self, length: usize) -> Result<(), OracleError> {
        if length > crate::array_limits::MAX_ARRAY_LEN {
            return Err(semantic_tool(
                "ARRAY_SIZE_LIMIT",
                "array length limit exceeded",
            ));
        }
        if self
            .live
            .checked_add(length)
            .is_none_or(|live| live > crate::array_limits::MAX_LIVE_ELEMENTS)
        {
            return Err(semantic_tool(
                "ARRAY_SIZE_LIMIT",
                "live array element limit exceeded",
            ));
        }
        if self
            .live
            .checked_add(length)
            .is_none_or(|live| live > self.cap)
        {
            return Err(tool(
                ToolFailure::Allocation,
                "live array element budget exhausted",
            ));
        }
        Ok(())
    }
}

/// A borrowed strided window over a live integer owner. Views are copyable
/// non-owners: they never transfer, release or record cleanup.
#[derive(Clone, Copy, Debug)]
pub(super) struct View {
    array: usize,
    start: u32,
    len: u32,
    stride: u32,
    mutable: bool,
}

impl View {
    /// Physical element of `index`, or an index trap at `span`.
    fn physical(self, index: u32, span: Span) -> Result<usize, OracleError> {
        let bounds = || {
            OracleError::from(EvalError::Trap {
                category: TrapCategory::IndexOutOfBounds,
                span,
            })
        };
        if index >= self.len {
            return Err(bounds());
        }
        let offset = index.checked_mul(self.stride).ok_or_else(bounds)?;
        let physical = self.start.checked_add(offset).ok_or_else(bounds)?;
        usize::try_from(physical).map_err(|_| bounds())
    }
}

pub(super) fn is_view_intrinsic(name: &str) -> bool {
    matches!(name, "view" | "view_mut")
}

impl Arena {
    pub(super) fn view_value(&self, view: View) -> Result<Value, OracleError> {
        let range = crate::eval::ViewRange {
            start: view.start,
            len: view.len,
            stride: view.stride,
            mutable: view.mutable,
        };
        Ok(match self.get(view.array)? {
            Elements::I32(v) => Value::ViewI32(Rc::new(RefCell::new(v.clone())), range),
            Elements::U32(v) => Value::ViewU32(Rc::new(RefCell::new(v.clone())), range),
            Elements::F32(_) => return Err(internal()),
        })
    }
}

/// `view`/`view_mut(source, start, end[, stride])` over an integer owner or view.
/// Range validation precedes any composition, and the last selected element
/// must lie inside the live owner; failures trap at the call span.
pub(super) fn make_view(
    name: &str,
    args: &[mir::ValueId],
    frame: &Frame,
    span: Span,
    arena: &Arena,
) -> Result<Scalar, OracleError> {
    let bounds = || {
        OracleError::from(EvalError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            span,
        })
    };
    let (source, bounds_args) = args.split_first().ok_or_else(internal)?;
    let numbers = bounds_args
        .iter()
        .map(|id| match frame.get(*id)? {
            Scalar::U32(x) => Ok(x),
            _ => Err(internal()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (start, end, stride) = match numbers.as_slice() {
        [start, end] => (*start, *end, 1),
        [start, end, stride] => (*start, *end, *stride),
        _ => return Err(internal()),
    };
    let base = match frame.get(*source)? {
        Scalar::Array(id) => {
            let length = arena.get(id)?.len();
            if matches!(arena.get(id)?, Elements::F32(_)) {
                return Err(internal());
            }
            View {
                array: id,
                start: 0,
                len: u32::try_from(length).map_err(|_| internal())?,
                stride: 1,
                mutable: true,
            }
        }
        Scalar::View(view) => view,
        _ => return Err(internal()),
    };
    let mutable = name == "view_mut";
    if start > end || end > base.len || stride == 0 {
        return Err(bounds());
    }
    if mutable && !base.mutable {
        return Err(internal());
    }
    let step = base.stride.checked_mul(stride).ok_or_else(bounds)?;
    let origin = start
        .checked_mul(base.stride)
        .and_then(|offset| base.start.checked_add(offset))
        .ok_or_else(bounds)?;
    let len = (end - start).div_ceil(stride);
    if len > 0 {
        let last = (len - 1)
            .checked_mul(step)
            .and_then(|offset| origin.checked_add(offset))
            .ok_or_else(bounds)?;
        if u64::from(last) >= arena.get(base.array)?.len() as u64 {
            return Err(bounds());
        }
    }
    Ok(Scalar::View(View {
        array: base.array,
        start: origin,
        len,
        stride: step,
        mutable,
    }))
}

fn reserve<T>(length: usize) -> Result<Vec<T>, OracleError> {
    let mut items = Vec::new();
    items
        .try_reserve_exact(length)
        .map_err(|_| tool(ToolFailure::Allocation, "array storage allocation failed"))?;
    Ok(items)
}

/// Float reductions read owner words through the independent fsum reference;
/// the net machine keeps demand, fuel and cleanup.
pub(super) fn reduce(
    kind: crate::fsum::Kind,
    args: &[mir::ValueId],
    frame: &Frame,
    span: Span,
    arena: &Arena,
) -> Result<Scalar, OracleError> {
    use crate::{fsum::Kind, fsum_reference as reference};
    if args.len() != if kind.is_dot() { 2 } else { 1 } {
        return Err(internal());
    }
    let floats = |arg: mir::ValueId| match frame.get(arg)? {
        Scalar::Array(id) => match arena.get(id)? {
            Elements::F32(values) => Ok(values),
            _ => Err(internal()),
        },
        _ => Err(internal()),
    };
    let a = floats(args[0])?;
    let b = match args.get(1) {
        Some(arg) => floats(*arg)?,
        None => a,
    };
    if a.len() != b.len() {
        return Err(OracleError::from(EvalError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            span,
        }));
    }
    Ok(Scalar::F32(match kind {
        Kind::SumTree => reference::sum_tree(a),
        Kind::DotTree => reference::dot_tree(a, b),
        Kind::SumExact => reference::sum_exact(a),
        Kind::DotExact => reference::dot_exact(a, b),
    }))
}

pub(super) fn filled(
    name: &str,
    args: &[mir::ValueId],
    frame: &Frame,
    arena: &mut Arena,
) -> Result<Scalar, OracleError> {
    let [length, value] = args else {
        return Err(internal());
    };
    let Scalar::U32(length) = frame.get(*length)? else {
        return Err(internal());
    };
    let length = usize::try_from(length).map_err(|_| internal())?;
    arena.size(length)?;
    match (name, frame.get(*value)?) {
        ("filled_i32", Scalar::I32(value)) => {
            let mut v = reserve(length)?;
            v.resize(length, value);
            arena.alloc(Elements::I32(v))
        }
        ("filled_u32", Scalar::U32(value)) => {
            let mut v = reserve(length)?;
            v.resize(length, value);
            arena.alloc(Elements::U32(v))
        }
        ("filled_f32", Scalar::F32(value)) => {
            let mut v = reserve(length)?;
            v.resize(length, value);
            arena.alloc(Elements::F32(v))
        }
        _ => Err(internal()),
    }
}

/// Stage the merged radix operation over arena words. Only successful work
/// commits owner contents; the net machine retains demand, fuel and cleanup.
pub(super) fn sort(
    name: &str,
    args: &[mir::ValueId],
    frame: &Frame,
    span: Span,
    arena: &mut Arena,
) -> Result<Scalar, OracleError> {
    use crate::sorting::{self, KeyKind};
    let id = |arg| match frame.get(arg)? {
        Scalar::Array(id) => Ok(id),
        _ => Err(internal()),
    };
    let key = id(*args.first().ok_or_else(internal)?)?;
    let kind = match arena.get(key)? {
        Elements::I32(_) => KeyKind::I32,
        Elements::U32(_) => KeyKind::U32,
        Elements::F32(_) => KeyKind::F32,
    };
    let mut keys = arena.words(key)?;
    if name == "argsort" {
        let order = sorting::argsort_words(&keys, kind, 1)
            .map_err(|error| sorting::failure(error, span))?;
        arena.size(order.len())?;
        return arena.alloc(Elements::U32(order));
    }
    let payload_id = (name == "sort_by_key")
        .then(|| id(*args.get(1).ok_or_else(internal)?))
        .transpose()?;
    let mut payload = payload_id.map(|id| arena.words(id)).transpose()?;
    let offsets = if name == "segmented_sort" {
        let offset = id(*args.get(1).ok_or_else(internal)?)?;
        if !matches!(arena.get(offset)?, Elements::U32(_)) {
            return Err(internal());
        }
        Some(arena.words(offset)?)
    } else {
        None
    };
    sorting::sort_words(
        &mut keys,
        payload.as_deref_mut(),
        kind,
        offsets.as_deref(),
        1,
    )
    .map_err(|error| sorting::failure(error, span))?;
    arena.commit_words(key, &keys)?;
    if let (Some(id), Some(payload)) = (payload_id, payload) {
        arena.commit_words(id, &payload)?;
    }
    Ok(Scalar::Unit)
}

pub(super) fn operation(
    op: &mir::Operation,
    frame: &mut Frame,
    ty: Option<&Type>,
    span: Span,
    arena: &mut Arena,
) -> Result<Option<Scalar>, OracleError> {
    let bounds = || {
        OracleError::from(EvalError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            span,
        })
    };
    Ok(match op {
        mir::Operation::Move(name) => Some(
            frame
                .places
                .get_mut(name)
                .and_then(|slot| slot.value.take())
                .ok_or_else(internal)?,
        ),
        mir::Operation::ArrayRef { name, .. } => Some(
            frame
                .places
                .get(name)
                .and_then(|slot| slot.value)
                .ok_or_else(internal)?,
        ),
        mir::Operation::Array(ids) => {
            arena.size(ids.len())?;
            Some(match ty {
                Some(Type::ArrayI32) => {
                    let mut items = reserve(ids.len())?;
                    for id in ids {
                        let Scalar::I32(x) = frame.get(*id)? else {
                            return Err(internal());
                        };
                        items.push(x);
                    }
                    arena.alloc(Elements::I32(items))?
                }
                Some(Type::ArrayU32) => {
                    let mut items = reserve(ids.len())?;
                    for id in ids {
                        let Scalar::U32(x) = frame.get(*id)? else {
                            return Err(internal());
                        };
                        items.push(x);
                    }
                    arena.alloc(Elements::U32(items))?
                }
                Some(Type::ArrayF32) => {
                    let mut items = reserve(ids.len())?;
                    for id in ids {
                        let Scalar::F32(x) = frame.get(*id)? else {
                            return Err(internal());
                        };
                        items.push(x);
                    }
                    arena.alloc(Elements::F32(items))?
                }
                _ => return Err(internal()),
            })
        }
        mir::Operation::Index { array, index } => {
            let (id, index) = match (frame.get(*array)?, frame.get(*index)?) {
                (Scalar::Array(id), Scalar::U32(index)) => {
                    (id, usize::try_from(index).map_err(|_| bounds())?)
                }
                (Scalar::View(view), Scalar::U32(index)) => {
                    (view.array, view.physical(index, span)?)
                }
                _ => return Err(internal()),
            };
            Some(match arena.get(id)? {
                Elements::I32(items) => Scalar::I32(*items.get(index).ok_or_else(bounds)?),
                Elements::U32(items) => Scalar::U32(*items.get(index).ok_or_else(bounds)?),
                Elements::F32(items) => Scalar::F32(*items.get(index).ok_or_else(bounds)?),
            })
        }
        mir::Operation::Length(name) => match frame.places.get(name).and_then(|slot| slot.value) {
            Some(Scalar::Array(id)) => Some(Scalar::U32(
                u32::try_from(arena.get(id)?.len()).map_err(|_| internal())?,
            )),
            Some(Scalar::View(view)) => {
                arena.get(view.array)?;
                Some(Scalar::U32(view.len))
            }
            _ => return Err(internal()),
        },
        mir::Operation::StoreIndex { name, index, value } => {
            let Scalar::U32(index) = frame.get(*index)? else {
                return Err(internal());
            };
            let (id, index) = match frame.places.get(name).and_then(|slot| slot.value) {
                Some(Scalar::Array(id)) => (id, usize::try_from(index).map_err(|_| bounds())?),
                Some(Scalar::View(view)) if view.mutable => {
                    (view.array, view.physical(index, span)?)
                }
                _ => return Err(internal()),
            };
            let data = arena
                .slots
                .get_mut(id)
                .and_then(Option::as_mut)
                .ok_or_else(internal)?;
            match (data, frame.get(*value)?) {
                (Elements::I32(items), Scalar::I32(x)) => {
                    *items.get_mut(index).ok_or_else(bounds)? = x
                }
                (Elements::U32(items), Scalar::U32(x)) => {
                    *items.get_mut(index).ok_or_else(bounds)? = x
                }
                (Elements::F32(items), Scalar::F32(x)) => {
                    *items.get_mut(index).ok_or_else(bounds)? = x
                }
                _ => return Err(internal()),
            }
            None
        }
        _ => return Err(internal()),
    })
}

#[cfg(test)]
mod tests {
    use super::super::Slot;
    use super::*;
    use crate::{check, front};
    use std::collections::BTreeMap;
    fn frame() -> Frame {
        let c = check::check(front::parse(b"fn main()->i32{return 0;}").unwrap(), true).unwrap();
        let p = mir::lower(&c).unwrap();
        Frame::new(&p, 0, Vec::new(), 9).unwrap()
    }
    fn bits(arena: &Arena, id: usize) -> Vec<u32> {
        let Value::ArrayF32(v) = arena.value(id).unwrap() else {
            panic!("array type")
        };
        v.borrow().iter().map(|x| x.to_bits()).collect()
    }
    fn id(value: Scalar) -> usize {
        let Scalar::Array(i) = value else {
            panic!("array handle")
        };
        i
    }
    #[test]
    fn tc_completion_refuses_output_alias_without_mutation() {
        let mut arena = Arena::new(2);
        let a = id(arena
            .alloc(Elements::F32(vec![F32Bits::from_bits(0x3f80_0000)]))
            .unwrap());
        let b = id(arena
            .alloc(Elements::F32(vec![F32Bits::from_bits(0x4000_0000)]))
            .unwrap());
        for output in [a, b] {
            assert!(
                arena
                    .finish_tc([a, b], [1, 1, 1], Scalar::Array(output))
                    .is_err()
            );
        }
        assert_eq!(bits(&arena, a), [0x3f80_0000]);
        assert_eq!(bits(&arena, b), [0x4000_0000]);
    }
    #[test]
    fn f32_bits_reconstruction_mixed_owner_ids_and_release() {
        let mut a = Arena::new(6);
        let i = id(a.alloc(Elements::I32(vec![-1])).unwrap());
        let u = id(a.alloc(Elements::U32(vec![u32::MAX])).unwrap());
        let f = id(a
            .alloc(Elements::F32(
                [0, 0x80000000, 0x7fa00001, 0xffc12345]
                    .map(F32Bits::from_bits)
                    .to_vec(),
            ))
            .unwrap());
        assert_eq!((i, u, f), (0, 1, 2));
        assert_eq!(bits(&a, f), [0, 0x80000000, 0x7fa00001, 0xffc12345]);
        assert_eq!(a.live, 6);
        let Value::ArrayF32(snapshot) = a.value(f).unwrap() else {
            panic!("float snapshot")
        };
        snapshot.borrow_mut()[0] = F32Bits::from_bits(1);
        assert_eq!(bits(&a, f)[0], 0);
        assert_eq!(a.release_owner(i).unwrap(), 1);
        assert_eq!(a.release_owner(f).unwrap(), 4);
        assert_eq!(a.live, 1);
        assert!(a.value(i).is_err());
        assert!(a.release_owner(f).is_err());
        assert_eq!(a.release_owner(u).unwrap(), 1);
        assert_eq!(a.live, 0);
    }
    #[test]
    fn f32_literal_filled_index_store_and_move_are_exact() {
        let mut f = frame();
        let mut a = Arena::new(8);
        f.values = BTreeMap::from([
            (0, Scalar::F32(F32Bits::NEG_ZERO)),
            (1, Scalar::F32(F32Bits::from_bits(0x7fa00001))),
            (2, Scalar::U32(2)),
            (3, Scalar::U32(1)),
        ]);
        let span = Span { start: 21, end: 28 };
        let literal = operation(
            &mir::Operation::Array(vec![0, 1]),
            &mut f,
            Some(&Type::ArrayF32),
            span,
            &mut a,
        )
        .unwrap()
        .unwrap();
        let owner = id(literal);
        assert_eq!(bits(&a, owner), [0x80000000, 0x7fa00001]);
        f.values.insert(4, literal);
        f.places.insert(
            "xs".into(),
            Slot {
                value: Some(literal),
                owned: true,
            },
        );
        let x = operation(
            &mir::Operation::Index { array: 4, index: 3 },
            &mut f,
            Some(&Type::F32),
            span,
            &mut a,
        )
        .unwrap()
        .unwrap();
        assert!(matches!(x,Scalar::F32(v) if v.to_bits()==0x7fa00001));
        operation(
            &mir::Operation::StoreIndex {
                name: "xs".into(),
                index: 3,
                value: 0,
            },
            &mut f,
            None,
            span,
            &mut a,
        )
        .unwrap();
        assert_eq!(bits(&a, owner), [0x80000000, 0x80000000]);
        let length = operation(
            &mir::Operation::Length("xs".into()),
            &mut f,
            Some(&Type::U32),
            span,
            &mut a,
        )
        .unwrap()
        .unwrap();
        assert!(matches!(length, Scalar::U32(2)));
        let filled = id(filled("filled_f32", &[2, 1], &f, &mut a).unwrap());
        assert_eq!(bits(&a, filled), [0x7fa00001, 0x7fa00001]);
        let loan = operation(
            &mir::Operation::ArrayRef {
                name: "xs".into(),
                loan: Some(crate::front::Mode::Read),
            },
            &mut f,
            Some(&Type::ArrayF32),
            span,
            &mut a,
        )
        .unwrap()
        .unwrap();
        assert_eq!(id(loan), owner);
        assert!(f.places["xs"].value.is_some());
        let moved = operation(
            &mir::Operation::Move("xs".into()),
            &mut f,
            Some(&Type::ArrayF32),
            span,
            &mut a,
        )
        .unwrap()
        .unwrap();
        assert_eq!(id(moved), owner);
        assert!(f.places["xs"].value.is_none());
        assert!(
            operation(
                &mir::Operation::Move("xs".into()),
                &mut f,
                Some(&Type::ArrayF32),
                span,
                &mut a
            )
            .is_err()
        );
    }
    #[test]
    fn f32_index_traps_and_kind_mismatch_preserve_data() {
        let mut f = frame();
        let mut a = Arena::new(3);
        let h = a.alloc(Elements::F32(vec![F32Bits::NEG_ZERO])).unwrap();
        let owner = id(h);
        f.values = BTreeMap::from([(0, h), (1, Scalar::U32(1)), (2, Scalar::I32(0))]);
        f.places.insert(
            "xs".into(),
            Slot {
                value: Some(h),
                owned: true,
            },
        );
        let span = Span { start: 12, end: 20 };
        let error = operation(
            &mir::Operation::Index { array: 0, index: 1 },
            &mut f,
            Some(&Type::F32),
            span,
            &mut a,
        )
        .unwrap_err();
        assert!(
            matches!(error,OracleError::Semantic(EvalError::Trap{category:TrapCategory::IndexOutOfBounds,span:s}) if s==span)
        );
        f.values.insert(1, Scalar::U32(0));
        assert!(
            operation(
                &mir::Operation::StoreIndex {
                    name: "xs".into(),
                    index: 1,
                    value: 2
                },
                &mut f,
                None,
                span,
                &mut a
            )
            .is_err()
        );
        assert_eq!(bits(&a, owner), [0x80000000]);
        assert_eq!(a.live, 1);
        assert!(
            operation(
                &mir::Operation::Array(vec![2]),
                &mut f,
                Some(&Type::ArrayF32),
                span,
                &mut a
            )
            .is_err()
        );
        assert_eq!(a.live, 1);
        assert_eq!(a.slots.len(), 1);
    }
    #[test]
    fn f32_allocation_failure_and_zero_owners_are_transactional() {
        let mut a = Arena::new(1);
        let first = id(a.alloc(Elements::F32(vec![F32Bits::NAN])).unwrap());
        let before = (a.live, a.slots.len());
        assert!(matches!(
            a.alloc(Elements::F32(vec![F32Bits::POS_ZERO])),
            Err(OracleError::Tool {
                kind: ToolFailure::Allocation,
                ..
            })
        ));
        assert_eq!((a.live, a.slots.len()), before);
        assert_eq!(bits(&a, first), [F32Bits::NAN.to_bits()]);
        assert_eq!(a.release_owner(first).unwrap(), 1);
        let empty = id(a.alloc(Elements::F32(Vec::new())).unwrap());
        assert_eq!(empty, 1);
        assert_eq!(a.release_owner(empty).unwrap(), 0);
        assert_eq!(a.live, 0);
        assert!(matches!(
            a.size(crate::array_limits::MAX_ARRAY_LEN + 1),
            Err(OracleError::Semantic(EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            }))
        ));
        assert!(a.release_units(1).is_err());
    }
}
