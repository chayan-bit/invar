//! ABI6 nominal shells. Members are scalar words or affine integer-array IDs.
//! Layout descriptors are storage tags, not executable graph instructions.
use super::{
    Error, Program, Status,
    emit::{Writer, value},
    fail, owned,
};
use crate::{front::Type, mir};

pub const MAX_RECORD_FIELDS: usize = 16;
pub const TEMPORARY_CLEANUP_PLACE: u64 = u64::MAX;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregateForm {
    Record(Vec<Type>),
    Variant(Vec<Option<Type>>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateKind {
    id: u32,
    name: String,
    form: AggregateForm,
}
impl AggregateKind {
    pub fn id(&self) -> u32 {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn form(&self) -> &AggregateForm {
        &self.form
    }
}
#[derive(Clone, Debug)]
pub struct AggregateLayout {
    kinds: Vec<AggregateKind>,
}
impl AggregateLayout {
    pub(super) fn new(mir: &mir::Program) -> Result<Self, Error> {
        Self::build(mir, false)
    }
    pub(super) fn device(mir: &mir::Program) -> Result<Self, Error> {
        if !mir.variants.is_empty()
            || mir.records.iter().any(|r| {
                r.fields
                    .iter()
                    .any(|f| !matches!(f.ty, Type::I32 | Type::U32 | Type::F32 | Type::Bool))
            })
        {
            return Err(Error::UnsupportedType);
        }
        Self::build(mir, true)
    }
    fn build(mir: &mir::Program, numeric: bool) -> Result<Self, Error> {
        let count = mir
            .records
            .len()
            .checked_add(mir.variants.len())
            .ok_or(Error::Limit)?;
        if count > 256 {
            return Err(Error::Limit);
        }
        let mut kinds = Vec::new();
        kinds.try_reserve_exact(count).map_err(|_| Error::Limit)?;
        for record in &mir.records {
            if record.fields.len() > MAX_RECORD_FIELDS {
                return Err(Error::Limit);
            }
            if record
                .fields
                .iter()
                .any(|f| !(member(&f.ty) || numeric && f.ty == Type::F32))
            {
                return Err(Error::UnsupportedType);
            }
            let mut fields = Vec::new();
            fields
                .try_reserve_exact(record.fields.len())
                .map_err(|_| Error::Limit)?;
            fields.extend(record.fields.iter().map(|f| f.ty.clone()));
            kinds.push(AggregateKind {
                id: kinds.len() as u32 + 2,
                name: copy_name(&record.name)?,
                form: AggregateForm::Record(fields),
            });
        }
        for variant in &mir.variants {
            if variant.arms.len() > 256 {
                return Err(Error::Limit);
            }
            if variant
                .arms
                .iter()
                .filter_map(|a| a.payload.as_ref())
                .any(|t| !matches!(t, Type::I32 | Type::U32 | Type::Bool))
            {
                return Err(Error::UnsupportedType);
            }
            let mut arms = Vec::new();
            arms.try_reserve_exact(variant.arms.len())
                .map_err(|_| Error::Limit)?;
            arms.extend(variant.arms.iter().map(|a| a.payload.clone()));
            kinds.push(AggregateKind {
                id: kinds.len() as u32 + 2,
                name: copy_name(&variant.name)?,
                form: AggregateForm::Variant(arms),
            });
        }
        Ok(Self { kinds })
    }
    pub fn kinds(&self) -> &[AggregateKind] {
        &self.kinds
    }
    pub fn temporary_cleanup_place(&self) -> u64 {
        TEMPORARY_CLEANUP_PLACE
    }
    pub fn max_record_fields(&self) -> usize {
        MAX_RECORD_FIELDS
    }
    pub(super) fn accepts(&self, ty: &Type) -> bool {
        matches!(ty, Type::Named(name) if self.kinds.iter().any(|k| k.name == *name))
    }
    fn kind(&self, name: &str) -> Result<&AggregateKind, Error> {
        self.kinds
            .iter()
            .find(|k| k.name == name)
            .ok_or(Error::UnsupportedType)
    }
    fn typed(&self, ty: &Type) -> Result<&AggregateKind, Error> {
        let Type::Named(name) = ty else {
            return Err(Error::State);
        };
        self.kind(name)
    }
}
fn copy_name(name: &str) -> Result<String, Error> {
    if name.len() > 4096 {
        return Err(Error::Limit);
    }
    let mut copy = String::new();
    copy.try_reserve_exact(name.len())
        .map_err(|_| Error::Limit)?;
    copy.push_str(name);
    Ok(copy)
}
fn member(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::U32 | Type::Bool) || owned::array(ty)
}
pub(super) fn operation_supported(op: &mir::Operation) -> bool {
    matches!(
        op,
        mir::Operation::RecordRef { .. }
            | mir::Operation::Record { .. }
            | mir::Operation::Field { .. }
            | mir::Operation::MoveField { .. }
            | mir::Operation::Variant { .. }
            | mir::Operation::VariantTag(_)
            | mir::Operation::VariantPayload { .. }
            | mir::Operation::VariantTakePayload { .. }
            | mir::Operation::DropValue(_)
    )
}

pub(super) fn preamble(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    let Some(layout) = &p.aggregate else {
        return Ok(());
    };
    let pool = p.owned.as_ref().ok_or(Error::State)?.allocations() * 4;
    w.push("GNET_FN gnet_u gnet_leaf_length(GNET_MEM gnet_w* heap, gnet_w h, gnet_u allocated, GNET_PRIVATE gnet_w& status) { if (!gnet_valid(heap,h,allocated)) { status=7; return 0; } gnet_u d=((gnet_u)h-1)*4; if (heap[d+2]!=0 && heap[d+2]!=1) { status=7; return 0; } return (gnet_u)heap[d+1]; }")?;
    w.push(&format!("GNET_FN gnet_u gnet_release_aggregate(GNET_MEM gnet_w* heap, gnet_u descriptor, gnet_u allocated, GNET_PRIVATE gnet_w& status) {{ gnet_u length=(gnet_u)heap[descriptor+1], base={pool}+(gnet_u)heap[descriptor], present=(gnet_u)heap[descriptor+3]; switch(heap[descriptor+2]) {{ case 0: case 1: return length;"))?;
    if p.numeric.is_some() {
        w.push("case 258: return length;")?;
    }
    for kind in layout.kinds() {
        w.push(&format!("case {}: {{", kind.id))?;
        match &kind.form {
            AggregateForm::Record(fields) => {
                w.push(&format!(
                    "if (length!={}) {{ status=7; return 0; }}",
                    fields.len()
                ))?;
                for (i, _) in fields.iter().enumerate().filter(|(_, t)| owned::array(t)) {
                    w.push(&format!("if (present & (1ULL<<{})) {{ length+=gnet_leaf_length(heap,heap[base+{i}],allocated,status); if (status!=0) return 0; }}", i+1))?;
                }
                for (i, _) in fields.iter().enumerate().filter(|(_, t)| owned::array(t)) {
                    w.push(&format!(
                        "if (present & (1ULL<<{})) heap[((gnet_u)heap[base+{i}]-1)*4+3]=0;",
                        i + 1
                    ))?;
                }
            }
            AggregateForm::Variant(arms) => {
                w.push(&format!("if (length!=2 || ((gnet_u)heap[base]&4294967295ULL)>={}) {{ status=7; return 0; }}", arms.len()))?;
            }
        }
        w.push("return length; }")?;
    }
    w.push("default: status=7; return 0; } }")
}

fn descriptor(w: &mut Writer, handle: &str, kind: &AggregateKind) -> Result<(), Error> {
    w.push(&format!("if (!gnet_valid(heap,{handle},allocated)) {{ {} }} gnet_u descriptor=((gnet_u){handle}-1)*4; if(heap[descriptor+2]!={}) {{ {} }}", fail(Status::InvalidState,None),kind.id,fail(Status::InvalidState,None)))
}
pub(super) fn operation(
    w: &mut Writer,
    p: &Program<'_>,
    f: usize,
    i: &mir::Instruction,
) -> Result<bool, Error> {
    let Some(layout) = &p.aggregate else {
        return Ok(false);
    };
    use mir::Operation as O;
    let dest = i
        .output
        .as_ref()
        .map(|(id, _)| value(p, f, *id))
        .transpose()?;
    let pool = p.owned.as_ref().ok_or(Error::State)?.allocations() * 4;
    match &i.kind {
        O::RecordRef { name, .. } => {
            let handle = format!("fr.p[{}]", p.place(f, name)?);
            w.push("{")?;
            descriptor(
                w,
                &handle,
                layout.typed(&p.mir.functions[f].locals[name].ty)?,
            )?;
            w.push(&format!("{}={handle}; }}", dest.ok_or(Error::State)?))?;
        }
        O::Record { name, fields } => {
            let kind = layout.kind(name)?;
            let AggregateForm::Record(types) = &kind.form else {
                return Err(Error::State);
            };
            let d = dest.ok_or(Error::State)?;
            w.push(&format!("{{ {d}=gnet_alloc(heap,{}, {},allocated,live,status); if(status!=0) {{ GNET_STOP; }} gnet_u descriptor=((gnet_u){d}-1)*4, base={pool}+(gnet_u)heap[descriptor]; heap[descriptor+3]={}ULL;",types.len(),kind.id,(1u64<<(types.len()+1))-1))?;
            for (index, id) in fields {
                w.push(&format!("heap[base+{index}]={};", value(p, f, *id)?))?;
            }
            w.push("}")?;
        }
        O::Field { record, index } => {
            let kind = layout.typed(p.ty(f, *record)?)?;
            w.push("{")?;
            descriptor(w, &value(p, f, *record)?, kind)?;
            w.push(&format!("if (!((gnet_u)heap[descriptor+3] & (1ULL<<{}))) {{ {} }} {}=heap[{pool}+(gnet_u)heap[descriptor]+{index}]; }}",index+1,fail(Status::InvalidState,None),dest.ok_or(Error::State)?))?;
        }
        O::MoveField { name, index } => {
            let kind = layout.typed(&p.mir.functions[f].locals[name].ty)?;
            w.push("{")?;
            descriptor(w, &format!("fr.p[{}]", p.place(f, name)?), kind)?;
            w.push(&format!("if (!((gnet_u)heap[descriptor+3] & (1ULL<<{}))) {{ {} }} {}=heap[{pool}+(gnet_u)heap[descriptor]+{index}]; heap[descriptor+3]=(gnet_w)((gnet_u)heap[descriptor+3] & ~(1ULL<<{})); }}",index+1,fail(Status::InvalidState,None),dest.ok_or(Error::State)?,index+1))?;
        }
        O::Variant { name, tag, payload } => {
            let kind = layout.kind(name)?;
            let d = dest.ok_or(Error::State)?;
            let x = payload
                .map(|id| value(p, f, id))
                .transpose()?
                .unwrap_or_else(|| "0".into());
            w.push(&format!("{{ {d}=gnet_alloc(heap,2,{},allocated,live,status); if(status!=0) {{ GNET_STOP; }} gnet_u base={pool}+(gnet_u)heap[((gnet_u){d}-1)*4]; heap[base]={}ULL; heap[base+1]={x}; }}",kind.id,*tag as u64+if payload.is_some(){1u64<<32}else{0}))?;
        }
        O::VariantTag(id) => {
            w.push("{")?;
            descriptor(w, &value(p, f, *id)?, layout.typed(p.ty(f, *id)?)?)?;
            w.push(&format!(
                "{}=(gnet_w)((gnet_u)heap[{pool}+(gnet_u)heap[descriptor]]&4294967295ULL); }}",
                dest.ok_or(Error::State)?
            ))?;
        }
        O::VariantPayload { value: id, tag } | O::VariantTakePayload { value: id, tag } => {
            w.push("{")?;
            descriptor(w, &value(p, f, *id)?, layout.typed(p.ty(f, *id)?)?)?;
            w.push(&format!("gnet_u base={pool}+(gnet_u)heap[descriptor]; if((gnet_u)heap[base]!=(4294967296ULL+{tag})) {{ {} }} {}=heap[base+1];",fail(Status::InvalidState,None),dest.ok_or(Error::State)?))?;
            if matches!(i.kind, O::VariantTakePayload { .. }) {
                w.push(&format!("heap[base]={tag};"))?;
            }
            w.push("}")?;
        }
        O::DropValue(id) => {
            if p.handle(p.ty(f, *id)?) {
                w.push(&format!("if (!gnet_drop(heap,cleanup,{}, {f},18446744073709551615ULL,fr.id,allocated,live,cleanup_used,status)) {{ GNET_STOP; }}",value(p,f,*id)?))?;
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}
