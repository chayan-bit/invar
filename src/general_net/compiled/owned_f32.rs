//! Direct binary32 array helpers for the separate CPU ABI10 owned-numeric
//! profile frozen by main and its ABI11 continuation. ABI1–9 remain unchanged. The
//! admission/routing contract is
//! in sprint-12-f32-arrays/HOOK_CONTRACT.md.
use super::{Error, Target};
use crate::{
    front::{Span, Type},
    mir,
    numeric_contract::NumericPolicy,
};
use std::collections::BTreeSet;

/// Frozen noncolliding tag: existing integer arrays0/1, aggregate tags2..257.
pub(super) const ARRAY_KIND: u16 = 258;

pub(super) fn direct_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::I32
            | Type::U32
            | Type::F32
            | Type::Bool
            | Type::Unit
            | Type::ArrayI32
            | Type::ArrayU32
            | Type::ArrayF32
    )
}
/// Supplementary admission after the main verified-net/proof boundary. External
/// owners, nominals/host rules and other device profiles stay unsupported.
/// Callers choose ABI10 or ABI11/ABI12; continuation admission is otherwise
/// identical. Only resumable owners (ABI12) admit CUDA; Metal is always refused.
pub(super) fn validate(
    program: &mir::Program,
    entry: &str,
    target: Target,
    continuation: bool,
) -> Result<(), Error> {
    if target != Target::Cpu && !(continuation && target == Target::Cuda) {
        return Err(Error::NumericTarget);
    }
    if program.numeric_policy != NumericPolicy::STRICT_F32
        || !program.records.is_empty()
        || !program.variants.is_empty()
    {
        return Err(Error::UnsupportedType);
    }
    let entry = program
        .functions
        .iter()
        .find(|f| f.name == entry)
        .ok_or(Error::Entry)?;
    if entry.params.iter().any(|p| {
        matches!(p.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32) || !direct_type(&p.ty)
    }) {
        return Err(Error::UnsupportedType);
    }
    let names = program
        .functions
        .iter()
        .map(|f| f.name.as_str())
        .collect::<BTreeSet<_>>();
    for f in &program.functions {
        if f.foreign_oracle.is_some()
            || !direct_type(&f.result)
            || f.params.iter().any(|p| !direct_type(&p.ty))
            || f.locals.values().any(|p| !direct_type(&p.ty))
        {
            return Err(Error::UnsupportedType);
        }
        for b in &f.blocks {
            if b.params.iter().any(|(_, ty)| !direct_type(ty)) {
                return Err(Error::UnsupportedType);
            }
            for i in &b.instructions {
                if i.output.as_ref().is_some_and(|(_, ty)| !direct_type(ty)) {
                    return Err(Error::UnsupportedType);
                }
                match &i.kind {
                    mir::Operation::Call { name, .. }
                        if !names.contains(name.as_str())
                            && !matches!(
                                crate::math::f32_trig_base(name),
                                "filled_i32"
                                    | "filled_u32"
                                    | "filled_f32"
                                    | "wrapping_add"
                                    | "wrapping_mul"
                                    | "fma"
                                    | "f32_from_i32"
                                    | "f32_from_u32"
                                    | "sin"
                                    | "cos"
                                    | "sqrt"
                            ) =>
                    {
                        return Err(Error::UnsupportedType);
                    }
                    mir::Operation::Blob { .. }
                    | mir::Operation::MoveField { .. }
                    | mir::Operation::RecordRef { .. }
                    | mir::Operation::Record { .. }
                    | mir::Operation::Field { .. }
                    | mir::Operation::Variant { .. }
                    | mir::Operation::VariantTag(_)
                    | mir::Operation::VariantPayload { .. }
                    | mir::Operation::VariantTakePayload { .. }
                    | mir::Operation::DropValue(_) => return Err(Error::UnsupportedType),
                    _ => {}
                }
            }
        }
    }
    Ok(())
}
/// Needs existing gnet_alloc/gnet_valid/gnet_drop and exact scalar numeric
/// helpers. Each checked binary32 encoding occupies one existing 64-bit word.
pub(super) fn preamble(
    allocations: usize,
    elements: usize,
    aggregates: bool,
) -> Result<String, Error> {
    if allocations == 0 || allocations > 10000 || elements > crate::array_limits::MAX_LIVE_ELEMENTS
    {
        return Err(Error::Limit);
    }
    let live_flag = if aggregates {
        "(heap[d+3]&1)"
    } else {
        "heap[d+3]"
    };
    let kind_check = if aggregates {
        "heap[d+2]<0 || heap[d+2]>258"
    } else {
        "heap[d+2]!=0 && heap[d+2]!=1 && heap[d+2]!=258"
    };
    let pool = allocations.checked_mul(4).ok_or(Error::Limit)?;
    Ok(format!(
        r#"
GNET_FN bool gnet_array_f32_bits(gnet_w x) {{return x>=0 && (gnet_u)x<=4294967295ULL;}}
GNET_FN bool gnet_array_f32_descriptor(GNET_MEM const gnet_w* heap,gnet_w handle,gnet_u allocated,GNET_PRIVATE gnet_u& descriptor,GNET_PRIVATE gnet_w& status) {{
 if(allocated>{allocations} || !gnet_valid(heap,handle,allocated)){{status=7;return false;}}
 descriptor=((gnet_u)handle-1)*4;
 if(heap[descriptor+2]!={kind} || heap[descriptor]<0 || heap[descriptor+1]<0 || (gnet_u)heap[descriptor]>{elements} || (gnet_u)heap[descriptor+1]>{elements}-(gnet_u)heap[descriptor]){{status=7;return false;}}
 return true;
}}
GNET_FN gnet_w gnet_array_f32_allocate(GNET_MEM gnet_w* heap,gnet_u length,GNET_PRIVATE gnet_u& allocated,GNET_PRIVATE gnet_u& live,GNET_PRIVATE gnet_w& status) {{
 if(allocated>{allocations} || live>{elements}){{status=7;return 0;}}
 gnet_u sum=0,previous_end=0;
 for(gnet_u i=0;i<allocated;++i){{gnet_u d=i*4;
  if({live_flag}!=0 && {live_flag}!=1){{status=7;return 0;}}
  if(!{live_flag})continue;
  if(({kind_check}) || heap[d]<0 || heap[d+1]<0 || (gnet_u)heap[d]>{elements} || (gnet_u)heap[d+1]>{elements}-(gnet_u)heap[d]){{status=7;return 0;}}
  gnet_u offset=(gnet_u)heap[d],n=(gnet_u)heap[d+1];
  if(offset<previous_end || n>{elements}-sum){{status=7;return 0;}}sum+=n;previous_end=offset+n;
 }}
 if(sum!=live){{status=7;return 0;}}
 return gnet_alloc(heap,length,{kind},allocated,live,status);
}}
GNET_FN gnet_w gnet_array_f32_literal(GNET_MEM gnet_w* heap,GNET_PRIVATE const gnet_w* values,gnet_u length,GNET_PRIVATE gnet_u& allocated,GNET_PRIVATE gnet_u& live,GNET_PRIVATE gnet_w& status) {{
 if(length>{max_array}){{status=10;return 0;}}
 if(length && !values){{status=7;return 0;}}
 for(gnet_u j=0;j<length;++j)if(!gnet_array_f32_bits(values[j])){{status=7;return 0;}}
 gnet_w handle=gnet_array_f32_allocate(heap,length,allocated,live,status);if(status)return 0;
 gnet_u offset={pool}+(gnet_u)heap[((gnet_u)handle-1)*4];
 for(gnet_u j=0;j<length;++j)heap[offset+j]=values[j];return handle;
}}
GNET_FN gnet_w gnet_array_f32_filled(GNET_MEM gnet_w* heap,gnet_u length,gnet_w value,GNET_PRIVATE gnet_u& allocated,GNET_PRIVATE gnet_u& live,GNET_PRIVATE gnet_w& status) {{
 if(!gnet_array_f32_bits(value)){{status=7;return 0;}}
 gnet_w handle=gnet_array_f32_allocate(heap,length,allocated,live,status);if(status)return 0;
 gnet_u offset={pool}+(gnet_u)heap[((gnet_u)handle-1)*4];
 for(gnet_u j=0;j<length;++j)heap[offset+j]=value;return handle;
}}
GNET_FN bool gnet_array_f32_length(GNET_MEM const gnet_w* heap,gnet_w handle,gnet_u allocated,GNET_FRAME gnet_w& result,GNET_PRIVATE gnet_w& status) {{
 gnet_u d=0;if(!gnet_array_f32_descriptor(heap,handle,allocated,d,status))return false;
 result=heap[d+1];return true;
}}
GNET_FN bool gnet_array_f32_index(GNET_MEM const gnet_w* heap,gnet_w handle,gnet_u index,gnet_u allocated,GNET_FRAME gnet_w& result,GNET_PRIVATE gnet_w& status) {{
 gnet_u d=0;if(!gnet_array_f32_descriptor(heap,handle,allocated,d,status))return false;
 if(index>=(gnet_u)heap[d+1]){{status=9;return false;}}
 gnet_w x=heap[{pool}+(gnet_u)heap[d]+index];if(!gnet_array_f32_bits(x)){{status=7;return false;}}result=x;return true;
}}
GNET_FN bool gnet_array_f32_store(GNET_MEM gnet_w* heap,gnet_w handle,gnet_u index,gnet_w value,gnet_u allocated,GNET_PRIVATE gnet_w& status) {{
 gnet_u d=0;if(!gnet_array_f32_descriptor(heap,handle,allocated,d,status))return false;
 if(index>=(gnet_u)heap[d+1]){{status=9;return false;}}
 if(!gnet_array_f32_bits(value)){{status=7;return false;}}
 heap[{pool}+(gnet_u)heap[d]+index]=value;return true;
}}
"#,
        max_array = crate::array_limits::MAX_ARRAY_LEN,
        kind = ARRAY_KIND
    ))
}
/// Typed, program-specific calls for the main emitter. Slot/value expressions
/// come from the verified emitter, never from source strings or runtime opcodes.
pub(super) enum Operation<'a> {
    Literal {
        destination: &'a str,
        values: &'a [&'a str],
    },
    Filled {
        destination: &'a str,
        length: &'a str,
        value: &'a str,
    },
    Index {
        destination: &'a str,
        handle: &'a str,
        index: &'a str,
    },
    Store {
        handle: &'a str,
        index: &'a str,
        value: &'a str,
    },
    Length {
        destination: &'a str,
        handle: &'a str,
    },
}
pub(super) fn emit_operation(op: Operation<'_>, span: Span) -> String {
    let fail = format!(
        "if(status==9){{start={};end={};}} GNET_STOP;",
        span.start, span.end
    );
    match op {
        Operation::Literal {
            destination,
            values,
        } => {
            let data = if values.is_empty() {
                "nullptr".to_owned()
            } else {
                "gnet_f32_values".to_owned()
            };
            let declaration = if values.is_empty() {
                String::new()
            } else {
                format!(
                    "gnet_w gnet_f32_values[{}]={{{}}};",
                    values.len(),
                    values.join(",")
                )
            };
            format!(
                "{{{declaration}{destination}=gnet_array_f32_literal(heap,{data},{},allocated,live,status);if(status){{GNET_STOP;}}}}",
                values.len()
            )
        }
        Operation::Filled {
            destination,
            length,
            value,
        } => format!(
            "{{{destination}=gnet_array_f32_filled(heap,(gnet_u){length},{value},allocated,live,status);if(status){{GNET_STOP;}}}}"
        ),
        Operation::Index {
            destination,
            handle,
            index,
        } => format!(
            "if(!gnet_array_f32_index(heap,{handle},(gnet_u){index},allocated,{destination},status)){{{fail}}}"
        ),
        Operation::Store {
            handle,
            index,
            value,
        } => format!(
            "if(!gnet_array_f32_store(heap,{handle},(gnet_u){index},{value},allocated,status)){{{fail}}}"
        ),
        Operation::Length {
            destination,
            handle,
        } => format!(
            "if(!gnet_array_f32_length(heap,{handle},allocated,{destination},status)){{GNET_STOP;}}"
        ),
    }
}

/// Route only typed direct-f32 array operations; scalar math and affine
/// move/bind/cleanup stay with their existing emitters.
pub(super) fn operation(
    w: &mut super::emit::Writer,
    p: &super::Program<'_>,
    f: usize,
    i: &mir::Instruction,
) -> Result<bool, Error> {
    if p.numeric.is_none() || p.owned.is_none() {
        return Ok(false);
    }
    use mir::Operation as O;
    let dest = || {
        i.output
            .as_ref()
            .map(|(id, _)| super::emit::value(p, f, *id))
            .transpose()?
            .ok_or(Error::State)
    };
    let v = |id| super::emit::value(p, f, id);
    let code = match &i.kind {
        O::Array(ids) if i.output.as_ref().is_some_and(|(_, t)| *t == Type::ArrayF32) => {
            let d = dest()?;
            let values = ids.iter().map(|id| v(*id)).collect::<Result<Vec<_>, _>>()?;
            let refs = values.iter().map(String::as_str).collect::<Vec<_>>();
            emit_operation(
                Operation::Literal {
                    destination: &d,
                    values: &refs,
                },
                i.span,
            )
        }
        O::Call { name, args } if name == "filled_f32" => {
            let [n, x] = args.as_slice() else {
                return Err(Error::State);
            };
            let d = dest()?;
            let n = v(*n)?;
            let x = v(*x)?;
            emit_operation(
                Operation::Filled {
                    destination: &d,
                    length: &n,
                    value: &x,
                },
                i.span,
            )
        }
        O::Index { array, index } if p.ty(f, *array)? == &Type::ArrayF32 => {
            let d = dest()?;
            let h = v(*array)?;
            let index = v(*index)?;
            emit_operation(
                Operation::Index {
                    destination: &d,
                    handle: &h,
                    index: &index,
                },
                i.span,
            )
        }
        O::StoreIndex { name, index, value }
            if p.mir.functions[f]
                .locals
                .get(name)
                .is_some_and(|l| l.ty == Type::ArrayF32) =>
        {
            let h = format!("fr.p[{}]", p.place(f, name)?);
            let index = v(*index)?;
            let x = v(*value)?;
            emit_operation(
                Operation::Store {
                    handle: &h,
                    index: &index,
                    value: &x,
                },
                i.span,
            )
        }
        O::Length(name)
            if p.mir.functions[f]
                .locals
                .get(name)
                .is_some_and(|l| l.ty == Type::ArrayF32) =>
        {
            let d = dest()?;
            let h = format!("fr.p[{}]", p.place(f, name)?);
            emit_operation(
                Operation::Length {
                    destination: &d,
                    handle: &h,
                },
                i.span,
            )
        }
        _ => return Ok(false),
    };
    w.push(&code)?;
    Ok(true)
}
