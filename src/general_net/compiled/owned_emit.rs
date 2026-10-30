//! Static array operations and heap ABI helpers for program-specific code.
use super::{
    Error, Program, Status, Target,
    emit::{Writer, value},
    fail, owned,
};
use crate::{
    front::{Mode, Type},
    mir,
};

pub(super) fn preamble(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    let Some(layout) = &p.owned else {
        return w.push(&super::resume::exit_macro(
            p,
            "GNET_STOP",
            if p.resumable {
                "gnet_finish(output,status,result,start,end,steps,transitions,max_depth,used); task->done=1; GNET_SAVE; return true"
            } else {
                "gnet_finish(output,status,result,start,end,steps,transitions,max_depth,used); return"
            },
            "gnet_stop",
        ));
    };
    let slots = layout.allocations();
    let elements = layout.elements();
    let clean = layout.cleanup_events();
    let max_array = crate::array_limits::MAX_ARRAY_LEN;
    let max_live = crate::array_limits::MAX_LIVE_ELEMENTS;
    let stop = if p.resumable {
        "task->done=1; GNET_SAVE; return true"
    } else {
        "return"
    };
    w.push(&super::resume::exit_macro(
        p,
        "GNET_STOP",
        &format!("gnet_finish(output,status,result,start,end,steps,transitions,max_depth,used); output[8]=(gnet_w)cleanup_used; output[9]=(gnet_w)allocated; {stop}"),
        "gnet_stop",
    ))?;
    if p.numeric.is_none()
        && p.aggregate.is_none()
        && p.states
            .iter()
            .filter(|state| state.reachable)
            .all(|state| {
                let function = &p.mir.functions[state.function];
                !p.handle(&function.result)
                    && function.params.iter().all(|param| !p.handle(&param.ty))
                    && function.locals.values().all(|local| !p.handle(&local.ty))
                    && p.layout[state.function]
                        .values
                        .values()
                        .all(|(_, ty)| !p.handle(ty))
            })
    {
        return Ok(());
    }
    if p.aggregate.is_some() {
        w.push("GNET_FN gnet_u gnet_release_aggregate(GNET_MEM gnet_w* heap, gnet_u descriptor, gnet_u allocated, GNET_PRIVATE gnet_w& status);")?;
    }
    let valid = if p.aggregate.is_some() {
        "(heap[((gnet_u)handle-1)*4+3]&1)==1"
    } else {
        "heap[((gnet_u)handle-1)*4+3]==1"
    };
    let dead = if p.aggregate.is_some() {
        "(heap[descriptor+3]&1)!=1"
    } else {
        "heap[descriptor+3]!=1"
    };
    let length = if p.aggregate.is_some() {
        "gnet_release_aggregate(heap,descriptor,allocated,status); if(status!=0) return false"
    } else {
        "(gnet_u)heap[descriptor+1]"
    };
    w.push(&format!(r#"
GNET_FN bool gnet_valid(GNET_MEM const gnet_w* heap, gnet_w handle, gnet_u allocated) {{
    return handle>0 && (gnet_u)handle<=allocated && {valid};
}}
GNET_FN gnet_w gnet_alloc(GNET_MEM gnet_w* heap, gnet_u length, gnet_w kind,
    GNET_PRIVATE gnet_u& allocated, GNET_PRIVATE gnet_u& live, GNET_PRIVATE gnet_w& status) {{
    if (length>{max_array} || live>{max_live} || length>{max_live}-live) {{ status=10; return 0; }}
    if (live+length>{elements}) {{ status=11; return 0; }}
    if (allocated=={slots}) {{ status=12; return 0; }}
    // Stable allocation IDs survive relocation. Forward copies are safe: each
    // live interval starts at or after its destination in allocation order.
    gnet_u cursor=0;
    for (gnet_u i=0; i<allocated; ++i) {{
        gnet_u descriptor=i*4;
        if ({dead}) continue;
        gnet_u old=(gnet_u)heap[descriptor], n=(gnet_u)heap[descriptor+1];
        for (gnet_u j=0; j<n; ++j) heap[{slots}*4+cursor+j]=heap[{slots}*4+old+j];
        heap[descriptor]=(gnet_w)cursor; cursor+=n;
    }}
    gnet_u descriptor=allocated*4;
    heap[descriptor]=(gnet_w)cursor; heap[descriptor+1]=(gnet_w)length;
    heap[descriptor+2]=kind; heap[descriptor+3]=1; ++allocated; live+=length;
    return (gnet_w)allocated;
}}
GNET_FN bool gnet_drop(GNET_MEM gnet_w* heap, GNET_MEM gnet_u* cleanup,
    gnet_w handle, gnet_u function, gnet_u place, gnet_u frame,
    gnet_u allocated, GNET_PRIVATE gnet_u& live, GNET_PRIVATE gnet_u& cleanup_used,
    GNET_PRIVATE gnet_w& status) {{
    if (handle==0) return true;
    if (!gnet_valid(heap,handle,allocated)) {{ status=7; return false; }}
    if (cleanup_used=={clean}) {{ status=13; return false; }}
    gnet_u descriptor=((gnet_u)handle-1)*4, length={length};
    gnet_u offset=cleanup_used*4;
    cleanup[offset]=function; cleanup[offset+1]=place; cleanup[offset+2]=length; cleanup[offset+3]=frame;
    ++cleanup_used; heap[descriptor+3]=0; live-=length; return true;
}}
"#))?;
    super::aggregate::preamble(w, p)
}

pub(super) fn initialize(
    w: &mut Writer,
    p: &Program<'_>,
    f: usize,
    frame: &str,
) -> Result<(), Error> {
    if p.owned.is_some() {
        for name in &p.mir.functions[f].declaration_order {
            if p.handle(&p.mir.functions[f].locals[name].ty) {
                w.push(&format!("{frame}.p[{}]=0;", p.place(f, name)?))?;
            }
        }
    }
    Ok(())
}

pub(super) fn cleanup(
    w: &mut Writer,
    p: &Program<'_>,
    f: usize,
    names: &[String],
) -> Result<(), Error> {
    if p.owned.is_none() {
        return Ok(());
    }
    for name in names {
        let local = &p.mir.functions[f].locals[name];
        if !p.handle(&local.ty) {
            continue;
        }
        let place = p.place(f, name)?;
        if !matches!(local.mode, Some(Mode::Read | Mode::Mut)) {
            w.push(&format!("if (!gnet_drop(heap,cleanup,fr.p[{place}],{f},{place},fr.id,allocated,live,cleanup_used,status)) {{ GNET_STOP; }}"))?;
        }
        w.push(&format!("fr.p[{place}]=0;"))?;
    }
    Ok(())
}

fn descriptor(w: &mut Writer, handle: &str) -> Result<(), Error> {
    w.push(&format!("if (!gnet_valid(heap,{handle},allocated)) {{ {} }} gnet_u descriptor=((gnet_u){handle}-1)*4;",fail(Status::InvalidState,None)))
}

pub(super) fn operation(
    w: &mut Writer,
    p: &Program<'_>,
    f: usize,
    i: &mir::Instruction,
) -> Result<bool, Error> {
    if p.owned.is_none() {
        return Ok(false);
    }
    if super::owned_f32::operation(w, p, f, i)? {
        return Ok(true);
    }
    use mir::Operation as O;
    let dest = i
        .output
        .as_ref()
        .map(|(id, _)| value(p, f, *id))
        .transpose()?;
    let array_output = i.output.as_ref().is_some_and(|(_, ty)| owned::array(ty));
    let pool = p.owned.as_ref().ok_or(Error::State)?.allocations() * 4;
    match &i.kind {
        O::Move(name) | O::ArrayRef { name, .. } => {
            let place = p.place(f, name)?;
            w.push(&format!("{}=fr.p[{place}];", dest.ok_or(Error::State)?))?;
            if matches!(i.kind, O::Move(_)) {
                w.push(&format!("fr.p[{place}]=0;"))?;
            }
        }
        O::Array(ids) => {
            let kind = usize::from(
                i.output
                    .as_ref()
                    .is_some_and(|(_, ty)| *ty == Type::ArrayU32),
            );
            let d = dest.ok_or(Error::State)?;
            w.push(&format!("{{ {d}=gnet_alloc(heap,{}, {kind}, allocated,live,status); if (status!=0) {{ GNET_STOP; }} gnet_u offset={pool}+(gnet_u)heap[((gnet_u){d}-1)*4];",ids.len()))?;
            for (j, id) in ids.iter().enumerate() {
                w.push(&format!("heap[offset+{j}]={};", value(p, f, *id)?))?;
            }
            w.push("}")?;
        }
        O::Call { name, args } if matches!(name.as_str(), "filled_i32" | "filled_u32") => {
            let [n, x] = args.as_slice() else {
                return Err(Error::State);
            };
            let n = value(p, f, *n)?;
            let x = value(p, f, *x)?;
            let d = dest.ok_or(Error::State)?;
            let kind = usize::from(name == "filled_u32");
            w.push(&format!("{{ {d}=gnet_alloc(heap,(gnet_u){n},{kind},allocated,live,status); if (status!=0) {{ GNET_STOP; }} gnet_u offset={pool}+(gnet_u)heap[((gnet_u){d}-1)*4]; for (gnet_u j=0; j<(gnet_u){n}; ++j) heap[offset+j]={x}; }}"))?;
        }
        O::Bind { name, value: id } | O::Store { name, value: id }
            if p.handle(&p.mir.functions[f].locals[name].ty) =>
        {
            cleanup(w, p, f, std::slice::from_ref(name))?;
            w.push(&format!(
                "fr.p[{}]={};",
                p.place(f, name)?,
                value(p, f, *id)?
            ))?;
        }
        O::Drop(names) => cleanup(w, p, f, names)?,
        O::Length(name) => {
            w.push("{")?;
            let handle = format!("fr.p[{}]", p.place(f, name)?);
            descriptor(w, &handle)?;
            w.push(&format!(
                "{}=heap[descriptor+1]; }}",
                dest.ok_or(Error::State)?
            ))?;
        }
        O::Index { array, index } => {
            w.push("{")?;
            descriptor(w, &value(p, f, *array)?)?;
            let index = value(p, f, *index)?;
            w.push(&format!("if ((gnet_u){index}>=(gnet_u)heap[descriptor+1]) {{ {} }} {}=heap[{pool}+(gnet_u)heap[descriptor]+(gnet_u){index}]; }}",fail(Status::IndexOutOfBounds,Some(i.span)),dest.ok_or(Error::State)?))?;
        }
        O::StoreIndex {
            name,
            index,
            value: id,
        } => {
            w.push("{")?;
            descriptor(w, &format!("fr.p[{}]", p.place(f, name)?))?;
            let index = value(p, f, *index)?;
            w.push(&format!("if ((gnet_u){index}>=(gnet_u)heap[descriptor+1]) {{ {} }} heap[{pool}+(gnet_u)heap[descriptor]+(gnet_u){index}]={}; }}",fail(Status::IndexOutOfBounds,Some(i.span)),value(p,f,*id)?))?;
        }
        _ if array_output => {
            return Err(Error::UnsupportedRule(
                super::coverage::operation(&i.kind).0,
            ));
        }
        _ => return Ok(false),
    }
    Ok(true)
}

pub(super) fn kernel(
    w: &mut Writer,
    p: &Program<'_>,
    target: Target,
    args: usize,
) -> Result<(), Error> {
    let (name, output, parameters, call) = if let Some(layout) = &p.owned {
        (
            owned::KERNEL_NAME,
            owned::OUTPUT_WORDS,
            ", GNET_MEM gnet_w* heap, GNET_MEM gnet_u* cleanup".to_owned(),
            format!(
                ", heap+lane*{}, cleanup+lane*{}",
                layout.heap_words(),
                layout.cleanup_words()
            ),
        )
    } else {
        (
            super::KERNEL_NAME,
            super::OUTPUT_WORDS,
            String::new(),
            String::new(),
        )
    };
    match target {
        Target::Cpu=>Ok(()),
        Target::Cuda=>w.push(&format!(r#"extern "C" __global__ void {name}(const gnet_w* input, gnet_w* output, gnet_u* trace, const gnet_u* config{parameters}) {{
    gnet_u lane=(gnet_u)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane>=config[0]) return;
    gnet_run(input+lane*{args},output+lane*{output},trace+lane*config[3]*7,config[1],config[2],config[3]{call});
}}"#)),
        Target::Metal=>{
            let buffers=if p.owned.is_some() {", device gnet_w* heap [[buffer(4)]], device gnet_u* cleanup [[buffer(5)]]"} else {""};
            w.push(&format!(r#"kernel void {name}(device const gnet_w* input [[buffer(0)]],device gnet_w* output [[buffer(1)]],device gnet_u* trace [[buffer(2)]],constant gnet_u* config [[buffer(3)]]{buffers},uint index [[thread_position_in_grid]]) {{
    gnet_u lane=(gnet_u)index;
    if (lane>=config[0]) return;
    gnet_run(input+lane*{args},output+lane*{output},trace+lane*config[3]*7,config[1],config[2],config[3]{call});
}}"#))
        }
    }
}
