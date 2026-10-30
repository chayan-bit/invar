use super::super::coverage;
use super::{EVENT_WORDS, Error, MAX_FRAMES, Program, State, Status, Target, fail};
use crate::{
    front::{BinaryOp, Type, UnaryOp},
    mir,
};

pub(super) struct Writer {
    text: String,
    cap: usize,
}
impl Writer {
    pub(super) fn push(&mut self, text: &str) -> Result<(), Error> {
        let addition = text.len().checked_add(1).ok_or(Error::Limit)?;
        let length = self.text.len().checked_add(addition).ok_or(Error::Limit)?;
        if length > self.cap {
            return Err(Error::Limit);
        }
        self.text.try_reserve(addition).map_err(|_| Error::Limit)?;
        self.text.push_str(text);
        self.text.push('\n');
        Ok(())
    }
}

pub(super) fn event(
    writer: &mut Writer,
    kind: u32,
    f: usize,
    b: usize,
    pos: usize,
    rule: coverage::Rule,
) -> Result<(), Error> {
    writer.push(&format!(
        "if (!gnet_event(trace, used, trace_cap, {kind}, fr, {f}, {b}, {pos}, {})) {{ {} }}",
        rule as u32,
        fail(Status::TraceLimit, None)
    ))
}

/// Every resumable state begins with the same yield and transition checks, so
/// under `resume::compact` they run once before the state switch. A cursor
/// outside the emitted states still fails first, exactly as its `default`.
fn hoisted_prologue(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    let mut guard = String::from("switch(fr.pc) { ");
    for state in p.states.iter().filter(|state| state.reachable) {
        guard.push_str(&format!("case {}: ", state.pc));
    }
    guard.push_str(&format!(
        "break; default: {{ {} }} }}",
        fail(Status::InvalidState, None)
    ));
    w.push(&guard)?;
    transition(w, true)
}

fn transition(writer: &mut Writer, resumable: bool) -> Result<(), Error> {
    if resumable {
        writer.push("if (slice_steps==quantum) { GNET_YIELD; } ++slice_steps;")?;
    }
    writer.push(&format!(
        "if (transitions == transition_cap) {{ {} }} ++transitions;",
        fail(Status::TransitionLimit, None)
    ))
}

pub(super) fn value(program: &Program<'_>, f: usize, id: mir::ValueId) -> Result<String, Error> {
    Ok(format!("fr.v[{}]", program.value(f, id)?))
}

pub(super) fn source(program: &Program<'_>, target: Target, cap: usize) -> Result<String, Error> {
    let mut w = Writer {
        text: String::new(),
        cap: cap.min(invar_device::MAX_GNET_SOURCE_BYTES),
    };
    w.push(match target {
        Target::Cpu => "#define GNET_FN inline\n#define GNET_MEM\n#define GNET_PRIVATE\ntypedef long long gnet_w; typedef unsigned long long gnet_u;",
        Target::Cuda => "#define GNET_FN __device__ inline\n#define GNET_MEM\n#define GNET_PRIVATE\ntypedef long long gnet_w; typedef unsigned long long gnet_u;",
        Target::Metal => "#include <metal_stdlib>\nusing namespace metal;\n#define GNET_FN inline\n#define GNET_MEM device\n#define GNET_PRIVATE thread\ntypedef long gnet_w; typedef ulong gnet_u;",
    })?;
    if program.entry_layout.is_some() && target == Target::Metal {
        w.push("#pragma METAL fp contract(off)")?;
    }
    if program.device {
        w.push(if target==Target::Metal {
            "#define GNET_CONFIG constant\n#define GNET_SHARED threadgroup\n#define GNET_GROUP_BARRIER threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)"
        } else {
            "#define GNET_CONFIG\n#define GNET_SHARED\n#define GNET_GROUP_BARRIER __syncthreads()"
        })?;
    }
    w.push(if target == Target::Metal && program.resumable {
        "#define GNET_FRAME device"
    } else {
        "#define GNET_FRAME GNET_PRIVATE"
    })?;
    w.push(&format!("struct GnetFrame {{ gnet_w v[{}]; gnet_w p[{}]; gnet_u pc, ret; gnet_u id, activation; }};", program.vsize,program.psize))?;
    w.push(&format!(
        r#"
GNET_FN bool gnet_event(GNET_MEM gnet_u* trace, GNET_PRIVATE gnet_u& used,
    gnet_u cap, gnet_u kind, GNET_FRAME GnetFrame& fr,
    gnet_u function, gnet_u block, gnet_u position, gnet_u rule) {{
    if (used >= cap) return false;
    gnet_u offset = used * {EVENT_WORDS};
    trace[offset]=kind; trace[offset+1]=fr.id; trace[offset+2]=fr.activation;
    trace[offset+3]=function; trace[offset+4]=block;
    trace[offset+5]=position; trace[offset+6]=rule;
    ++used; return true;
}}
GNET_FN void gnet_finish(GNET_MEM gnet_w* output, gnet_w status, gnet_w result,
    gnet_w start, gnet_w end, gnet_u steps, gnet_u transitions,
    gnet_u max_depth, gnet_u used) {{
    output[0]=status; output[1]=result; output[2]=start; output[3]=end;
    output[4]=(gnet_w)steps; output[5]=(gnet_w)transitions;
    output[6]=(gnet_w)max_depth; output[7]=(gnet_w)used;
}}
"#
    ))?;
    super::f32::preamble(&mut w, program)?;
    super::resume::preamble(&mut w, program)?;
    super::speculation::preamble(&mut w, program)?;
    super::owned_emit::preamble(&mut w, program)?;
    if program.numeric.is_some()
        && let Some(layout) = &program.owned
    {
        w.push(&super::owned_f32::preamble(
            layout.allocations(),
            layout.elements(),
            program.aggregate.is_some(),
        )?)?;
    }
    let extra = if program.owned.is_some() {
        ", GNET_MEM gnet_w* heap, GNET_MEM gnet_u* cleanup"
    } else {
        ""
    };
    if program.resumable {
        super::resume::begin(&mut w, program)?;
    } else {
        w.push(&format!(
            r#"
GNET_FN void gnet_run(GNET_MEM const gnet_w* input, GNET_MEM gnet_w* output,
    GNET_MEM gnet_u* trace, gnet_u fuel, gnet_u transition_cap, gnet_u trace_cap{extra}) {{
    GnetFrame frames[{MAX_FRAMES}];
    unsigned int depth=0, max_depth=0;
    gnet_u steps=0, transitions=0, used=0, next_frame=1;
    gnet_w status=0, result=0, start=0, end=0;
"#
        ))?;
        if let Some(layout) = &program.owned {
            w.push(&format!("gnet_u allocated=0, live=0, cleanup_used=0; for (gnet_u i=0; i<{}; ++i) heap[i*4+3]=0;",layout.allocations()))?;
        }
        w.push(&format!("if (fuel==0 || fuel>1000000 || transition_cap>1000000 || trace_cap>1000000) {{ {} }} frames[0].ret=0; frames[0].id=0; frames[0].activation=0;",fail(Status::InvalidInput,None)))?;
    }
    super::owned_emit::initialize(&mut w, program, program.entry, "frames[0]")?;
    super::entry::initialize(&mut w, program)?;
    let entry = &program.mir.functions[program.entry];
    for (i, param) in entry.params.iter().enumerate() {
        let predicate = match param.ty {
            Type::I32 => format!("input[{i}] < -2147483648 || input[{i}] > 2147483647"),
            Type::F32 if program.numeric.is_some() => {
                format!("input[{i}] < 0 || (gnet_u)input[{i}] > 4294967295U")
            }
            Type::U32 => format!("input[{i}] < 0 || (gnet_u)input[{i}] > 4294967295U"),
            Type::Bool => format!("input[{i}] != 0 && input[{i}] != 1"),
            _ if program.entry_layout.is_some() && program.handle(&param.ty) => {
                format!("!gnet_valid(heap,input[{i}],allocated)")
            }
            _ => return Err(Error::UnsupportedType),
        };
        w.push(&format!(
            "if ({predicate}) {{ {} }} frames[0].p[{}]=input[{i}];",
            fail(Status::InvalidInput, None),
            program.place(program.entry, &param.name)?
        ))?;
    }
    w.push(&format!(
        "depth=1; max_depth=1; frames[0].pc={};",
        program.pc(program.entry, entry.entry, 0)?
    ))?;
    if program.resumable {
        w.push("task->started=1; task->fuel=fuel; task->transition_cap=transition_cap; task->trace_cap=trace_cap; }")?;
    }
    let compact = super::resume::compact(program);
    if program.body_regions.is_empty() && !compact {
        w.push("while (true) { GNET_FRAME GnetFrame& fr=frames[depth-1]; switch(fr.pc) {")?;
    } else {
        w.push("while (true) { GNET_FRAME GnetFrame& fr=frames[depth-1];")?;
        super::body_map::boundary(&mut w, program)?;
        if compact {
            hoisted_prologue(&mut w, program)?;
        }
        w.push("switch(fr.pc) {")?;
    }
    for state in &program.states {
        if state.reachable {
            state_source(&mut w, program, *state)?;
        }
    }
    w.push(&format!(
        "default: {{ {} }} }} }}",
        fail(Status::InvalidState, None)
    ))?;
    if compact {
        super::resume::exits(&mut w)?;
    }
    w.push("}")?;
    let args = entry.params.len();
    if program.resumable {
        // A CUDA device image launches only its group kernel; a second entry
        // point would make NVRTC compile the continuation twice.
        if !(program.device && target == Target::Cuda) {
            super::resume::kernel(&mut w, program, target, args)?;
        }
    } else {
        super::owned_emit::kernel(&mut w, program, target, args)?;
    }
    super::speculation::source(&mut w, program)?;
    if program.device {
        super::device::kernel(&mut w, target)?;
    }
    Ok(super::f32::trim_unused(w.text))
}

fn state_source(w: &mut Writer, p: &Program<'_>, state: State) -> Result<(), Error> {
    let State {
        function: f,
        block: b,
        start,
        pc,
        ..
    } = state;
    let block = &p.mir.functions[f].blocks[b];
    w.push(&format!("case {pc}: {{"))?;
    let prologue = !super::resume::compact(p);
    for (i, instruction) in block
        .instructions
        .iter()
        .enumerate()
        .skip(start)
        .take(if p.resumable { 1 } else { usize::MAX })
    {
        if prologue {
            transition(w, p.resumable)?;
        }
        let rule = coverage::operation(&instruction.kind).0;
        if rule == coverage::Rule::Tick {
            w.push(&format!(
                "if (steps == fuel) {{ {} }} ++steps;",
                fail(Status::FuelLimit, None)
            ))?;
            if p.resumable {
                w.push(&format!("fr.pc={}; break; }}", p.pc(f, b, i + 1)?))?;
                return Ok(());
            }
            continue;
        }
        event(w, 0, f, b, i, rule)?;
        if let Some(&callee) = p.calls.get(&(f, b, i)) {
            w.push(&format!(
                "if (depth=={MAX_FRAMES}) {{ {} }}",
                fail(Status::CallDepthLimit, None)
            ))?;
            let mir::Operation::Call { args, .. } = &instruction.kind else {
                return Err(Error::State);
            };
            let target = &p.mir.functions[callee];
            let ret = instruction
                .output
                .as_ref()
                .map(|(id, _)| p.value(f, *id))
                .transpose()?
                .map_or(p.vsize, |slot| slot);
            super::speculation::call(w, p, f, b, i, ret)?;
            w.push(&format!("GNET_FRAME GnetFrame& child=frames[depth]; child.pc={}; child.ret={ret}; child.id=next_frame++; child.activation=0;",
                p.pc(callee,target.entry,0)?))?;
            super::owned_emit::initialize(w, p, callee, "child")?;
            for (arg, param) in args.iter().zip(&target.params) {
                w.push(&format!(
                    "child.p[{}]={};",
                    p.place(callee, &param.name)?,
                    value(p, f, *arg)?
                ))?;
            }
            w.push(&format!(
                "fr.pc={}; ++depth; if (depth>max_depth) max_depth=depth; break; }}",
                p.pc(f, b, i + 1)?
            ))?;
            return Ok(());
        }
        operation(w, p, f, instruction)?;
        event(w, 1, f, b, i, rule)?;
        if p.resumable {
            w.push(&format!("fr.pc={}; break; }}", p.pc(f, b, i + 1)?))?;
            return Ok(());
        }
    }
    if prologue {
        transition(w, p.resumable)?;
    }
    let term = block.end.as_ref().ok_or(Error::State)?;
    let (rule, _, _) = coverage::terminator(term);
    event(w, 0, f, b, block.instructions.len(), rule)?;
    match term {
        mir::Terminator::Goto { target, args, .. } => {
            // Resolve all phi arguments before changing any target value.
            for (i, arg) in args.iter().enumerate() {
                w.push(&format!("gnet_w phi{i}={};", value(p, f, *arg)?))?;
            }
            for (i, (id, _)) in p.mir.functions[f].blocks[*target].params.iter().enumerate() {
                w.push(&format!("fr.v[{}]=phi{i};", p.value(f, *id)?))?;
            }
            event(w, 1, f, b, block.instructions.len(), rule)?;
            w.push(&format!(
                "fr.pc={}; ++fr.activation; break;",
                p.pc(f, *target, 0)?
            ))?;
        }
        mir::Terminator::Branch {
            condition, yes, no, ..
        } => {
            let c = value(p, f, *condition)?;
            event(w, 1, f, b, block.instructions.len(), rule)?;
            w.push(&format!(
                "fr.pc=({c})?{}:{}; ++fr.activation; break;",
                p.pc(f, *yes, 0)?,
                p.pc(f, *no, 0)?
            ))?;
        }
        mir::Terminator::Return {
            value: returned,
            cleanup,
            ..
        } => {
            let returned = returned
                .map(|id| value(p, f, id))
                .transpose()?
                .unwrap_or_else(|| "0".into());
            w.push(&format!("gnet_w returned={returned}; gnet_u ret=fr.ret;"))?;
            super::owned_emit::cleanup(w, p, f, cleanup)?;
            event(w, 1, f, b, block.instructions.len(), rule)?;
            w.push("--depth; if (depth==0) { result=returned; GNET_STOP; }")?;
            w.push(&format!(
                "GNET_FRAME GnetFrame& parent=frames[depth-1]; if (ret<{}) parent.v[ret]=returned;",
                p.vsize
            ))?;
            // These are statically generated call continuation cases, not
            // runtime call metadata or a graph opcode dispatcher.
            w.push("switch(parent.pc) {")?;
            for (&(cf, cb, ci), &callee) in &p.calls {
                if callee != f {
                    continue;
                }
                w.push(&format!(
                    "case {}: {{ GNET_FRAME GnetFrame& fr=parent;",
                    p.pc(cf, cb, ci + 1)?
                ))?;
                event(w, 1, cf, cb, ci, coverage::Rule::Call)?;
                w.push("break; }")?;
            }
            w.push(&format!(
                "default: {{ {} }} }} break;",
                fail(Status::InvalidState, None)
            ))?;
        }
    }
    w.push("}")
}

pub(super) fn operation(
    w: &mut Writer,
    p: &Program<'_>,
    f: usize,
    i: &mir::Instruction,
) -> Result<(), Error> {
    use mir::Operation as O;
    if super::f32::operation(w, p, f, i)?
        || super::aggregate::operation(w, p, f, i)?
        || super::owned_emit::operation(w, p, f, i)?
    {
        return Ok(());
    }
    let dest = i
        .output
        .as_ref()
        .map(|(id, _)| value(p, f, *id))
        .transpose()?;
    let expr = match &i.kind {
        O::I32(n) => n.to_string(),
        O::U32(n) => format!("(gnet_w){n}U"),
        O::Bool(n) => usize::from(*n).to_string(),
        O::Read(name) => format!("fr.p[{}]", p.place(f, name)?),
        O::Bind { name, value: id } | O::Store { name, value: id } => {
            return w.push(&format!(
                "fr.p[{}]={};",
                p.place(f, name)?,
                value(p, f, *id)?
            ));
        }
        O::Drop(_) => return Ok(()),
        O::ForNext(id) => {
            let v = value(p, f, *id)?;
            w.push(&format!(
                "if ({v} == 4294967295U) {{ {} }}",
                fail(Status::InvalidState, None)
            ))?;
            format!("{v}+1")
        }
        O::Unary {
            op,
            value: id,
            operator,
        } => {
            let v = value(p, f, *id)?;
            match op {
                UnaryOp::Not => format!("!{v}"),
                UnaryOp::Neg => {
                    w.push(&format!(
                        "if ({v} == -2147483648) {{ {} }}",
                        fail(Status::IntegerOverflow, Some(*operator))
                    ))?;
                    format!("-{v}")
                }
            }
        }
        O::Binary { .. } => return binary(w, p, f, i, dest.ok_or(Error::State)?),
        O::Call { name, args } => {
            let [a, b] = args.as_slice() else {
                return Err(Error::State);
            };
            let op = match name.as_str() {
                "wrapping_add" => "+",
                "wrapping_mul" => "*",
                _ => return Err(Error::UnsupportedCall),
            };
            format!(
                "(gnet_w)((unsigned int)((gnet_u){} {op} (gnet_u){}))",
                value(p, f, *a)?,
                value(p, f, *b)?
            )
        }
        _ => return Err(Error::UnsupportedRule(coverage::operation(&i.kind).0)),
    };
    w.push(&format!("{}={expr};", dest.ok_or(Error::State)?))
}

fn binary(
    w: &mut Writer,
    p: &Program<'_>,
    f: usize,
    instruction: &mir::Instruction,
    dest: String,
) -> Result<(), Error> {
    use BinaryOp as B;
    let mir::Operation::Binary {
        op,
        left,
        right,
        operator: span,
    } = instruction.kind
    else {
        return Err(Error::State);
    };
    let a = value(p, f, left)?;
    let b = value(p, f, right)?;
    let unsigned = p.ty(f, left)? == &Type::U32;
    let operator = match op {
        B::Add => "+",
        B::Sub => "-",
        B::Mul => "*",
        B::Div => "/",
        B::Rem => "%",
        B::Eq => "==",
        B::Ne => "!=",
        B::Lt => "<",
        B::Le => "<=",
        B::Gt => ">",
        B::Ge => ">=",
        B::And => "&&",
        B::Or => "||",
    };
    if matches!(op, B::Div | B::Rem) {
        w.push(&format!(
            "if ({b}==0) {{ {} }}",
            fail(Status::DivisionByZero, Some(span))
        ))?;
    }
    if matches!(op, B::Add | B::Sub | B::Mul | B::Div | B::Rem) {
        let expr = if unsigned && op == B::Mul {
            format!("(gnet_u){a} * (gnet_u){b}")
        } else {
            format!("{a} {operator} {b}")
        };
        w.push(&format!(
            "{{ {} wide={expr};",
            if unsigned && op == B::Mul {
                "gnet_u"
            } else {
                "gnet_w"
            }
        ))?;
        let range = if unsigned {
            if op == B::Mul {
                "wide>4294967295U"
            } else {
                "wide<0 || wide>4294967295U"
            }
        } else {
            "wide<(-2147483648) || wide>2147483647"
        };
        w.push(&format!(
            "if ({range}) {{ {} }} {dest}=(gnet_w)wide; }}",
            fail(Status::IntegerOverflow, Some(span))
        ))
    } else {
        w.push(&format!("{dest}=({a} {operator} {b});"))
    }
}
