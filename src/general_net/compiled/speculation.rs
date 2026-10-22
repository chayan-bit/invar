//! Compiled CPU orchestration over private ABI3 child continuations. Scalar
//! transcripts commit in source order; local-owner children replay at demand.
use super::{Error, Program, emit::Writer};

#[derive(Clone, Copy, Debug)]
pub struct SpeculationLimits {
    pub slots: usize,
    pub fuel: u64,
    pub transitions: u64,
    pub trace_events: usize,
    pub quantum: u64,
    pub lookahead: usize,
    pub scratch_bytes: usize,
}
impl Default for SpeculationLimits {
    fn default() -> Self {
        Self {
            slots: 2,
            fuel: 10_000,
            transitions: 100_000,
            trace_events: 10_000,
            quantum: 32,
            lookahead: 128,
            scratch_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeculationLayout {
    limits: SpeculationLimits,
    device: bool,
    slot_bytes: usize,
    heap_words: usize,
    cleanup_words: usize,
    bytes: usize,
}
impl SpeculationLayout {
    pub(super) fn new(l: SpeculationLimits, p: &Program<'_>) -> Result<Self, Error> {
        if l.slots > 8
            || l.fuel == 0
            || l.fuel > 1_000_000
            || l.transitions > 1_000_000
            || l.trace_events > 1_000_000
            || l.quantum == 0
            || l.quantum > 1_000_000
            || l.lookahead == 0
            || l.lookahead > 256
            || l.scratch_bytes > 64 * 1024 * 1024
        {
            return Err(Error::Limit);
        }
        let owned = p.owned.as_ref().ok_or(Error::State)?;
        let slot_bytes = super::ContinuationLayout::new(p.bytes)
            .task_bytes()
            .checked_add(18 * 8)
            .ok_or(Error::Limit)?;
        let words = owned
            .heap_words()
            .checked_add(owned.cleanup_words())
            .and_then(|n| l.trace_events.checked_mul(7).and_then(|t| n.checked_add(t)))
            .ok_or(Error::Limit)?;
        let bytes = words
            .checked_mul(8)
            .and_then(|n| n.checked_add(slot_bytes))
            .and_then(|n| n.checked_mul(l.slots))
            .ok_or(Error::Limit)?;
        if bytes > l.scratch_bytes {
            return Err(Error::Limit);
        }
        Ok(Self {
            limits: l,
            device: p.device,
            slot_bytes,
            heap_words: owned.heap_words(),
            cleanup_words: owned.cleanup_words(),
            bytes,
        })
    }
    pub fn slots(&self) -> usize {
        self.limits.slots
    }
    pub fn limits(&self) -> SpeculationLimits {
        self.limits
    }
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    pub fn heap_words(&self) -> usize {
        self.heap_words * self.limits.slots
    }
    pub fn trace_words(&self) -> usize {
        self.limits.trace_events * 7 * self.limits.slots
    }
    pub fn cleanup_words(&self) -> usize {
        self.cleanup_words * self.limits.slots
    }
    pub fn scratch_bytes(&self) -> usize {
        self.bytes
    }
    pub fn entry_point(&self) -> &'static str {
        if self.device {
            super::device::KERNEL_NAME
        } else {
            "gnet_speculate"
        }
    }
}

pub(super) fn preamble(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    let Some((layout, plan)) = &p.speculation else {
        return Ok(());
    };
    w.push(r#"

struct GnetSpecSlot {
    GnetTask task; gnet_w output[10];
    gnet_u valid,generation,completion_generation,parent_frame,parent_activation,call_pc,callee,owner_replay;
};
struct GnetSpecState {
    GNET_MEM GnetSpecSlot* slots; GNET_MEM gnet_w* heap; GNET_MEM gnet_u* trace; GNET_MEM gnet_u* cleanup;
    gnet_u capacity,next_generation,spawned,committed,cancelled,replayed,guard_failed,capacity_serialized,spec_transitions,owned_replays,last_reason;
};
GNET_FN int gnet_cache(GNET_PRIVATE GnetSpecState*,GNET_FRAME GnetFrame*,GNET_PRIVATE gnet_u&,GNET_PRIVATE gnet_u&,GNET_PRIVATE gnet_u&,GNET_PRIVATE gnet_u&,GNET_PRIVATE gnet_u&,GNET_PRIVATE gnet_u&,GNET_MEM gnet_u*,gnet_u,gnet_u,gnet_u,gnet_u,gnet_u,GNET_PRIVATE gnet_w&,GNET_PRIVATE gnet_w&,GNET_PRIVATE gnet_w&);
"#)?;
    w.push(&format!(
        "static_assert(sizeof(GnetSpecSlot)=={},\"GNET speculative slot layout mismatch\");",
        layout.slot_bytes
    ))?;
    let sites: std::collections::BTreeSet<_> = plan.candidates.iter().map(|c| c.source).collect();
    for site in sites {
        w.push(&format!(
            "GNET_FN void gnet_discover_{}(GNET_FRAME const GnetFrame&,GNET_PRIVATE GnetSpecState*);",
            p.pc(site.function, site.block, site.position)?
        ))?;
    }
    Ok(())
}

pub(super) fn call(
    w: &mut Writer,
    p: &Program<'_>,
    f: usize,
    b: usize,
    i: usize,
    ret: usize,
) -> Result<(), Error> {
    let Some((_, plan)) = &p.speculation else {
        return Ok(());
    };
    let source = super::CallSite {
        function: f,
        block: b,
        position: i,
    };
    if plan.candidates.iter().any(|c| c.source == source) {
        w.push(&format!(
            "if (spec && spec->capacity) gnet_discover_{}(fr,spec);",
            p.pc(f, b, i)?
        ))?;
    }
    w.push(&format!("if (spec) {{ int cached=gnet_cache(spec,frames,depth,max_depth,steps,transitions,used,next_frame,trace,fuel,transition_cap,trace_cap,{},{ret},status,start,end); if (cached) fr.pc={}; if (cached==2) {{ GNET_STOP; }} if (cached==1) {{",p.pc(f,b,i)?,p.pc(f,b,i+1)?))?;
    super::emit::event(w, 1, f, b, i, super::coverage::Rule::Call)?;
    w.push(&format!("fr.pc={}; break; }} }}", p.pc(f, b, i + 1)?))
}

pub(super) fn source(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    let Some((layout, plan)) = &p.speculation else {
        return Ok(());
    };
    let l = layout.limits;
    let heap = layout.heap_words;
    let cleanup = layout.cleanup_words;
    w.push(r#"
GNET_FN void gnet_discard(GNET_PRIVATE GnetSpecState* spec,GNET_MEM GnetSpecSlot& slot,gnet_u reason) {
    if (!slot.valid) return;
    slot.valid=0; ++spec->cancelled; spec->last_reason=reason;
}
GNET_FN int gnet_cache(GNET_PRIVATE GnetSpecState* spec,GNET_FRAME GnetFrame* frames,GNET_PRIVATE gnet_u& depth,
    GNET_PRIVATE gnet_u& max_depth,GNET_PRIVATE gnet_u& steps,GNET_PRIVATE gnet_u& transitions,GNET_PRIVATE gnet_u& used,GNET_PRIVATE gnet_u& next_frame,
    GNET_MEM gnet_u* trace,gnet_u fuel,gnet_u transition_cap,gnet_u trace_cap,gnet_u call_pc,gnet_u ret,
    GNET_PRIVATE gnet_w& status,GNET_PRIVATE gnet_w& start,GNET_PRIVATE gnet_w& end) {
    GNET_FRAME GnetFrame& parent=frames[depth-1];
    for (gnet_u lane=0; lane<spec->capacity; ++lane) {
        GNET_MEM GnetSpecSlot& slot=spec->slots[lane];
        if (!slot.valid || slot.parent_frame!=parent.id || slot.parent_activation!=parent.activation || slot.call_pc!=call_pc) continue;
        gnet_u reason=0;
        if (!slot.task.done) reason=3;
        else if (slot.generation!=slot.completion_generation) reason=7;
        else if (slot.owner_replay) {reason=6; ++spec->owned_replays;}
        else if (slot.output[0]!=0 && slot.output[0]!=1 && slot.output[0]!=2) reason=4;
        else if (depth+slot.task.max_depth>64 || steps+slot.task.steps>fuel || transitions+slot.task.transitions>transition_cap || used+slot.task.used>trace_cap) reason=5;
        if (reason) {++spec->replayed; gnet_discard(spec,slot,reason); return 0;}
        gnet_u base=next_frame;
"#)?;
    w.push(&format!(
        "GNET_MEM gnet_u* private_trace=spec->trace; if ({}) private_trace+=lane*{}*7;",
        l.trace_events, l.trace_events
    ))?;
    w.push(r#"
        for (gnet_u i=0; i<slot.task.used; ++i) {
            for (gnet_u j=0; j<7; ++j) trace[(used+i)*7+j]=private_trace[i*7+j];
            trace[(used+i)*7+1]+=base;
        }
        used+=slot.task.used; steps+=slot.task.steps; transitions+=slot.task.transitions;
        if (depth+slot.task.max_depth>max_depth) max_depth=depth+slot.task.max_depth;
        next_frame+=slot.task.next_frame;
        ++spec->committed; slot.valid=0;
        if (slot.output[0]==0) {if (ret<sizeof(parent.v)/sizeof(parent.v[0])) parent.v[ret]=slot.output[1]; return 1;}
        // Scalar-only trap stacks contain no owner to duplicate. Materialize
        // their exact live continuation before reporting the source trap.
        for (gnet_u i=0; i<slot.task.depth; ++i) {frames[depth+i]=slot.task.frames[i]; frames[depth+i].id+=base;}
        if (slot.task.depth) frames[depth].ret=ret;
        depth+=slot.task.depth;
        status=slot.output[0]; start=slot.output[2]; end=slot.output[3]; return 2;
    }
    return 0;
}
// Projection can trap but cannot publish a source observation or debit fuel.
#undef GNET_STOP
#define GNET_STOP return false
"#)?;
    for (index, c) in plan.candidates.iter().enumerate() {
        let f = c.source.function;
        let block = &p.mir.functions[f].blocks[c.source.block];
        let crate::mir::Operation::Call { args, .. } = &block.instructions[c.target.position].kind
        else {
            return Err(Error::State);
        };
        w.push(&format!("GNET_FN bool gnet_project_{index}(GNET_FRAME const GnetFrame& original,GNET_PRIVATE gnet_w* args) {{ GnetFrame fr={{}}; gnet_w status=0,start=0,end=0;"))?;
        for id in &c.scalar_values {
            let v = p.value(f, *id)?;
            w.push(&format!("fr.v[{v}]=original.v[{v}];"))?;
        }
        for name in &c.scalar_places {
            let place = p.place(f, name)?;
            w.push(&format!("fr.p[{place}]=original.p[{place}];"))?;
        }
        for &pos in &c.projection {
            super::emit::operation(w, p, f, &block.instructions[pos])?;
        }
        for (i, id) in args.iter().enumerate() {
            w.push(&format!("args[{i}]={};", super::emit::value(p, f, *id)?))?;
        }
        w.push("return true; }")?;
    }
    let sites: std::collections::BTreeSet<_> = plan.candidates.iter().map(|c| c.source).collect();
    for site in sites {
        w.push(&format!(
            "GNET_FN void gnet_discover_{}(GNET_FRAME const GnetFrame& original,GNET_PRIVATE GnetSpecState* spec) {{",
            p.pc(site.function, site.block, site.position)?
        ))?;
        for (index, c) in plan
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.source == site)
        {
            let function = &p.mir.functions[c.callee];
            let target_pc = p.pc(c.target.function, c.target.block, c.target.position)?;
            w.push(&format!(r#"{{
    bool duplicate=false; gnet_u free_lane=spec->capacity;
    for (gnet_u lane=0; lane<spec->capacity; ++lane) {{
        GNET_MEM GnetSpecSlot& slot=spec->slots[lane];
        if (slot.valid && slot.parent_frame==original.id && slot.parent_activation==original.activation && slot.call_pc=={target_pc}) duplicate=true;
        if (!slot.valid && free_lane==spec->capacity) free_lane=lane;
    }}
    if (!duplicate) {{
        if (free_lane==spec->capacity) {{ ++spec->capacity_serialized; spec->last_reason=1; }}
        else if (spec->next_generation==18446744073709551615ULL) {{ spec->last_reason=9; }}
        else {{
            gnet_w args[{}]={{}};
            if (!gnet_project_{index}(original,args)) {{ ++spec->guard_failed; spec->last_reason=2; }}
            else {{
                GNET_MEM GnetSpecSlot& slot=spec->slots[free_lane];
                slot.task=GnetTask{{}};
                slot.generation=++spec->next_generation; slot.completion_generation=0;
                slot.valid=1; slot.parent_frame=original.id; slot.parent_activation=original.activation;
                slot.call_pc={target_pc}; slot.callee={}; slot.owner_replay={};
                slot.task.started=1; slot.task.depth=1; slot.task.max_depth=1; slot.task.next_frame=1;
                slot.task.fuel={}; slot.task.transition_cap={}; slot.task.trace_cap={};
                GNET_FRAME GnetFrame& child=slot.task.frames[0]; child.pc={};
"#,function.params.len().max(1),c.callee,usize::from(c.replay_owners),l.fuel,l.transitions,l.trace_events,p.pc(c.callee,function.entry,0)?))?;
            super::owned_emit::initialize(w, p, c.callee, "child")?;
            for (i, param) in function.params.iter().enumerate() {
                w.push(&format!(
                    "child.p[{}]=args[{i}];",
                    p.place(c.callee, &param.name)?
                ))?;
            }
            let allocations = p.owned.as_ref().ok_or(Error::State)?.allocations();
            w.push(&format!("for (gnet_u i=0;i<{allocations};++i) spec->heap[free_lane*{heap}+i*4+3]=0; ++spec->spawned; }} }} }} }}"))?;
        }
        w.push("}")?;
    }
    if p.device {
        return super::device::source(w, p);
    }
    w.push("#include <cstdint>")?;
    w.push(
        r#"
GNET_FN bool gnet_disjoint(const void* a,gnet_u an,const void* b,gnet_u bn) {
    if (an==0 || bn==0) return true;
    if (!a || !b) return false;
    std::uintptr_t x=(std::uintptr_t)a,y=(std::uintptr_t)b;
    if (an>UINTPTR_MAX-x || bn>UINTPTR_MAX-y) return false;
    return x+an<=y || y+bn<=x;
}
"#,
    )?;
    let params = p.mir.functions[p.entry].params.len();
    let task = super::ContinuationLayout::new(p.bytes).task_bytes();
    w.push(&format!(r#"
GNET_FN gnet_w gnet_speculate(const gnet_w* input,gnet_w* output,gnet_u* trace,
    gnet_u fuel,gnet_u transition_cap,gnet_u trace_cap,gnet_w* heap,gnet_u* cleanup,
    GnetTask* task,GNET_PRIVATE GnetSpecState* spec,gnet_u quantum,gnet_u slices) {{
    if (!spec || spec->capacity>{} || slices>1000000) return 8;
    if (fuel==0 || fuel>1000000 || transition_cap>1000000 || trace_cap>1000000 || quantum==0 || quantum>1000000) {{
        gnet_resume(input,output,trace,fuel,transition_cap,trace_cap,heap,cleanup,task,quantum,nullptr);
        spec->capacity=0; ++spec->guard_failed; spec->last_reason=8;
        return 0;
    }}
    if (spec->capacity) {{
        const void* main_buffers[7]={{input,output,trace,heap,cleanup,task,spec}};
        gnet_u main_bytes[7]={{{params}*8,80,trace_cap*7*8,{heap}*8,{cleanup}*8,{task},sizeof(GnetSpecState)}};
        const void* scratch[4]={{spec->slots,spec->heap,spec->trace,spec->cleanup}};
        gnet_u scratch_bytes[4]={{spec->capacity*sizeof(GnetSpecSlot),spec->capacity*{heap}*8,spec->capacity*{}*7*8,spec->capacity*{cleanup}*8}};
        bool separate=true;
        for (gnet_u i=0;i<4;++i) {{
            if (scratch_bytes[i] && !scratch[i]) separate=false;
            for (gnet_u j=0;j<7;++j) if (!gnet_disjoint(scratch[i],scratch_bytes[i],main_buffers[j],main_bytes[j])) separate=false;
            for (gnet_u j=0;j<i;++j) if (!gnet_disjoint(scratch[i],scratch_bytes[i],scratch[j],scratch_bytes[j])) separate=false;
        }}
        if (!separate) {{ spec->capacity=0; ++spec->guard_failed; spec->last_reason=10; }}
    }}
    for (gnet_u slice=0;slice<slices;++slice) {{
        if (gnet_resume(input,output,trace,fuel,transition_cap,trace_cap,heap,cleanup,task,quantum,spec)) {{
            for (gnet_u lane=0;lane<spec->capacity;++lane) gnet_discard(spec,spec->slots[lane],8);
            return 0;
        }}
        // Oldest frontier always advanced first. Each scratch task has its
        // separate cap and finite quantum; private failure never fails main.
        for (gnet_u lane=0;lane<spec->capacity;++lane) {{
            GNET_MEM GnetSpecSlot& slot=spec->slots[lane];
            if (!slot.valid || slot.task.done) continue;
            gnet_u before=slot.task.transitions;
            gnet_u* child_trace=spec->trace; if ({}) child_trace+=lane*{}*7;
            gnet_u* child_cleanup=spec->cleanup; if ({cleanup}) child_cleanup+=lane*{cleanup};
            if (gnet_resume(nullptr,slot.output,child_trace,{},{},{},spec->heap+lane*{heap},child_cleanup,&slot.task,{},nullptr)) slot.completion_generation=slot.generation;
            spec->spec_transitions+=slot.task.transitions-before;
        }}
    }}
    return task->done?0:14;
}}
"#,l.slots,l.trace_events,l.trace_events,l.trace_events,l.fuel,l.transitions,l.trace_events,l.quantum))
}
