//! Per-root continuation ABI. A host worklist owns each task exclusively and
//! schedules successive quanta only after the previous submission is fenced.
use super::{Error, MAX_FRAMES, Program, Target, emit::Writer};

const TASK_COUNTERS: usize = 18;
pub const KERNEL_NAME: &str = "invar_gnet_resume_v3";
/// ABI12 `cuda-owned-numeric-continuation-v1`; never shared with ABI3 caches.
pub const OWNED_NUMERIC_KERNEL_NAME: &str = "invar_gnet_owned_numeric_resume_v1";
#[derive(Clone, Debug)]
pub struct ContinuationLayout {
    task_bytes: usize,
    kernel: &'static str,
}
impl ContinuationLayout {
    pub(super) fn new(frame_bytes: usize) -> Self {
        Self {
            task_bytes: frame_bytes + TASK_COUNTERS * 8,
            kernel: KERNEL_NAME,
        }
    }
    /// Same task bytes as ABI11; only the device kernel symbol differs.
    pub(super) fn owned_numeric_cuda(frame_bytes: usize) -> Self {
        Self {
            kernel: OWNED_NUMERIC_KERNEL_NAME,
            ..Self::new(frame_bytes)
        }
    }
    pub fn task_bytes(&self) -> usize {
        self.task_bytes
    }
    pub fn kernel_name(&self) -> &'static str {
        self.kernel
    }
    pub fn config_words(&self) -> usize {
        5
    }
    pub fn max_roots(&self) -> usize {
        1024
    }
    pub fn max_slices(&self) -> usize {
        1_000_000
    }
}

pub(super) fn preamble(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    if !p.resumable {
        return Ok(());
    }
    w.push(r#"
struct GnetTask {
    GnetFrame frames[64];
    gnet_u started,done,depth,max_depth,steps,transitions,used,next_frame;
    gnet_u allocated,live,cleanup_used,fuel,transition_cap,trace_cap;
    gnet_w status,result,start,end;
};
#define GNET_SAVE task->depth=depth; task->max_depth=max_depth; task->steps=steps; task->transitions=transitions; task->used=used; task->next_frame=next_frame; task->allocated=allocated; task->live=live; task->cleanup_used=cleanup_used; task->status=status; task->result=result; task->start=start; task->end=end"#)?;
    let yield_macro = exit_macro(p, "gnet_yield", "GNET_SAVE; return false");
    w.push(&format!("{yield_macro}\n"))?;
    w.push(&format!(
        "static_assert(sizeof(GnetTask)=={}, \"GNET task layout mismatch\");",
        ContinuationLayout::new(p.bytes).task_bytes()
    ))?;
    if p.numeric.is_some() && p.owned.is_some() && p.target != Target::Cpu {
        // ABI12: header-free equivalent of the ABI11 layout assertion (NVRTC
        // has no standard headers by default). `frames` is the first member,
        // so an exact frame-array size and total size pin `started` at
        // offset `frame bytes` with the 18 counters following.
        w.push(&format!(
            "static_assert(sizeof(GnetFrame)=={} && sizeof(GnetTask::frames)=={} && sizeof(GnetTask)=={}, \"GNET ABI12 layout mismatch\");",
            p.bytes / MAX_FRAMES,
            p.bytes,
            p.bytes + TASK_COUNTERS * 8
        ))?;
    } else if p.numeric.is_some() && p.owned.is_some() {
        // ABI11 only: older generated sources stay byte-identical.
        w.push(&format!(
            "#include <cstddef>\nstatic_assert(sizeof(GnetFrame)=={} && offsetof(GnetTask,started)=={} && sizeof(GnetTask)-offsetof(GnetTask,started)=={}, \"GNET ABI11 layout mismatch\");",
            p.bytes / MAX_FRAMES,
            p.bytes,
            TASK_COUNTERS * 8
        ))?;
    }
    Ok(())
}

/// CUDA continuations hoist the per-state prologue out of the state switch
/// and share one yield and one stop epilogue (`exits`). The code is the same
/// protocol; NVRTC time grew with thousands of inlined copies of both (#278).
/// Metal keeps the inline form: Metal Shading Language has no `goto`.
pub(super) fn compact(p: &Program<'_>) -> bool {
    p.resumable && p.target == Target::Cuda
}

/// `#define LABEL BODY` (the macro is the upper-case label), or under
/// `compact` a jump to the shared `label` that runs `LABEL_EXIT`.
pub(super) fn exit_macro(p: &Program<'_>, label: &str, body: &str) -> String {
    let name = label.to_ascii_uppercase();
    if compact(p) {
        format!("#define {name}_EXIT {body}\n#define {name} goto {label}")
    } else {
        format!("#define {name} {body}")
    }
}

/// The shared epilogues at the end of `gnet_resume`, reached only by `goto`.
pub(super) fn exits(w: &mut Writer) -> Result<(), Error> {
    w.push("gnet_yield: GNET_YIELD_EXIT;\ngnet_stop: GNET_STOP_EXIT;")
}

pub(super) fn begin(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    let slots = p.owned.as_ref().map_or(0, |layout| layout.allocations());
    let extra = if p.speculation.is_some() {
        ", GNET_PRIVATE GnetSpecState* spec=nullptr"
    } else {
        ""
    };
    let region = if p.body_regions.is_empty() {
        ""
    } else {
        ", GNET_MEM gnet_u* p2=nullptr"
    };
    w.push(&format!(r#"
GNET_FN bool gnet_resume(GNET_MEM const gnet_w* input,GNET_MEM gnet_w* output,
    GNET_MEM gnet_u* trace,gnet_u fuel,gnet_u transition_cap,gnet_u trace_cap,
    GNET_MEM gnet_w* heap,GNET_MEM gnet_u* cleanup,GNET_MEM GnetTask* task,gnet_u quantum{extra}{region}) {{
    if (task->done) return true;
    GNET_FRAME GnetFrame* frames=task->frames;
    gnet_u depth=task->depth,max_depth=task->max_depth,steps=task->steps;
    gnet_u transitions=task->transitions,used=task->used,next_frame=task->next_frame;
    gnet_u allocated=task->allocated,live=task->live,cleanup_used=task->cleanup_used;
    gnet_w status=task->status,result=task->result,start=task->start,end=task->end;
    gnet_u slice_steps=0;
    if (quantum==0 || quantum>1000000 || fuel==0 || fuel>1000000 || transition_cap>1000000 || trace_cap>1000000 ||
        (task->started && (task->fuel!=fuel || task->transition_cap!=transition_cap || task->trace_cap!=trace_cap))) {{ status=8; GNET_STOP; }}
    if (!task->started) {{
        depth=0; max_depth=0; steps=0; transitions=0; used=0; next_frame=1;
        allocated=0; live=0; cleanup_used=0; status=0; result=0; start=0; end=0;
        for (gnet_u i=0; i<{slots}; ++i) heap[i*4+3]=0;
        frames[0].ret=0; frames[0].id=0; frames[0].activation=0;
"#))
}

pub(super) fn kernel(
    w: &mut Writer,
    p: &Program<'_>,
    target: Target,
    args: usize,
) -> Result<(), Error> {
    let heap = p.owned.as_ref().map_or(0, |layout| layout.heap_words());
    let clean = p.owned.as_ref().map_or(0, |layout| layout.cleanup_words());
    let outputs = if p.owned.is_some() { 10 } else { 8 };
    let heap_arg = if heap == 0 {
        "heap".to_owned()
    } else {
        format!("heap+lane*{heap}")
    };
    let input = if args == 0 {
        "input".to_owned()
    } else {
        format!("input+lane*{args}")
    };
    // Only ABI12 strict-f32 owned continuations reach a device kernel with a
    // numeric policy; they get a distinct symbol.
    let name = if p.numeric.is_some() && !p.device {
        OWNED_NUMERIC_KERNEL_NAME
    } else {
        KERNEL_NAME
    };
    let cleanup = if clean == 0 {
        "cleanup".to_owned()
    } else {
        format!("cleanup+lane*{clean}")
    };
    let call = format!(
        "gnet_resume({input},output+lane*{outputs},(config[3]?trace+lane*config[3]*7:trace),config[1],config[2],config[3],{heap_arg},{cleanup},tasks+lane,config[4]);"
    );
    match target {
        Target::Cpu=>w.push(&format!(r#"
// Finite round-robin worklist over exclusively owned roots. A yielded root
// re-enters at its generated continuation. Completion publishes only that root.
GNET_FN gnet_w gnet_worklist(const gnet_w* input,gnet_w* output,gnet_u* trace,
    gnet_u roots,gnet_u fuel,gnet_u transition_cap,gnet_u trace_cap,
    gnet_w* heap,gnet_u* cleanup,GnetTask* tasks,gnet_u quantum,gnet_u slices,
    gnet_u& submitted,gnet_u& completed) {{
    submitted=0; completed=0;
    if (roots>1024 || slices>1000000 || quantum==0 || quantum>1000000 || fuel==0 || fuel>1000000 || transition_cap>1000000 || trace_cap>1000000) return 8;
    for (gnet_u lane=0; lane<roots; ++lane) if (tasks[lane].done) ++completed;
    gnet_u cursor=0;
    while (completed<roots) {{
        gnet_u lane=cursor; cursor=(cursor+1)%roots;
        if (tasks[lane].done) continue;
        if (submitted==slices) return 14;
        ++submitted;
        if ({call}) ++completed;
    }}
    return 0;
}}
"#,call=call.trim_end_matches(';').replace("config[3]","trace_cap").replace("config[1]","fuel").replace("config[2]","transition_cap").replace("config[4]","quantum"))),
        Target::Cuda=>w.push(&format!(r#"extern "C" __global__ void {name}(const gnet_w* input,gnet_w* output,gnet_u* trace,const gnet_u* config,gnet_w* heap,gnet_u* cleanup,GnetTask* tasks) {{
    gnet_u lane=(gnet_u)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane>=config[0]) return;
    {call}
}}"#)),
        Target::Metal=>w.push(&format!(r#"kernel void {KERNEL_NAME}(device const gnet_w* input [[buffer(0)]],device gnet_w* output [[buffer(1)]],device gnet_u* trace [[buffer(2)]],constant gnet_u* config [[buffer(3)]],device gnet_w* heap [[buffer(4)]],device gnet_u* cleanup [[buffer(5)]],device GnetTask* tasks [[buffer(6)]],uint index [[thread_position_in_grid]]) {{
    gnet_u lane=(gnet_u)index;
    if (lane>=config[0]) return;
    {call}
}}"#)),
    }
}
