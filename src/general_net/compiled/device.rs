//! One fixed group owns one root. Barriers separate source mutation, private
//! child execution, and completion publication; host fences separate epochs.
use super::{Error, Program, Target, emit::Writer};

pub const KERNEL_NAME: &str = "invar_gnet_children_v5";
pub const BUFFERS: usize = 12;
pub const STATE_WORDS: usize = 14;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceConfig {
    pub fuel: u64,
    pub transitions: u64,
    pub trace_events: usize,
    pub quantum: u64,
    pub rounds: u64,
    pub capacity: usize,
}
impl DeviceConfig {
    pub fn words(self) -> [u64; 6] {
        [
            self.fuel,
            self.transitions,
            self.trace_events as u64,
            self.quantum,
            self.rounds,
            self.capacity as u64,
        ]
    }
}

/// Backend-issued device and allocation identities, never fabricated addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRange {
    pub device: u64,
    pub allocation: u64,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug)]
pub struct DeviceLayout {
    target: Target,
    base: [usize; BUFFERS],
    slots: usize,
}
#[derive(Clone, Debug)]
pub struct Workset {
    config: DeviceConfig,
    bytes: [usize; BUFFERS],
    ranges: [BufferRange; BUFFERS],
    target: Target,
}
impl Workset {
    pub fn config(&self) -> DeviceConfig {
        self.config
    }
    pub fn buffer_bytes(&self) -> &[usize; BUFFERS] {
        &self.bytes
    }
    pub fn ranges(&self) -> &[BufferRange; BUFFERS] {
        &self.ranges
    }
    pub fn target(&self) -> Target {
        self.target
    }
}
impl DeviceLayout {
    pub(super) fn new(p: &Program<'_>, target: Target) -> Result<Self, Error> {
        let (scratch, _) = p.speculation.as_ref().ok_or(Error::State)?;
        let owned = p.owned.as_ref().ok_or(Error::State)?;
        let words = |n: usize| n.max(1).checked_mul(8).ok_or(Error::Limit);
        Ok(Self {
            target,
            slots: scratch.slots(),
            base: [
                words(
                    p.entry_layout
                        .as_ref()
                        .map_or(p.mir.functions[p.entry].params.len(), |e| e.input_words()),
                )?,
                80,
                8,
                48,
                words(owned.heap_words())?,
                words(owned.cleanup_words())?,
                super::ContinuationLayout::new(p.bytes).task_bytes(),
                scratch
                    .slot_bytes()
                    .checked_mul(scratch.slots())
                    .ok_or(Error::Limit)?
                    .max(8),
                words(scratch.heap_words())?,
                words(scratch.trace_words())?,
                words(scratch.cleanup_words())?,
                STATE_WORDS * 8,
            ],
        })
    }
    pub fn kernel_name(&self) -> &'static str {
        KERNEL_NAME
    }
    pub fn group_threads(&self) -> usize {
        32
    }
    pub fn groups(&self) -> usize {
        1
    }
    pub fn max_rounds(&self) -> u64 {
        64
    }
    pub fn state_words(&self) -> usize {
        STATE_WORDS
    }
    pub fn buffer_bytes(&self, config: DeviceConfig) -> Result<[usize; BUFFERS], Error> {
        if config.fuel == 0
            || config.fuel > 1_000_000
            || config.transitions > 1_000_000
            || config.trace_events > 1_000_000
            || config.quantum == 0
            || config.quantum > 1_000_000
            || config.rounds == 0
            || config.rounds > 64
            || config.capacity > self.slots
        {
            return Err(Error::Limit);
        }
        let mut bytes = self.base;
        bytes[2] = config
            .trace_events
            .checked_mul(7 * 8)
            .ok_or(Error::Limit)?
            .max(8);
        Ok(bytes)
    }
    pub fn workset(
        &self,
        config: DeviceConfig,
        ranges: [BufferRange; BUFFERS],
    ) -> Result<Workset, Error> {
        let bytes = self.buffer_bytes(config)?;
        for (i, r) in ranges.iter().enumerate() {
            if r.device == 0
                || r.device != ranges[0].device
                || r.allocation == 0
                || r.bytes < bytes[i]
                || r.offset % 8 != 0
                || r.offset.checked_add(r.bytes).is_none()
            {
                return Err(Error::State);
            }
            for other in &ranges[..i] {
                if r.allocation == other.allocation
                    && r.offset < other.offset + other.bytes
                    && other.offset < r.offset + r.bytes
                {
                    return Err(Error::State);
                }
            }
        }
        Ok(Workset {
            config,
            bytes,
            ranges,
            target: self.target,
        })
    }
}

pub(super) fn source(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    let (scratch, _) = p.speculation.as_ref().ok_or(Error::State)?;
    let owned = p.owned.as_ref().ok_or(Error::State)?;
    let l = scratch.limits();
    let heap = owned.heap_words();
    let clean = owned.cleanup_words();
    super::fault::hook(w, p.target, p.fault)?;
    let children_fault = super::fault::call(p, super::fault::Phase::Children);
    let publication_fault = super::fault::call(p, super::fault::Phase::Publication);
    // The target is passed through the emitter's macro definition, so this
    // common body is exactly the same protocol for CUDA and Metal.
    w.push(&format!(r#"
GNET_FN void gnet_device_group(GNET_MEM const gnet_w* input,GNET_MEM gnet_w* output,
    GNET_MEM gnet_u* trace,GNET_CONFIG const gnet_u* config,GNET_MEM gnet_w* heap,
    GNET_MEM gnet_u* cleanup,GNET_MEM GnetTask* task,GNET_MEM GnetSpecSlot* slots,
    GNET_MEM gnet_w* private_heap,GNET_MEM gnet_u* private_trace,
    GNET_MEM gnet_u* private_cleanup,GNET_MEM gnet_u* state,
    gnet_u index,GNET_SHARED gnet_u& root_done,GNET_SHARED gnet_u* deltas) {{
    if (config[0]==0 || config[0]>1000000 || config[1]>1000000 || config[2]>1000000 || config[3]==0 || config[3]>1000000 || config[4]==0 || config[4]>64 || config[5]>{}) {{
        if (index==0) state[12]=8; return;
    }}
    GnetSpecState spec={{}};
    if (index==0) {{
        spec.slots=slots; spec.heap=private_heap; spec.trace=private_trace; spec.cleanup=private_cleanup;
        spec.capacity=config[5]; spec.next_generation=state[1]; spec.spawned=state[2]; spec.committed=state[3];
        spec.cancelled=state[4]; spec.replayed=state[5]; spec.guard_failed=state[6];
        spec.capacity_serialized=state[7]; spec.spec_transitions=state[8]; spec.owned_replays=state[9]; spec.last_reason=state[10];
    }}
    for (gnet_u round=0;round<config[4];++round) {{
        if (index==0) {{
            root_done=gnet_resume(input,output,trace,config[0],config[1],config[2],heap,cleanup,task,config[3],&spec)?1:0;
            ++state[11];
            if (root_done) for (gnet_u lane=0;lane<spec.capacity;++lane) gnet_discard(&spec,slots[lane],8);
        }}
        // All writers are quiescent before a slot can be reused/cancelled.
        GNET_GROUP_BARRIER;
        if (root_done) break;
        if (index>0 && index<={}) {{
            gnet_u lane=index-1; deltas[lane]=0;
            if (lane<config[5]) {{
                GNET_MEM GnetSpecSlot& slot=slots[lane];
                if (slot.valid && !slot.task.done) {{
                    gnet_u before=slot.task.transitions;
                    GNET_MEM gnet_u* child_trace=private_trace; if ({}) child_trace+=lane*{}*7;
                    GNET_MEM gnet_u* child_cleanup=private_cleanup; if ({clean}) child_cleanup+=lane*{clean};
                    if (gnet_resume(nullptr,slot.output,child_trace,{},{},{},private_heap+lane*{heap},child_cleanup,&slot.task,{},nullptr)) slot.completion_generation=slot.generation;
                    deltas[lane]=slot.task.transitions-before;
                }}
            }}
        }}{children_fault}
        // Completion generation and private transcript become readable only
        // after every child writer reaches this device-memory barrier.
        GNET_GROUP_BARRIER;{publication_fault}
        if (index==0) {{ gnet_u advanced=0; for (gnet_u lane=0;lane<{};++lane) {{ spec.spec_transitions+=deltas[lane]; if (deltas[lane]) ++advanced; }} if (advanced>state[13]) state[13]=advanced; }}
    }}
    if (index==0) {{
        state[0]=spec.capacity; state[1]=spec.next_generation; state[2]=spec.spawned; state[3]=spec.committed;
        state[4]=spec.cancelled; state[5]=spec.replayed; state[6]=spec.guard_failed;
        state[7]=spec.capacity_serialized; state[8]=spec.spec_transitions; state[9]=spec.owned_replays; state[10]=spec.last_reason;
        state[12]=root_done?0:14;
    }}
}}
"#,l.slots,l.slots,l.trace_events,l.trace_events,l.fuel,l.transitions,l.trace_events,l.quantum,l.slots))?;
    Ok(())
}

pub(super) fn kernel(w: &mut Writer, target: Target) -> Result<(), Error> {
    let params = "GNET_MEM const gnet_w* input,GNET_MEM gnet_w* output,GNET_MEM gnet_u* trace,GNET_CONFIG const gnet_u* config,GNET_MEM gnet_w* heap,GNET_MEM gnet_u* cleanup,GNET_MEM GnetTask* task,GNET_MEM GnetSpecSlot* slots,GNET_MEM gnet_w* private_heap,GNET_MEM gnet_u* private_trace,GNET_MEM gnet_u* private_cleanup,GNET_MEM gnet_u* state";
    let args = "input,output,trace,config,heap,cleanup,task,slots,private_heap,private_trace,private_cleanup,state,index,root_done,deltas";
    match target {
        Target::Cuda=>w.push(&format!(r#"extern "C" __global__ void {KERNEL_NAME}({params}) {{
    gnet_u index=threadIdx.x;
    if (gridDim.x!=1 || gridDim.y!=1 || gridDim.z!=1 || blockDim.x!=32 || blockDim.y!=1 || blockDim.z!=1) {{ if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) state[12]=8; return; }}
    __shared__ gnet_u root_done; __shared__ gnet_u deltas[8];
    gnet_device_group({args});
}}"#)),
        Target::Metal=> {
            let params=params.split(',').enumerate().map(|(i,p)|format!("{p} [[buffer({i})]]")).collect::<Vec<_>>().join(",");
            w.push(&format!(r#"kernel void {KERNEL_NAME}({params},uint index [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]],uint3 groups [[threadgroups_per_grid]],uint3 size [[threads_per_threadgroup]]) {{
    if (groups.x!=1 || groups.y!=1 || groups.z!=1 || size.x!=32 || size.y!=1 || size.z!=1) {{ if (group.x==0 && group.y==0 && group.z==0 && index==0) state[12]=8; return; }}
    threadgroup gnet_u root_done; threadgroup gnet_u deltas[8];
    gnet_device_group({args});
}}"#))
        },
        Target::Cpu=>Err(Error::SpeculationTarget),
    }
}
