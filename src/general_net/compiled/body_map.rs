//! P2 capsules over P1 regions. The frame and every lane use the existing
//! program-specific continuation emitter; there is no second MIR evaluator.
//!
//! Protocol v1 uses twelve retained buffers. A frame stops before the loop
//! branch, after its first header evaluation. Lane zero starts at that branch;
//! later lanes restart the header with their own induction value. Every lane
//! stops on the backedge. The frame executes the final false header itself.
//! P5 can replace the host fence between capsules; P6 chooses placement, never
//! legality. No certificate, frame image or device address is accepted from a
//! source caller. Runtime images belong exclusively to the host launcher.
use super::{emit::Writer, *};
use crate::general_net::{
    normal_form::{self, ArrayLaw, IntTy, Node, Rep, certificate},
    regions::{self, Counting, RegionId},
};

pub const KERNEL: &str = "invar_gnet_body_map_v1";
pub const RESOLVE_KERNEL: &str = "invar_gnet_body_resolve_v1";
pub const THREADS: usize = 256;
pub const MAX_GROUPS: usize = 65_535;
pub const CONTROL_WORDS: usize = 14;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Map,
    First,
    Frontier,
}

#[derive(Clone, Debug)]
pub struct Site {
    pub region: RegionId,
    pub shape: Shape,
    /// Generated continuation states, not MIR opcodes.
    pub(crate) head: u32,
    pub(crate) branch: u32,
    pub(crate) induction: usize,
    pub(crate) induction_place: bool,
    pub(crate) bound: usize,
    /// Monotone array frame slot and its certified constant flag value.
    pub(crate) flags: Vec<(usize, i64)>,
    /// Frontier carried sums (all other representations belong to P8).
    pub(crate) sums: Vec<(usize, bool)>,
}

#[derive(Debug)]
pub enum Rejection {
    Compiler(Error),
    Regions(regions::Error),
    Certificate(certificate::Rejection),
}
impl From<Error> for Rejection {
    fn from(e: Error) -> Self {
        Self::Compiler(e)
    }
}

/// One compiled native frame and its device capsules. Opaque serial regions
/// remain in the frame. Unsupported collective representations are explicit
/// misses, so their P8 owner can replace the serial continuation later.
#[derive(Debug)]
pub struct Compiled {
    pub(crate) frame: Artifact,
    pub(crate) lanes: Artifact,
    pub(crate) sites: Vec<Site>,
    pub(crate) misses: Vec<(RegionId, &'static str)>,
    pub(crate) frame_words: usize,
    pub(crate) values: usize,
    fingerprint: [u8; 32],
}
impl Compiled {
    pub fn target(&self) -> Target {
        self.lanes.target()
    }
    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }
    pub fn misses(&self) -> &[(RegionId, &'static str)] {
        &self.misses
    }
    pub fn device_source(&self) -> &str {
        self.lanes.source()
    }
    pub fn groups(lanes: usize) -> Result<usize, Error> {
        if lanes == 0 {
            return Err(Error::Limit);
        }
        Ok(lanes.div_ceil(THREADS).min(MAX_GROUPS))
    }
}

/// Re-derive both partitions and all law obligations before emitting any code.
/// The supplied form cannot add a site by editing a node or dropping a guard.
pub fn emit(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    form: &normal_form::NormalForm,
    limits: Limits,
    owned: OwnedLimits,
) -> Result<Compiled, Rejection> {
    if target == Target::Cpu {
        return Err(Error::SpeculationTarget.into());
    }
    certificate::check_all(net, proof, form).map_err(Rejection::Certificate)?;
    let partition = regions::partition(net, proof).map_err(Rejection::Regions)?;
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    let private = SpeculationLimits {
        slots: 0,
        ..Default::default()
    };
    let mut frame =
        emit_device_speculative(net, proof, entry, Target::Cuda, limits, owned, private)?;
    let mut lanes = emit_device_speculative(net, proof, entry, target, limits, owned, private)?;
    let mut p = Program::new(
        &mir,
        entry,
        limits,
        frame.owned.clone(),
        true,
        frame.aggregate.clone(),
        frame.numeric,
    )?;
    p.entry_layout = frame.entry_layout.clone();
    let mut sites = Vec::new();
    let mut misses = Vec::new();
    for function in &form.functions {
        for lp in &function.loops {
            let Node::Parallel(node) = &lp.node else {
                continue;
            };
            let Some(head) = lp.region.head else {
                return Err(Error::State.into());
            };
            let region = partition.region(lp.region).ok_or(Error::State)?;
            // A capsule must not nest another selected capsule: its whole body
            // (including calls and inner loops) runs in its lane's private frame.
            if sites.iter().any(|s: &Site| {
                s.region.function == lp.region.function
                    && partition
                        .region(s.region)
                        .is_some_and(|r| r.blocks.contains(&head))
            }) {
                continue;
            }
            let frontier = node
                .arrays
                .iter()
                .any(|(_, a)| matches!(a, ArrayLaw::Monotone { .. }));
            if node.first_exit && frontier {
                misses.push((
                    lp.region,
                    "first/frontier composition requires a collective bridge",
                ));
                continue;
            }
            let supported = node
                .arrays
                .iter()
                .all(|(_, a)| matches!(a, ArrayLaw::Own | ArrayLaw::Monotone { .. }))
                && node
                    .carried
                    .iter()
                    .all(|c| frontier && !c.scan && matches!(c.rep, Rep::LiftedSum(_)))
                && lp.certificates.iter().all(|c| c.guards.is_empty());
            if !supported {
                misses.push((
                    lp.region,
                    "collective representation or runtime guard requires P8",
                ));
                continue;
            }
            let function = &mir.functions[lp.region.function];
            let block = &function.blocks[head];
            let Some(mir::Terminator::Branch { condition, .. }) = &block.end else {
                misses.push((lp.region, "noncanonical counted header"));
                continue;
            };
            let Some(mir::Instruction {
                kind:
                    mir::Operation::Binary {
                        op: crate::front::BinaryOp::Lt,
                        left,
                        right,
                        ..
                    },
                ..
            }) = block
                .instructions
                .iter()
                .find(|i| i.output.as_ref().is_some_and(|(id, _)| id == condition))
            else {
                misses.push((lp.region, "noncanonical counted comparison"));
                continue;
            };
            let (induction, induction_place) = match &node.counting {
                Counting::For { .. } => (p.value(lp.region.function, *left)?, false),
                Counting::CountedWhile { induction } => {
                    (p.place(lp.region.function, induction)?, true)
                }
                Counting::Uncounted => {
                    misses.push((lp.region, "uncounted loop"));
                    continue;
                }
            };
            let mut flags = Vec::new();
            for (array, law) in &node.arrays {
                if matches!(law, ArrayLaw::Monotone { own_writer: true }) {
                    let value = lp
                        .certificates
                        .iter()
                        .find(|c| c.subject == normal_form::Subject::Array(array.clone()))
                        .and_then(|c| {
                            c.facts.iter().find_map(|f| match f {
                                normal_form::Fact::Store { value, .. } => value.literal(),
                                _ => None,
                            })
                        })
                        .ok_or(Error::State)?;
                    flags.push((p.place(lp.region.function, array)?, value));
                }
            }
            let sums = node
                .carried
                .iter()
                .map(|c| {
                    Ok((
                        p.place(lp.region.function, &c.var)?,
                        matches!(c.rep, Rep::LiftedSum(IntTy::U32)),
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            // Only certified iteration-local ownership is admitted by P1; the
            // private heap also covers allocations/calls without shared IDs.
            let _ = region;
            sites.push(Site {
                region: lp.region,
                shape: if node.first_exit {
                    Shape::First
                } else if frontier {
                    Shape::Frontier
                } else {
                    Shape::Map
                },
                head: p.pc(lp.region.function, head, 0)?,
                branch: p.pc(lp.region.function, head, block.instructions.len())?,
                induction,
                induction_place,
                bound: p.value(lp.region.function, *right)?,
                flags,
                sums,
            });
        }
    }
    p.body_regions = sites.clone();
    p.target = Target::Cuda;
    // The native CPU model lowers this CUDA spelling into a confined compiled
    // C++ library. Only thread zero runs the residual native frame.
    frame.source = emit::source(&p, Target::Cuda, limits.source_bytes)?;
    if !sites.is_empty() {
        frame.source.push_str(&frame_kernel(true));
    } else {
        // No region hook is needed when the entire program stays native.
        frame.source.push_str(&frame_kernel(false));
    }
    p.target = target;
    lanes.source = emit::source(&p, target, limits.source_bytes)?;
    if !sites.is_empty() {
        lanes.source.push_str(&lane_kernel(&p, target)?);
    }
    if frame.source.len() > limits.source_bytes || lanes.source.len() > limits.source_bytes {
        return Err(Error::Limit.into());
    }
    Ok(Compiled {
        frame,
        lanes,
        sites,
        misses,
        frame_words: p.bytes / 8,
        values: p.vsize,
        fingerprint: partition.fingerprint,
    })
}

pub(super) fn boundary(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    if p.body_regions.is_empty() {
        return Ok(());
    }
    w.push("if (p2!=nullptr) {")?;
    // A lane stops before the next header (or records its real early return).
    for (k, site) in p.body_regions.iter().enumerate() {
        w.push(&format!("if(p2[0]==2 && p2[1]=={} && depth==p2[2] && fr.pc=={} && p2[8]) {{ p2[6]=1; GNET_YIELD; }}", k+1, site.head))?;
        w.push(&format!(
            "if(p2[0]==2 && p2[1]=={} && depth==p2[2] && fr.pc=={})p2[8]=1;",
            k + 1,
            site.branch
        ))?;
        // An exit from a mapped callee must stop before unwinding into the
        // native caller. Its winning iteration will replay through cleanup.
        for state in &p.states {
            if state.function == site.region.function {
                let block = &p.mir.functions[state.function].blocks[state.block];
                if state.start == block.instructions.len()
                    && matches!(block.end, Some(mir::Terminator::Return { .. }))
                {
                    w.push(&format!("if(p2[0]==2 && p2[1]=={} && depth==p2[2] && fr.pc=={}){{p2[6]=2;GNET_YIELD;}}",k+1,state.pc))?;
                }
            }
        }
        let induction = if site.induction_place { "p" } else { "v" };
        w.push(&format!("if(p2[0]==1 && fr.pc=={} && fr.{}[{}]<fr.v[{}]) {{ p2[1]={};p2[2]=depth;p2[3]=(gnet_u)fr.{}[{}];p2[4]=(gnet_u)fr.v[{}];p2[6]=0; GNET_YIELD; }}",
            site.branch, induction, site.induction, site.bound, k+1, induction, site.induction, site.bound))?;
    }
    w.push("}")
}

fn frame_kernel(regions: bool) -> String {
    let extra = if regions { ",state" } else { "" };
    format!(
        r#"
struct GnetSpecSlot {{ gnet_u unused; }};
extern "C" __global__ void invar_gnet_children_v5(const gnet_w* input,gnet_w* output,
 gnet_u* trace,const gnet_u* config,gnet_w* heap,gnet_u* cleanup,GnetTask* task,
 GnetSpecSlot*,gnet_w*,gnet_u*,gnet_u*,gnet_u* state) {{
 if(threadIdx.x!=0)return;
 state[1]=0;
 gnet_resume(input,output,trace,config[0],config[1],config[2],heap,cleanup,task,config[3]{extra});
 ++state[11];
}}
"#
    )
}

fn lane_kernel(p: &Program<'_>, target: Target) -> Result<String, Error> {
    let heap = p.owned.as_ref().ok_or(Error::State)?.heap_words();
    let clean = p.owned.as_ref().ok_or(Error::State)?.cleanup_words().max(1);
    let params = "GNET_MEM const gnet_w* input,GNET_MEM gnet_w* output,GNET_MEM gnet_u* trace,GNET_CONFIG const gnet_u* config,GNET_MEM gnet_w* heaps,GNET_MEM gnet_u* cleanups,GNET_MEM GnetTask* tasks,GNET_MEM GnetTask* seed,GNET_MEM const gnet_w* initial,GNET_MEM const gnet_w* prefixes,GNET_MEM const gnet_w* active,GNET_MEM gnet_u* controls";
    let mut text = String::from(if target == Target::Metal {
        "\n#define GNET_CONFIG constant\n"
    } else {
        "\n#define GNET_CONFIG\n"
    });
    text.push_str(&format!(
        r#"
GNET_FN void gnet_p2_lane({params},gnet_u lane) {{
 GNET_MEM GnetTask* task=tasks+lane;
 GNET_MEM gnet_w* heap=heaps+lane*{heap};
 GNET_MEM gnet_u* cleanup=cleanups+lane*{clean};
 GNET_MEM gnet_u* control=controls+lane*14;
 for(gnet_u j=0;j<{heap};++j)heap[j]=initial[j];
 *task=*seed;task->steps=0;task->transitions=0;task->used=0;task->cleanup_used=0;
 task->fuel=config[1];task->transition_cap=config[2];task->trace_cap=config[3];
 control[0]=2;control[1]=config[4];control[2]=seed->depth;control[8]=0;
 GNET_FRAME GnetFrame& fr=task->frames[task->depth-1];
 switch(config[4]) {{
"#
    ));
    for (k, site) in p.body_regions.iter().enumerate() {
        let place = if site.induction_place { "p" } else { "v" };
        text.push_str(&format!(
            "case {}: {{ fr.{place}[{}]=(gnet_w)(config[5]+lane);if(lane)fr.pc={};",
            k + 1,
            site.induction,
            site.head
        ));
        // Pass 2 prefixes are compiler-owned frame slots, not algebraic guesses.
        for (j, (slot, _)) in site.sums.iter().enumerate() {
            text.push_str(&format!(
                "if(prefixes[0])fr.p[{slot}]=prefixes[1+lane*{}+{j}];",
                site.sums.len()
            ));
        }
        // ponytail: private heap snapshots and O(n^2) reconstruction; P5/P8
        // can replace storage/prefix work after the exact capsule is qualified.
        for (j, (slot, flag)) in site.flags.iter().enumerate() {
            let pool = p.owned.as_ref().ok_or(Error::State)?.allocations() * 4;
            text.push_str(&format!("if(prefixes[0]){{gnet_u d=((gnet_u)fr.p[{slot}]-1)*4;for(gnet_u v=0;v<lane;++v)if(active[v*{}+{j}]){{gnet_u at=config[5]+v;if(at<(gnet_u)heap[d+1])heap[{pool}+(gnet_u)heap[d]+at]={flag};}}}}",site.flags.len()));
        }
        text.push_str("break; }");
    }
    text.push_str(r#"default:control[6]=8;return;}
 gnet_resume(input,output+lane*10,trace+lane*config[3]*7,config[1],config[2],config[3],heap,cleanup,task,1000000,control);
 // A quantum/resource cut is unresolved and must replay in the native frame.
 if(!control[6])control[6]=task->done?2:3;
}
"#);
    let reduction = r#"
 gnet_u best=config[0];
 for(gnet_u lane=index;lane<config[0];lane+=256)if(controls[lane*14+6]!=1)best=min(best,lane);
 cuts[index]=best;
 GNET_P2_BARRIER;
 for(gnet_u gap=128;gap;gap/=2){if(index<gap)cuts[index]=min(cuts[index],cuts[index+gap]);GNET_P2_BARRIER;}
 if(index==0)controls[9]=cuts[0];
"#;
    match target {
        Target::Cuda => {
            // Avoid target library overloads: source positions are u64 words.
            let reduction = reduction
                .replace("min(best,lane)", "(best<lane?best:lane)")
                .replace(
                    "min(cuts[index],cuts[index+gap])",
                    "(cuts[index]<cuts[index+gap]?cuts[index]:cuts[index+gap])",
                );
            text.push_str(&format!(
                r#"
#define GNET_P2_BARRIER __syncthreads()
extern "C" __global__ void {RESOLVE_KERNEL}({params}) {{
 gnet_u index=threadIdx.x;__shared__ gnet_u cuts[256];
 {reduction}
}}
"#
            ));
        }
        Target::Metal => {
            let args = params
                .split(',')
                .enumerate()
                .map(|(i, p)| format!("{p} [[buffer({i})]]"))
                .collect::<Vec<_>>()
                .join(",");
            text.push_str(&format!(
                r#"
#define GNET_P2_BARRIER threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)
kernel void {RESOLVE_KERNEL}({args},uint index [[thread_index_in_threadgroup]]) {{
 threadgroup gnet_u cuts[256];
 {reduction}
}}
"#
            ));
        }
        Target::Cpu => return Err(Error::SpeculationTarget),
    }
    match target {
        Target::Cuda => {
            text.push_str(&format!(r#"extern "C" __global__ void {KERNEL}({params}) {{
 for(gnet_u lane=(gnet_u)blockIdx.x*blockDim.x+threadIdx.x;lane<config[0];lane+=(gnet_u)gridDim.x*blockDim.x)
 gnet_p2_lane(input,output,trace,config,heaps,cleanups,tasks,seed,initial,prefixes,active,controls,lane);
}}
// CPU model entry: same lane body under the deterministic native driver.
extern "C" __global__ void invar_gnet_children_v5({params}) {{
 for(gnet_u lane=threadIdx.x;lane<config[0];lane+=32)
 gnet_p2_lane(input,output,trace,config,heaps,cleanups,tasks,seed,initial,prefixes,active,controls,lane);
}}
"#));
        }
        Target::Metal => {
            let args = params
                .split(',')
                .enumerate()
                .map(|(i, p)| format!("{p} [[buffer({i})]]"))
                .collect::<Vec<_>>()
                .join(",");
            text.push_str(&format!(r#"kernel void {KERNEL}({args},uint index [[thread_position_in_grid]],uint groups [[threadgroups_per_grid]]) {{
 for(gnet_u lane=index;lane<config[0];lane+=(gnet_u)groups*256)
 gnet_p2_lane(input,output,trace,config,heaps,cleanups,tasks,seed,initial,prefixes,active,controls,lane);
}}
"#));
        }
        Target::Cpu => return Err(Error::SpeculationTarget),
    }
    Ok(text)
}
