//! Task-region driver: arena nodes, the two schedules (staged single-group
//! scheduler plus run kernel, or one persistent cascade kernel), the serial
//! frame machine and the target dialects (CPU model, CUDA, Metal).
//!
//! Buffers (32-bit words): 0 entry arguments, 1 state, 2 arena nodes,
//! 3 lists (staged: two scheduler lists and the task list, `arena` words
//! each; unused by the cascade).
//!
//! State words: 0 initialized, 1 done (1 finished, 2 cascade stalled),
//! 2 status, 3 value, 4 span start, 5 span end, 6 arena top, 7 current list,
//! 8-9 list counts, 10-11 ready counts, 12 task count, 13-15 run threadgroups
//! (x, y, z), 16 scheduler rounds, 17 tasks run (cascade: serial runs),
//! 18 scheduler passes, 19 arena high water, 20-22 next scheduler threadgroups
//! (zero once the root is done, so a runner that dispatches later passes
//! indirectly pays nothing for idle ones), 24 cascade claims, 25 call
//! sequence (host-written, nonzero; tags the cascade's published nodes so a
//! previous call's node words are never mistaken for this call's), 26 cascade
//! pending nodes minus one (published children minus finished claims, wrapping;
//! the root counts from the start, so 0xffffffff means nothing is pending).
//!
//! Node words: 0 state (cascade: the call's tag once published), 1 function,
//! 2 pc, 3 parent, 4 sibling slot, 5 call depth, 6 children, 7 children done,
//! 8 first child, 9-12 unused, 13-15 pending sibling-argument failure,
//! 16 ret, 17 cancelled (an earlier sibling failed, so this node is never
//! observed), 18.. frame, then one result slot per sibling (status, value,
//! span start, span end) that the finished child writes into its parent.

use std::fmt::Write;

use super::{Mutation, Plan, Program, Refusal, Schedule, lower};
use crate::general_net::compiled::Target;

pub const BUFFERS: usize = 4;
pub const STATE_WORDS: usize = 32;
const HEADER_WORDS: usize = 18;
const RESULT_WORDS: usize = 4;
/// Kernel names of the task ABI.
pub const SCHED_KERNEL: &str = "gt_sched";
pub const RUN_KERNEL: &str = "gt_run";
pub const CASCADE_KERNEL: &str = "gt_cascade";

/// Emitted task-region source for one target and plan.
#[derive(Clone, Debug)]
pub struct Source {
    pub text: String,
    pub target: Target,
    pub plan: Plan,
    /// Word count of each buffer.
    pub words: [usize; BUFFERS],
}

/// Word count of each buffer for `program` under `plan`.
pub fn buffer_words(program: &Program, plan: &Plan) -> [usize; BUFFERS] {
    let arena = plan.arena as usize;
    [
        program.params.max(1),
        STATE_WORDS,
        arena * node_words(program),
        arena * 3,
    ]
}

/// Words of one arena node: the header, the frame, then one result slot
/// (status, value, span start, span end) per sibling, which the children
/// write so a resumed join reads only its own node.
fn node_words(program: &Program) -> usize {
    HEADER_WORDS + program.frame_words + RESULT_WORDS * program.siblings
}

const COMMON: &str = r#"
#define GT_R_RETURN 0u
#define GT_R_CALL 1u
#define GT_R_FORK 2u
#define GT_R_FAIL 3u
#define GT_R_YIELD 4u
#define GT_READY 1u
#define GT_WAIT 2u
#define GT_TASK 3u
#define GT_DONE 4u
#define GT_NONE 0xffffffffu
struct GtSt{uint status;uint lo;uint hi;uint steps;};
struct GtFrame{uint fn;uint pc;int ret;int v[GT_FW];};
struct GtIo{uint k;uint callee[GT_K];int args[GT_K*GT_P];int res[GT_K];int value;};
GT_FN bool gt_add_i(int a,int b,GT_TH int& r){gt_i64 w=(gt_i64)a+(gt_i64)b;r=(int)w;return (gt_i64)r==w;}
GT_FN bool gt_sub_i(int a,int b,GT_TH int& r){gt_i64 w=(gt_i64)a-(gt_i64)b;r=(int)w;return (gt_i64)r==w;}
GT_FN bool gt_mul_i(int a,int b,GT_TH int& r){gt_i64 w=(gt_i64)a*(gt_i64)b;r=(int)w;return (gt_i64)r==w;}
GT_FN bool gt_add_u(uint a,uint b,GT_TH uint& r){r=a+b;return r>=a;}
GT_FN bool gt_sub_u(uint a,uint b,GT_TH uint& r){r=a-b;return a>=b;}
GT_FN bool gt_mul_u(uint a,uint b,GT_TH uint& r){gt_u64 w=(gt_u64)a*(gt_u64)b;r=(uint)w;return (w>>32)==0ull;}
"#;

const DRIVER: &str = r#"
GT_FN GT_DEV uint* gt_node(GT_DEV uint* nodes,uint n){return nodes+(gt_u64)n*(gt_u64)GT_NODE;}
GT_FN void gt_load(GT_DEV uint* b,GT_TH GtFrame& fr){fr.fn=b[1];fr.pc=b[2];fr.ret=(int)b[16];for(uint i=0u;i<GT_FW;++i)fr.v[i]=(int)b[GT_HDR+i];}
GT_FN void gt_save(GT_DEV uint* b,GT_TH GtFrame& fr){b[1]=fr.fn;b[2]=fr.pc;b[16]=(uint)fr.ret;for(uint i=0u;i<GT_FW;++i)b[GT_HDR+i]=(uint)fr.v[i];}
// Counters `cn`: 0-1 list counts, 2-3 ready counts, 4 task count, 5 arena
// top. The staged scheduler keeps them in group memory; its run kernel passes
// state+8, whose words match 0-4 (it never allocates, so never touches 5).
// Push the `k` consecutive nodes `first..first+k` with one atomic per counter.
template<typename C> GT_FN void gt_push(C cn,GT_DEV uint* lists,uint list,uint first,uint k,bool ready){uint i=gt_atomic_add(cn+list,k);for(uint j=0u;j<k;++j)lists[list*GT_ARENA+i+j]=first+j;if(ready)gt_atomic_add(cn+2u+list,k);}
template<typename C> GT_FN void gt_task(C cn,GT_DEV uint* lists,uint n){uint i=gt_atomic_add(cn+4u,1u);lists[2u*GT_ARENA+i]=n;}
// Whether `n` or one of its next `levels` ancestors was cancelled. Cancellation
// only saves work: a CANCELLED status always sits under a flagged node, which
// follows a failed sibling the join reports first, so a shallow check is exact.
GT_FN bool gt_cancelled(GT_DEV uint* nodes,uint n,uint levels){
 for(uint d=0u;d<=levels&&n!=GT_NONE;++d){GT_DEV uint* b=gt_node(nodes,n);if(gt_load_word(b+17u)!=0u)return true;n=b[3];}
 return false;
}
// Finish node `n`; return its parent when this was the parent's last child
// (the join is ready), else GT_NONE. The fences order the result words before
// the join counter (release) and the siblings' results after it (acquire).
GT_FN uint gt_finish(GT_DEV uint* state,GT_DEV uint* nodes,uint n,uint status,int value,uint lo,uint hi){
 GT_DEV uint* b=gt_node(nodes,n);uint parent=b[3];uint slot=b[4];b[0]=GT_DONE;
 if(parent==GT_NONE){state[2]=status;state[3]=(uint)value;state[4]=lo;state[5]=hi;GT_FENCE;gt_store_word(state+1u,1u);return GT_NONE;}
 GT_DEV uint* q=gt_node(nodes,parent);
 GT_DEV uint* r=q+GT_RES+4u*slot;r[0]=status;r[1]=(uint)value;r[2]=lo;r[3]=hi;
 // A failed child decides its parent's failure: later siblings are never observed.
 if(status!=0u){uint first=q[8];for(uint s=GT_CANCEL_FROM;s<GT_CANCEL_TO;++s)gt_store_word(gt_node(nodes,first+s)+17u,1u);}
 GT_FENCE;
 if(gt_atomic_add(q+7u,1u)+1u==GT_JOIN_AT){GT_FENCE;return parent;}
 return GT_NONE;
}
GT_FN uint gt_serial(GT_DEV uint* state,GT_DEV uint* nodes,uint n){
 GT_DEV uint* b=gt_node(nodes,n);
 if(gt_cancelled(nodes,n,GT_MAX_DEPTH))return gt_finish(state,nodes,n,8u,0,0u,0u);
 uint base=b[5];
 GtFrame stack[GT_MAX_DEPTH];GtFrame cur;gt_load(b,cur);
 GtIo io={};GtSt st={0u,0u,0u,0u};
 uint sp=0u;
 for(;;){
  uint r=gt_step(cur,io,st,false,base+sp);
  if(r==GT_R_RETURN){if(sp==0u)return gt_finish(state,nodes,n,0u,io.value,0u,0u);int v=io.value;--sp;cur=stack[sp];cur.ret=v;continue;}
  if(r==GT_R_YIELD)continue;
  if(r==GT_R_CALL){
   if(base+sp>=GT_MAX_DEPTH)return gt_finish(state,nodes,n,4u,0,0u,0u);
   if(++st.steps>GT_BUDGET)return gt_finish(state,nodes,n,5u,0,0u,0u);
   if((st.steps&255u)==0u&&gt_cancelled(nodes,n,GT_MAX_DEPTH))return gt_finish(state,nodes,n,8u,0,0u,0u);
   stack[sp]=cur;++sp;cur.fn=io.callee[0];cur.pc=0u;cur.ret=0;
   for(uint q=0u;q<GT_P;++q)cur.v[q]=io.args[q];
   continue;
  }
  return gt_finish(state,nodes,n,st.status!=0u?st.status:7u,0,st.lo,st.hi);
 }
}
// Advance node `n` until it returns, fails or spawns children; return the
// parent whose join became ready, else GT_NONE. `GT_ALLOC`, `GT_PUBLISH` and
// `GT_FALLBACK` are the schedule's arena allocation, child publication and
// arena-full fallback (staged: group counters and lists; cascade: device
// counters, tagged node words and an in-lane serial run).
template<typename C> GT_FN uint gt_advance(GT_DEV uint* state,C cn,GT_DEV uint* nodes,GT_DEV uint* lists,uint next,uint n){
 GT_DEV uint* b=gt_node(nodes,n);
 if(gt_cancelled(nodes,n,GT_CANCEL_LEVELS))return gt_finish(state,nodes,n,8u,0,0u,0u);
 uint depth=b[5];
 GtFrame fr;gt_load(b,fr);
 GtIo io={};GtSt st={0u,0u,0u,0u};
 if(b[0]==GT_WAIT){
  uint k=b[6];
  GT_EAGER_PENDING
  for(uint j=0u;j<k;++j){GT_DEV uint* c=b+GT_RES+4u*(GT_CHILD_ORDER);if(c[0]!=0u)return gt_finish(state,nodes,n,c[0],0,c[2],c[3]);}
  if(b[13]!=0u)return gt_finish(state,nodes,n,b[13],0,b[14],b[15]);
  for(uint j=0u;j<k;++j)io.res[j]=(int)b[GT_RES+4u*j+1u];
  fr.ret=io.res[0];
 }
 GtFrame good=fr;
 for(;;){
  uint r=gt_step(fr,io,st,true,depth);
  if(r==GT_R_YIELD){good=fr;continue;}
  if(r==GT_R_RETURN)return gt_finish(state,nodes,n,0u,io.value,0u,0u);
  if(r!=GT_R_CALL&&r!=GT_R_FORK)return gt_finish(state,nodes,n,st.status!=0u?st.status:7u,0,st.lo,st.hi);
  uint k=io.k;
  if(k==0u)return gt_finish(state,nodes,n,st.status,0,st.lo,st.hi);
  if(depth>=GT_MAX_DEPTH)return gt_finish(state,nodes,n,4u,0,0u,0u);
  uint base=GT_ALLOC(k);
  if(base>GT_ARENA-k){gt_save(b,GT_FALLBACK_FRAME);GT_FALLBACK}
  for(uint j=0u;j<k;++j){
   GT_DEV uint* c=gt_node(nodes,base+j);
   c[0]=0u;c[1]=io.callee[j];c[2]=0u;c[3]=n;c[4]=j;c[5]=GT_CHILD_DEPTH;
   for(uint i=6u;i<GT_HDR;++i)c[i]=0u;
   for(uint q=0u;q<GT_FW;++q)c[GT_HDR+q]=q<GT_P?(uint)io.args[j*GT_P+q]:0u;
  }
  b[6]=k;b[7]=0u;b[8]=base;b[13]=st.status;b[14]=st.lo;b[15]=st.hi;gt_save(b,fr);b[0]=GT_WAIT;
  GT_PUBLISH
  return GT_NONE;
 }
}
"#;

const SCHED_BODY: &str = r#"
GT_PHASE_BEGIN
 if(tid==0u){
  if(state[0]==0u){
   GT_DEV uint* b=gt_node(nodes,0u);
   for(uint i=0u;i<GT_NODE;++i)b[i]=0u;
   b[0]=GT_READY;b[1]=GT_ENTRY;b[3]=GT_NONE;b[5]=1u;
   for(uint q=0u;q<GT_ENTRY_P;++q)b[GT_HDR+q]=(uint)params[q];
   state[6]=1u;state[7]=0u;state[8]=1u;state[9]=0u;state[10]=1u;state[11]=0u;lists[0]=0u;state[0]=1u;
  }
  for(uint i=0u;i<4u;++i)cnt[i]=state[8u+i];
  cnt[4]=0u;cnt[5]=state[6];
  uint cur=state[7];sh[1]=cur;sh[0]=state[1]!=0u?0u:cnt[cur];sh[2]=cnt[2u+cur]<GT_W?1u:0u;
 }
GT_PHASE_END
while(sh[0]!=0u){
 GT_PHASE_BEGIN
  uint cur=sh[1];uint ncur=sh[0];bool expand=sh[2]!=0u;
  for(uint i=tid;i<ncur;i+=GT_SCHED_T){
   uint n=lists[cur*GT_ARENA+i];
   GT_DEV uint* b=gt_node(nodes,n);
   if(b[0]==GT_READY&&!expand){b[0]=GT_TASK;gt_task(cnt,lists,n);}
   else{uint p=gt_advance(state,cnt,nodes,lists,cur^1u,n);if(p!=GT_NONE)gt_push(cnt,lists,cur^1u,p,1u,false);}
  }
 GT_PHASE_END
 GT_PHASE_BEGIN
  if(tid==0u){uint cur=sh[1];cnt[cur]=0u;cnt[2u+cur]=0u;cur^=1u;sh[1]=cur;sh[0]=cnt[cur];sh[2]=cnt[2u+cur]<GT_W?1u:0u;state[16]+=1u;}
 GT_PHASE_END
}
GT_PHASE_BEGIN
 if(tid==0u){for(uint i=0u;i<5u;++i)state[8u+i]=cnt[i];state[6]=cnt[5];state[7]=sh[1];uint t=cnt[4];state[13]=(t+GT_RUN_T-1u)/GT_RUN_T;state[14]=1u;state[15]=1u;state[20]=state[1]!=0u?0u:1u;state[21]=1u;state[22]=1u;state[17]+=t;state[18]+=1u;if(state[6]>state[19])state[19]=state[6];}
GT_PHASE_END
"#;

const RUN_BODY: &str = "if(state[1]!=0u||i>=state[12])return;uint p=gt_serial(state,nodes,lists[2u*GT_ARENA+i]);if(p!=GT_NONE)gt_push(state+8u,lists,state[7],p,1u,false);";

// Staged schedule hooks: group counters, ready lists and the task list.
const STAGED_HOOKS: &str = "#define GT_CANCEL_LEVELS 1u\n#define GT_ALLOC(k) gt_atomic_add(cn+5u,k)\n#define GT_PUBLISH {for(uint j=0u;j<k;++j)gt_node(nodes,base+j)[0]=GT_READY;gt_push(cn,lists,next,base,k,true);}\n#define GT_FALLBACK {b[0]=GT_TASK;gt_task(cn,lists,n);return GT_NONE;}\n";

// Cascade schedule hooks: device counters (state 6 arena top), a node is
// published by storing the call's tag after a device fence: state 25
// sequence and READY, or TASK once the arena top reached `width` (the
// claiming lane then runs it serially, decided without another device
// round trip). An arena-full fork runs serially in the lane. Cancellation
// only saves work, so a lane checks just the node's own flag (one round trip
// fewer per node; descendants of a cancelled node finish unobserved).
const CASCADE_HOOKS: &str = "#define GT_CANCEL_LEVELS 0u\n#define GT_ALLOC(k) gt_atomic_add(state+6u,k)\n#define GT_PUBLISH {gt_atomic_add(state+26u,k);GT_FENCE;uint tag=(state[25]<<3u)|(base+k>=GT_W?GT_TASK:GT_READY);for(uint j=0u;j<k;++j)gt_store_word(gt_node(nodes,base+j),tag);}\n#define GT_FALLBACK {gt_atomic_add(state+17u,1u);return gt_serial(state,nodes,n);}\n";

// One cascade worker per SIMD group (the target's `GT_WORKER` prologue
// returns the other lanes): the step machines diverge, so nodes sharing a
// SIMD group would run one after another. Worker 0 allocates and publishes
// the root; every worker then claims arena indices in allocation order
// (state 24), waits for the claimed node's tag (a poll per loop iteration, so
// a SIMD group never spins inside a branch while a sibling owes it the node),
// and runs it: serially when the tag says TASK, else forking; a ready join is
// resumed in the worker. An index at or past the arena is never published:
// the worker only polls for the root's completion. State 26 counts pending
// nodes (minus one): the root from the start, children when they are allocated
// and before their tags are stored, and a claim is subtracted only once its
// worker is done with it. A pending node is published or being run, and a
// running lane always ends (its steps are budgeted), so an idle worker counts
// polls only while nothing is pending. `GT_POLL_LIMIT` such polls mean a claim
// that can never be published (a defect, never a result): unless the call has
// already finished, it marks the call stalled.
const CASCADE_BODY: &str = r#"
 GT_WORKER
 uint seq=state[25];
 if(gid==0u){
  uint r=gt_atomic_add(state+6u,1u);
  GT_DEV uint* b=gt_node(nodes,r);
  for(uint i=1u;i<GT_NODE;++i)b[i]=0u;
  b[1]=GT_ENTRY;b[3]=GT_NONE;b[5]=1u;
  for(uint q=0u;q<GT_ENTRY_P;++q)b[GT_HDR+q]=(uint)params[q];
  GT_FENCE;gt_store_word(b,(seq<<3u)|(1u>=GT_W?GT_TASK:GT_READY));
 }
 uint mine=GT_NONE;uint polls=0u;
 for(;;){
  if(gt_load_word(state+1u)!=0u)break;
  if(mine==GT_NONE)mine=gt_atomic_add(state+24u,1u);
  uint t=mine<GT_ARENA?gt_load_word(gt_node(nodes,mine)):0u;
  if((t>>3u)!=seq){
   if(gt_load_word(state+26u)!=0xffffffffu){polls=0u;continue;}
   GT_FENCE;
   if(gt_load_word(state+1u)!=0u)break;
   if(++polls>=GT_POLL_LIMIT){gt_store_word(state+1u,2u);break;}
   continue;
  }
  GT_FENCE;polls=0u;
  uint n=mine;mine=GT_NONE;
  if((t&7u)==GT_TASK){gt_atomic_add(state+17u,1u);n=gt_serial(state,nodes,n);}
  else n=gt_advance(state,state,nodes,lists,0u,n);
  while(n!=GT_NONE)n=gt_advance(state,state,nodes,lists,0u,n);
  GT_FENCE;gt_atomic_add(state+26u,0xffffffffu);
 }
"#;

const CPU_CASCADE_HARNESS: &str = r#"
#include <cstdio>
#include <vector>
int main(int argc,char** argv){
 FILE* in=argc>1?std::fopen(argv[1],"r"):stdin;
 if(in==nullptr)return 2;
 std::vector<int> params(GT_ENTRY_P+1u,0);std::vector<uint> state(32u,0u);
 std::vector<uint> nodes((size_t)GT_ARENA*GT_NODE,0u);std::vector<uint> lists((size_t)GT_ARENA*3u,0u);
 uint seq=0u;
 for(;;){
  bool ok=true;for(uint q=0u;q<GT_ENTRY_P;++q){if(std::fscanf(in,"%d",&params[q])!=1){ok=false;break;}}
  if(!ok)break;
  for(uint i=0u;i<32u;++i)state[i]=0u;
  state[25]=++seq;
  gt_cascade(params.data(),state.data(),nodes.data(),lists.data());
  std::printf("%u %u %u %u %u %u %u %u %u\n",state[1],state[2],state[3],state[4],state[5],state[24],state[17],1u,state[6]);
  std::fflush(stdout);
 }
 return 0;
}
"#;

const CPU_HARNESS: &str = r#"
#include <cstdio>
#include <vector>
int main(int argc,char** argv){
 FILE* in=argc>1?std::fopen(argv[1],"r"):stdin;
 if(in==nullptr)return 2;
 std::vector<int> params(GT_ENTRY_P+1u,0);std::vector<uint> state(32u,0u);
 std::vector<uint> nodes((size_t)GT_ARENA*GT_NODE,0u);std::vector<uint> lists((size_t)GT_ARENA*3u,0u);
 for(;;){
  bool ok=true;for(uint q=0u;q<GT_ENTRY_P;++q){if(std::fscanf(in,"%d",&params[q])!=1){ok=false;break;}}
  if(!ok)break;
  for(uint i=0u;i<32u;++i)state[i]=0u;
  gt_sched(params.data(),state.data(),nodes.data(),lists.data());
  for(uint round=0u;round<100000u&&state[1]==0u;++round){
   if(state[12]==0u){state[2]=7u;break;}
   gt_run(params.data(),state.data(),nodes.data(),lists.data(),state[13]);
   gt_sched(params.data(),state.data(),nodes.data(),lists.data());
  }
  std::printf("%u %u %u %u %u %u %u %u %u\n",state[1],state[2],state[3],state[4],state[5],state[16],state[17],state[18],state[19]);
  std::fflush(stdout);
 }
 return 0;
}
"#;

fn dialect(target: Target, schedule: Schedule) -> &'static str {
    match (target, schedule) {
        (Target::Metal, Schedule::Staged) => {
            "#include <metal_stdlib>\nusing namespace metal;\n#define GT_FN inline\n#define GT_TH thread\n#define GT_DEV device\n#define GT_FENCE\ntypedef long gt_i64;typedef ulong gt_u64;\nGT_FN uint gt_atomic_add(device uint* p,uint v){return atomic_fetch_add_explicit((device atomic_uint*)p,v,memory_order_relaxed);}\nGT_FN uint gt_atomic_add(threadgroup uint* p,uint v){return atomic_fetch_add_explicit((threadgroup atomic_uint*)p,v,memory_order_relaxed);}\nGT_FN uint gt_load_word(device uint* p){return atomic_load_explicit((device atomic_uint*)p,memory_order_relaxed);}\nGT_FN void gt_store_word(device uint* p,uint v){atomic_store_explicit((device atomic_uint*)p,v,memory_order_relaxed);}\n"
        }
        // MSL 3.2: device-coherent buffers and device-scope fences make one
        // dispatch's groups see each other's node words.
        (Target::Metal, Schedule::Cascade) => {
            "#include <metal_stdlib>\nusing namespace metal;\n#define GT_FN inline\n#define GT_TH thread\n#define GT_DEV coherent(device) device\ntypedef long gt_i64;typedef ulong gt_u64;\n#define GT_FENCE atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);\nGT_FN uint gt_atomic_add(GT_DEV uint* p,uint v){return atomic_fetch_add_explicit((GT_DEV atomic_uint*)p,v,memory_order_relaxed);}\nGT_FN uint gt_load_word(GT_DEV uint* p){return atomic_load_explicit((GT_DEV atomic_uint*)p,memory_order_relaxed);}\nGT_FN void gt_store_word(GT_DEV uint* p,uint v){atomic_store_explicit((GT_DEV atomic_uint*)p,v,memory_order_relaxed);}\n"
        }
        // Node words other lanes wrote are read volatile (past the SM's L1).
        (Target::Cuda, _) => {
            "typedef unsigned int uint;typedef long long gt_i64;typedef unsigned long long gt_u64;\n#define GT_FN __device__ inline\n#define GT_TH\n#define GT_DEV volatile\n#define GT_FENCE __threadfence();\nGT_FN uint gt_atomic_add(volatile uint* p,uint v){return atomicAdd((uint*)p,v);}\nGT_FN uint gt_load_word(volatile uint* p){return *p;}\nGT_FN void gt_store_word(volatile uint* p,uint v){atomicExch((uint*)p,v);}\n"
        }
        (Target::Cpu, _) => {
            "typedef unsigned int uint;typedef long long gt_i64;typedef unsigned long long gt_u64;\n#define GT_FN static inline\n#define GT_TH\n#define GT_DEV\n#define GT_FENCE\nGT_FN uint gt_atomic_add(uint* p,uint v){uint o=*p;*p=o+v;return o;}\nGT_FN uint gt_load_word(uint* p){return *p;}\nGT_FN void gt_store_word(uint* p,uint v){*p=v;}\n"
        }
    }
}

fn staged_kernels(target: Target) -> String {
    let params =
        "GT_DEV const int* params,GT_DEV uint* state,GT_DEV uint* nodes,GT_DEV uint* lists";
    match target {
        Target::Metal => format!(
            "#define GT_PHASE_BEGIN {{uint tid=ltid;\n#define GT_PHASE_END }}threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);\n\
             kernel void {SCHED_KERNEL}(device const int* params [[buffer(0)]],device uint* state [[buffer(1)]],device uint* nodes [[buffer(2)]],device uint* lists [[buffer(3)]],uint ltid [[thread_index_in_threadgroup]]){{threadgroup uint sh[4];threadgroup uint cnt[6];{SCHED_BODY}}}\n\
             kernel void {RUN_KERNEL}(device const int* params [[buffer(0)]],device uint* state [[buffer(1)]],device uint* nodes [[buffer(2)]],device uint* lists [[buffer(3)]],uint ltid [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]){{uint i=gid*GT_RUN_T+ltid;{RUN_BODY}}}\n"
        ),
        Target::Cuda => format!(
            "#define GT_PHASE_BEGIN {{uint tid=ltid;\n#define GT_PHASE_END }}__syncthreads();\n\
             extern \"C\" __global__ void {SCHED_KERNEL}({params}){{__shared__ uint sh[4];__shared__ uint cnt[6];uint ltid=threadIdx.x;{SCHED_BODY}}}\n\
             extern \"C\" __global__ void {RUN_KERNEL}({params}){{uint i=blockIdx.x*GT_RUN_T+threadIdx.x;{RUN_BODY}}}\n"
        ),
        Target::Cpu => format!(
            "#define GT_PHASE_BEGIN for(uint tid=0u;tid<GT_SCHED_T;++tid){{\n#define GT_PHASE_END }}\n\
             static void {SCHED_KERNEL}({params}){{uint sh[4]={{0u,0u,0u,0u}};uint cnt[6]={{0u,0u,0u,0u,0u,0u}};{SCHED_BODY}}}\n\
             static void gt_run_lane({params},uint i){{{RUN_BODY}}}\n\
             static void {RUN_KERNEL}({params},uint groups){{for(uint i=0u;i<groups*GT_RUN_T;++i)gt_run_lane(params,state,nodes,lists,i);}}\n{CPU_HARNESS}"
        ),
    }
}

fn cascade_kernel(target: Target) -> String {
    let params =
        "GT_DEV const int* params,GT_DEV uint* state,GT_DEV uint* nodes,GT_DEV uint* lists";
    match target {
        Target::Metal => format!(
            "kernel void {CASCADE_KERNEL}(device const int* params [[buffer(0)]],GT_DEV uint* state [[buffer(1)]],GT_DEV uint* nodes [[buffer(2)]],GT_DEV uint* lists [[buffer(3)]],uint gid [[thread_position_in_grid]],uint simd [[thread_index_in_simdgroup]]){{\n#define GT_WORKER if(simd!=0u)return;\n{CASCADE_BODY}}}\n"
        ),
        Target::Cuda => format!(
            "extern \"C\" __global__ void {CASCADE_KERNEL}(const int* params,GT_DEV uint* state,GT_DEV uint* nodes,GT_DEV uint* lists){{uint gid=blockIdx.x*blockDim.x+threadIdx.x;\n#define GT_WORKER if((threadIdx.x&31u)!=0u)return;\n{CASCADE_BODY}}}\n"
        ),
        // One lane: it claims indices in allocation order, and every index it
        // reaches before the root finishes is already published.
        Target::Cpu => format!(
            "static void {CASCADE_KERNEL}({params}){{uint gid=0u;\n#define GT_WORKER\n{CASCADE_BODY}}}\n{CPU_CASCADE_HARNESS}"
        ),
    }
}

/// Emit the task-region source of `program` for `target` under `plan`.
pub fn emit(
    program: &Program,
    target: Target,
    plan: &Plan,
    mutation: Mutation,
) -> Result<Source, Refusal> {
    let mut text = String::new();
    text.push_str(dialect(target, plan.schedule));
    let entry = &program.functions[&program.entry];
    let _ = writeln!(
        text,
        "#define GT_POLL_LIMIT {}u\n#define GT_FW {}u\n#define GT_K {}u\n#define GT_P {}u\n#define GT_NODE {}u\n#define GT_ARENA {}u\n#define GT_W {}u\n#define GT_SCHED_T {}u\n#define GT_RUN_T {}u\n#define GT_BUDGET {}u\n#define GT_MAX_DEPTH {}u\n#define GT_ENTRY {}u\n#define GT_ENTRY_P {}u\n#define GT_HDR {}u\n#define GT_RES (GT_HDR+GT_FW)",
        plan.poll_limit.max(1),
        program.frame_words,
        program.siblings,
        program.params,
        node_words(program),
        plan.arena,
        plan.width,
        plan.sched_threads,
        plan.run_threads,
        plan.budget,
        super::MAX_CALL_DEPTH,
        program.entry,
        entry.tree.params.len(),
        HEADER_WORDS,
    );
    let (order, eager, depth, fallback) = (
        if mutation == Mutation::LastFailure {
            "(k-1u-j)"
        } else {
            "j"
        },
        if mutation == Mutation::EagerArgumentFailure {
            "if(b[13]!=0u)return gt_finish(state,nodes,n,b[13],0,b[14],b[15]);"
        } else {
            ""
        },
        if mutation == Mutation::DepthOffByOne {
            "depth"
        } else {
            "depth+1u"
        },
        if mutation == Mutation::StaleFallback {
            "fr"
        } else {
            "good"
        },
    );
    let join_at = if mutation == Mutation::EarlyJoin {
        "1u"
    } else {
        "q[6]"
    };
    let (cancel_from, cancel_to) = if mutation == Mutation::CancelEarlier {
        ("0u", "b[4]")
    } else {
        ("b[4]+1u", "q[6]")
    };
    let _ = writeln!(
        text,
        "#define GT_CHILD_ORDER {order}\n#define GT_EAGER_PENDING {eager}\n#define GT_CHILD_DEPTH {depth}\n#define GT_FALLBACK_FRAME {fallback}\n#define GT_CANCEL_FROM {cancel_from}\n#define GT_CANCEL_TO {cancel_to}\n#define GT_JOIN_AT {join_at}"
    );
    text.push_str(COMMON);
    text.push_str(&lower::functions(program, mutation)?);
    text.push_str(match plan.schedule {
        Schedule::Staged => STAGED_HOOKS,
        Schedule::Cascade => CASCADE_HOOKS,
    });
    text.push_str(DRIVER);
    text.push_str(&match plan.schedule {
        Schedule::Staged => staged_kernels(target),
        Schedule::Cascade => cascade_kernel(target),
    });
    Ok(Source {
        text,
        target,
        plan: plan.clone(),
        words: buffer_words(program, plan),
    })
}
