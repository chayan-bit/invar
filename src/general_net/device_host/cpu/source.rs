use super::*;
pub(super) fn emit(artifact: &compiled::Artifact, symbol: &str) -> Result<String, String> {
    let mut shim = String::from(
        r#"
#include <barrier>
#include <thread>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <new>
std::barrier group_barrier(32);
#define __device__
#define __global__
#define __shared__ static
#define __syncthreads() group_barrier.arrive_and_wait()
struct Dim {unsigned x=1,y=1,z=1;};
Dim gridDim,blockDim{32,1,1};thread_local Dim threadIdx{0,0,0},blockIdx{0,0,0};
"#,
    );
    let (source, bridge) = match artifact.target() {
        Target::Cuda => (artifact.source().to_owned(), String::new()),
        Target::Metal => {
            shim.push_str(
                r#"
#define device
#define thread
#define threadgroup
#define constant
#define kernel
using uint=unsigned;using ulong=std::uint64_t;
#if __SIZEOF_LONG__ == 4
#define long long long
#endif
struct uint3{uint x,y,z;};
namespace metal{enum class mem_flags{mem_device=1,mem_threadgroup=2};
inline unsigned operator|(mem_flags a,mem_flags b){return unsigned(a)|unsigned(b);}
inline void threadgroup_barrier(unsigned){group_barrier.arrive_and_wait();}}
"#,
            );
            (
                shared_exits(
                    &artifact
                        .source()
                        .replace("#include <metal_stdlib>", "")
                        .replace(
                            "threadgroup gnet_u root_done; threadgroup gnet_u deltas[8];",
                            "static gnet_u root_done; static gnet_u deltas[8];",
                        ),
                )?,
                r#"
#undef thread
#undef device
#undef threadgroup
#undef constant
#undef kernel
#undef long
"#
                .into(),
            )
        }
        _ => return Err("CPU model requires a device artifact".into()),
    };
    let call = if artifact.target() == Target::Cuda {
        "invar_gnet_children_v5(a,o,t,c,h,cl,task.get(),slot(),ph,pt,pc,s);"
    } else {
        "invar_gnet_children_v5(a,o,t,c,h,cl,task.get(),slot(),ph,pt,pc,s,threadIdx.x,uint3{0,0,0},uint3{1,1,1},uint3{32,1,1});"
    };
    let driver=r#"
#ifdef _WIN32
#define INVAR_MODEL_EXPORT __declspec(dllexport)
#else
#define INVAR_MODEL_EXPORT __attribute__((visibility("default")))
#endif
struct ModelWork {
 std::array<std::vector<gnet_u>,12> b;
 std::array<size_t,12> sizes{};
 std::unique_ptr<GnetTask> task;
 std::vector<GnetSpecSlot> slots;
 std::unique_ptr<gnet_u> empty_slot;
 std::vector<std::thread> workers;
 std::mutex mutex;std::condition_variable wake,finished;
 bool stopping=false,busy=false,submitted=false;uint64_t epoch=0;unsigned done=0;
 GnetSpecSlot* slot(){return slots.empty()?reinterpret_cast<GnetSpecSlot*>(empty_slot.get()):slots.data();}
 void* data(size_t i){if(i==6)return task.get();if(i==7)return slots.empty()?static_cast<void*>(empty_slot.get()):static_cast<void*>(slots.data());return b[i].data();}
 void kernel_epoch(unsigned lane){threadIdx.x=lane;
  auto a=reinterpret_cast<const gnet_w*>(data(0));auto o=reinterpret_cast<gnet_w*>(data(1));
  auto t=static_cast<gnet_u*>(data(2));auto c=static_cast<const gnet_u*>(data(3));
  auto h=reinterpret_cast<gnet_w*>(data(4));auto cl=static_cast<gnet_u*>(data(5));
  auto ph=reinterpret_cast<gnet_w*>(data(8));auto pt=static_cast<gnet_u*>(data(9));
  auto pc=static_cast<gnet_u*>(data(10));auto s=static_cast<gnet_u*>(data(11));
  @CALL@
 }
 void start(){
  try {for(unsigned lane=0;lane<32;++lane)workers.emplace_back([this,lane]{
   uint64_t seen=0;for(;;){std::unique_lock lock(mutex);wake.wait(lock,[&]{return stopping||epoch>seen;});if(stopping)return;seen=epoch;lock.unlock();kernel_epoch(lane);lock.lock();if(++done==32){busy=false;finished.notify_one();}}
  });}catch(...){shutdown();throw;}
 }
 void shutdown(){ {std::lock_guard lock(mutex);stopping=true;}wake.notify_all();for(auto& worker:workers)if(worker.joinable())worker.join();}
 ~ModelWork(){shutdown();}
};
extern "C" INVAR_MODEL_EXPORT uint32_t @SYMBOL@(uint32_t command,void** opaque,const uint64_t* input,size_t n,uint64_t* output,size_t m) noexcept {
 if(!opaque)return 2;
 try{
 if(command==0){if(*opaque||!input||n!=18+@ARGS@||!output||m!=12)return 2;
  size_t total=0;for(size_t i=0;i<12;++i){if(!input[i]||input[i]>8*1024*1024||total>8*1024*1024-input[i])return 2;total+=input[i];}
  if(input[6]*8!=sizeof(GnetTask)||input[7]*8!=(@SLOTS@?sizeof(GnetSpecSlot)*@SLOTS@:8)||input[3]!=6||input[0]<@ARGS@)return 2;
  auto work=std::make_unique<ModelWork>();for(size_t i=0;i<12;++i)work->sizes[i]=size_t(input[i]);
  for(size_t i=0;i<6;++i)work->b[i].resize(work->sizes[i],0);work->task=std::make_unique<GnetTask>();
  work->slots.resize(@SLOTS@);if(work->slots.empty())work->empty_slot=std::make_unique<gnet_u>(0);
  for(size_t i=8;i<12;++i)work->b[i].resize(work->sizes[i],0);
  std::memcpy(work->data(3),input+12,48);if(@ARGS@)std::memcpy(work->data(0),input+18,@ARGS@*8);
  work->start();for(size_t i=0;i<12;++i)output[i]=uint64_t(reinterpret_cast<uintptr_t>(work->data(i)));*opaque=work.release();return 0;
 }
 auto work=static_cast<ModelWork*>(*opaque);if(!work)return 2;
 if(command==1){std::unique_lock lock(work->mutex);if(work->busy)return 3;work->done=0;work->busy=true;work->submitted=true;++work->epoch;work->wake.notify_all();work->finished.wait(lock,[&]{return !work->busy;});return 0;}
 if(command==2){if(!output||m!=1)return 2;std::lock_guard lock(work->mutex);output[0]=work->submitted&&!work->busy;return 0;}
 if(command==3){std::lock_guard lock(work->mutex);if(!work->submitted||work->busy)return 3;size_t total=0;for(auto size:work->sizes)total+=size;if(!output||m!=total)return 2;size_t at=0;for(size_t i=0;i<12;++i){std::memcpy(output+at,work->data(i),work->sizes[i]*8);at+=work->sizes[i];}return 0;}
 if(command==4){ {std::lock_guard lock(work->mutex);if(work->busy)return 3;}delete work;*opaque=nullptr;return 0;}
 return 2;
 }catch(const std::bad_alloc&){return 1;}catch(...){return 4;}
}
"#.replace("@CALL@",call).replace("@SYMBOL@",symbol).replace("@ARGS@",&artifact.entry_layout().map_or(artifact.params().len(), |e|e.input_words()).to_string()).replace("@SLOTS@",&artifact.speculation_layout().ok_or("missing private layout")?.slots().to_string());
    Ok(format!("{shim}\n{source}\n{bridge}\n{driver}"))
}

/// Metal has no `goto`, so every one of the thousands of `GNET_STOP` and
/// `GNET_YIELD` sites in a Metal `gnet_resume` expands its whole epilogue
/// inline; clang -O1 with UBSan spent about 270 CPU-s on one such model. C++
/// has `goto`, so, as the CUDA emitter does (`resume::exit_macro`), every site
/// jumps to one copy of the same epilogue at the end of `gnet_resume`. The
/// state bodies stay byte-identical; a site outside `gnet_resume`, or one
/// that skips a declaration, fails to compile instead of changing meaning.
fn shared_exits(source: &str) -> Result<String, String> {
    let shape = || "CPU device model: unexpected Metal continuation shape".to_owned();
    let resume = source
        .find("\nGNET_FN bool gnet_resume(")
        .ok_or_else(shape)?;
    let define = |head: &str| {
        let at = source[..resume].rfind(head).ok_or_else(shape)? + 1;
        let end = at + source[at..].find('\n').ok_or_else(shape)?;
        Ok::<_, String>((at..end, &source[at + head.len() - 1..end]))
    };
    let (yield_line, yield_body) = define("\n#define GNET_YIELD ")?;
    let (stop_line, stop_body) = define("\n#define GNET_STOP ")?;
    // gnet_resume closes at column 0, right before the next kernel.
    let close = resume + source[resume..].find("\n}\nkernel ").ok_or_else(shape)?;
    let mut lines = [
        (yield_line, "#define GNET_YIELD goto gnet_yield".to_owned()),
        (stop_line, "#define GNET_STOP goto gnet_stop".to_owned()),
        (
            close..close,
            format!("\ngnet_yield: {yield_body};\ngnet_stop: {stop_body};"),
        ),
    ];
    lines.sort_by_key(|(range, _)| range.start);
    let mut out = String::with_capacity(source.len() + 1024);
    let mut at = 0;
    for (range, text) in lines {
        out.push_str(&source[at..range.start]);
        out.push_str(&text);
        at = range.end;
    }
    out.push_str(&source[at..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn metal_exits_jump_to_one_shared_epilogue_in_gnet_resume() {
        let source = "x\n#define GNET_YIELD GNET_SAVE; return false\n#define GNET_STOP gnet_finish(); return true\nGNET_FN bool gnet_resume(int x) {\nif (x) { GNET_STOP; } GNET_YIELD;\n}\nkernel void k() {}\n#define GNET_STOP return false\n";
        assert_eq!(
            super::shared_exits(source).unwrap(),
            "x\n#define GNET_YIELD goto gnet_yield\n#define GNET_STOP goto gnet_stop\nGNET_FN bool gnet_resume(int x) {\nif (x) { GNET_STOP; } GNET_YIELD;\ngnet_yield: GNET_SAVE; return false;\ngnet_stop: gnet_finish(); return true;\n}\nkernel void k() {}\n#define GNET_STOP return false\n"
        );
        assert!(super::shared_exits(&source.replace("kernel", "void")).is_err());
        assert!(super::shared_exits(&source.replace("GNET_YIELD G", "GNET_Y G")).is_err());
    }
}
