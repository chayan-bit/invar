#![forbid(unsafe_code)]
//! ABI11 owned-numeric continuations: real emitted C++ under UBSan compared
//! word-for-word across quanta and against the independent graph oracle.

use crate::{child_deadline, gnet_words};

use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
    numeric_contract::{F32Bits, NumericPolicy},
};
use std::{fs, path::PathBuf, process::Stdio};

mod cases;

const QUANTA: [u64; 5] = [1, 2, 3, 17, 1_000_000];

fn source(name: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let own = root
        .join("issue_gnet_11_numeric_owned_state_fixtures")
        .join(name);
    fs::read_to_string(if own.exists() {
        own
    } else {
        root.join("issue_gnet_10_f32_arrays_fixtures").join(name)
    })
    .unwrap()
}
fn net(s: &str) -> Net {
    let c = check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&c).unwrap();
    Net::lower(&c, general_net::Limits::default()).unwrap()
}
fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 1000,
        transitions: 100000,
        trace_events: 10000,
        array_elements: 128,
    }
}
fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 128,
        allocations: 128,
        cleanup_events: 128,
    }
}
fn carriers(xs: &[eval::Value]) -> Vec<i64> {
    xs.iter()
        .map(|x| match x {
            eval::Value::F32(x) => i64::from(x.to_bits()),
            eval::Value::U32(x) => i64::from(*x),
            eval::Value::I32(x) => i64::from(*x),
            _ => panic!("typed inputs"),
        })
        .collect()
}

/// Shared driver body. Every buffer is guarded; output stays at the sentinel
/// until the terminal packet. Each line: calls, ten header words, trace,
/// cleanup and the complete heap.
const COMMON: &str = r#"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
static const gnet_w S=-31337; static const gnet_u U=912345;
static bool guarded(const std::vector<gnet_w>& o,const std::vector<gnet_w>& h,const std::vector<gnet_u>& t,const std::vector<gnet_u>& c){
 return o.front()==S&&o.back()==S&&h.front()==S&&h.back()==S&&t.front()==U&&t.back()==U&&c.front()==U&&c.back()==U;}
static void print(gnet_u calls,const gnet_w* output,const gnet_u* trace,const gnet_u* cleanup,const gnet_w* heap){
 std::cout<<calls<<' ';for(unsigned j=0;j<10;++j)std::cout<<output[j]<<' ';
 for(gnet_u j=0;j<(gnet_u)output[7]*7;++j)std::cout<<trace[j]<<' ';
 for(gnet_u j=0;j<(gnet_u)output[8]*4;++j)std::cout<<cleanup[j]<<' ';
 for(gnet_u j=0;j<@HEAP@;++j)std::cout<<heap[j]<<' ';std::cout<<'\n';}
int main(){
 std::string mode;
 while(std::cin>>mode){gnet_u fuel,trans,cap,quantum;if(!(std::cin>>fuel>>trans>>cap>>quantum)||cap>1000000)return 2;
  std::vector<gnet_w> input(@PARAMS@);for(auto& x:input)if(!(std::cin>>x))return 2;
  std::vector<gnet_w> og(12,S),hg(@HEAP@+2,S);gnet_w* output=og.data()+1;gnet_w* heap=hg.data()+1;std::memset(heap,0,@HEAP@*8);
  std::vector<gnet_u> tg(cap*7+2,U),cg(@CLEAN@+2,U);gnet_u* trace=tg.data()+1;gnet_u* cleanup=cg.data()+1;
  std::memset(trace,0,cap*56);std::memset(cleanup,0,@CLEAN@*8);gnet_u calls=0;
@BODY@
  if(!guarded(og,hg,tg,cg))return 3;
  if(output[7]<0||(gnet_u)output[7]>cap||output[8]<0||(gnet_u)output[8]>@CLEANUP_CAP@||output[9]<0||(gnet_u)output[9]>@SLOTS@)return 4;
  print(calls,output,trace,cleanup,heap);}
}
"#;
/// CPU worklist driver over `roots` lanes. Every pending lane keeps its
/// output sentinel; completed lanes publish only their own buffers. A finished
/// worklist is replayed and must not change any byte. Rejected arguments must
/// leave every buffer and task untouched.
const WORKLIST: &str = r#"
#include <iostream>
#include <vector>
#include <cstring>
static const gnet_w S=-31337; static const gnet_u U=912345;
int main(){
 gnet_u roots,fuel,trans,cap,quantum,slices;
 while(std::cin>>roots>>fuel>>trans>>cap>>quantum>>slices){
  gnet_u lanes=roots<64?roots:64;if(cap>100000)return 2;gnet_u n=lanes?lanes:1;
  std::vector<gnet_w> input(n*@PARAMS@+1,0);for(gnet_u j=0;j<lanes*@PARAMS@;++j)if(!(std::cin>>input[j]))return 2;
  std::vector<gnet_w> og(n*10+2,S),hg(n*@HEAP@+2,S);gnet_w* output=og.data()+1;gnet_w* heap=hg.data()+1;std::memset(heap,0,n*@HEAP@*8);
  std::vector<gnet_u> tg(n*cap*7+2,U),cg(n*@CLEAN@+2,U);gnet_u* trace=tg.data()+1;gnet_u* cleanup=cg.data()+1;
  std::memset(trace,0,n*cap*56);std::memset(cleanup,0,n*@CLEAN@*8);
  std::vector<GnetTask> tasks(n);std::memset((void*)tasks.data(),0,n*sizeof(GnetTask));
  auto snapshot=[&](){std::vector<unsigned char> t(n*sizeof(GnetTask));std::memcpy(t.data(),(const void*)tasks.data(),t.size());return t;};
  std::vector<unsigned char> zero=snapshot();std::vector<gnet_w> o0=og,h0=hg;std::vector<gnet_u> t0=tg,c0=cg;
  gnet_w status=0;gnet_u rounds=0,previous=0;
  do{gnet_u submitted=7777,completed=7777;
   status=gnet_worklist(input.data(),output,trace,roots,fuel,trans,cap,heap,cleanup,tasks.data(),quantum,slices,submitted,completed);
   std::cout<<"c "<<status<<' '<<submitted<<' '<<completed<<'\n';
   if(status!=0&&status!=14){if(submitted||completed||o0!=og||h0!=hg||t0!=tg||c0!=cg||zero!=snapshot())return 13;break;}
   if(completed<previous||completed>roots)return 5;previous=completed;
   gnet_u done=0;for(gnet_u l=0;l<lanes;++l){if(tasks[l].done){++done;continue;}for(unsigned j=0;j<10;++j)if(output[l*10+j]!=S)return 8;}
   if(done!=completed)return 6;
   if(++rounds>1000000)return 7;
  }while(status==14&&slices>0);
  if(og.front()!=S||og.back()!=S||hg.front()!=S||hg.back()!=S||tg.front()!=U||tg.back()!=U||cg.front()!=U||cg.back()!=U)return 3;
  if(status==0&&lanes){
   for(gnet_u l=0;l<lanes;++l){gnet_w* o=output+l*10;gnet_u* t=trace+l*cap*7;gnet_u* c=cleanup+l*@CLEAN@;gnet_w* h=heap+l*@HEAP@;
    if(o[7]<0||(gnet_u)o[7]>cap||o[8]<0||(gnet_u)o[8]>@CLEANUP_CAP@||o[9]<0||(gnet_u)o[9]>@SLOTS@)return 4;
    for(gnet_u j=(gnet_u)o[7]*7;j<cap*7;++j)if(t[j])return 9;
    for(gnet_u j=(gnet_u)o[8]*4;j<@CLEAN@;++j)if(c[j])return 10;
    std::cout<<"l ";for(unsigned j=0;j<10;++j)std::cout<<o[j]<<' ';
    for(gnet_u j=0;j<(gnet_u)o[7]*7;++j)std::cout<<t[j]<<' ';
    for(gnet_u j=0;j<(gnet_u)o[8]*4;++j)std::cout<<c[j]<<' ';
    for(gnet_u j=0;j<@HEAP@;++j)std::cout<<h[j]<<' ';std::cout<<'\n';}
   std::vector<gnet_w> o2=og,h2=hg;std::vector<gnet_u> t2=tg,c2=cg;std::vector<unsigned char> k2=snapshot();
   for(gnet_u q:{(gnet_u)1,quantum}){gnet_u submitted=7777,completed=7777;
    if(gnet_worklist(input.data(),output,trace,roots,fuel,trans,cap,heap,cleanup,tasks.data(),q,0,submitted,completed)!=0||submitted!=0||completed!=roots)return 11;}
   if(o2!=og||h2!=hg||t2!=tg||c2!=cg||k2!=snapshot())return 12;}
  std::cout<<"e\n";}
}
"#;
const ABI11_BODY: &str = r#"
  auto task=std::make_unique<GnetTask>();std::memset(task.get(),0,sizeof(GnetTask));bool done=false;
  if(mode=="run"){
   while(!done){if(++calls>2000000)return 7;
    done=gnet_resume(input.data(),output,trace,fuel,trans,cap,heap,cleanup,task.get(),quantum);
    if(!done)for(unsigned j=0;j<10;++j)if(output[j]!=S)return 8;}
  }else{
   calls=1;done=gnet_resume(input.data(),output,trace,fuel,trans,cap,heap,cleanup,task.get(),quantum);
   if(done){std::cout<<"done\n";continue;}
   for(unsigned j=0;j<10;++j)if(output[j]!=S)return 8;
   gnet_u f=fuel,t=trans,c=cap,q=quantum;
   if(mode=="fuel")f=fuel+1;else if(mode=="transitions")t=trans+1;else if(mode=="trace")c=cap?cap-1:1;
   else if(mode=="zero")q=0;else if(mode=="huge")q=1000001;else return 2;
   ++calls;if(!gnet_resume(input.data(),output,trace,f,t,c,heap,cleanup,task.get(),q))return 9;
  }
  if(!task->done)return 10;
  // Terminal replay: a finished task never rewrites any buffer, even with
  // otherwise invalid arguments.
  std::vector<gnet_w> o2=og,h2=hg;std::vector<gnet_u> t2=tg,c2=cg;
  std::vector<unsigned char> k2(sizeof(GnetTask));std::memcpy(k2.data(),task.get(),sizeof(GnetTask));
  for(gnet_u q:{(gnet_u)0,(gnet_u)1,quantum})if(!gnet_resume(input.data(),output,trace,fuel+5,trans+5,cap,heap,cleanup,task.get(),q))return 11;
  if(o2!=og||h2!=hg||t2!=tg||c2!=cg||std::memcmp(k2.data(),task.get(),sizeof(GnetTask)))return 12;
"#;

struct Cpu {
    dir: PathBuf,
}
impl Cpu {
    fn build(a: &compiled::Artifact, body: &str) -> Self {
        Self::build_driver(a, COMMON, body)
    }
    fn build_driver(a: &compiled::Artifact, template: &str, body: &str) -> Self {
        let layout = a.owned_layout().unwrap();
        let dir = child_deadline::scratch("gnet-abi11-state");
        let driver = template
            .replace("@BODY@", body)
            .replace("@HEAP@", &layout.heap_words().to_string())
            .replace("@CLEAN@", &layout.cleanup_words().to_string())
            .replace("@CLEANUP_CAP@", &layout.cleanup_events().to_string())
            .replace("@SLOTS@", &layout.allocations().to_string())
            .replace("@PARAMS@", &a.params().len().to_string());
        fs::write(dir.join("source.cpp"), format!("{}\n{driver}", a.source())).unwrap();
        child_deadline::cxx(
            child_deadline::CXX17_UBSAN_O2,
            &dir.join("source.cpp"),
            &dir.join("run"),
        );
        Self { dir }
    }
    /// Rows: (mode, limits, quantum, carriers).
    fn batch(&self, rows: &[(&str, oracle::Limits, u64, Vec<i64>)]) -> Vec<Option<Vec<i64>>> {
        let mut input = String::new();
        for (mode, l, q, xs) in rows {
            input.push_str(&format!(
                "{mode} {} {} {} {q}",
                l.fuel, l.transitions, l.trace_events
            ));
            for x in xs {
                input.push_str(&format!(" {x}"))
            }
            input.push('\n')
        }
        let path = self.dir.join("inputs.txt");
        fs::write(&path, input).unwrap();
        let out = child_deadline::command(120)
            .arg(self.dir.join("run"))
            .stdin(Stdio::from(fs::File::open(path).unwrap()))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "native run {:?}:{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        let lines: Vec<_> = String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|line| {
                (line != "done").then(|| {
                    line.split_whitespace()
                        .map(|x| x.parse().unwrap())
                        .collect()
                })
            })
            .collect();
        assert_eq!(lines.len(), rows.len());
        lines
    }
}
/// One `gnet_worklist` execution over independent roots.
struct Worklist {
    /// (status, submitted, completed) per worklist call.
    calls: Vec<[i64; 3]>,
    /// Terminal packet, trace, cleanup and heap per lane, as `print` emits.
    lanes: Vec<Vec<i64>>,
}
impl Cpu {
    fn worklist(
        &self,
        l: oracle::Limits,
        roots: u64,
        quantum: u64,
        slices: u64,
        inputs: &[Vec<i64>],
    ) -> Worklist {
        let mut line = format!(
            "{roots} {} {} {} {quantum} {slices}",
            l.fuel, l.transitions, l.trace_events
        );
        for x in inputs.iter().flatten() {
            line.push_str(&format!(" {x}"))
        }
        let path = self.dir.join("worklist.txt");
        fs::write(&path, line + "\n").unwrap();
        let out = child_deadline::command(120)
            .arg(self.dir.join("run"))
            .stdin(Stdio::from(fs::File::open(path).unwrap()))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "worklist run {:?}:{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        let mut result = Worklist {
            calls: Vec::new(),
            lanes: Vec::new(),
        };
        let mut ended = false;
        for line in String::from_utf8(out.stdout).unwrap().lines() {
            let mut words = line.split_whitespace();
            let tag = words.next().unwrap();
            let words: Vec<i64> = words.map(|x| x.parse().unwrap()).collect();
            match tag {
                "c" => result.calls.push(words.try_into().unwrap()),
                "l" => result.lanes.push(words),
                "e" => ended = true,
                x => panic!("{x}"),
            }
        }
        assert!(ended);
        result
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// One ABI11 native for one entry and storage profile.
struct Pair {
    abi11: Cpu,
    layout: compiled::OwnedLayout,
}
impl Pair {
    fn new(n: &Net, entry: &str, s: compiled::OwnedLimits) -> Self {
        let proof = n.verify().unwrap();
        let cpu = compiled::Target::Cpu;
        let limits = compiled::Limits::default();
        let eleven =
            compiled::emit_resumable_owned_numeric(n, &proof, entry, cpu, limits, s).unwrap();
        assert_eq!(eleven.abi_version(), 11);
        assert_eq!(eleven.numeric_policy(), Some(NumericPolicy::STRICT_F32));
        assert!(
            eleven.speculation_layout().is_none()
                && eleven.device_layout().is_none()
                && eleven.aggregate_layout().is_none()
                && eleven.fork_plan().is_none()
        );
        let c = eleven.continuation_layout().unwrap();
        assert_eq!(c.task_bytes(), eleven.frame_bytes() + 18 * 8);
        assert_eq!(c.kernel_name(), "invar_gnet_resume_v3");
        let layout = eleven.owned_layout().unwrap().clone();
        assert_eq!(layout.output_words(), 10);
        assert!(eleven.source().contains("GNET ABI11 layout mismatch"));
        Self {
            abi11: Cpu::build(&eleven, ABI11_BODY),
            layout,
        }
    }
    /// Runs every row once per quantum; every quantum must publish the same
    /// terminal packet, trace, cleanup and heap words as the single-slice run.
    fn run(&self, rows: &[(oracle::Limits, Vec<i64>)]) -> Vec<Vec<i64>> {
        let base: Vec<_> = rows
            .iter()
            .map(|(l, x)| ("run", *l, 1_000_000, x.clone()))
            .collect();
        let expected: Vec<Vec<i64>> = self
            .abi11
            .batch(&base)
            .into_iter()
            .map(|w| w.unwrap()[1..].to_vec())
            .collect();
        let mut resumed = Vec::new();
        for q in QUANTA {
            for (l, x) in rows {
                resumed.push(("run", *l, q, x.clone()))
            }
        }
        let actual = self.abi11.batch(&resumed);
        for (i, w) in actual.into_iter().enumerate() {
            let w = w.unwrap();
            let q = QUANTA[i / rows.len()];
            let e = &expected[i % rows.len()];
            assert_eq!(&w[1..], e.as_slice(), "quantum {q} row {}", i % rows.len());
            // Every nonfinal slice runs exactly `q` transitions. A transition
            // limit reached on a slice boundary yields once more first.
            let t = e[5] as u64;
            let calls = w[0] as u64;
            let minimum = t.div_ceil(q).max(1);
            assert!(
                calls == minimum || (calls == minimum + 1 && t.is_multiple_of(q)),
                "quantum {q}: {calls} calls for {t} transitions"
            );
        }
        expected
    }
}

/// Independent oracle comparison of status, counters, trace, cleanup lengths
/// and root carriers. Full storage equality is checked across quanta.
fn compare(layout: &compiled::OwnedLayout, o: &oracle::Observation, w: &[i64]) {
    assert_eq!((w[0], w[2], w[3]), gnet_words::status(o), "{o:?}");
    assert_eq!(
        &w[4..9],
        &[
            o.semantic.steps_used as i64,
            o.transitions as i64,
            o.max_frames as i64,
            o.events.len() as i64,
            o.semantic.cleanup_trace.len() as i64
        ]
    );
    let trace_end = 10 + o.events.len() * 7;
    let events: Vec<i64> = o.events.iter().flat_map(gnet_words::event_i64).collect();
    assert_eq!(&w[10..trace_end], events.as_slice());
    let cleanup = &w[trace_end..trace_end + o.semantic.cleanup_trace.len() * 4];
    for (row, c) in cleanup.chunks_exact(4).zip(&o.semantic.cleanup_trace) {
        assert_eq!(row[2], c.length as i64)
    }
    let heap = &w[trace_end + cleanup.len()..];
    assert_eq!(heap.len(), layout.heap_words());
    match &o.semantic.result {
        Ok(eval::Value::F32(x)) => assert!(x.strict_eq(F32Bits::from_bits(w[1] as u32))),
        Ok(eval::Value::I32(x)) => assert_eq!(w[1], i64::from(*x)),
        Ok(eval::Value::U32(x)) => assert_eq!(w[1], i64::from(*x)),
        Ok(eval::Value::ArrayF32(xs)) => {
            let d = usize::try_from(w[1] - 1).unwrap() * 4;
            assert_eq!((heap[d + 2], heap[d + 3]), (258, 1));
            let start = layout.allocations() * 4 + heap[d] as usize;
            let xs = xs.borrow();
            assert_eq!(heap[d + 1] as usize, xs.len());
            for (a, b) in heap[start..start + xs.len()].iter().zip(xs.iter()) {
                assert!(b.strict_eq(F32Bits::from_bits(*a as u32)))
            }
        }
        Err(_) => assert_eq!(w[1], 0),
        x => panic!("{x:?}"),
    }
}
