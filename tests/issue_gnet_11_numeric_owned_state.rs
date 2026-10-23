#![forbid(unsafe_code)]
//! ABI11 owned-numeric continuations: real emitted C++ under UBSan compared
//! word-for-word across quanta and against the independent graph oracle.
#[allow(dead_code)]
#[path = "../test-support/child_deadline.rs"]
mod child_deadline;

use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
    numeric_contract::{F32Bits, NumericPolicy},
};
use std::{
    fs,
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicUsize, Ordering},
};

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
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gnet-abi11-state-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let driver = template
            .replace("@BODY@", body)
            .replace("@HEAP@", &layout.heap_words().to_string())
            .replace("@CLEAN@", &layout.cleanup_words().to_string())
            .replace("@CLEANUP_CAP@", &layout.cleanup_events().to_string())
            .replace("@SLOTS@", &layout.allocations().to_string())
            .replace("@PARAMS@", &a.params().len().to_string());
        fs::write(dir.join("source.cpp"), format!("{}\n{driver}", a.source())).unwrap();
        let out = child_deadline::command(120)
            .args([
                "c++",
                "-std=c++17",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ])
            .arg(dir.join("source.cpp"))
            .arg("-o")
            .arg(dir.join("run"))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{:?}:{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
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

fn status(o: &oracle::Observation) -> (i64, i64, i64) {
    match &o.semantic.result {
        Ok(_) => (0, 0, 0),
        Err(eval::EvalError::Trap { category, span }) => (
            match category {
                eval::TrapCategory::IntegerOverflow => 1,
                eval::TrapCategory::DivisionByZero => 2,
                eval::TrapCategory::IndexOutOfBounds => 9,
                _ => panic!("trap"),
            },
            span.start as i64,
            span.end as i64,
        ),
        Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0),
        Err(eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        }) => (10, 0, 0),
        Err(eval::EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            ..
        }) => (4, 0, 0),
        Err(_) => (
            match o.failure {
                Some(oracle::ToolFailure::TransitionLimit) => 5,
                Some(oracle::ToolFailure::TraceLimit) => 6,
                Some(oracle::ToolFailure::Allocation) => 11,
                _ => panic!("{o:?}"),
            },
            0,
            0,
        ),
    }
}
fn events(o: &oracle::Observation) -> Vec<i64> {
    o.events
        .iter()
        .flat_map(|e| {
            [
                i64::from(e.kind == oracle::EventKind::Complete),
                e.frame as i64,
                e.activation as i64,
                e.token.block.function as i64,
                e.token.block.block as i64,
                e.token.position as i64,
                e.rule as i64,
            ]
        })
        .collect()
}
/// Independent oracle comparison of status, counters, trace, cleanup lengths
/// and root carriers. Full storage equality is checked across quanta.
fn compare(layout: &compiled::OwnedLayout, o: &oracle::Observation, w: &[i64]) {
    assert_eq!((w[0], w[2], w[3]), status(o), "{o:?}");
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
    assert_eq!(&w[10..trace_end], events(o).as_slice());
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

#[test]
fn issue_gnet_11_every_fuel_prefix_matches_oracle_across_quanta() {
    let names = [
        "literals.inv",
        "filled_empty.inv",
        "update_fma.inv",
        "loans_moves.inv",
        "replacement.inv",
        "mixed_relocation.inv",
        "root_return.inv",
        "element_copy.inv",
        "nested_loans.inv",
        "index_trap.inv",
        "store_trap.inv",
        "older_integer_trap.inv",
        "division_trap.inv",
        "array_size_trap.inv",
        "call_depth_trap.inv",
    ];
    let mut comparisons = 0;
    for name in names {
        let n = net(&source(name));
        let proof = n.verify().unwrap();
        let elements = if name == "mixed_relocation.inv" {
            3
        } else {
            128
        };
        let pair = Pair::new(
            &n,
            "main",
            compiled::OwnedLimits {
                elements,
                ..storage()
            },
        );
        let l = |fuel| oracle::Limits {
            fuel,
            array_elements: elements,
            ..limits()
        };
        let complete = oracle::observe(&n, &proof, l(1000)).unwrap();
        if name.contains("trap") {
            assert!(complete.semantic.result.is_err(), "{name}")
        } else {
            assert_eq!(complete.semantic.result, Ok(eval::Value::I32(7)), "{name}")
        }
        let mut rows = Vec::new();
        let mut expected = Vec::new();
        for fuel in 1..=complete.semantic.steps_used + 1 {
            rows.push((l(fuel), vec![]));
            expected.push(oracle::observe(&n, &proof, l(fuel)).unwrap());
        }
        for (w, o) in pair.run(&rows).iter().zip(&expected) {
            compare(&pair.layout, o, w);
            comparisons += QUANTA.len();
        }
    }
    eprintln!(
        "ABI11: {comparisons} quantum/fuel observations equal across quanta and to the oracle"
    );
}

#[test]
fn issue_gnet_11_scalar_arguments_bits_and_owner_roots() {
    let n = net(&source("runtime.inv"));
    let proof = n.verify().unwrap();
    let values = [
        0, 0x80000000, 1, 0x007fffff, 0x3f800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fa00001,
        0xffc12345,
    ];
    for entry in ["run", "make"] {
        let pair = Pair::new(&n, entry, storage());
        let mut rows = Vec::new();
        let mut expected = Vec::new();
        for bits in values {
            for length in [0, 1, 3] {
                if entry == "run" && length == 0 {
                    continue;
                }
                let args = [
                    eval::Value::U32(length),
                    eval::Value::F32(F32Bits::from_bits(bits)),
                ];
                rows.push((limits(), carriers(&args)));
                expected.push(oracle::observe_call(&n, &proof, entry, &args, limits()).unwrap());
            }
        }
        for ((w, o), (_, x)) in pair.run(&rows).iter().zip(&expected).zip(&rows) {
            compare(&pair.layout, o, w);
            if entry == "run" {
                // Signaling NaN payloads and signed zero copy exactly.
                assert_eq!(w[1], x[1])
            }
        }
    }
    // Invalid external carriers are refused before any owner is created.
    let pair = Pair::new(&n, "make", storage());
    for bad in [vec![-1, 0], vec![0, 1 << 32], vec![1 << 32, -1]] {
        for w in pair.run(&[(limits(), bad)]) {
            assert_eq!(&w[..10], &[8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        }
    }
}

#[test]
fn issue_gnet_11_storage_trace_and_transition_budgets_across_quanta() {
    let n = net(&source("mixed_relocation.inv"));
    let proof = n.verify().unwrap();
    let pair = Pair::new(&n, "main", storage());
    let mut rows = Vec::new();
    let mut expected = Vec::new();
    for cap in [0, 1, 2, 3, 7, 17, 31, 34] {
        for l in [
            oracle::Limits {
                trace_events: cap as usize,
                ..limits()
            },
            oracle::Limits {
                transitions: cap,
                ..limits()
            },
        ] {
            rows.push((l, vec![]));
            expected.push(oracle::observe(&n, &proof, l).unwrap());
        }
    }
    for (w, o) in pair.run(&rows).iter().zip(&expected) {
        compare(&pair.layout, o, w)
    }
    let heap_limited = oracle::Limits {
        array_elements: 2,
        ..limits()
    };
    let o = oracle::observe(&n, &proof, heap_limited).unwrap();
    assert_eq!(o.failure, Some(oracle::ToolFailure::Allocation));
    let pair = Pair::new(
        &n,
        "main",
        compiled::OwnedLimits {
            elements: 2,
            ..storage()
        },
    );
    compare(&pair.layout, &o, &pair.run(&[(heap_limited, vec![])])[0]);
    // Private backend storage failures stop identically at every quantum.
    for (s, code) in [
        (
            compiled::OwnedLimits {
                allocations: 2,
                ..storage()
            },
            12,
        ),
        (
            compiled::OwnedLimits {
                cleanup_events: 0,
                ..storage()
            },
            13,
        ),
        (
            compiled::OwnedLimits {
                cleanup_events: 1,
                ..storage()
            },
            13,
        ),
    ] {
        let pair = Pair::new(&n, "main", s);
        assert_eq!(pair.run(&[(limits(), vec![])])[0][0], code);
    }
}

#[test]
fn issue_gnet_11_fixed_budget_mutation_and_invalid_quantum_finish_with_status8() {
    let n = net(&source("mixed_relocation.inv"));
    let pair = Pair::new(&n, "main", storage());
    let full = pair.run(&[(limits(), vec![])]).remove(0);
    for q in [1u64, 3, 17] {
        let rows: Vec<_> = ["fuel", "transitions", "trace", "zero", "huge"]
            .into_iter()
            .map(|m| (m, limits(), q, vec![]))
            .collect();
        for w in pair.abi11.batch(&rows) {
            let w = w.expect("first quantum must yield");
            assert_eq!(w[0], 2);
            let w = &w[1..];
            // Terminal status 8 keeps the committed prefix of the fixed run.
            assert_eq!((w[0], w[1], w[5]), (8, 0, q as i64));
            let events = w[7] as usize * 7;
            assert_eq!(&w[10..10 + events], &full[10..10 + events]);
            assert!(w[4] <= full[4] && w[8] <= full[8] && w[9] <= full[9]);
        }
    }
    // Invalid first-call arguments finish an unstarted task with zero counters.
    for q in [0u64, 1_000_001] {
        let w = pair
            .abi11
            .batch(&[("run", limits(), q, vec![])])
            .remove(0)
            .unwrap();
        assert_eq!(&w[..11], &[1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
    for fuel in [0u64, 1_000_001] {
        let l = oracle::Limits { fuel, ..limits() };
        let w = pair
            .abi11
            .batch(&[("run", l, 1, vec![])])
            .remove(0)
            .unwrap();
        assert_eq!(w[1], 8);
    }
}

#[test]
fn issue_gnet_11_identity_admission_and_legacy_profiles_unchanged() {
    let n = net(&source("literals.inv"));
    let proof = n.verify().unwrap();
    let cpu = compiled::Target::Cpu;
    let l = compiled::Limits::default();
    let abi = |r: Result<compiled::Artifact, compiled::Error>| r.unwrap().abi_version();
    assert_eq!(
        abi(compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            cpu,
            l,
            storage()
        )),
        11
    );
    // Metal stays refused. Direct CUDA emission is the distinct ABI12 profile
    // (a compile receipt only since stage 2b; never loaded or executed, see the
    // issue_gnet_12 tests).
    assert_eq!(
        compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            compiled::Target::Metal,
            l,
            storage()
        )
        .unwrap_err(),
        compiled::Error::NumericTarget
    );
    assert_eq!(
        compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            compiled::Target::Cuda,
            l,
            storage()
        )
        .unwrap()
        .abi_version(),
        12
    );
    // No other profile is widened to numeric owners.
    for r in [
        compiled::emit_resumable(&n, &proof, "main", cpu, l, storage()),
        compiled::emit_resumable_aggregates(&n, &proof, "main", cpu, l, storage()),
    ] {
        assert_eq!(r.unwrap_err(), compiled::Error::UnsupportedType);
    }
    for (s, entry) in [
        ("fn entry(xs:read Array<f32>)->i32{return 7;}", "entry"),
        ("fn entry(xs:take Array<i32>)->i32{return 7;}", "entry"),
        ("struct Numeric{x:f32} fn main()->i32{return 7;}", "main"),
        ("struct Old{x:i32} fn main()->i32{return 7;}", "main"),
    ] {
        let n = net(s);
        let proof = n.verify().unwrap();
        assert_eq!(
            compiled::emit_resumable_owned_numeric(&n, &proof, entry, cpu, l, storage())
                .unwrap_err(),
            compiled::Error::UnsupportedType,
            "{s}"
        );
    }
    // Legacy scalar profiles keep their numbers and emit no ABI11 layout check.
    let scalar = net("fn run(x:i32)->i32{return x;} fn main()->i32{return 7;}");
    let p = scalar.verify().unwrap();
    for (r, version) in [
        (compiled::emit(&scalar, &p, "run", cpu, l), 1),
        (
            compiled::emit_owned(&scalar, &p, "run", cpu, l, storage()),
            2,
        ),
        (
            compiled::emit_resumable(&scalar, &p, "run", cpu, l, storage()),
            3,
        ),
        (
            compiled::emit_aggregates(&scalar, &p, "run", cpu, l, storage()),
            6,
        ),
        (
            compiled::emit_resumable_aggregates(&scalar, &p, "run", cpu, l, storage()),
            7,
        ),
        (
            compiled::emit_resumable_owned_numeric(&scalar, &p, "run", cpu, l, storage()),
            11,
        ),
    ] {
        let a = r.unwrap();
        assert_eq!(a.abi_version(), version);
        assert_eq!(
            a.source().contains("GNET ABI11 layout mismatch"),
            version == 11
        );
    }
    // Storage limits are validated before emission.
    assert_eq!(
        compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            cpu,
            l,
            compiled::OwnedLimits {
                allocations: 0,
                ..storage()
            }
        )
        .unwrap_err(),
        compiled::Error::Limit
    );
}

#[test]
fn issue_gnet_11_cpu_worklist_independent_roots_slices_and_lane_isolation() {
    let n = net(&source("runtime.inv"));
    let proof = n.verify().unwrap();
    let eleven = compiled::emit_resumable_owned_numeric(
        &n,
        &proof,
        "run",
        compiled::Target::Cpu,
        compiled::Limits::default(),
        storage(),
    )
    .unwrap();
    let worklist = Cpu::build_driver(&eleven, WORKLIST, "");
    let pair = Pair::new(&n, "run", storage());
    // Success, signed zero, signaling NaN, an index trap (empty owner), a
    // private heap limit and infinity occupy adjacent independent lanes.
    let inputs: Vec<Vec<i64>> = vec![
        vec![3, 0x80000000],
        vec![0, 0x3f800000],
        vec![1, 0x7fa00001],
        vec![70, 0],
        vec![7, 0xff800000],
    ];
    let mut statuses = Vec::new();
    let mut observations = 0;
    for l in [
        limits(),
        oracle::Limits {
            trace_events: 5,
            ..limits()
        },
        oracle::Limits {
            transitions: 40,
            ..limits()
        },
    ] {
        let rows: Vec<_> = inputs.iter().map(|x| (l, x.clone())).collect();
        // Separate single-root runs (identical at every quantum) are the
        // reference for each lane.
        let single = pair.run(&rows);
        for q in [1u64, 3, 17, 1_000_000] {
            let resumed: Vec<_> = inputs.iter().map(|x| ("run", l, q, x.clone())).collect();
            let quanta: i64 = pair
                .abi11
                .batch(&resumed)
                .into_iter()
                .map(|w| w.unwrap()[0])
                .sum();
            for slices in [1u64, 2, 5, 1_000_000] {
                let w = worklist.worklist(l, inputs.len() as u64, q, slices, &inputs);
                assert_eq!(w.lanes, single, "quantum {q} slices {slices}");
                let (last, sliced) = w.calls.split_last().unwrap();
                assert_eq!(last[0], 0);
                assert_eq!(last[2], inputs.len() as i64);
                for c in sliced {
                    assert_eq!((c[0], c[1]), (14, slices as i64));
                }
                // Every submission is one exclusive root quantum.
                let submitted: i64 = w.calls.iter().map(|c| c[1]).sum();
                assert_eq!(submitted, quanta, "quantum {q} slices {slices}");
                observations += 1;
            }
        }
        statuses.extend(single.iter().map(|w| w[0]));
    }
    // Lanes reached success, index trap, heap limit, trace and transition limits.
    for code in [0, 9, 11, 6, 5] {
        assert!(statuses.contains(&code), "status {code}: {statuses:?}");
    }
    // Zero roots complete immediately; zero slices with pending roots yield 14
    // without submitting work, and a later call resumes them.
    let empty = worklist.worklist(limits(), 0, 1, 1, &[]);
    assert_eq!((empty.calls, empty.lanes.len()), (vec![[0, 0, 0]], 0));
    let stalled = worklist.worklist(limits(), 2, 1, 0, &inputs[..2]);
    assert_eq!(stalled.calls, vec![[14, 0, 0]]);
    // Invalid worklist arguments return 8 before touching any task or buffer.
    for (roots, fuel, quantum, slices) in [
        (1025, 1000, 1, 1),
        (2, 1000, 0, 1),
        (2, 1000, 1_000_001, 1),
        (2, 1000, 1, 1_000_001),
        (2, 0, 1, 1),
        (2, 1_000_001, 1, 1),
    ] {
        let lanes = usize::try_from(roots).unwrap().min(64);
        let filler = vec![vec![1, 0]; lanes];
        let l = oracle::Limits { fuel, ..limits() };
        let w = worklist.worklist(l, roots, quantum, slices, &filler);
        assert_eq!(w.calls, vec![[8, 0, 0]]);
        assert!(w.lanes.is_empty());
    }
    eprintln!("ABI11 worklist: {observations} multi-root executions equal to single roots");
}
