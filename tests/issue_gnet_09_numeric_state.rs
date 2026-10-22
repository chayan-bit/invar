#![forbid(unsafe_code)]
use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
    numeric_contract::F32Bits,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

const SOURCE: &str = "fn recur(a:f32,b:f32,c:f32,n:u32)->f32 {if n==0u32{return a;}return recur(fma(a,b,c),b,c,n-1u32);} fn run(a:f32,b:f32,c:f32,n:u32)->f32 {var x=a;for i in 0u32..n{x=recur(x,b,c,2u32);}return x;}";
const INPUTS: [[i64; 4]; 3] = [
    [0x7f7fffff, 0x40000000, 0xff7fffff, 1],
    [0x80000000, 0x3f800000, 0x80000000, 2],
    [0x7fa12345, 0x3f800001, 1, 3],
];
const FUEL: u64 = 10000;
const CAP: usize = 10000;

struct Cpu {
    dir: PathBuf,
    state_words: usize,
    net: Net,
}
impl Cpu {
    fn new() -> Self {
        let c = check::check(front::parse(SOURCE.as_bytes()).unwrap(), false).unwrap();
        let net = Net::lower(&c, general_net::Limits::default()).unwrap();
        let artifact = compiled::emit_resumable_numeric(
            &net,
            &net.verify().unwrap(),
            "run",
            compiled::Target::Cpu,
            compiled::Limits::default(),
        )
        .unwrap();
        let state_words = artifact.continuation_layout().unwrap().task_bytes() / 8;
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gnet-numeric-state-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let driver = DRIVER
            .replace("@FUEL@", &FUEL.to_string())
            .replace("@CAP@", &CAP.to_string())
            .replace("@BYTES@", &(state_words * 8).to_string());
        fs::write(
            dir.join("source.cpp"),
            format!("{}\n{driver}", artifact.source()),
        )
        .unwrap();
        let output = Command::new("c++")
            .args([
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
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self {
            dir,
            state_words,
            net,
        }
    }
    fn run(&self, mode: u8, input: &[i64]) -> Vec<(String, Vec<i64>)> {
        let output = Command::new(self.dir.join("run"))
            .arg(mode.to_string())
            .args(input.iter().map(ToString::to_string))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "mode {mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| {
                let mut words = line.split_whitespace();
                (
                    words.next().unwrap().to_owned(),
                    words.map(|x| x.parse().unwrap()).collect(),
                )
            })
            .collect()
    }
    fn expected(&self, input: &[i64]) -> Vec<i64> {
        let mut args: Vec<_> = input[..3]
            .iter()
            .map(|x| eval::Value::F32(F32Bits::from_bits(*x as u32)))
            .collect();
        args.push(eval::Value::U32(input[3] as u32));
        let o = oracle::observe_call(
            &self.net,
            &self.net.verify().unwrap(),
            "run",
            &args,
            oracle::Limits {
                fuel: FUEL,
                transitions: 100000,
                trace_events: CAP,
                ..Default::default()
            },
        )
        .unwrap();
        let Ok(eval::Value::F32(result)) = o.semantic.result else {
            panic!("{o:?}")
        };
        let mut words = vec![
            0,
            i64::from(result.to_bits()),
            0,
            0,
            o.semantic.steps_used as i64,
            o.transitions as i64,
            o.max_frames as i64,
            o.events.len() as i64,
        ];
        for event in o.events {
            words.extend([
                i64::from(event.kind == oracle::EventKind::Complete),
                event.frame as i64,
                event.activation as i64,
                event.token.block.function as i64,
                event.token.block.block as i64,
                event.token.position as i64,
                event.rule as i64,
            ]);
        }
        words
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}

#[test]
fn issue_gnet_09_each_yield_preserves_complete_task_and_exact_event_prefix() {
    let cpu = Cpu::new();
    for input in INPUTS {
        let expected = cpu.expected(&input);
        let mut reference = BTreeMap::new();
        for (kind, state) in cpu.run(0, &input) {
            if kind == "R" {
                assert_eq!(state, expected);
                continue;
            }
            assert_eq!(kind, "P");
            assert_eq!(state[0], 1); // Input captured exactly once.
            assert_eq!(&state[8..11], &[0, 0, 0]);
            assert_eq!(&state[11..14], &[FUEL as i64, 100000, CAP as i64]);
            assert_eq!(state.len(), cpu.state_words + state[6] as usize * 7);
            let events = &state[cpu.state_words..];
            assert_eq!(events, &expected[8..8 + events.len()]);
            if state[1] == 0 {
                assert!(state[2] > 0);
                assert_eq!(state[14], 0);
            }
            assert!(reference.insert(state[5], state).is_none());
        }
        assert!(reference.len() > 30);
        for (kind, state) in cpu.run(1, &input) {
            if kind == "R" {
                assert_eq!(state, expected);
            } else {
                assert_eq!(&state, reference.get(&state[5]).unwrap());
            }
        }
    }
}

#[test]
fn issue_gnet_09_changed_resume_limits_fail_before_more_work() {
    let cpu = Cpu::new();
    for mode in 2..=6 {
        let rows = cpu.run(mode, &INPUTS[0]);
        assert_eq!(rows.len(), 3);
        let before = &rows[0].1;
        let after = &rows[1].1;
        assert_eq!(&before[2..14], &after[2..14]);
        assert_eq!(&before[18..], &after[18..]);
        assert_eq!(after[1], 1);
        assert_eq!(after[14], compiled::Status::InvalidInput as i64);
        assert_eq!(&rows[2].1[..4], &[8, 0, 0, 0]);
        assert_eq!(
            &rows[2].1[4..8],
            &[before[4], before[5], before[3], before[6]]
        );
    }
}

#[test]
fn issue_gnet_09_worklist_owns_three_scalar_tasks_and_eight_word_outputs() {
    let cpu = Cpu::new();
    let inputs: Vec<_> = INPUTS.into_iter().flatten().collect();
    let rows = cpu.run(7, &inputs);
    assert_eq!(rows[0].0, "W");
    assert_eq!(rows[0].1, [14, 3, 0]);
    let mut completed = 0;
    for (kind, words) in &rows {
        if kind == "W" {
            assert!(words[2] >= completed);
            completed = words[2];
        }
    }
    assert_eq!(completed, 3);
    let results: Vec<_> = rows.iter().filter(|(kind, _)| kind == "R").collect();
    assert_eq!(results.len(), 3);
    for (row, input) in results.iter().zip(INPUTS) {
        assert_eq!(row.1, cpu.expected(&input));
    }
    assert_eq!(rows.last().unwrap().1, [0, 0, 3]);
}

const DRIVER: &str = r#"
#include <iostream>
#include <vector>
#include <cstring>
#include <cfenv>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
static_assert(sizeof(GnetTask)==@BYTES@);
void state(const GnetTask& t,const gnet_u* trace) {
 std::cout<<"P ";
 for(auto x:{t.started,t.done,t.depth,t.max_depth,t.steps,t.transitions,t.used,t.next_frame,t.allocated,t.live,t.cleanup_used,t.fuel,t.transition_cap,t.trace_cap}) std::cout<<x<<' ';
 for(auto x:{t.status,t.result,t.start,t.end}) std::cout<<x<<' ';
 for(const auto& f:t.frames) {for(auto x:f.v) std::cout<<x<<' ';for(auto x:f.p) std::cout<<x<<' ';for(auto x:{f.pc,f.ret,f.id,f.activation}) std::cout<<x<<' ';}
 for(gnet_u i=0;i<t.used*7;++i) std::cout<<trace[i]<<' ';
 std::cout<<'\n';
}
void result(const gnet_w* output,const gnet_u* trace) {
 std::cout<<"R ";for(int i=0;i<8;++i) std::cout<<output[i]<<' ';
 for(gnet_u i=0;i<(gnet_u)output[7]*7;++i) std::cout<<trace[i]<<' ';std::cout<<'\n';
}
int main(int argc,char** argv) {
 int mode=std::stoi(argv[1]); std::vector<gnet_w> input;
 for(int i=2;i<argc;++i) input.push_back(std::stoll(argv[i]));
 if(std::fesetround(FE_DOWNWARD)) return 77;
#if defined(__SSE__)
 _mm_setcsr(_mm_getcsr()|0x8040);
#endif
 if(mode==7) {
  GnetTask tasks[3]={};std::vector<gnet_w> outputs(26,123456789);std::vector<gnet_u> trace(3*@CAP@*7+1,987654321);
  gnet_u submitted,completed;gnet_w code=gnet_worklist(input.data(),outputs.data()+1,trace.data(),3,@FUEL@,100000,@CAP@,nullptr,nullptr,tasks,1,3,submitted,completed);
  std::cout<<"W "<<code<<' '<<submitted<<' '<<completed<<'\n';if(code!=14||submitted!=3||completed!=0) return 10;
  for(auto& x:input) x=-1;for(auto x:outputs) if(x!=123456789) return 11;
  for(int round=0;code==14&&round<100000;++round) {
   code=gnet_worklist(input.data(),outputs.data()+1,trace.data(),3,@FUEL@,100000,@CAP@,nullptr,nullptr,tasks,round%3+1,2,submitted,completed);
   std::cout<<"W "<<code<<' '<<submitted<<' '<<completed<<'\n';
   for(int lane=0;lane<3;++lane) {auto& t=tasks[lane];if(t.allocated||t.live||t.cleanup_used) return 12;if(!t.done) for(int j=0;j<8;++j) if(outputs[1+lane*8+j]!=123456789) return 13;}
  }
  if(code!=0||completed!=3||outputs.front()!=123456789||outputs.back()!=123456789||trace.back()!=987654321) return 14;
  for(int lane=0;lane<3;++lane) result(outputs.data()+1+lane*8,trace.data()+lane*@CAP@*7);
  GnetTask copy[3];std::memcpy(copy,tasks,sizeof(tasks));auto old_outputs=outputs;auto old_trace=trace;
  code=gnet_worklist(nullptr,outputs.data()+1,trace.data(),3,@FUEL@,100000,@CAP@,nullptr,nullptr,tasks,1,1,submitted,completed);
  if(code||submitted||completed!=3||std::memcmp(copy,tasks,sizeof(tasks))||outputs!=old_outputs||trace!=old_trace) return 15;
  std::cout<<"W "<<code<<' '<<submitted<<' '<<completed<<'\n';return 0;
 }
 GnetTask task={};gnet_w output[8];for(auto& x:output) x=123456789;std::vector<gnet_u> trace(@CAP@*7);
 bool done=false;gnet_u round=0;
 do {
  int rounding=mode==1?(round%4):2;if(std::fesetround(rounding==0?FE_TONEAREST:rounding==1?FE_UPWARD:rounding==2?FE_DOWNWARD:FE_TOWARDZERO)) return 77;
  gnet_u quantum=mode==1?((round%3==0)?1:(round%3==1)?3:17):1;
  done=gnet_resume(round?nullptr:input.data(),output,trace.data(),@FUEL@,100000,@CAP@,nullptr,nullptr,&task,quantum);
  state(task,trace.data());if(!done) for(auto x:output) if(x!=123456789) return 20;
  if(!round) for(auto& x:input) x=-1;
  if(mode>=2&&!done) {
   done=gnet_resume(nullptr,output,trace.data(),@FUEL@+(mode==2),100000+(mode==3),@CAP@+(mode==4),nullptr,nullptr,&task,mode==5?0:mode==6?1000001:1);
   if(!done) return 21;state(task,trace.data());break;
  }
  if(++round>100000) return 22;
 } while(!done);
 GnetTask copy;std::memcpy(&copy,&task,sizeof(task));gnet_w before[8];std::memcpy(before,output,sizeof(output));auto old_trace=trace;
 if(!gnet_resume(nullptr,output,trace.data(),0,0,0,nullptr,nullptr,&task,0)||std::memcmp(&copy,&task,sizeof(task))||std::memcmp(before,output,sizeof(output))||trace!=old_trace) return 23;
 result(output,trace.data());
}
"#;
