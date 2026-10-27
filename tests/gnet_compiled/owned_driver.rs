//! The C++ driver shared by the owned (ABI2/3) and aggregate (ABI6/7) CPU
//! tests. It runs an emitted owned-storage artifact once, or resumes it by
//! `quantum` transitions until it completes and then checks that one more
//! resume is a no-op; it prints the header, trace, cleanup and heap words.

use crate::child_deadline;
use invar::general_net::{compiled, oracle};
use std::{
    fmt::Debug,
    fs,
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
};

/// Compile `artifact` with the driver into a fresh `invar-{name}-*` directory.
pub fn build(name: &str, artifact: &compiled::Artifact, quantum: Option<u64>) -> PathBuf {
    let run = if let Some(quantum) = quantum {
        let bytes = artifact.continuation_layout().unwrap().task_bytes();
        format!(
            "static_assert(sizeof(GnetTask)=={bytes}); GnetTask task={{}}; unsigned rounds=0; while(!gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,{quantum})) {{ if(++rounds>1000001) return 3; }} auto before=task.transitions; if(!gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,{quantum}) || task.transitions!=before) return 4;"
        )
    } else {
        "gnet_run(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data());"
            .to_owned()
    };
    let layout = artifact.owned_layout().unwrap();
    let dir = child_deadline::scratch(&format!("invar-{name}"));
    let driver = format!(
        r#"
#include <iostream>
#include <vector>
#include <string>
int main(int argc,char** argv) {{
    gnet_u fuel=std::stoull(argv[1]),trans=std::stoull(argv[2]),cap=std::stoull(argv[3]);
    std::vector<gnet_w> input; for(int i=4;i<argc;++i) input.push_back(std::stoll(argv[i]));
    gnet_w output[10]={{}}; std::vector<gnet_u> trace(cap*7),cleanup({clean}); std::vector<gnet_w> heap({heap});
    {run}
    for(auto word:output) std::cout<<word<<' ';
    for(gnet_u i=0;i<(gnet_u)output[7]*7;++i) std::cout<<trace[i]<<' ';
    for(gnet_u i=0;i<(gnet_u)output[8]*4;++i) std::cout<<cleanup[i]<<' ';
    for(auto word:heap) std::cout<<word<<' ';
}}
"#,
        run = run,
        clean = layout.cleanup_words(),
        heap = layout.heap_words()
    );
    fs::write(
        dir.join("source.cpp"),
        format!("{}\n{driver}", artifact.source()),
    )
    .unwrap();
    child_deadline::cxx(
        child_deadline::CXX17_UBSAN_O1,
        &dir.join("source.cpp"),
        &dir.join("run"),
    );
    dir
}

/// Run the driver built in `dir` on `args` under `limits` and parse its words.
pub fn run<T: FromStr>(dir: &Path, args: &[i64], limits: oracle::Limits) -> Vec<T>
where
    T::Err: Debug,
{
    let output = Command::new(dir.join("run"))
        .args([
            limits.fuel.to_string(),
            limits.transitions.to_string(),
            limits.trace_events.to_string(),
        ])
        .args(args.iter().map(ToString::to_string))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}
