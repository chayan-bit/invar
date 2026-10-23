#![cfg(unix)]
#![forbid(unsafe_code)]
#[path = "../support/executable.rs"]
mod executable;

use std::{fs, os::unix::fs::DirBuilderExt, process::Command};

#[test]
fn generated_executables_survive_parallel_forks() {
    let root = std::env::temp_dir().join(format!("invar-exec-stress-{}", std::process::id()));
    fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
    // Large comments widen the old writable-fd window without changing behavior.
    let mut script = b"#!/bin/sh\nprintf 'generated-once\\n'\n#".to_vec();
    script.extend(std::iter::repeat_n(b'x', 1024 * 1024));
    script.push(b'\n');
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for worker in 0..16 {
            let root = &root;
            let script = &script;
            workers.push(scope.spawn(move || {
                for iteration in 0..8 {
                    let path = root.join(format!("{worker}-{iteration}"));
                    executable::write(&path, script).unwrap();
                    let output = Command::new(&path).output().unwrap();
                    assert!(output.status.success(), "{:?}", output);
                    assert_eq!(output.stdout, b"generated-once\n");
                    assert!(output.stderr.is_empty());
                    fs::remove_file(path).unwrap();
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
    fs::remove_dir(root).unwrap();
}
