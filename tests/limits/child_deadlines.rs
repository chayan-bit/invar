#![forbid(unsafe_code)]
#[cfg_attr(not(unix), allow(dead_code))]
#[path = "../../test-support/child_deadline.rs"]
mod child_deadline;

use std::{process::Command, time::Duration};

#[test]
fn scale_is_explicit_bounded_and_cannot_disable_deadlines() {
    for invalid in ["", "0", "65", "-1", "1.5", " 2", "2 ", "4294967296"] {
        assert!(child_deadline::parse_scale(invalid).is_err(), "{invalid:?}");
    }
    assert_eq!(child_deadline::parse_scale("1"), Ok(1));
    assert_eq!(child_deadline::parse_scale("64"), Ok(64));
    if let Ok(expected) = std::env::var("INVAR_DEADLINE_EXPECT_SECONDS") {
        assert_eq!(child_deadline::seconds(10).to_string(), expected);
        assert_eq!(
            child_deadline::wall(Duration::from_millis(25)),
            Duration::from_millis(25 * expected.parse::<u64>().unwrap() / 10)
        );
        return;
    }
    for scale in [None, Some("1"), Some("8"), Some("64")] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "child_deadlines::scale_is_explicit_bounded_and_cannot_disable_deadlines",
            ])
            .env_remove("INVAR_TEST_TIMEOUT_SCALE")
            .env(
                "INVAR_DEADLINE_EXPECT_SECONDS",
                (10 * scale.unwrap_or("1").parse::<u64>().unwrap()).to_string(),
            );
        if let Some(scale) = scale {
            command.env("INVAR_TEST_TIMEOUT_SCALE", scale);
        }
        assert!(command.status().unwrap().success());
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_deadlines::scale_is_explicit_bounded_and_cannot_disable_deadlines",
        ])
        .env("INVAR_DEADLINE_EXPECT_SECONDS", "10")
        .env("INVAR_TEST_TIMEOUT_SCALE", "0")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("INVAR_TEST_TIMEOUT_SCALE"));
}

#[cfg(unix)]
#[test]
fn hung_child_is_killed_even_when_it_ignores_term() {
    if std::env::var_os("INVAR_DEADLINE_HUNG_CHILD").is_some() {
        let output = child_deadline::command(1)
            .args(["sh", "-c", "trap '' TERM; while :; do sleep 1; done"])
            .output()
            .unwrap();
        use std::os::unix::process::ExitStatusExt;
        assert!(
            output.status.code() == Some(137) || output.status.signal() == Some(9),
            "{output:?}"
        );
        eprintln!(
            "hung child: forced kill (SIGKILL/137), stdout={} stderr={}",
            output.stdout.len(),
            output.stderr.len()
        );
        return;
    }
    // Isolated environment: the normal suite's scale cannot prolong this kill check.
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_deadlines::hung_child_is_killed_even_when_it_ignores_term",
            "--nocapture",
        ])
        .env("INVAR_TEST_TIMEOUT_SCALE", "2")
        .env("INVAR_DEADLINE_HUNG_CHILD", "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
}

#[cfg(unix)]
#[test]
fn cxx_builds_a_harness_and_reports_compiler_errors() {
    let dir = std::env::temp_dir().join(format!("invar-cxx-helper-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("ok.cpp"), "int main() { return 0; }\n").unwrap();
    child_deadline::cxx(
        child_deadline::CXX17_UBSAN_O1,
        &dir.join("ok.cpp"),
        &dir.join("ok"),
    );
    assert!(Command::new(dir.join("ok")).status().unwrap().success());
    std::fs::write(
        dir.join("bad.cpp"),
        "int main() { return undeclared_name; }\n",
    )
    .unwrap();
    let failure = std::panic::catch_unwind(|| {
        child_deadline::cxx(
            child_deadline::CXX17_UBSAN_O2,
            &dir.join("bad.cpp"),
            &dir.join("bad"),
        )
    })
    .unwrap_err();
    let message = failure.downcast_ref::<String>().unwrap();
    assert!(message.contains("undeclared_name"), "{message}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[test]
fn assert_success_accepts_a_clean_exit_and_panics_with_stderr() {
    let ok = Command::new("sh").args(["-c", "exit 0"]).output().unwrap();
    child_deadline::assert_success(&ok);
    let failed = Command::new("sh")
        .args(["-c", "echo child-stderr >&2; exit 3"])
        .output()
        .unwrap();
    let failure = std::panic::catch_unwind(|| child_deadline::assert_success(&failed)).unwrap_err();
    let message = failure.downcast_ref::<String>().unwrap();
    assert!(message.contains("child-stderr"), "{message}");
}
