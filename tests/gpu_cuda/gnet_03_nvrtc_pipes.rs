//! The bounded NVRTC subprocess keeps only a capped prefix of its log and
//! image but must drain both pipes: a child whose stderr was closed after the
//! cap failed its next write (EPIPE) and a successful compile became a crash.
//! Draining must still end within the compile deadline: a descendant that
//! keeps the pipes open after the compiler exits is killed with its process
//! group, and one that escaped the group ends in a typed timeout.
//! A stand-in compiler makes this deterministic without a GPU or NVRTC.
#![forbid(unsafe_code)]

use invar_cuda::{
    Error,
    cuda::{CompiledCubin, compile_cubin_bounded, compile_gnet_cubin_bounded},
};
use invar_device::Capabilities;
use std::{
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    sync::OnceLock,
    time::{Duration, Instant},
};

const CAPS: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 0,
    sm: 120,
};
const SOURCE: &str = "extern \"C\" __global__ void k() {}";
/// The ordinary (non-general-net) compile deadline.
const DEADLINE: Duration = Duration::from_secs(20);

/// Stand-in compiler bodies. Each consumes the source, then runs its body.
const COMPILERS: &[(&str, &str)] = &[
    // About 200 KiB of warnings in many small writes, far past the pipe
    // buffer and the 512-byte log cap; any failed write exits 7.
    (
        "long-log",
        "i=0; while [ $i -lt 4000 ]; do printf 'warning %05d: variable was set but never used\\n' $i >&2 || exit 7; i=$((i+1)); done; printf cubin",
    ),
    // Above the ordinary 1 MiB cap, below the GNET 16 MiB cap.
    ("gnet-image", "head -c 2097152 /dev/zero || exit 7"),
    // Twice the 16 MiB general-net CUBIN cap, well past what the capped
    // reader and the pipe buffer hold.
    ("big-image", "head -c 33554432 /dev/zero || exit 7"),
    // Over both caps, then a background child keeps both pipes open for 26 s
    // after the compiler exits.
    (
        "lingering",
        "head -c 2097152 /dev/zero; head -c 4096 /dev/zero >&2; sleep 26 & exit 0",
    ),
    ("hang", "sleep 1000"),
    // Exits only once the descendant leads its own session, so the group kill
    // at exit can never reach it.
    (
        "escaped",
        "setsid sleep 45 & p=$!; until [ \"$(cut -d' ' -f6 /proc/$p/stat)\" = \"$p\" ]; do :; done; printf cubin; exit 0",
    ),
];

/// Path of a stand-in compiler. Every script is written once, before any test
/// can spawn: a fork while a script is still open for writing would make its
/// exec fail with ETXTBSY.
fn compiler(name: &str) -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("gnet03-nvrtc-pipes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, body) in COMPILERS {
            let path = dir.join(name);
            std::fs::write(&path, format!("#!/bin/sh\ncat >/dev/null\n{body}\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        dir
    })
    .join(name)
}

/// The result without the image, for assertion messages.
fn shown(result: &Result<CompiledCubin, Error>) -> Result<usize, &Error> {
    result.as_ref().map(|c| c.image.len())
}

#[test]
fn issue_gnet_03_nvrtc_long_log_keeps_a_successful_compile() {
    let compiled = compile_gnet_cubin_bounded(SOURCE, &compiler("long-log"), CAPS)
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(compiled.image, b"cubin");
    assert!(compiled.log.starts_with("warning 00000"));
    assert!(compiled.log.len() <= 513, "{}", compiled.log.len());
}

#[test]
fn issue_gnet_03_nvrtc_image_uses_the_general_net_cap() {
    let compiled = compile_gnet_cubin_bounded(SOURCE, &compiler("gnet-image"), CAPS)
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(compiled.image.len(), 2 * 1024 * 1024);
    let ordinary = compile_cubin_bounded(SOURCE, &compiler("gnet-image"), CAPS);
    assert!(
        matches!(ordinary, Err(Error::Limit)),
        "{:?}",
        shown(&ordinary)
    );
}

#[test]
fn issue_gnet_03_nvrtc_oversized_image_is_a_limit_not_a_crash() {
    let result = compile_gnet_cubin_bounded(SOURCE, &compiler("big-image"), CAPS);
    assert!(matches!(result, Err(Error::Limit)), "{:?}", shown(&result));
}

#[test]
fn issue_gnet_03_nvrtc_descendant_holding_pipes_is_killed_at_exit() {
    let begun = Instant::now();
    let result = compile_cubin_bounded(SOURCE, &compiler("lingering"), CAPS);
    let took = begun.elapsed();
    assert!(matches!(result, Err(Error::Limit)), "{:?}", shown(&result));
    assert!(took < Duration::from_secs(5), "{took:?}");
}

#[test]
fn issue_gnet_03_nvrtc_compiler_that_never_exits_times_out() {
    let begun = Instant::now();
    let result = compile_cubin_bounded(SOURCE, &compiler("hang"), CAPS);
    let took = begun.elapsed();
    assert!(
        matches!(result, Err(Error::Timeout)),
        "{:?}",
        shown(&result)
    );
    assert!(
        took >= DEADLINE && took < DEADLINE + Duration::from_secs(5),
        "{took:?}"
    );
}

/// A descendant in its own session survives the group kill and holds the
/// pipes; the compile still ends at the deadline with a typed timeout.
#[cfg(target_os = "linux")]
#[test]
fn issue_gnet_03_nvrtc_escaped_descendant_ends_at_the_deadline() {
    let begun = Instant::now();
    let result = compile_cubin_bounded(SOURCE, &compiler("escaped"), CAPS);
    let took = begun.elapsed();
    assert!(
        matches!(result, Err(Error::Timeout)),
        "{:?}",
        shown(&result)
    );
    assert!(
        took >= DEADLINE && took < DEADLINE + Duration::from_secs(5),
        "{took:?}"
    );
}
