//! CPU-only production metadata, compiler-controller and owning-fence tests.
//! No CUDA open, library probe, NVRTC call or hardware launch.
use crate::child_deadline;

use super::*;
use invar_device::{Provenance, Span, floating::Buffer};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

const CAPS: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 4,
    sm: 86,
};

#[test]
fn issue_m4_transpose_runtime_capacity_shortfall_precedes_storage() {
    let p = plan([3, 17], 2);
    let s = shape(&p, CAPS, 55).unwrap();
    let allocations = Cell::new(0);
    let result = admit_before_storage(
        || crate::validate_device_capacity(s.transfer_bytes, s.transfer_bytes),
        || {
            allocations.set(allocations.get() + 1);
            Ok(())
        },
    );
    assert_eq!(allocations.get(), 0);
    assert!(matches!(result, Err(Error::ResourceLimit(_))));
}

fn origin() -> Provenance {
    Provenance {
        source_identity: 71,
        graph_revision: 8,
    }
}
fn plan(dimensions: [u32; 2], padding: usize) -> ir::Verified {
    let [r, c] = dimensions;
    let buffer = |id, shape: [u32; 2]| Buffer {
        id,
        elements: u64::from(shape[0]) * u64::from(shape[1]) + padding as u64 * 2,
        shape,
        strides: [u64::from(shape[1]), 1],
        offset: padding as u64,
        alignment: 4,
    };
    ir::verify(
        ir::Ir {
            version: 1,
            layout_version: 1,
            span: Span { start: 1, end: 90 },
            source_identity: 71,
            graph_revision: 8,
            function_contract: [17; 32],
            dimensions,
            input: buffer(1, [r, c]),
            output: buffer(2, [c, r]),
        },
        CAPS,
        origin(),
    )
    .unwrap()
}

#[test]
fn issue_m4_transpose_runtime_exact_capacities_caps_budget_and_empty() {
    let p = plan([3, 17], 2);
    let s = shape(&p, CAPS, 55).unwrap();
    assert_eq!(s.lengths, [55, 55]);
    assert_eq!((s.logical, s.offset, s.transfer_bytes), (51, 2, 55 * 3 * 4));
    for n in [0, 51, 54, 56] {
        assert!(matches!(shape(&p, CAPS, n), Err(Error::Limit)));
    }
    assert!(matches!(
        shape(
            &p,
            Capabilities::Cuda {
                major: 12,
                minor: 5,
                sm: 86
            },
            55
        ),
        Err(Error::Limit)
    ));
    assert!(matches!(
        crate::validate_transfer_budget(crate::MAX_TRANSFER_BYTES + 1, 0),
        Err(Error::ResourceLimit(_))
    ));
    let square = plan([16384, 16384], 0);
    let s = shape(&square, CAPS, 1 << 28).unwrap();
    assert_eq!(s.transfer_bytes, 3 * 1024 * 1024 * 1024);
    // The family cap counts 2 * input + 3 * output words, not just transfers.
    // The square is exactly 5 GiB; padding either buffer already exceeds it.
    for (input, output) in [
        ((1 << 28) + 1, 1 << 28),
        (1 << 28, (1 << 28) + 1),
        (1 << 29, 1 << 28),
        (1 << 28, 1 << 29),
    ] {
        let mut padded = p.ir().clone();
        padded.input.elements = input;
        padded.output.elements = output;
        let padded = ir::verify(padded, CAPS, origin()).unwrap();
        assert!(matches!(
            shape(&padded, CAPS, input as usize),
            Err(Error::ResourceLimit("M4-02 owned storage exceeds 5 GiB"))
        ));
    }
    let mut over = p.ir().clone();
    over.input.elements = 1 << 29;
    over.output.elements = 1 << 29;
    let over = ir::verify(over, CAPS, origin()).unwrap();
    assert!(matches!(
        shape(&over, CAPS, 1 << 29),
        Err(Error::ResourceLimit(_))
    ));
    let mut large = p.ir().clone();
    large.input.elements = invar_device::floating::MAX_ELEMENTS;
    large.output.elements = invar_device::floating::MAX_ELEMENTS;
    let large = ir::verify(large, CAPS, origin()).unwrap();
    assert!(matches!(
        shape(&large, CAPS, invar_device::floating::MAX_ELEMENTS as usize),
        Err(Error::ResourceLimit(_))
    ));
    for d in [[0, 17], [3, 0], [0, 0]] {
        for padding in [0, 2] {
            let p = plan(d, padding);
            let s = shape(&p, CAPS, p.ir().input.elements as usize).unwrap();
            assert_eq!(s.logical, 0);
            validate_output(&s, &zeros(s.lengths[1]).unwrap()).unwrap();
            assert!(!ir::emit_cuda(&p).unwrap().contains("__syncthreads"));
        }
    }
}

#[test]
fn issue_m4_transpose_runtime_distinct_physical_capacities_and_base_alignment() {
    let mut p = plan([3, 17], 2).ir().clone();
    p.input.elements += 3;
    p.output.elements += 7;
    let p = ir::verify(p, CAPS, origin()).unwrap();
    let s = shape(&p, CAPS, 58).unwrap();
    assert_eq!(s.lengths, [58, 62]);
    assert_eq!(s.transfer_bytes, (58 + 62 * 2) * 4);
    let mut output = vec![0; 62];
    output[2..53].fill(0x7f800001);
    validate_output(&s, &output).unwrap();
    for (ptr, alignment) in [(0, 4), (2, 4), (4, 8), (16, 3), (16, 0), (16, 2)] {
        assert!(matches!(
            validate_address(ptr, alignment),
            Err(Error::Limit)
        ));
    }
    for pair in [(4, 4), (256, 4), (256, 256)] {
        validate_address(pair.0, pair.1).unwrap();
    }
}

#[test]
fn issue_m4_transpose_runtime_exact_options_and_early_helper_refusals() {
    let p = plan([3, 17], 1);
    let source = ir::emit_cuda(&p).unwrap();
    assert_eq!(plan_options(&source, CAPS).unwrap(), p.nvrtc_options());
    for bad in [
        source.replace("element-bytes=4", "element-bytes=8"),
        source.replace("layout-version=1", "layout-version=2"),
        source.replace("--modify-stack-limit=false", "--modify-stack-limit=true"),
        source.replace("compute_86", "compute_80"),
        source.replace("caps=Cuda", "caps=Unknown"),
    ] {
        assert!(matches!(plan_options(&bad, CAPS), Err(Error::Compile(_))));
        // This fails before the optional library probe, not an NVRTC execution.
        assert!(matches!(
            compile_transpose_cubin(&bad, CAPS),
            Err(Error::Compile(_))
        ));
    }
    for bad in ["\0".into(), " ".repeat(invar_device::MAX_SOURCE_BYTES + 1)] {
        assert_eq!(plan_options(&bad, CAPS), Err(Error::Limit));
    }
    for caps in [
        Capabilities::Metal {
            apple_family: 7,
            msl_major: 3,
            msl_minor: 0,
        },
        Capabilities::Cuda {
            major: 11,
            minor: 8,
            sm: 86,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 3,
            sm: 86,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 4,
            sm: 81,
        },
    ] {
        assert_eq!(plan_options(&source, caps), Err(Error::Unavailable));
    }
}

#[test]
fn issue_m4_transpose_runtime_forged_and_stale_descriptor_guards() {
    let original = plan([3, 17], 1);
    let cases: Vec<(ir::Ir, ir::Error)> = [
        (0, ir::Error::Version),
        (1, ir::Error::Layout),
        (2, ir::Error::Graph),
        (3, ir::Error::Contract),
        (4, ir::Error::Shape),
        (5, ir::Error::Layout),
        (6, ir::Error::Bounds),
        (7, ir::Error::Alignment),
        (8, ir::Error::Alias),
    ]
    .into_iter()
    .map(|(which, error)| {
        let mut p = original.ir().clone();
        match which {
            0 => p.version = 2,
            1 => p.layout_version = 2,
            2 => p.graph_revision += 1,
            3 => p.function_contract = [0; 32],
            4 => p.output.shape = [3, 17],
            5 => p.input.strides = [18, 1],
            6 => p.input.elements = 51,
            7 => p.output.alignment = 3,
            8 => p.output.id = p.input.id,
            _ => unreachable!(),
        };
        (p, error)
    })
    .collect();
    for (p, error) in cases {
        assert_eq!(ir::verify(p, CAPS, origin()).unwrap_err(), error);
    }
    assert_eq!(
        ir::verify(
            original.ir().clone(),
            CAPS,
            Provenance {
                source_identity: 72,
                graph_revision: 8
            }
        )
        .unwrap_err(),
        ir::Error::Graph
    );
}

#[test]
fn issue_m4_transpose_runtime_all_word_classes_and_padding() {
    let p = plan([1, 1], 1);
    let s = shape(&p, CAPS, 3).unwrap();
    for word in [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc00000, 0x7f800001, 0xff800001, 0xffffffff,
    ] {
        validate_output(&s, &[0, word, 0]).unwrap();
    }
    for bad in [&[0, 0][..], &[0, 0, 0, 0], &[1, 0, 0], &[0, 0, 1]] {
        assert!(validate_output(&s, bad).is_err());
    }
}

// Explicit CPU model of the emitter's padded tile/barrier phases. This is
// schedule qualification, not execution or qualification of CUDA hardware.
fn tile_model(p: &ir::Verified, input: &[u32], grid: usize) -> Vec<u32> {
    let [rows, cols] = p.ir().dimensions.map(|n| n as usize);
    let mut output = vec![0; p.ir().output.elements as usize];
    let tile_cols = cols.div_ceil(16);
    let tiles = rows.div_ceil(16) * tile_cols;
    for block in 0..grid {
        for t in (block..tiles).step_by(grid) {
            let mut tile = [[None; 17]; 16];
            for (y, row) in tile.iter_mut().enumerate() {
                for (x, word) in row.iter_mut().take(16).enumerate() {
                    let r = t / tile_cols * 16 + y;
                    let c = t % tile_cols * 16 + x;
                    if r < rows && c < cols {
                        *word = Some(input[p.ir().input.offset as usize + r * cols + c]);
                    }
                }
            }
            // First uniform barrier: all valid tile slots are now initialized.
            for y in 0..16 {
                for (x, row) in tile.iter().enumerate() {
                    let r = t % tile_cols * 16 + y;
                    let c = t / tile_cols * 16 + x;
                    if r < cols && c < rows {
                        output[p.ir().output.offset as usize + r * rows + c] = row[y].unwrap();
                    }
                }
            }
            // Second uniform barrier precedes any reuse of this block's tile.
        }
    }
    output
}
#[test]
fn issue_m4_transpose_runtime_odd_padded_multitile_bits_model() {
    let words = [
        0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xff800001, 0xffffffff,
    ];
    for dims in [
        [1, 1],
        [3, 17],
        [17, 3],
        [16, 16],
        [17, 33],
        [33, 17],
        [0, 17],
        [17, 0],
    ] {
        let p = plan(dims, 2);
        let input: Vec<_> = (0..p.ir().input.elements as usize)
            .map(|i| words[i % words.len()])
            .collect();
        let [rows, cols] = dims.map(|n| n as usize);
        let mut expected = vec![0; p.ir().output.elements as usize];
        for r in 0..rows {
            for c in 0..cols {
                expected[2 + c * rows + r] = input[2 + r * cols + c];
            }
        }
        for grid in [1, 2, 3] {
            let got = tile_model(&p, &input, grid);
            assert_eq!(got, expected);
            validate_output(&shape(&p, CAPS, input.len()).unwrap(), &got).unwrap();
        }
        let source = ir::emit_cuda(&p).unwrap();
        if rows * cols > 0 {
            assert_eq!(source.matches("__syncthreads();").count(), 2);
            assert!(source.contains("tile[16][17]"));
        }
    }
}

#[derive(Default)]
struct State {
    submitted: usize,
    reads: usize,
    polls: usize,
    dropped: Vec<&'static str>,
    complete: bool,
    ready_after: usize,
    submit_error: Option<Error>,
    poll_error: Option<Error>,
    read_error: Option<Error>,
    panic_poll: bool,
}
struct Lease {
    name: &'static str,
    state: Rc<RefCell<State>>,
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.state.borrow_mut().dropped.push(self.name);
    }
}
struct MockWork {
    state: Rc<RefCell<State>>,
    _leases: Vec<Lease>,
    output: Vec<u32>,
}
fn work(state: &Rc<RefCell<State>>) -> Box<MockWork> {
    let names = [
        "context",
        "stream",
        "module",
        "function",
        "fence",
        "input-host",
        "output-host",
        "input-device",
        "output-device",
    ];
    Box::new(MockWork {
        state: state.clone(),
        _leases: names
            .into_iter()
            .map(|name| Lease {
                name,
                state: state.clone(),
            })
            .collect(),
        output: vec![0, 0x3f800000, 0],
    })
}
impl PendingWork for MockWork {
    fn submit(&mut self) -> Result<(), Error> {
        let mut s = self.state.borrow_mut();
        s.submitted += 1;
        s.submit_error.clone().map_or(Ok(()), Err)
    }
    fn complete(&self) -> Result<bool, Error> {
        let mut s = self.state.borrow_mut();
        s.polls += 1;
        if s.panic_poll {
            panic!("injected host poll unwind");
        }
        s.poll_error
            .clone()
            .map_or(Ok(s.complete && s.polls > s.ready_after), Err)
    }
    fn read(&self) -> Result<Vec<u32>, Error> {
        let mut s = self.state.borrow_mut();
        s.reads += 1;
        s.read_error
            .clone()
            .map_or_else(|| Ok(self.output.clone()), Err)
    }
}
fn reap(q: &mut Quarantine<Box<MockWork>>) {
    q.reap(|w| w.state.borrow().complete);
}

#[test]
fn issue_m4_transpose_runtime_every_unknown_submit_phase_retains_all_owners() {
    for phase in [
        "input-upload",
        "output-upload",
        "launch",
        "download",
        "fence-record",
    ] {
        let error = Error::Driver(phase.into());
        let state = Rc::new(RefCell::new(State {
            submit_error: Some(error.clone()),
            ..State::default()
        }));
        let mut q = Quarantine::new(1);
        assert_eq!(
            finish(work(&state), |w| q.push(w), Duration::from_secs(1), || None),
            Err(error)
        );
        assert!(q.blocks_dispatch());
        assert_eq!(q.len(), 1);
        reap(&mut q);
        assert!(state.borrow().dropped.is_empty());
        assert_eq!(state.borrow().reads, 0);
        state.borrow_mut().complete = true;
        reap(&mut q);
        reap(&mut q);
        assert_eq!(state.borrow().dropped.len(), 9);
        assert!(!q.blocks_dispatch());
    }
}

#[test]
fn issue_m4_transpose_runtime_poll_error_deadline_and_cancellation_never_publish() {
    for expected in [
        Error::Driver("query failed".into()),
        Error::Timeout,
        Error::Canceled,
    ] {
        let state = Rc::new(RefCell::new(State::default()));
        if matches!(expected, Error::Driver(_)) {
            state.borrow_mut().poll_error = Some(expected.clone());
        }
        let mut q = Quarantine::new(1);
        let timeout = if expected == Error::Timeout {
            Duration::ZERO
        } else {
            Duration::from_secs(1)
        };
        let cancel = expected == Error::Canceled;
        assert_eq!(
            finish(
                work(&state),
                |w| q.push(w),
                timeout,
                || cancel.then_some(Error::Canceled)
            ),
            Err(expected)
        );
        assert_eq!(state.borrow().submitted, 1);
        assert_eq!(state.borrow().reads, 0);
        assert!(q.blocks_dispatch());
        reap(&mut q);
        assert!(state.borrow().dropped.is_empty());
        state.borrow_mut().complete = true;
        reap(&mut q);
        assert_eq!(state.borrow().dropped.len(), 9);
    }
}

#[test]
fn issue_m4_transpose_runtime_pending_then_confirmed_fence_reads_once_and_releases() {
    let state = Rc::new(RefCell::new(State {
        complete: true,
        ready_after: 2,
        ..State::default()
    }));
    let mut q = Quarantine::new(1);
    let output = finish(work(&state), |w| q.push(w), Duration::from_secs(1), || None).unwrap();
    assert_eq!(output, [0, 0x3f800000, 0]);
    assert_eq!(state.borrow().polls, 3);
    assert_eq!(state.borrow().reads, 1);
    assert_eq!(state.borrow().dropped.len(), 9);
    assert!(!q.blocks_dispatch());
    for error in [Error::OutOfMemory, Error::Driver("host read failed".into())] {
        let state = Rc::new(RefCell::new(State {
            complete: true,
            read_error: Some(error.clone()),
            ..State::default()
        }));
        assert_eq!(
            finish(work(&state), |w| q.push(w), Duration::from_secs(1), || None),
            Err(error)
        );
        assert_eq!(state.borrow().dropped.len(), 9);
        assert!(!q.blocks_dispatch());
    }
}

#[test]
fn issue_m4_transpose_runtime_guard_quarantines_host_unwind_before_fence() {
    let state = Rc::new(RefCell::new(State {
        panic_poll: true,
        ..State::default()
    }));
    let mut q = Quarantine::new(1);
    // This is a host-library unwind ownership test, not Invar trap simulation.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        finish(work(&state), |w| q.push(w), Duration::from_secs(1), || None)
    }));
    assert!(result.is_err());
    assert_eq!(state.borrow().reads, 0);
    assert!(state.borrow().dropped.is_empty());
    assert!(q.blocks_dispatch());
    state.borrow_mut().complete = true;
    reap(&mut q);
    assert_eq!(state.borrow().dropped.len(), 9);
}

#[test]
fn issue_m4_transpose_runtime_runtime_drop_and_overflow_never_release_unknown_work() {
    let state = Rc::new(RefCell::new(State::default()));
    {
        let mut q = Quarantine::new(1);
        q.push(work(&state));
        q.push(work(&state));
        assert_eq!(q.len(), 1);
        assert!(q.blocks_dispatch());
        state.borrow_mut().complete = true;
        reap(&mut q);
        assert_eq!(state.borrow().dropped.len(), 9);
        assert!(q.blocks_dispatch());
    }
    assert_eq!(state.borrow().dropped.len(), 9); // overflow work remains intentionally retained
    let state = Rc::new(RefCell::new(State::default()));
    {
        let mut q = Quarantine::new(1);
        q.push(work(&state));
    }
    assert!(state.borrow().dropped.is_empty());
}

#[cfg(unix)]
mod compiler {
    use super::*;
    use std::path::PathBuf;

    // Committed executables never have a write fd for sibling forks to inherit.
    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/transpose-compiler")
            .join(name)
    }
    #[test]
    fn issue_m4_transpose_runtime_child_policy_protocol_and_original_diagnostic() {
        let source = ir::emit_cuda(&plan([3, 17], 0)).unwrap();
        let script = fixture("success.sh");
        let result = compile_bounded(
            &source,
            &script,
            CAPS,
            child_deadline::wall(Duration::from_secs(1)),
        )
        .unwrap();
        assert_eq!(result.image, b"mock-cubin");
        assert_eq!(result.log, "original warning");
        let script = fixture("diagnostic.sh");
        assert!(
            matches!(compile_bounded(&source,&script,CAPS,child_deadline::wall(Duration::from_secs(1))),Err(Error::Compile(message)) if message.contains("original transpose-option diagnostic")&&message.contains("27"))
        );
    }
    #[test]
    fn issue_m4_transpose_runtime_child_empty_image_and_output_caps_fail_closed() {
        let source = ir::emit_cuda(&plan([3, 17], 0)).unwrap();
        for name in ["empty.sh", "oversized-image.sh", "oversized-log.sh"] {
            let script = fixture(name);
            assert!(matches!(
                compile_bounded(
                    &source,
                    &script,
                    CAPS,
                    child_deadline::wall(Duration::from_secs(2))
                ),
                Err(Error::Limit)
            ));
        }
    }
    /// A helper that exits but leaves a descendant on its pipes must neither
    /// stall the compile to its deadline nor leak the descendant.
    #[test]
    fn issue_m4_transpose_runtime_child_lingering_descendant_neither_stalls_nor_leaks() {
        use crate::cuda::compiler_child::tests::{descendant, fixture, lingers};
        let source = ir::emit_cuda(&plan([3, 17], 0)).unwrap();
        let deadline = Duration::from_secs(10);
        let begun = Instant::now();
        let compiled = compile_bounded(&source, &fixture("lingering.sh"), CAPS, deadline).unwrap();
        assert!(
            begun.elapsed() < Duration::from_secs(5),
            "{:?}",
            begun.elapsed()
        );
        assert_eq!(compiled.image, b"mock-cubin");
        assert!(!lingers(descendant(&compiled.log)), "{}", compiled.log);
        let begun = Instant::now();
        let result = compile_bounded(&source, &fixture("lingering-log.sh"), CAPS, deadline);
        assert!(
            matches!(result, Err(Error::Limit)),
            "{:?}",
            result.as_ref().err()
        );
        assert!(
            begun.elapsed() < Duration::from_secs(5),
            "{:?}",
            begun.elapsed()
        );
    }
    #[test]
    fn issue_m4_transpose_runtime_child_deadline_covers_nonreading_stdin() {
        let mut source = ir::emit_cuda(&plan([3, 17], 0)).unwrap();
        source.push_str("//");
        source.extend(std::iter::repeat_n(
            ' ',
            invar_device::MAX_SOURCE_BYTES - source.len(),
        ));
        let script = fixture("nonreading-stdin.sh");
        let begun = Instant::now();
        assert!(matches!(
            compile_bounded(&source, &script, CAPS, Duration::from_millis(50)),
            Err(Error::Timeout)
        ));
        assert!(begun.elapsed() < child_deadline::wall(Duration::from_secs(2)));
    }
}

#[test]
fn issue_m4_02_cap_2_28_cuda_transfer_admission() {
    let p = plan([16384, 16384], 0);
    assert_eq!(
        shape(&p, CAPS, 1 << 28).unwrap().transfer_bytes,
        3 * (1_usize << 30)
    );
}
