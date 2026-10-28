//! Strict binary32 contraction and the D1 probe on Metal, from the same
//! verified `invar_device::floating` plan the CUDA runtime executes. The root
//! owns source-loop admission and publication; this module returns only the
//! completed finite physical C storage, and only Completed work publishes.

use super::*;
use invar_device::{f32_probe, floating as ir, tc};

/// One threadgroup; the direct and probe kernels are grid-strided.
const LANES: u32 = 32;

/// (threadgroups, lanes) of the tiled schedule; its kernel raises the flag on
/// any other lane count, so a mismatch is discarded, never published.
fn tiled_launch(plan: &ir::Verified) -> (u32, u32) {
    let geometry = if plan.is_tc() {
        tc::geometry(plan)
    } else {
        invar_device::gemm::geometry(plan)
    };
    (geometry.groups, geometry.threads)
}

#[derive(Debug)]
pub struct FloatingResult {
    /// Fresh physical C allocation, including zeroed prefix/tail padding.
    pub output: Vec<u32>,
    pub source_hash: u64,
    pub transfer_bytes: usize,
    pub compile_ms: f64,
    pub kernel_ms: f64,
}

struct Shape {
    lengths: [usize; 3],
    logical: usize,
    offset: usize,
}

fn finite(word: u32) -> bool {
    word & 0x7f80_0000 != 0x7f80_0000
}

fn shape(
    plan: &ir::Verified,
    caps: Capabilities,
    lengths: [usize; 2],
    max_buffer: usize,
) -> Result<Shape, Error> {
    if plan.capabilities() != caps {
        return Err(Error::Limit);
    }
    let mut physical = [0; 3];
    for (slot, buffer) in plan
        .ir()
        .inputs
        .iter()
        .chain(std::iter::once(&plan.ir().output))
        .enumerate()
    {
        physical[slot] = usize::try_from(buffer.elements).map_err(|_| Error::Limit)?;
    }
    if lengths != [physical[0], physical[1]] {
        return Err(Error::Limit);
    }
    let [rows, _, cols] = plan.ir().dimensions;
    let logical = usize::try_from(u64::from(rows) * u64::from(cols)).map_err(|_| Error::Limit)?;
    let offset = usize::try_from(plan.ir().output.offset).map_err(|_| Error::Limit)?;
    if offset
        .checked_add(logical)
        .is_none_or(|end| end > physical[2])
    {
        return Err(Error::Limit);
    }
    let bytes = physical[0]
        .max(1)
        .checked_add(physical[1].max(1))
        .and_then(|n| n.checked_add(physical[2]))
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::Limit)?;
    crate::check_aggregate_limit(bytes)?;
    for length in physical {
        let bytes = length.max(1).checked_mul(4).ok_or(Error::Limit)?;
        if bytes > max_buffer {
            return Err(Error::MatrixResourceLimit {
                resource: "Metal maxBufferLength",
                requested: bytes,
                limit: max_buffer,
            });
        }
    }
    Ok(Shape {
        lengths: physical,
        logical,
        offset,
    })
}

fn validate_inputs(plan: &ir::Verified, inputs: [&[u32]; 2]) -> Result<(), Error> {
    if plan.ir().inputs[0].id == plan.ir().inputs[1].id && inputs[0] != inputs[1] {
        return Err(Error::Limit);
    }
    if inputs.into_iter().flatten().any(|&word| !finite(word)) {
        return Err(Error::Device("floating nonfinite input refusal".into()));
    }
    Ok(())
}

fn validate_output(shape: &Shape, values: &[u32]) -> Result<(), Error> {
    if values.len() != shape.lengths[2] || values.iter().any(|&word| !finite(word)) {
        return Err(Error::Device(
            "floating output length/nonfinite refusal".into(),
        ));
    }
    let end = shape
        .offset
        .checked_add(shape.logical)
        .ok_or(Error::Limit)?;
    let padding = values
        .get(..shape.offset)
        .ok_or(Error::Limit)?
        .iter()
        .chain(values.get(end..).ok_or(Error::Limit)?);
    if padding.into_iter().any(|&word| word != 0) {
        return Err(Error::Device("floating output padding modified".into()));
    }
    Ok(())
}

struct Owned {
    output: Vec<u32>,
    /// The kernel's dispatch flag: the result may differ from strict IEEE.
    flagged: bool,
    status: u32,
    compile_ms: f64,
    kernel_ms: f64,
}

/// Raw device words from a non-publishing qualification run.
#[derive(Debug)]
pub struct Diagnostic {
    pub output: Vec<u32>,
    pub flagged: bool,
}

impl MetalDevice {
    /// Qualification-only mutation entry; no source execution calls this.
    #[doc(hidden)]
    pub fn diagnose_f32_ops_source(
        &mut self,
        plan: &invar_device::v2::Verified,
        packed: &[u32],
        source: &str,
    ) -> Result<Diagnostic, Error> {
        self.admit()?;
        let p = invar_device::f32_ops::program(plan).map_err(|_| Error::Limit)?;
        if plan.capabilities() != self.caps
            || packed.len() != p.packed_len()
            || source.len() > invar_device::MAX_SOURCE_BYTES
        {
            return Err(Error::Limit);
        }
        let owned = self.submit_owned(
            source,
            invar_device::f32_ops::KERNEL_NAME,
            [packed, &[]],
            &vec![0; p.output_len() + 1],
            (1, LANES),
        )?;
        Ok(Diagnostic {
            output: owned.output[..p.output_len()].to_vec(),
            flagged: owned.flagged,
        })
    }
    /// Run strict rank-1 f32 work through the same owned dispatch as contraction.
    /// Guarded results are discarded before publication; the caller may replay
    /// the exact source only after this method confirms terminal completion.
    pub fn run_f32_ops(
        &mut self,
        plan: &invar_device::v2::Verified,
        packed: &[u32],
    ) -> Result<Vec<u32>, Error> {
        self.admit()?;
        let p = invar_device::f32_ops::program(plan).map_err(|_| Error::Limit)?;
        if plan.capabilities() != self.caps || packed.len() != p.packed_len() {
            return Err(Error::Limit);
        }
        let ops = invar_device::f32_ops::probe_ops(p);
        let report = self.strict_f32_probe()?;
        if !report.admits_strict_with_fallback(&ops)
            || !invar_device::f32_ops::admits_sqrt(p, report)
        {
            return Err(Error::Unavailable(
                "strict f32 operation probe refused".into(),
            ));
        }
        if p.output_len() == 0 {
            return Ok(Vec::new());
        }
        let source =
            invar_device::f32_ops::emit(plan).map_err(|e| Error::Toolchain(format!("{e:?}")))?;
        let initial = vec![0; p.output_len() + 1];
        let owned = self.submit_owned(
            &source,
            invar_device::f32_ops::KERNEL_NAME,
            [packed, &[]],
            &initial,
            (1, LANES),
        )?;
        if owned.status & ir::guard::OVERFLOW != 0 {
            return Err(Error::IntegerOverflow);
        }
        if owned.flagged {
            return Err(Error::SubnormalFallback);
        }
        let mut output = owned.output;
        if output.pop() != Some(0) {
            return Err(Error::Device("f32 private status padding modified".into()));
        }
        Ok(output)
    }
    /// D1 per-process preflight, cached for this opened device: the shared
    /// probe table compiled with the strict Metal options and submitted through
    /// the same owned path as contractions. Strict work is refused unless the
    /// returned report admits it exactly or under the underflow guard.
    pub fn strict_f32_probe(&mut self) -> Result<&f32_probe::Report, Error> {
        if self.strict_probe.is_none() {
            self.admit()?;
            let words = f32_probe::words();
            // Two words per case: result and guard flag, both unwritten.
            let sentinel = f32_probe::unwritten();
            let owned = self.submit_owned(
                &f32_probe::emit_msl(),
                f32_probe::KERNEL_NAME,
                [&words, &[0]],
                &sentinel,
                (1, LANES),
            )?;
            let report = f32_probe::Report::evaluate(&owned.output).map_err(Error::InvalidProbe)?;
            self.strict_probe = Some(report);
        }
        self.strict_probe
            .as_ref()
            .ok_or_else(|| Error::Device("Metal probe state lost".into()))
    }

    /// Whether this device may run strict contractions: exactly, or with the
    /// underflow guard when every probe mismatch in the contraction's
    /// operations raised the device guard flag.
    pub fn admits_strict_contraction(&mut self) -> Result<bool, Error> {
        let report = self.strict_f32_probe()?;
        Ok(
            report.admits_strict()
                || report.admits_strict_with_fallback(f32_probe::CONTRACTION_OPS),
        )
    }

    /// Run a verified strict contraction after the D1 probe admits this device.
    /// Exact physical input capacities are required; the output is fresh C
    /// storage whose verified logical region the caller may publish. When the
    /// underflow guard fires, the device words are discarded here and
    /// `Error::SubnormalFallback` tells the caller to run the exact CPU path.
    pub fn run_contraction(
        &mut self,
        plan: &ir::Verified,
        inputs: [&[u32]; 2],
    ) -> Result<FloatingResult, Error> {
        let source = ir::emit_msl(plan)
            .map_err(|e| Error::Toolchain(format!("strict MSL emission: {e:?}")))?;
        self.run_admitted(plan, &source, inputs, (1, LANES))
    }

    /// The M4-03 tiled schedule of the same verified plan, with the same D1
    /// admission, underflow guard, CPU fallback signal and output checks.
    pub fn run_tiled_contraction(
        &mut self,
        plan: &ir::Verified,
        inputs: [&[u32]; 2],
    ) -> Result<FloatingResult, Error> {
        if plan.is_tc() {
            return self.run_tc_contraction(plan, inputs);
        }
        let source = invar_device::gemm::emit_msl(plan)
            .map_err(|e| Error::Toolchain(format!("tiled MSL emission: {e:?}")))?;
        self.run_admitted(plan, &source, inputs, tiled_launch(plan))
    }

    /// `tc-f16-f32-v1`: inputs are rounded to binary16 here (an element past
    /// the binary16 range or rounding to a binary16 subnormal is refused
    /// before submission), then the
    /// `simdgroup_float8x8` schedule runs with the same admission and checks.
    /// Binary16 values, their products and nonzero sums are normal binary32
    /// numbers, so the underflow guard has nothing to flag in this kernel.
    fn run_tc_contraction(
        &mut self,
        plan: &ir::Verified,
        inputs: [&[u32]; 2],
    ) -> Result<FloatingResult, Error> {
        shape(
            plan,
            self.caps,
            [inputs[0].len(), inputs[1].len()],
            self.device.maxBufferLength(),
        )?;
        let source =
            tc::emit_msl(plan).map_err(|e| Error::Toolchain(format!("tc MSL emission: {e:?}")))?;
        let refused = |r: tc::InputRange| Error::Device(r.to_string());
        let a = tc::round_inputs(0, inputs[0]).map_err(refused)?;
        let b = tc::round_inputs(1, inputs[1]).map_err(refused)?;
        self.run_admitted(plan, &source, [&a, &b], tiled_launch(plan))
    }

    fn run_admitted(
        &mut self,
        plan: &ir::Verified,
        source: &str,
        inputs: [&[u32]; 2],
        launch: (u32, u32),
    ) -> Result<FloatingResult, Error> {
        self.admit()?;
        if !self.admits_strict_contraction()? {
            let summary = self.strict_f32_probe()?.summary();
            return Err(Error::Unavailable(format!(
                "strict-f32-v1 is ineligible on this Metal device (D1): {summary}"
            )));
        }
        let (result, flagged) = self.execute_contraction(plan, source, inputs, true, launch)?;
        if flagged {
            return Err(Error::SubnormalFallback);
        }
        Ok(result)
    }

    /// Qualification evidence only: execute the verified kernel without the D1
    /// admission and return the raw words and guard flag. The result is never
    /// an Invar result and no source path calls this.
    #[doc(hidden)]
    pub fn diagnose_unadmitted_contraction(
        &mut self,
        plan: &ir::Verified,
        inputs: [&[u32]; 2],
    ) -> Result<Diagnostic, Error> {
        let source = ir::emit_msl(plan)
            .map_err(|e| Error::Toolchain(format!("strict MSL emission: {e:?}")))?;
        self.diagnose_source(plan, &source, inputs)
    }

    /// Qualification evidence only: like `diagnose_unadmitted_contraction`
    /// but for caller-supplied MSL, so tests can prove that a kernel mutation
    /// which disables the guard is caught. Never reachable from source.
    #[doc(hidden)]
    pub fn diagnose_source(
        &mut self,
        plan: &ir::Verified,
        source: &str,
        inputs: [&[u32]; 2],
    ) -> Result<Diagnostic, Error> {
        self.diagnose_launch(plan, source, inputs, (1, LANES))
    }

    /// Qualification evidence only: `diagnose_source` for tiled-schedule MSL,
    /// launched with the plan's tiled geometry. Never reachable from source.
    #[doc(hidden)]
    pub fn diagnose_tiled_source(
        &mut self,
        plan: &ir::Verified,
        source: &str,
        inputs: [&[u32]; 2],
    ) -> Result<Diagnostic, Error> {
        self.diagnose_launch(plan, source, inputs, tiled_launch(plan))
    }

    fn diagnose_launch(
        &mut self,
        plan: &ir::Verified,
        source: &str,
        inputs: [&[u32]; 2],
        launch: (u32, u32),
    ) -> Result<Diagnostic, Error> {
        self.admit()?;
        let (result, flagged) = self.execute_contraction(plan, source, inputs, false, launch)?;
        Ok(Diagnostic {
            output: result.output,
            flagged,
        })
    }

    fn execute_contraction(
        &mut self,
        plan: &ir::Verified,
        source: &str,
        inputs: [&[u32]; 2],
        publishing: bool,
        launch: (u32, u32),
    ) -> Result<(FloatingResult, bool), Error> {
        let shape = shape(
            plan,
            self.caps,
            [inputs[0].len(), inputs[1].len()],
            self.device.maxBufferLength(),
        )?;
        let hash = source_hash(source);
        let mut output = Vec::new();
        output
            .try_reserve_exact(shape.lengths[2])
            .map_err(|_| Error::OutOfMemory)?;
        output.resize(shape.lengths[2], 0);
        if shape.logical == 0 {
            let empty = FloatingResult {
                output,
                source_hash: hash,
                transfer_bytes: 0,
                compile_ms: 0.0,
                kernel_ms: 0.0,
            };
            return Ok((empty, false));
        }
        validate_inputs(plan, inputs)?;
        let owned = self.submit_owned(source, ir::KERNEL_NAME, inputs, &output, launch)?;
        // A flagged dispatch is discarded unread by the publishing path.
        if publishing && !owned.flagged {
            validate_output(&shape, &owned.output)?;
        }
        let result = FloatingResult {
            output: owned.output,
            source_hash: hash,
            transfer_bytes: (shape.lengths[0] + shape.lengths[1] + 2 * shape.lengths[2]) * 4,
            compile_ms: owned.compile_ms,
            kernel_ms: owned.kernel_ms,
        };
        Ok((result, owned.flagged))
    }

    /// Compile `source` with the pinned strict options, copy every input into
    /// fresh private shared buffers, bind an UNWRITTEN status word at slot 3, run
    /// `launch` = (threadgroups, lanes), and return C and the flag only after a
    /// Completed command; unknown completion quarantines all of it.
    fn submit_owned(
        &mut self,
        source: &str,
        kernel: &str,
        inputs: [&[u32]; 2],
        initial: &[u32],
        launch: (u32, u32),
    ) -> Result<Owned, Error> {
        let started = Instant::now();
        let (pipeline, _) = self.compile(source, kernel)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let words = |n: usize| n.max(1).checked_mul(4).ok_or(Error::Limit);
        let sizes = [
            words(inputs[0].len())?,
            words(inputs[1].len())?,
            words(initial.len())?,
            4,
        ];
        let total = sizes
            .iter()
            .try_fold(0usize, |sum, size| sum.checked_add(*size))
            .ok_or(Error::Limit)?;
        self.check_submission(*sizes.iter().max().ok_or(Error::Limit)?, total)?;
        let buffers = [
            self.buffer(sizes[0])?,
            self.buffer(sizes[1])?,
            self.buffer(sizes[2])?,
            self.buffer(sizes[3])?,
        ];
        for (values, buffer) in inputs
            .into_iter()
            .chain([initial, &[ir::guard::UNWRITTEN][..]])
            .zip(&buffers)
        {
            // SAFETY: each fresh shared buffer holds at least `values.len()` u32
            // slots, is disjoint from `values`, and no command uses it yet.
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().as_ptr().cast(),
                    values.len(),
                );
            }
        }
        let command = command_buffer(&self.queue)?;
        let refs = buffers.each_ref().map(|b| &**b);
        self.encode(&command, &pipeline, &refs, launch.0, launch.1)?;
        let mut work = MetalWork::prepared(command.clone(), buffers.to_vec(), vec![pipeline]);
        work.submit()?;
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal command failed: {:?}",
                command.error()
            )));
        }
        let kernel_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
        let mut output = Vec::new();
        output
            .try_reserve_exact(initial.len())
            .map_err(|_| Error::OutOfMemory)?;
        output.resize(initial.len(), 0);
        // SAFETY: Completed is terminal; the retained C buffer holds
        // `initial.len()` u32 slots and the destination is disjoint.
        unsafe {
            ptr::copy_nonoverlapping(
                buffers[2].contents().as_ptr().cast(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        // SAFETY: Completed is terminal; the retained flag buffer holds one u32.
        let flag = unsafe { ptr::read(buffers[3].contents().as_ptr().cast::<u32>()) };
        // Every kernel that binds F claims it; the probe's F is unused (its
        // per-case flags are decoded by Report) and stays UNWRITTEN.
        let legal = match kernel {
            ir::KERNEL_NAME => Some(ir::guard::GUARD),
            invar_device::f32_ops::KERNEL_NAME => Some(invar_device::f32_ops::METAL_STATUS),
            _ => None,
        };
        let status = match legal {
            Some(legal) => ir::guard::decode_bits(flag, legal).map_err(Error::InvalidGuard)?,
            None => 0,
        };
        Ok(Owned {
            output,
            flagged: status & ir::guard::GUARD != 0,
            status,
            compile_ms,
            kernel_ms,
        })
    }
}

#[cfg(test)]
mod cap_tests {
    use super::*;
    use invar_device::{Provenance, Span};

    fn plan([r, k, c]: [u32; 3]) -> ir::Verified {
        let buffer = |id, shape: [u32; 2]| ir::Buffer {
            id,
            elements: u64::from(shape[0]) * u64::from(shape[1]),
            shape,
            strides: [u64::from(shape[1]), 1],
            offset: 0,
            alignment: 4,
        };
        ir::verify(
            ir::Ir {
                version: 1,
                numeric_version: 1,
                numeric_policy: ir::NUMERIC_POLICY.into(),
                layout_version: 1,
                span: Span { start: 1, end: 2 },
                source_identity: 1,
                graph_revision: 1,
                function_contract: [1; 32],
                dimensions: [r, k, c],
                inputs: [buffer(0, [r, k]), buffer(1, [k, c])],
                output: buffer(2, [r, c]),
            },
            Capabilities::Metal {
                apple_family: 9,
                msl_major: 3,
                msl_minor: 1,
            },
            Provenance {
                source_identity: 1,
                graph_revision: 1,
            },
        )
        .unwrap()
    }

    #[test]
    fn issue_m4_03_matrix_size_2_28_shared_metal_budget_and_max_buffer_are_typed() {
        let square = plan([16384; 3]);
        let s = shape(&square, square.capabilities(), [1 << 28; 2], usize::MAX).unwrap();
        assert_eq!(s.lengths, [1 << 28; 3]);
        assert_eq!(s.logical, 1 << 28);
        let skinny = plan([16384, 1, 16384]);
        assert!(shape(&skinny, skinny.capabilities(), [16384; 2], 1 << 30).is_ok());
        assert!(matches!(
            shape(&skinny, skinny.capabilities(), [16384; 2], (1 << 30) - 1),
            Err(Error::MatrixResourceLimit {
                resource: "Metal maxBufferLength",
                requested: 1_073_741_824,
                limit: 1_073_741_823
            })
        ));
    }
}
