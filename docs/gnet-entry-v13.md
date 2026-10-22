# GNET entry ABI13

P0b extends the existing one-group GNET device route. It changes neither launch geometry, child extraction nor source-order speculation. The M4-01 numeric contracts were merged normally for their strict-f32 prerequisites. No schedule-family implementation is changed.

An expanded entry uses ABI13; scalar integer entries keep ABI5. Parameters are i32/u32/Bool/f32, `read` or `mut Array<i32|u32|f32>`, or `read` flat records of i32/u32/Bool/f32. Flat scalar records and owned `Array<i32|u32|f32>` results are supported. `take` parameters, variants, nested records and records containing owners are typed refusals. At most eight parameters and sixteen fields per record are admitted. Field order is checked-source declaration order, never host struct padding.

The twelve physical buffers, task counters, group of 32 threads and kernel symbol remain unchanged. Buffer 0 contains parameter words, allocation/live counters, then a canonical owned heap image. Root initialization copies the image once before the first source step. Each scalar/field/element occupies an eight-byte word: sign-extended i32, zero-extended u32/f32 bits, and Bool 0/1. Record and array parameters are stable allocation IDs into the existing checked heap layout. Records carry the existing per-field presence mask.

Host entry uses safe `eval::Value` containers. Shared/exclusive RefCell leases prevent conflicting caller access while a session owns borrowed arrays. Exact read aliases share a device owner. Any duplicate identity involving `mut` is refused conservatively, including empty owners; partial or strided views are not admitted. There is no raw-pointer, external-stride or caller-supplied record-padding ABI. Wrong field count/types or a strided Value produces `wrong_layout`; capacity/length produces `oversize`; mutable alias produces `alias`; unsupported owners produce `unsupported_type`; an already borrowed container produces `borrowed`. Each refusal precedes backend allocation.

Device writes remain private across continuation epochs. Only a successful source return after the completion fence reaches publication. Every destination descriptor, length, kind, live bit and scalar word is checked before the first mutable host write. Trap, fuel exhaustion, cancellation, malformed completion and unknown completion publish no host array changes. Unknown completion retains the existing quarantine and forbids replay. Array transfers and readbacks are copying; native/Python zero-copy remains a separate gate. Oracle receipts hash bounded inputs and alias topology before mutation, then retain an independent snapshot of flat record results. Returned arrays are copied into fresh host storage only after their issued handle, kind, live flag, length, heap range and canonical element words validate. A malformed result prevents mutable-argument publication as well.

Strict-f32-v1 uses the existing independently tested integer-software binary32 operations for every ABI13 operation on both targets. This is the exact fallback path, including all subnormal inputs/results, signed zeros, NaNs and explicit fma. Each physical backend runs the existing cached D1 live-device probe before preparing ABI13 work and records its completed report. Native floating arithmetic is never admitted; every operation uses the exact software fallback, including when the probe detects flushing or contraction. A failed or unknown probe completion refuses preparation through the existing poison/quarantine gate. There is no hardware-float fast path or CPU replay. Metal disables fast math and emits `#pragma METAL fp contract(off)`; CUDA disables contraction and FTZ and requests precise division/sqrt. Receipts identify software arithmetic, absence of native admission and the subnormal path. CPU net agreement compares exact words, steps, trap spans, transitions, Demand/Complete events and cleanup.

The original 1,000,000 source-array cap bounded allocation; it was repeated in the evaluator, MIR, native host and GNET allocation helpers, independently of device capacity. `array_limits` now defines a 2^24 per-array cap and a 2^28 live-element ceiling. Old source-limit fixtures now request 2^24+1, preserving their failure purpose. A 2^24-element host f32/u32/i32 array is 64 MiB, or 128 MiB in GNET's eight-byte carriers. The fixed packet plus heap requires about 256 MiB for one full-size input, before bounded frame/trace/scratch. Callers must explicitly raise `Limits.owned.elements` and `Limits.buffer_bytes`; the default remains conservative. The full-size test also sets `Limits.private.slots = 0`: unchanged speculation storage otherwise replicates the owner heap per private slot and can exceed its separate scratch budget. Source fuel, trace and per-plan storage limits are unchanged.

Native integer source allocation caps are qualified on x86_64 Linux with LLVM 21.1.8 in Debug and Release. Six per-array/aggregate cases match the reference, and the private HA-1 protocol cap is retained. This qualification covers admitted integer ordinary-source allocations; it does not renew device, performance or unsupported-type acceptance.

The hard storage ceiling is 8 GiB, following the 2^28 device caps policy. Before allocating the packet, host admission also checks four payload copies plus the retained input against that ceiling, covering adapter storage, byte capture, decoded checkpoints and readback overlap. Metal checks live maxBufferLength and recommended working set minus current allocations; CUDA checks threefold concurrent storage and free memory less a ten-percent margin. Checked arithmetic is retained for lengths, byte extents, heap sizes and totals. Allocation failure remains explicit. The CPU device model retains its smaller 64 MiB bound; full-size source acceptance and full-size Metal admission are separate tests.

The private `gnet-device` command accepts f32 as `{"bits":1065353216}`, arrays as JSON arrays of exact typed elements, and flat records as arrays in declaration order. f32 array elements use the same bits object. Expanded receipts include `arguments_after` after reclaim releases all leases. The CLI's OS argument-size limit is separate from the Rust host API's supported array size.

Historical qualification before the native-cap rerun on 18 October 2026:

Runnable checks are `issue_gnet_p0b_entry` (CPU group models, typed refusals, fence gate and source cap) and `issue_gnet_p0b_gpu` (ignored Metal/CUDA hardware tests). Physical CUDA passed 2/0/0 per profile on the RTX 5070 Ti. Native LLVM runtime acceptance of the raised cap remains unverified. No performance or parallel-region acceptance is claimed.

P5b packages the exact D1 probe MSL as a second offline library for ABI13.
Build and load derive the same inventory from the admitted artifact and reject
missing, stale, extra or wrong-entry libraries before installation. ABI5 keeps
its one-library container. The container format and production prohibition on
runtime source compilation are unchanged. See [P5b](gnet-p5b.md) for current
qualification and explicit robotics budget refusals.

P0b trig follow-up admits `sin`, `cos` and `sqrt` in the compiled C++, CUDA
and MSL numeric bodies, including ABI13 entries. `trig_f32.rs` generates the
frozen operation sequence over the existing exact integer f32 helpers; its
integer square-root primitive rounds correctly without device float arithmetic.
NaNs, signed zeros, infinities, subnormals and huge trig arguments follow
`invar-trig-f32-v1`. The launch, speculation and scheduling structure is unchanged.
