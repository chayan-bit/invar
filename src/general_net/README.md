# general_net

The general interaction-net route: a checked program becomes an executable net that is both an independent oracle and the input to general compiled lowering for CPU models, CUDA and Metal.

## Where to start

- `Net::lower` and `Net::encode` in [mod.rs](mod.rs) build a net; `Net::verify` returns a `Proof`; `Net::compiled_mir(proof)` is the boundary to compiled code.
- [oracle.rs](oracle.rs) and the `oracle_*.rs` files are the net's CPU oracle; [coverage.rs](coverage.rs) is the exhaustive rule inventory over MIR.
- [regions/](regions/mod.rs) partitions a net (P1a); [normal_form/](normal_form/mod.rs) derives laws, certificates and the parallel normal form (P1b); [path_sum/](path_sum/mod.rs), [task/](task/mod.rs) and [law_collective/](law_collective/mod.rs) lower specific shapes.
- [compiled/](compiled/mod.rs) emits program-specific code for the compiled ABIs; [device_host/](device_host/mod.rs) invokes it on CPU models, CUDA and Metal; [provenance.rs](provenance.rs) binds artifacts to their source bytes.
- [CONTRACT.md](CONTRACT.md) is the frozen interface description.

## Position in the pipeline

After [mir](../mir/README.md); the normal form is consulted by `schedule::lowering::check`. See the code tour step 6 in the [architecture code tour](../../docs/architecture.md#code-tour-one-program-end-to-end).

## Tests

- `cargo nextest run --locked --test gnet_net` (oracle against reference and MIR), `--test gnet_compiled`, `--test gnet_device_model` (generated C++ kernels on the CPU), `--test gnet_provenance`, `--test gnet_numeric_host`, `--test gnet_slices`, `--test normal_form`, `--test law_collective` and `--test gnet05_corpus`.
- [docs/testing.md](../../docs/testing.md) says what each covers; the `gnet_*` binaries compile generated C++ and are slow.

## Rules to keep

- Never trust a name, hash or benchmark id: certificates are re-checked against the current net.
- Interface changes need a `CONTRACT_VERSION` bump and a rerun of the GNET-01 and GNET-02 suites.

## Read next

- [GNET core contract](CONTRACT.md)
- [net normal form](../../docs/net-normal-form-v01.md)
- [path-sum lowering](../../docs/path-sum-v01.md)
- [task region](../../docs/gnet-task-v01.md)
- [law collective](../../docs/law-collective-v01.md)
