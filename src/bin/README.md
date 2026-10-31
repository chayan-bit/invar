# bin

Helper binaries the compiler spawns; the `invar` CLI itself is [../main.rs](../main.rs).

## Where to start

- [invar-sequencer.rs](invar-sequencer.rs) runs system sequencing; [invar-nvrtc.rs](invar-nvrtc.rs) compiles CUDA source out of process for the CUDA engine (feature `cuda`), with a parent-owned deadline and output cap.
- The acceptance, qualification and benchmark binaries (`invar-bench`, `invar-heldout`, `invar-performance-report`, `invar-qualify`, `invar-pathsum-gate`, `invar-m3-03-cold`) live in [tools](../../tools/README.md).

## Position in the pipeline

Spawned by `engine` and the system commands.

## Tests

- `cargo nextest run --locked --test gpu_cuda` needs `--features cuda` and a GPU; `tests/gpu_cuda/nvrtc_pipes.rs` covers the NVRTC pipes.

## Rules to keep

- The parent owns every deadline and byte cap; a helper's output is untrusted.
