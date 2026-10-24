# GNET-05 source-derived acceptance corpus

Ordinary Invar source for the general-net GPU track (GNET-05; obligations in `src/general_net/CONTRACT.md`).
It is additive: the hand-written fixture corpora elsewhere under `bench/general-net-v01` are unchanged.
Every file here is generator output: `generate.rs` is the only hand-written input.
Tests: `tests/gnet05_corpus/issue_gnet_05_corpus.rs`.

## Files

| File | Content |
| --- | --- |
| `generate.rs` | Seeded generator `gnet05-gen-v3` (SplitMix64, root seed `0x474e455430357631`), held-out split, normalized identity, large-input ladders, `calibrate`, per-input step budgets and the deep inputs (`DEEP_INPUTS`, `deepen`) |
| `deep_select.py` | Record of how the deep inputs were selected with the net-laws prototype of the normal-form design (Appendix C of `docs/design/net-parallel-laws-v01.md`, in git history up to commit `85dc53f`) |
| `programs/*.inv` | 72 programs, each with entry `run(a, b) -> i32` and a `main` calling it on one recorded input |
| `manifest.json` | Per-program provenance: family, ordinal, seed, drawn shape, inputs, negative inputs, large-input ladder, the calibrated large input and the deep input, the declared step budget of every input, split, source and normalized SHA-256; default and maximum budgets; pinned held-out digest |
| `validation.json` | Per-input reference status, steps and oracle outcome; per-target device admission; counted denominators; the deep inputs' outcomes, counted apart (Release run) |

Regenerate with `INVAR_GNET05_WRITE=1 cargo test --release --locked --test gnet05_corpus issue_gnet_05_corpus::` (Release writes `validation.json`, because the deep inputs run only there; about 40 seconds on a 32-core x86_64 host).
A Debug run skips the deep inputs and checks every other recorded outcome and denominator (about two minutes).
Without the variable the tests fail if any file differs from the generator.

## Composition

Six families, 12 programs each, 6 held out per family (36 tuning, 36 held out).
Each program has 4 recorded runtime inputs; input 0 is always a tiny control.
46 programs also record a calibrated large input (334 inputs in total; see below), and 30 programs record one more deep input with a larger declared step budget (see Step budgets).
Inputs, not the source, choose recursion depth, range, graph size and which branches run.

| Family | Shape | Source size (bytes / lines) |
| --- | --- | --- |
| divide | irregular recursive divide and conquer: data-dependent split point, arity 2 or 3, uneven depth budgets | 510-703 / 14-22 |
| branch | input-dependent branch trees: each node picks a serial, fan-out or leaf shape from runtime state | 561-864 / 19-29 |
| frontier | dynamically discovered graph frontiers: recursive discovery with runtime out-degree, and a serial owned-array BFS | 757-996 / 26-33 |
| serial | serial and negative controls: straight-line chains, optional runtime divide-by-zero trap, optional runtime-depth recursion | 343-601 / 12-24 |
| compose | cross-family compositions: divide and conquer with branch-tree leaves, and a discovered frontier whose vertex work is a divide and conquer | 1079-1555 / 32-45 |
| forkjoin | residual fork-join: divide and conquer or branch trees whose combine is not linear in the child results (degree-two products, a select on the children's order, or an extra recursive call that runs when the children's values satisfy a test) | 565-1089 / 17-35 |

Serial and negative share:
- 12 of 72 programs (17%) are the serial family; 16 of the 60 device-admitted programs have no independent call site (fork candidates 0) and run serially.
- 12 programs (8 compose, 1 branch, 3 forkjoin) are refused by device admission with `compiler_structural_budget`; the refusals are counted in `validation.json`.
- 12 of 334 inputs (4%) are negative: 5 runtime traps and 7 call-depth limits, all in the serial family.
- Positive inputs other than the deep inputs finish within 100,000 reference steps (largest: 89,819) and far below the 64-frame call-depth limit.

## Large inputs

The recorded inputs are small (median work 111 to 782 agents per family, measured for the normal-form design), so no GPU gate can be about parallelism at those sizes.
Every program whose work grows with its input carries a ladder of larger candidate inputs in `manifest.json`:
- divide and compose (divide and conquer over branch trees): the range size grows by about the square root of two per rung (64, 90, 128, ... 65,536) at a fixed depth budget;
- branch trees: the depth `n` from 6 to 48, one chain per recorded state (the state picks the tree's shape);
- serial-array BFS: the vertex count, with every round run.

`calibrate` records the rung with the most reference steps that returns a value within `LARGE_STEPS` = 90,000 steps as one more input.
That is 90% of the 100,000-step fuel the reference evaluator, the net oracle and the device configuration all run with, so a large input never depends on a raised budget.
The choice is a function of the source and the reference step count only, so it is reproducible and checked by the files test.
46 programs have a large input: median 73,296 steps, 48 to 204 times the median recorded input of their family (by reference steps).
The remaining 26 cannot grow: the serial family (sequential by construction), and the recursive-discovery frontiers whose depth is capped by the generator (at most 6).
Under the 100,000-step budget the largest average parallelism (the normal-form design's law model) at a large input is about 180; the deep inputs below reach about 1,000.

## Step budgets and deep inputs

Every recorded input declares its reference and CPU-oracle step budget in `manifest.json` (`budgets`, one per input).
The default is 100,000 steps (`DEFAULT_BUDGET`), the fuel every other GNET path runs with; the corpus caps a declared budget at 10,000,000 (`MAX_BUDGET`).
Only the deep inputs declare more.
`DEEP_INPUTS` gives 30 divide, compose and forkjoin programs one deep input each: the ladder rung `deep_select.py` selected with the net-laws prototype, namely the first rung of each chain whose law-directed average parallelism W/S (model c) reaches 1,000, with the least work across chains.
Its budget is the reference step count rounded up to a whole million (1 to 7 million).
22 of the 30 reach W/S of 1,015 to 2,271; the other 8 (compose-03 and -06, forkjoin-02, -04, -05, -08, -09 and -11) stop below 1,000 within the cap, at 87 to 923, and are marked short in the evidence.
Every deep input is a value the oracle reproduces (`deep oracle equal value`: 30), with the outcome frozen in `validation.json`.
The deep inputs are counted apart (`deep_counts`, `deep`), so the 334-input denominators are unchanged, and they run in Release only.
The device configuration caps fuel at 1,000,000, so the device harness runs a deep input only with `INVAR_GNET05_DEEP_DEVICE=1` and only when its budget is within that cap (10 of 30).

## Residual fork-join

Every combine in the other recursive families is linear modulo a prime, so a path-sum law removes their fork-join; `forkjoin` is the tier that keeps it (D5's residual fork-join gate).
Its 12 programs draw one of three non-linear combines and a divide-and-conquer or branch-tree shape; operands are reduced modulo 10007 first so products stay below 2^31.
`issue_gnet_05_corpus_forkjoin_combines_are_not_linear` checks that every forkjoin program multiplies two child results or branches on them, and that no other program does.
With the net-laws prototype at the large input, none of the 12 is a path-sum, each keeps 129 to 2,021 fork-join tasks, and its average parallelism is 9 to 158.
9 of the 12 are admitted by device admission today.

## Held-out discipline

`HeldOut::of` records the normalized identity (comments and whitespace removed, user identifiers canonicalized) of every held-out program; the sorted digest is pinned in `manifest.json`.
`HeldOut::admit_for_tuning` refuses any source whose normalized identity is held out, so a renamed, reformatted or recommented held-out program cannot reach a tuning path.
`tuning_programs` returns only tuning-split programs that pass that gate.

## Anti-gaming

The compiler must not key on fixture names, source hashes or benchmark IDs.
The tests prove it two ways:
- Renaming every user identifier, and adding comments that name benchmark IDs and fixture paths (a different source hash), leaves admission, the fork plan and the emitted device code unchanged for every program and both targets. Emitted code is compared after removing the rename prefix and the source byte offsets of trap spans, which legitimately follow the source text.
- No compiler, MIR, net or device source file mentions the corpus, its paths, program IDs or fixture names.

## Freeze relationship

Version 2 (`gnet05-gen-v2`) is a reviewed manifest bump: the 60 version-1 sources and their first four inputs are unchanged; it adds the forkjoin family, the large inputs and a new pinned held-out digest.
Version 3 (`gnet05-gen-v3`) changes no source, input or held-out digest; it adds the declared step budgets and the deep inputs.
This corpus is separate from the M5-01 permanent freeze and needs no hold on it.
The M5-01 held-out set covers the M4 families; later families append through reviewed manifest bumps, and the performance acceptance treats general-net as a separate prerequisite.
Freeze the GNET-05 held-out set (the pinned digest) before any GNET tuning, and keep it out of M5-01 calibration.
