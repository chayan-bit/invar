#!/usr/bin/env python3
"""Run the handwritten robotics baselines (GNET-05 comparison D) over every scored case.

Reads bench/general-net-v01/gnet05/robotics/{manifest,validation}.json, runs the Metal or CUDA
runner once per program and scored input (10 warmups, 30 samples per call), checks the full
output bit for bit against the frozen reference digest and appends one JSON line per case.

Each sample runs K complete calls back to back (K fixed per case, read from --batch, default 1)
so that a sample takes at least 1 ms of GPU time. --calibrate FILE starts every case at K = 1 and
rescales K by TARGET_MS (5 ms) over the shortest sample until every sample reaches it (batched
calls amortize launch overhead, so one rescale can fall short); FILE is then frozen and reused.

  metal: robotics-baselines.py metal <runner> <out-dir> [--only SUBSTRING] [--batch FILE]
  cuda:  robotics-baselines.py cuda <runner> <out-dir> [--only SUBSTRING] [--batch FILE]
"""
import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "bench/general-net-v01/gnet05/robotics"
KERNELS = ROOT / "bench/baselines/robotics-kernels.h"
# Calibration target per sample: 1 ms of GPU time with a 5x margin, because the GPU
# performance state moves a small case's per-call time by up to about 4x between sessions.
TARGET_MS = 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["metal", "cuda"])
    ap.add_argument("runner")
    ap.add_argument("out_dir")
    ap.add_argument("--only", default="")
    ap.add_argument("--batch", help="JSON {case: K} of calls per sample")
    ap.add_argument("--calibrate", help="measure the K table and write it to this file")
    args = ap.parse_args()
    batches = json.loads(pathlib.Path(args.batch).read_text()) if args.batch else {}
    calibrated = {}
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    validation = {p["id"]: p for p in json.loads((CORPUS / "validation.json").read_text())["programs"]}
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / f"{args.backend}-results.jsonl"
    failures = 0
    cases = 0
    with results.open("a") as sink:
        for program in manifest["programs"]:
            if args.only not in program["id"]:
                continue
            expected = {tuple(s["input"]): s for s in validation[program["id"]]["scored_inputs"]}
            for scored in program["scored_inputs"]:
                seed, n = scored[0], scored[1]
                m = scored[2] if len(scored) > 2 else 0
                name = f"{program['id']}-{'-'.join(map(str, scored))}"
                output = out_dir / f"{name}.bin"
                command = [str(pathlib.Path(args.runner).resolve())]
                if args.backend == "metal":
                    command.append(str(KERNELS))
                command += [program["family"], program["arm"],
                            "1" if program["transforms"] == "trigonometric" else "0",
                            str(seed), str(n), str(m), str(output)]
                k = 1 if args.calibrate else batches.get(name, 1)
                for _ in range(6 if args.calibrate else 1):
                    run = subprocess.run(command + [str(k)], capture_output=True, text=True)
                    if run.returncode != 0:
                        break
                    record = json.loads(run.stdout)
                    if record["min_sample_gpu_ms"] >= TARGET_MS:
                        break
                    k = math.ceil(k * TARGET_MS / record["min_sample_gpu_ms"])
                if run.returncode != 0:
                    print(f"FAIL {name}: exit {run.returncode} {run.stderr.strip()}", file=sys.stderr)
                    failures += 1
                    continue
                digest = hashlib.sha256(output.read_bytes()).hexdigest()
                want = expected[tuple(scored)]
                calibrated[name] = record["batch"]
                record.update({"case": name, "id": program["id"], "split": program["split"], "variant": program["variant"],
                               "input": scored, "output_sha256": digest,
                               "reference_sha256": want["sha256"],
                               "bitwise_equal": digest == want["sha256"] and record["output_words"] == want["words"]})
                sink.write(json.dumps(record) + "\n")
                cases += 1
                status = "ok" if record["bitwise_equal"] else "MISMATCH"
                failures += 0 if record["bitwise_equal"] else 1
                print(f"{status} {name} batch={record['batch']} "
                      f"min_sample_gpu_ms={record['min_sample_gpu_ms']:.4g} "
                      f"device_median_ms={record['device_median_ms']:.4g} "
                      f"host_median_ms={record['host_median_ms']:.4g}")
    if args.calibrate:
        pathlib.Path(args.calibrate).write_text(json.dumps(calibrated, indent=1, sort_keys=True) + "\n")
    print(f"cases={cases} failures={failures} results={results}")
    return 1 if failures or not cases else 0


if __name__ == "__main__":
    sys.exit(main())
