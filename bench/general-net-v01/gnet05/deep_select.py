"""Select the GNET-05 deep inputs with the #133 net-laws prototype.

Usage: python3 deep_select.py PROTOTYPE_DIR [CORPUS_DIR] > deep.json

PROTOTYPE_DIR holds `engine.py` and `laws.py` from #133 Appendix C.
For every program with a ladder (divide, compose divide-of-branch and
forkjoin), each chain is walked in ascending order and the first rung whose
law-directed average parallelism W/S (model c) reaches TARGET is a
candidate; the candidate with the least work across chains is selected.
A chain stops at a trap or at MAX_WORK. A program none of whose rungs reaches
TARGET within MAX_WORK selects its highest-W/S rung and is reported as short.
The selected inputs are frozen in `generate.rs` (`DEEP_INPUTS`) with their
reference step budgets; this script is the record of how they were chosen.
"""
import json
import pathlib
import sys

TARGET = 1000.0
# Prototype work is about half the reference evaluator's steps; the corpus
# step budget is capped at 10,000,000.
MAX_WORK = 4_500_000
FAMILIES = ("divide", "compose", "forkjoin")


def main():
    sys.path.insert(0, sys.argv[1])
    import engine

    root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(__file__).parent)
    manifest = json.loads((root / "manifest.json").read_text())
    out = []
    for p in manifest["programs"]:
        if p["family"] not in FAMILIES or not p["ladder"]:
            continue
        prog = engine.Program((root / p["file"]).read_text())
        conv = engine.U if p["param"] == "u32" else int
        best, top = None, None
        for chain in p["ladder"]:
            for rung in chain:
                r = engine.run(prog, "run", [conv(x) for x in rung], fuel=MAX_WORK)
                if r["status"] != "ok":
                    break
                s = r["span"]
                row = {"input": rung, "work": r["work"],
                       "ws": {m: r["work"] / max(s[m], 1) for m in ("a", "b", "c")},
                       "tasks": r["tasks"], "forms_c": r["forms"].get("c")}
                if top is None or row["ws"]["c"] > top["ws"]["c"]:
                    top = row
                if row["ws"]["c"] >= TARGET:
                    if best is None or row["work"] < best["work"]:
                        best = row
                    break
        chosen = best or top
        chosen.update(id=p["id"], family=p["family"], split=p["split"], short=best is None)
        print(json.dumps(chosen), file=sys.stderr, flush=True)
        out.append(chosen)
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
