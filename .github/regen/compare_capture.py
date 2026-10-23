"""Compare the regenerated Panda home capture with the committed one."""
import json, sys

old = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))
SHIFT = int(sys.argv[3])  # old robot geoms minus new robot geoms


def remap(contacts):
    return [[g if g == 0 else g - SHIFT for g in pair] for pair in contacts]


def same(a, b):
    for key in ("step", "time", "q", "dq", "forces", "controls", "witness"):
        if a[key] != b[key]:
            return f"{key}: {a[key]!r} != {b[key]!r}"
    if remap(a["contacts"]) != b["contacts"]:
        return f"contacts: {remap(a['contacts'])} != {b['contacts']}"
    return None


problems = []
if same(old["initial"], new["initial"]):
    problems.append(("initial", same(old["initial"], new["initial"])))
for name in ("executed",):
    assert len(old[name]) == len(new[name]), name
    for i, (a, b) in enumerate(zip(old[name], new[name])):
        if (d := same(a, b)):
            problems.append((name, i, d))
assert len(old["candidates"]) == len(new["candidates"])
for i, (a, b) in enumerate(zip(old["candidates"], new["candidates"])):
    if a["step"] != b["step"] or a["control"] != b["control"] or (d := same(a["state"], b["state"])):
        problems.append(("candidates", i, d if a["step"] == b["step"] else "step"))
print("compared", len(old["executed"]), "states", len(old["candidates"]), "candidates;", len(problems), "differences")
for p in problems[:10]:
    print(p)
sys.exit(1 if problems else 0)
