"""Temporary regeneration for the visual-mesh strip (deleted before merge)."""
import hashlib, json, os, shutil, subprocess, sys, tempfile

root = os.getcwd()
invar = os.path.join(root, "target/debug/invar")
out_dir = os.path.join(root, "regen-out")
os.makedirs(out_dir, exist_ok=True)


def run(args):
    return subprocess.run([invar, *args], cwd=root, capture_output=True)


def tree_hash(directory):
    files = []
    for base, _, names in os.walk(directory):
        for name in names:
            path = os.path.join(base, name)
            files.append((os.path.relpath(path, directory), path))
    files.sort()
    h = hashlib.sha256()
    for rel, path in files:
        data = open(path, "rb").read()
        h.update(f"{rel}\0{len(data)}\0".encode())
        h.update(data)
    return len(files), h.hexdigest()


# 1. examples/index.json expected IR and generated-tree hashes
index_path = os.path.join(root, "examples/index.json")
document = json.load(open(index_path))
changes = []
for entry in document["examples"]:
    expect = entry["expect"]
    full = os.path.join(root, entry["path"])
    if "ir_sha256" in expect:
        out = run(["system", "ir", full])
        assert out.returncode == 0, (entry["path"], out.stderr.decode())
        again = run(["system", "ir", full])
        assert out.stdout == again.stdout
        new = hashlib.sha256(out.stdout).hexdigest()
        if new != expect["ir_sha256"]:
            changes.append((entry["path"], "ir", expect["ir_sha256"], new))
            expect["ir_sha256"] = new
    for target in expect.get("generate", []):
        results = []
        for attempt in range(2):
            gen = os.path.join(tempfile.mkdtemp(), "out")
            r = run(["system", "gen", target["target"], full, "--out", gen])
            assert r.returncode == 0, (entry["path"], target["target"], r.stderr.decode())
            results.append(tree_hash(gen))
            shutil.rmtree(os.path.dirname(gen))
        assert results[0] == results[1]
        files, digest = results[0]
        if (files, digest) != (target["files"], target["tree_sha256"]):
            changes.append((entry["path"], target["target"], (target["files"], target["tree_sha256"]), (files, digest)))
            target["files"], target["tree_sha256"] = files, digest
open(os.path.join(out_dir, "index.json"), "w").write(json.dumps(document, indent=2) + "\n")
open(os.path.join(out_dir, "index-changes.txt"), "w").write("\n".join(map(str, changes)) + "\n")
print("index changes:", len(changes))

# 2. rendered scene identities (the binder's qualified model_xml) for both arms
identities = {}
for name, cell in (("panda", "cell.inv"), ("ur5e", "cell-ur5e.inv")):
    gen = os.path.join(out_dir, f"mjcf-{name}")
    shutil.rmtree(gen, ignore_errors=True)
    r = run(["system", "gen", "mjcf", os.path.join(root, "examples/cells/reach_place", cell), "--out", gen])
    assert r.returncode == 0, r.stderr.decode()
    identities[name] = hashlib.sha256(open(os.path.join(gen, "scene.xml"), "rb").read()).hexdigest()
json.dump(identities, open(os.path.join(out_dir, "scene-identities.json"), "w"), indent=2)
print(identities)
