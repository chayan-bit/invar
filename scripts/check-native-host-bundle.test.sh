#!/bin/sh
# Self-test for scripts/check-native-host-bundle.py: the real tree passes, and each class of bundle drift it exists
# for fails, on a scratch copy of the bundled sources.
set -eu
: "${TMPDIR:?Set a private TMPDIR before running the self-test}"
exec python3 - "$(cd "$(dirname "$0")/.." && pwd)" <<'PY'
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
script = root / 'scripts/check-native-host-bundle.py'
COPY = ['src', 'crates/invar-device/src', 'crates/invar-foreign/src']

def run(tree):
    result = subprocess.run([sys.executable, '-I', str(script), '--root', str(tree)], capture_output=True, text=True)
    return result.returncode, result.stdout

code, out = run(root)
assert code == 0, out
print('PASS: the tree passes:', out.strip())

def edit(tree, path, old, new):
    file = tree / path
    text = file.read_text()
    assert old in text, (path, old)
    file.write_text(text.replace(old, new, 1))

# (description, edits, text the failure must name)
CASES = [
    ('#377: numeric_contract declares `pub mod fold;` and the host does not inline it',
     [('src/native.rs', '("pub mod fold;", include_str!("numeric_contract/fold.rs")),', '')],
     'src/numeric_contract.rs:'),
    ('a new file module in an embedded source',
     [('src/numeric_contract.rs', 'mod ieee;', 'mod ieee;\nmod extra;')],
     '`mod extra;` is a file module'),
    ('#377: fold.rs reaches crate::fsum_reference and the host lacks the module',
     [('src/native.rs', 'mod fsum_reference {{ {} }}', 'mod fsum_ref_gone {{ {} }}')],
     'uses crate::fsum_reference'),
    ('the declaration changed, so the inline replacen no longer finds it',
     [('src/numeric_contract.rs', 'pub mod storage;', 'pub(crate) mod storage;')],
     'replacen would not find it'),
    ('a new invar-foreign module, neither stripped nor inlined',
     [('crates/invar-foreign/src/lib.rs', 'pub mod component;', 'pub mod component;\npub mod brand_new;')],
     '`pub mod brand_new;` is a file module'),
]
def fresh(work):
    tree = Path(work, 'tree')
    shutil.rmtree(tree, ignore_errors=True)
    for path in COPY:
        source, target = root / path, tree / path
        target.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if source.is_dir() else shutil.copy)(source, target)
    assert run(tree)[0] == 0, 'the scratch copy must pass before the edit'
    return tree

with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as work:
    for description, edits, expected in CASES:
        tree = fresh(work)
        for path, old, new in edits:
            edit(tree, path, old, new)
        code, out = run(tree)
        assert code == 1 and expected in out, (description, out)
        print('PASS: fails on', description)
    # A test-only module is compiled out of the host, so it needs no inline entry.
    tree = fresh(work)
    edit(tree, 'crates/invar-foreign/src/lib.rs', 'pub mod component;', 'pub mod component;\n#[cfg(test)]\nmod tests;')
    code, out = run(tree)
    assert code == 0, out
    print('PASS: a #[cfg(test)] module needs no inline entry')
PY
