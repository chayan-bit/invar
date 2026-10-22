"""Derive contract-v2.json from the frozen contract.json for the principal-inertia Panda.

Only the Panda model identity changes: `sha256` becomes the SHA-256 of the
vendored principal-form `panda.xml`, `derived_from_sha256` records the upstream
model (whose FK goldens still apply), and `derivation` names the rewrite
script by hash. Tolerances, inputs, libraries and UR5e stay byte-equal in value.
Usage: derive_contract_v2.py REPO
"""

import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(repo):
    repo = Path(repo)
    here = repo / "tests/fixtures/foreign-components/gpu-parity-oracles"
    contract = json.loads((here / "contract.json").read_text())
    panda = contract["models"][0]
    assert panda["name"] == "panda" and panda["file"] == "franka_emika_panda/panda.xml"
    model = repo / "examples/cells/reach_place/assets/menagerie" / panda["file"]
    script = repo / "tests/fixtures/s1_06/diagonalize_panda_inertia.py"
    panda["derived_from_sha256"] = panda["sha256"]
    panda["sha256"] = sha(model)
    panda["derivation"] = {"script": "tests/fixtures/s1_06/diagonalize_panda_inertia.py",
                           "script_sha256": sha(script),
                           "change": "every fullinertia written as the equivalent quat + diaginertia"}
    contract["status"] = "internal_cpu_oracle_contract_v2_principal_panda_inertia_frozen_before_generation"
    contract["supersedes"] = {"file": "contract.json", "sha256": sha(here / "contract.json")}
    (here / "contract-v2.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(sha(here / "contract-v2.json"))


if __name__ == "__main__":
    main(sys.argv[1])
