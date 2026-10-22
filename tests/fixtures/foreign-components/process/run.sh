#!/usr/bin/env bash
set -euo pipefail

out=${1:?usage: run.sh NEW_OUTPUT_DIR}
src=$(cd "$(dirname "$0")" && pwd)
component=$(cd "$src/../../../../examples/cells/reach_place/components/process_fixture" && pwd)
[[ ! -e $out ]] || { echo 'Refusing existing output directory' >&2; exit 1; }
mkdir -p "$out"
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fPIC -shared \
  "$component/process_reference.c" -lm \
  -Wl,"--version-script=$component/abi_exports.map" \
  -o "$out/libprocess_fixture.so"
python3 "$src/test_client.py" "$out/libprocess_fixture.so" "$src/fixture_server.py"
