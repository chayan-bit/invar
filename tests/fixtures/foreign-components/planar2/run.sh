#!/usr/bin/env bash
set -euo pipefail

out=${1:?usage: run.sh NEW_OUTPUT_DIR}
src=$(cd "$(dirname "$0")" && pwd)
[[ ! -e $out ]] || { echo 'Refusing existing output directory' >&2; exit 1; }
mkdir -p "$out"
cc -std=c11 -Wall -Wextra -Werror -pedantic -fPIC -shared \
  "$src/planar2.c" -lm -o "$out/libplanar2.so"
python3 "$src/generate_goldens.py" "$out/planar2_goldens.json"
cmp "$src/planar2_goldens.json" "$out/planar2_goldens.json"
python3 "$src/test_planar2.py" "$out/libplanar2.so" "$out/planar2_goldens.json"
