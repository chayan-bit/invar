#!/usr/bin/env bash
set -euo pipefail
out=${1:?usage: run.sh NEW_OUTPUT_DIR}
mkdir -p "$out"
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -shared -fPIC -pthread \
  examples/cells/reach_place/components/curobo_planner/planner_reference.c \
  -o "$out/libcurobo_planner_reference.so"
python3 tests/fixtures/foreign-components/curobo_planner/test_client.py \
  "$out/libcurobo_planner_reference.so"
