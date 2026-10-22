#!/bin/sh
[ "$1:$2:$3:$4:$#" = "12:4:86:strict-f32-v1:4" ] || exit 17
cat >/dev/null
printf mock-cubin
printf "original warning" >&2
