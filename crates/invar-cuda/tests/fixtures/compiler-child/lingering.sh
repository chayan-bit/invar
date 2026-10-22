#!/bin/sh
# Succeeds, but leaves a descendant holding stdout and stderr open after exit.
cat >/dev/null
sleep 30 &
printf mock-cubin
printf 'descendant=%s' "$!" >&2
