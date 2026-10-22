#!/bin/sh
# Over the 512-byte log cap, then a descendant keeps both pipes open after exit.
cat >/dev/null
sleep 30 &
head -c 513 /dev/zero >&2
printf mock-cubin
