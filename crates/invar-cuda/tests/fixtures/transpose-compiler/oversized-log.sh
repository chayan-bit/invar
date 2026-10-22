#!/bin/sh
cat >/dev/null
head -c 513 /dev/zero >&2
printf mock-cubin
