#!/bin/sh
# Exits once a descendant leads its own session, beyond any group kill; that
# descendant holds the pipes for 10 s.
cat >/dev/null
setsid sleep 10 &
p=$!
until [ "$(cut -d' ' -f6 "/proc/$p/stat")" = "$p" ]; do :; done
printf mock-cubin
