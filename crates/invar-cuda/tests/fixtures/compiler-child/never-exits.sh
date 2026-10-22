#!/bin/sh
# Never exits; its descendant holds the pipes and records its pid if asked.
cat >/dev/null
sleep 1000 &
[ -z "$INVAR_COMPILER_PIDFILE" ] || printf '%s\n' "$!" >"$INVAR_COMPILER_PIDFILE"
wait
