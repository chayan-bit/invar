#!/bin/sh
# Killed by a signal its parent never sent, as by the kernel OOM killer.
cat >/dev/null
kill -9 $$
