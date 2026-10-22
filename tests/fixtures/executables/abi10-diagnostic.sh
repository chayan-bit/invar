#!/bin/sh
if [ "$1" = --version ]; then printf 'trusted test compiler';exit 0;fi
printf 'ORIGINAL_ABI10_COMPILER_DIAGNOSTIC' >&2
exit 27
