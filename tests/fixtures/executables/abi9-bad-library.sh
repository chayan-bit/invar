#!/bin/sh
if [ "$1" = --version ];then printf 'trusted test compiler';exit 0;fi
for arg do dest="$arg";done
printf 'invalid library bytes' > "$dest"
exit 0
