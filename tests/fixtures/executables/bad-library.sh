#!/bin/sh
if [ "$1" = --version ];then printf test;exit 0;fi
for target do :;done
printf 'not a shared library' > "$target"
