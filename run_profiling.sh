#!/bin/bash

OPTION=$1
TARGET="./build/raytracer"
REPORTS_PATH="./reports"
DATE=$(date '+_%Y-%m-%d_%H-%M')

if [[ -z "$1" ]]; then
    echo 'use -asan for memchek, -tsan for racecheck'
    exit 1
fi

if [ "$OPTION" == '-asan' ]; then
    compute-sanitizer --tool memcheck "$TARGET"
elif [ "$OPTION" == '-tsan' ]; then
    compute-sanitizer --tool racecheck "$TARGET"
elif [ "$OPTION" == '-ncu' ]; then
    if [ ! -d $REPORTS_PATH ]; then
        mkdir -p reports
    fi
    sudo ncu -o "$REPORTS_PATH/report$DATE" "$TARGET"
fi