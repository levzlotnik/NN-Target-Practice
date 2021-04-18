#!/bin/bash

EXEC_CMD="$@"
echo "$EXEC_CMD"
FIRST_ARG=$1
EXEC_FILENAME=$(basename -- "$FIRST_ARG")
CALLGRIND_OUT_FILENAME="$EXEC_FILENAME.out"
DOT_FILENAME="$EXEC_FILENAME.dot"
STACK_SIZE=10000000

valgrind --tool=callgrind --main-stacksize=$STACK_SIZE --callgrind-out-file="$CALLGRIND_OUT_FILENAME" $EXEC_CMD
gprof2dot --format=callgrind --output="$DOT_FILENAME" "$CALLGRIND_OUT_FILENAME" && \
dot -Tsvg "$DOT_FILENAME" -o "$EXEC_FILENAME.profile.svg" && \
rm "$CALLGRIND_OUT_FILENAME" && \
rm "$DOT_FILENAME"
