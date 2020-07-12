EXEC_CMD=$1
EXEC_FILENAME=$(basename -- "$EXEC_CMD")
CALLGRIND_OUT_FILENAME="$EXEC_FILENAME.out"
DOT_FILENAME="$EXEC_FILENAME.dot"

valgrind --tool=callgrind --callgrind-out-file="$CALLGRIND_OUT_FILENAME" $EXEC_CMD
gprof2dot --format=callgrind --output="$DOT_FILENAME" "$CALLGRIND_OUT_FILENAME"
dot -Tsvg "$DOT_FILENAME" -o "$EXEC_FILENAME.profile.svg"
rm "$CALLGRIND_OUT_FILENAME"
rm "$DOT_FILENAME"
