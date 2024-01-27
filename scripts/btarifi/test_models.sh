#!/bin/bash

# Check if the directory path is provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]
then
    echo "Usage: $0 [model] [dataset] [project-name] [optional: file pattern]"
    exit 1
fi

# Assign the first argument to DIRECTORY
MODEL=$1
DATASET=$2
PROJECT_NAME=$3

DIRECTORY="$MASE/mase_output/$PROJECT_NAME/software/training_ckpts"

# Assign the second argument to PATTERN, default to '*.txt' if not provided
PATTERN=${4:-"best"}

FILES=$(ls "$DIRECTORY" | grep "$PATTERN")

# Using ls and grep in a for loop
for file in $FILES
do
    echo "Testing $file: $DIRECTORY/$file"
    touch "$DIRECTORY/$file-output.txt"
    $MACHOP/ch test $MODEL $DATASET --load "$DIRECTORY/$file" --project "${PROJECT_NAME}_test" --load-type pl | tee "$DIRECTORY/$file-output.txt"
done
