#!/usr/bin/env bash

# Define an array of batch sizes
batch_sizes=(128 256 512 1024 2048)

# Loop through each batch size
for batch_size in "${batch_sizes[@]}"
do
    echo "Training with batch size: $batch_size"
    $MASE/machop/ch train jsc-tiny jsc --max-epochs 10 --batch-size $batch_size
done

echo "Training complete for all batch sizes."

