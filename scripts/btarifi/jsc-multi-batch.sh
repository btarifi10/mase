#!/usr/bin/env bash

# Define an array of batch sizes
batch_sizes=(64 128 256 512 1024 2048)

# Loop through each batch size
for batch_size in "${batch_sizes[@]}"
do
    echo "Training with batch size: $batch_size"
    $MASE/machop/ch train jsc-tiny jsc --max-epochs 10 --batch-size $batch_size --project "lab-1_jsc-tiny_varying-batch-size"
done

echo "Training complete for all batch sizes."

