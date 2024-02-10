#!/usr/bin/env bash

# Define an array of epochs
lr=(1e-06 1e-05 1e-04 1e-04 1e-02 1e-01)

# Loop through each batch size
for rate in "${lr[@]}"
do
    echo "Training with rate: $rate"
    $MASE/machop/ch train jsc-tiny jsc --max-epochs 10 --batch-size 256 --project "lab-1_jsc-tiny_varying-lr" --learning-rate $rate
done

echo "Training complete."

