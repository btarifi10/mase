#!/usr/bin/env bash

# Define an array of epochs
epochs=(5 10 20 50 100 200)

# Loop through each batch size
for epoch in "${epochs[@]}"
do
    echo "Training with max epochs: $epoch"
    $MASE/machop/ch train jsc-tiny jsc --max-epochs $epoch --batch-size 128 --project "lab-1_jsc-tiny_varying-epoch"
done

echo "Training complete for all epochs."

