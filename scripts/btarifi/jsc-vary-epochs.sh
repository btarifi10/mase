#!/usr/bin/env bash

# Define an array of epochs
epochs=(5 10 50 100 200)

# Loop through each batch size
for epoch in "${epochs[@]}"
do
    echo "Training with max epochs: $epoch"
    $MASE/machop/ch train jsc-tiny jsc --max-epochs $epoch --batch-size 128 --project "jsc-tiny_classification_jsc_2024-01-26_epochs"
done

echo "Training complete for all epochs."

