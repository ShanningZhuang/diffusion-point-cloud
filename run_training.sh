#!/bin/bash

# Define the categories
CATEGORIES=("car" "airplane" "bag" "table")

# Loop through the categories and run the python script
for category in "${CATEGORIES[@]}"
do
  echo "Training for category: $category"
  python train_gen.py --categories "$category" --max_iters 20000 --test_freq 2000
done

echo "All categories have been trained." 