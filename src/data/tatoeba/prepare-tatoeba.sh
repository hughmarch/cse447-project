#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/../../../work/data/tatoeba"
TARGET_FILE="$TARGET_DIR/sentences-test.csv"
INPUT_FILE="$TARGET_DIR/sentences.csv"

# Define expected dataset directories
DATASET_SIZES=("test-001" "test-005" "test-025" "test-100")

if [ ! -f "$TARGET_FILE" ]; then
    mkdir -p "$TARGET_DIR"
    
    echo "Downloading and extracting dataset..."
    wget -c https://downloads.tatoeba.org/exports/sentences.tar.bz2 -O temp.tar.bz2
    tar xjf temp.tar.bz2 -C "$TARGET_DIR"
    rm temp.tar.bz2

    echo "Splitting dataset into train and test..."
    python "$SCRIPT_DIR/split_csv.py" "$INPUT_FILE" 0.3 --output_prefix "$TARGET_DIR/sentences" --seed 1

    echo "Removing original CSV to save space..."
    rm "$INPUT_FILE"

    echo "Processing complete!"
else
    echo "Dataset already processed. Skipping download and extraction."
fi

# Check if all expected preprocessed test datasets exist
ALL_DATASETS_PRESENT=true
for DATASET in "${DATASET_SIZES[@]}"; do
    if [ ! -d "$TARGET_DIR/$DATASET" ]; then
        ALL_DATASETS_PRESENT=false
        break
    fi
done

# Preprocess test dataset if not already done
if [ "$ALL_DATASETS_PRESENT" = false ]; then
    echo "Preprocessing test dataset..."
    python "$SCRIPT_DIR/prepare_test.py" "$TARGET_FILE" "$TARGET_DIR" 1 # num_splits per sentence
    echo "Test dataset preprocessing complete!"
else
    echo "All test datasets already preprocessed. Skipping."
fi
