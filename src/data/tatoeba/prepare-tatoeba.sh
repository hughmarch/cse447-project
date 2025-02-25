#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/../../../work/data/tatoeba"
TARGET_FILE="$TARGET_DIR/sentences-test.csv"
INPUT_FILE="$TARGET_DIR/sentences.csv"
PREPROCESSED_DIR="$TARGET_DIR/tatoeba-preprocessed-test"

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

# Preprocess test dataset if not already done
if [ ! -d "$PREPROCESSED_DIR" ]; then
    echo "Preprocessing test dataset..."
    python "$SCRIPT_DIR/prepare_test.py" "$TARGET_FILE" "$PREPROCESSED_DIR" 5
    echo "Test dataset preprocessing complete!"
else
    echo "Test dataset already preprocessed. Skipping."
fi
