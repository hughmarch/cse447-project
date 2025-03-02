#!/bin/bash

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
DATA_DIR="$PROJECT_ROOT/work/data/tatoeba"
EVAL_RESULTS_DIR="$PROJECT_ROOT/work/eval"

# Parse arguments
while getopts "s:" opt; do
  case ${opt} in
    s )
      TEST_SIZE="$OPTARG"
      ;;
    \? )
      echo "Usage: eval.sh -s {001|005|025|100}"
      exit 1
      ;;
  esac
done

# Validate test size
if [[ ! "$TEST_SIZE" =~ ^(001|005|025|100)$ ]]; then
  echo "Error: Invalid test size. Choose from 001, 005, 025, or 100."
  exit 1
fi

TEST_DIR="$DATA_DIR/test-$TEST_SIZE"

# Step 1: Prepare test dataset
bash "$PROJECT_ROOT/src/data/tatoeba/prepare-tatoeba.sh"

# Step 2: Run prediction
PREDICTIONS_FILE="$EVAL_RESULTS_DIR/pred.txt"
mkdir -p "$EVAL_RESULTS_DIR"
bash "$PROJECT_ROOT/src/predict.sh" "$TEST_DIR/input.txt" "$PREDICTIONS_FILE"

# Step 3: Run evaluation
python "$PROJECT_ROOT/src/eval/eval.py" \
  "$TEST_DIR/answer.txt" \
  "$PREDICTIONS_FILE" \
  "$TEST_DIR/language.txt" \
  "$EVAL_RESULTS_DIR"
