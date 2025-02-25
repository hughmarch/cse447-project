import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def prepare_test(input_file, output_dir, num_splits, seed=1):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Define input and answer directories
    input_dir = os.path.join(output_dir, "input")
    answer_dir = os.path.join(output_dir, "answer")

    # Create directories if they don’t exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)

    # Define column names for the input file
    column_names = ["id", "lang", "sentence"]

    # Read CSV file
    df = pd.read_csv(input_file, sep="\t", names=column_names, header=None, dtype=str)

    # Initialize file handles
    file_handles = {}

    # Progress bar
    progress_bar = tqdm(total=len(df), desc="Processing sentences", unit="sentence")

    for _, row in df.iterrows():
        lang, sentence = row["lang"], row["sentence"]
        if not isinstance(sentence, str) or len(sentence) == 0:
            continue  # Skip empty or invalid sentences

        # Open file handles if not already open
        if lang not in file_handles:
            input_path = os.path.join(input_dir, f"{lang}.txt")
            answer_path = os.path.join(answer_dir, f"{lang}.txt")
            file_handles[lang] = {
                "input": open(input_path, "a", encoding="utf-8"),
                "answer": open(answer_path, "a", encoding="utf-8")
            }

        # Generate `num_splits` random indices
        split_indices = np.random.choice(len(sentence), min(num_splits, len(sentence)), replace=False)
        
        # Write to files
        for idx in split_indices:
            file_handles[lang]["input"].write(sentence[:idx] + "\n")
            file_handles[lang]["answer"].write(sentence[idx] + "\n")

        progress_bar.update(1)

    # Close file handles
    for lang in file_handles:
        file_handles[lang]["input"].close()
        file_handles[lang]["answer"].close()

    progress_bar.close()
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a dataset by splitting sentences into input and answer files.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("num_splits", type=int, help="Number of splits per sentence")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")

    args = parser.parse_args()
    
    prepare_test(args.input_file, args.output_dir, args.num_splits, args.seed)
