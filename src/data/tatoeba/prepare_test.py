import os
import argparse
import numpy as np
from tqdm import tqdm

def prepare_test(input_file, output_dir, num_splits, seed=1):
    np.random.seed(seed)

    # Define dataset sizes (fractions of test-100)
    dataset_sizes = {
        "test-001": 0.01,
        "test-005": 0.05,
        "test-025": 0.25,
        "test-100": 1.00
    }

    # Create base directory
    os.makedirs(output_dir, exist_ok=True)

    # Create test-100 directory
    test_100_dir = os.path.join(output_dir, "test-100")
    os.makedirs(test_100_dir, exist_ok=True)

    # Open files for test-100
    input_100 = open(os.path.join(test_100_dir, "input.txt"), "w", encoding="utf-8")
    answer_100 = open(os.path.join(test_100_dir, "answer.txt"), "w", encoding="utf-8")
    lang_100 = open(os.path.join(test_100_dir, "language.txt"), "w", encoding="utf-8")

    # Process file line by line to avoid excessive memory usage
    total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))
    total_out_lines = 0

    with open(input_file, "r", encoding="utf-8") as f, tqdm(total=total_lines, desc="Processing test-100", unit="sentence") as progress_bar:
        for line in f:
            parts = line.strip().split()
            progress_bar.update(1)

            if len(parts) < 3:
                continue  # Skip malformed lines

            _, lang, sentence = parts[0], parts[1], " ".join(parts[2:])  # Extract language and sentence

            min_sequence = 2

            if len(sentence) < min_sequence + 1:
                continue  # Skip empty or very short sentences

            # Generate `num_splits` random indices in the range [min_sequence, len(sentence)]
            split_indices = np.random.choice(
              len(sentence) - min_sequence, min(num_splits, len(sentence) - min_sequence), replace=False) + min_sequence

            for idx in split_indices:
                input_100.write(sentence[:idx] + "\n")
                answer_100.write(sentence[idx] + "\n")
                lang_100.write(lang + "\n")
                total_out_lines += 1

    input_100.close()
    answer_100.close()
    lang_100.close()

    print("Finished processing test-100. Now creating smaller datasets...")

    # Create output directories and open file streams for all datasets
    dataset_streams = {}
    sample_limits = {}

    for dataset_name, fraction in dataset_sizes.items():
        if dataset_name == "test-100":
            continue  # Already created

        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        dataset_streams[dataset_name] = {
            "input": open(os.path.join(dataset_dir, "input.txt"), "w", encoding="utf-8"),
            "answer": open(os.path.join(dataset_dir, "answer.txt"), "w", encoding="utf-8"),
            "lang": open(os.path.join(dataset_dir, "language.txt"), "w", encoding="utf-8"),
        }

        sample_limits[dataset_name] = int(total_out_lines * fraction)  # Set max lines for each dataset

    # Copy lines from test-100 to smaller datasets without reading all at once
    with open(os.path.join(test_100_dir, "input.txt"), "r", encoding="utf-8") as full_input, \
         open(os.path.join(test_100_dir, "answer.txt"), "r", encoding="utf-8") as full_answer, \
         open(os.path.join(test_100_dir, "language.txt"), "r", encoding="utf-8") as full_lang:

        # Track how many lines we've written for each dataset
        dataset_counts = {name: 0 for name in dataset_sizes.keys() if name != "test-100"}

        with tqdm(total=total_out_lines, desc="Copying to smaller datasets", unit="line") as progress_bar:
            for input_line, answer_line, lang_line in zip(full_input, full_answer, full_lang):
                for dataset_name, stream in dataset_streams.items():
                    if dataset_counts[dataset_name] < sample_limits[dataset_name]:
                        stream["input"].write(input_line)
                        stream["answer"].write(answer_line)
                        stream["lang"].write(lang_line)
                        dataset_counts[dataset_name] += 1

                progress_bar.update(1)

    # Close all output streams
    for dataset_name, streams in dataset_streams.items():
        streams["input"].close()
        streams["answer"].close()
        streams["lang"].close()

    print(f"Created datasets: {', '.join(dataset_streams.keys())}")
    print("All datasets created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare test datasets at different sizes.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("num_splits", type=int, help="Number of splits per sentence")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")

    args = parser.parse_args()
    
    prepare_test(args.input_file, args.output_dir, args.num_splits, args.seed)
