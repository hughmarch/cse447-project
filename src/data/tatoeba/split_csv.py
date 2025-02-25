import argparse
import pandas as pd
import numpy as np

def split_csv(input_file, test_ratio, seed, output_prefix):
    # Define output file names
    train_file = f"{output_prefix}-train.csv"
    test_file = f"{output_prefix}-test.csv"
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Define column names manually
    column_names = ["id", "lang", "sentence"]

    # Read CSV in chunks with tab separator
    chunks = pd.read_csv(
        input_file,
        sep="\t",
        names=column_names,  # Assign column names explicitly
        header=None,  # No header in file
        chunksize=500000,
        low_memory=False,
        dtype={"id": str, "lang": str, "sentence": str}  # Ensure proper parsing
    )

    # Collect chunks into a shuffled dataframe
    df = pd.concat(chunks, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle

    # Split the data
    split_idx = int(len(df) * test_ratio)
    test_df = df.iloc[:split_idx]
    train_df = df.iloc[split_idx:]

    # Save to CSV with tab separator
    test_df.to_csv(test_file, sep="\t", index=False, header=False)
    train_df.to_csv(train_file, sep="\t", index=False, header=False)

    print(f"Saved {len(train_df)} rows to {train_file}")
    print(f"Saved {len(test_df)} rows to {test_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a test-train split on a large CSV file.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("test_ratio", type=float, help="Proportion of data to use for the test set (e.g., 0.3 for 30%)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for shuffling")
    parser.add_argument("--output_prefix", type=str, default="split", help="Prefix for output CSV files")

    args = parser.parse_args()
    
    split_csv(args.input_file, args.test_ratio, args.seed, args.output_prefix)
