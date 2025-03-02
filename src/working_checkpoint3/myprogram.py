import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import string
import random
import tqdm
from datasets import load_dataset
import aiohttp
import time
import logging
import numpy as np


def preprocess_opensubtitles(work_dir, lang1="en", lang2="fr", split_ratio=0.9):
    """
    Fetch and preprocess the OpenSubtitles dataset, splitting it into train/test subsets.
    """
    logging.info(f"Loading OpenSubtitles dataset for language pair {lang1}-{lang2}...")
    dataset = load_dataset("open_subtitles", lang1=lang1, lang2=lang2, split="train",
            storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
)

    logging.info("Processing dataset...")
    # Extract translation text for one language (e.g., lang1)
    all_texts = [example["translation"]["en"] for example in dataset]
    all_texts = [text.strip() for text in all_texts if text.strip()]  # Clean empty lines

    # Shuffle and split dataset
    split_idx = int(len(all_texts) * split_ratio)
    train_texts = all_texts[:split_idx]
    test_texts = all_texts[split_idx:]

    # Save to files
    train_path = os.path.join(work_dir, "train.txt")
    test_path = os.path.join(work_dir, "test.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_texts))

    logging.info(f"Train dataset saved to {train_path}")
    logging.info(f"Test dataset saved to {test_path}")

    return train_path, test_path



def preprocess_to_subset(file_path, output_path, subset_ratio=0.001):
    """
    Preprocess the dataset to include only a subset of the data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    subset_size = int(len(lines) * subset_ratio)
    subset_lines = lines[:subset_size]

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(subset_lines)

    logging.info(f"Subset of {subset_size} lines saved to {output_path}")

# ---- PREPROCESSING ----
def preprocess_text(text, remove_spaces=False):
    """
    Preprocess the input text for character-level modeling.
    - Optionally removes spaces or normalizes them.
    - Removes non-printable characters.

    Args:
        text (str): Raw input text.
        remove_spaces (bool): If True, removes all spaces.

    Returns:
        str: Cleaned text.
    """
    logging.info("Preprocessing text...")
    # Remove non-printable characters
    
    text = text[:2000000]
    if "'" in text:
        print("Apostrophe found in text")
    text = "".join(ch for ch in text if ch in string.printable)
    if "'" in text:
        print("Apostrophe found in text after preprocess")
    if remove_spaces:
        # Remove spaces completely
        text = text.replace(" ", "")
    else:
        # Normalize multiple spaces to a single space
        text = " ".join(text.split())

    return text

# ---- DATASET ----
class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)  # Number of unique characters

        # Encode text
        self.encoded_text = [self.char_to_idx[ch] for ch in text]

        # Create input-output pairs
        self.data = [
            (self.encoded_text[i : i + seq_len], self.encoded_text[i + seq_len])
            for i in range(len(self.encoded_text) - seq_len)
        ]

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Return a specific sample (input sequence and target character)
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ---- MODEL ----
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # print("HIDDEN DIM: " + str(hidden_dim))
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, dropout=0.4, \
            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, vocab_size) # hidden_dim*2 if bidir = True

    def forward(self, x, hidden=None):
        # print("Input shape (x):", x.shape)  # Print the shape of the input tensor x
        x = self.embed(x)
        # print("Embedded input shape:", x.shape)  # Print the shape after embedding

        out, hidden = self.rnn(x, hidden)
        # print("Output shape (out):", out.shape)  # Print the shape of the output from the RNN
        # print("Hidden state shape:", hidden[0].shape)  # Print the shape of the hidden state
        # print("Cell state shape:", hidden[1].shape)  # Print the shape of the cell state (if LSTM)

        logits = self.fc(out[:, -1, :])  # Only the last time step
        # print("Logits shape:", logits.shape)  # Print the shape of the logits

        return logits, hidden


# ---- TRAINING ----
def train_model(train_data, embed_dim, hidden_dim, num_layers, lr, epochs, batch_size, seq_len, output_dir):
    """
    Train the CharRNN model on the provided dataset.
    """
    logging.info("Preparing dataset...")
    dataset = CharDataset(train_data, seq_len)

    # Calculate vocab_size from the dataset
    vocab_size = dataset.vocab_size

    logging.info("Initializing model...")
    model = CharRNN(vocab_size, embed_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logging.info("Starting training...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print("LOADED THE DATASET")
    model.train()
    print("CALLED MODEL TRAIN")
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(dataloader)
        for i, (x, y) in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, leave=False)):
            optimizer.zero_grad()
            # Mixed precision forward and backward pass
            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            # Show progress at every 10% of the iterations
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    logging.info("Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    torch.save(dataset, os.path.join(output_dir, "dataset.pth"))
    logging.info("Model and dataset saved.")

# ---- PREDICTION ---- old without error handling for unseen characters

# def predict_next_chars(model_path, input_texts, top_k=3):
#     logging.info("Loading model and dataset...")
#     dataset = torch.load(os.path.join(model_path, "dataset.pth"))
#     model = CharRNN(dataset.vocab_size, embed_dim=128, hidden_dim=256, num_layers=2)
#     model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
#     model.eval()
#
#     predictions = []
#     for text in input_texts:
#         encoded_input = [dataset.char_to_idx[ch] for ch in text if ch in dataset.char_to_idx]
#         x = torch.tensor(encoded_input).unsqueeze(0)
#         logits, _ = model(x)
#
#         # Get logits for the next character
#         top_k_ids = torch.topk(logits, k=top_k + 10).indices[0].tolist()  # Retrieve extra candidates
#         top_k_chars = [dataset.idx_to_char[i] for i in top_k_ids]
#
#         # Filter out spaces
#         top_k_chars = [char for char in top_k_chars if char != " "]
#
#         # Ensure exactly `top_k` predictions
#         top_k_chars = top_k_chars[:top_k]
#         predictions.append("".join(top_k_chars))
#
#     return predictions
# ---- PREDICTION ----
import time
import torch
import os

def predict_next_chars(model_path, input_texts, top_k=3):
    print(torch.get_num_threads())
    char_to_idx = {' ': 0, '!': 1, '"': 2, '#': 3, '$': 4, '%': 5, "'": 6, '(': 7, ')': 8, '*': 9, '+': 10, ',': 11, '-': 12, '.': 13, '/': 14, '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20, '6': 21, '7': 22, '8': 23, '9': 24, ':': 25, ';': 26, '?': 27, '@': 28, 'A': 29, 'B': 30, 'C': 31, 'D': 32, 'E': 33, 'F': 34, 'G': 35, 'H': 36, 'I': 37, 'J': 38, 'K': 39, 'L': 40, 'M': 41, 'N': 42, 'O': 43, 'P': 44, 'Q': 45, 'R': 46, 'S': 47, 'T': 48, 'U': 49, 'V': 50, 'W': 51, 'X': 52, 'Y': 53, 'Z': 54, '[': 55, ']': 56, '_': 57, 'a': 58, 'b': 59, 'c': 60, 'd': 61, 'e': 62, 'f': 63, 'g': 64, 'h': 65, 'i': 66, 'j': 67, 'k': 68, 'l': 69, 'm': 70, 'n': 71, 'o': 72, 'p': 73, 'q': 74, 'r': 75, 's': 76, 't': 77, 'u': 78, 'v': 79, 'w': 80, 'x': 81, 'y': 82, 'z': 83, '{': 84, '}': 85, '~': 86}
    idx_to_char = {0: ' ', 1: '!', 2: '"', 3: '#', 4: '$', 5: '%', 6: "'", 7: '(', 8: ')', 9: '*', 10: '+', 11: ',', 12: '-', 13: '.', 14: '/', 15: '0', 16: '1', 17: '2', 18: '3', 19: '4', 20: '5', 21: '6', 22: '7', 23: '8', 24: '9', 25: ':', 26: ';', 27: '?', 28: '@', 29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F', 35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W', 52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 85: '}', 86: '~'}
    logging.info("Loading model and dataset...")

    # Measure dataset loading time
    dataset_start_time = time.time()
    # dataset = torch.load(os.path.join(model_path, "dataset.pth"))
    dataset_end_time = time.time()
    print(f"Time to load dataset: {dataset_end_time - dataset_start_time:.2f} seconds")

    # Measure model initialization time
    model_init_start_time = time.time()
    vocab_size = len(char_to_idx.keys())
    model = CharRNN(vocab_size, embed_dim=64, hidden_dim=128, num_layers=2)
    model_init_end_time = time.time()
    print(f"Time to initialize model: {model_init_end_time - model_init_start_time:.2f} seconds")

    # Measure model weights loading time
    weights_start_time = time.time()
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
    model.eval()
    weights_end_time = time.time()
    print(f"Time to load model weights: {weights_end_time - weights_start_time:.2f} seconds")

    total_time = weights_end_time - dataset_start_time
    print(f"Total loading time: {total_time:.2f} seconds")
    # print(dataset.char_to_idx)
    # print(dataset.idx_to_char)
    predictions = []
    
    
    with torch.no_grad():
        for text in input_texts:
            try:
                # Process each character in the input text
                encoded_input = []
                for ch in text:
                    if ch == '’':
                        ch = "'"
                    if ch in char_to_idx:
                        encoded_input.append(char_to_idx[ch])  # Encode known character
                    else:
                        # chars = ['a', 'e', '.']
                        predictions.append("ae.")
                        # random_chars = random.sample(
                        #     [char for char in chars],
                        #     k=top_k,
                        # )
                        # predictions.append("".join(random_chars))
                        break

                if len(encoded_input) > 0 and len(encoded_input) == len(text):
                    x = torch.tensor(encoded_input).unsqueeze(0)
                    logits, _ = model(x)
                    top_k_ids = torch.topk(logits, k=top_k + 10).indices[0].tolist()
                    top_k_chars = [idx_to_char[i] for i in top_k_ids]
                    top_k_chars = [char for char in top_k_chars if char != " "]
                    top_k_chars = top_k_chars[:top_k]
                    predictions.append("".join(top_k_chars))

            except Exception as e:
                logging.error(f"Error during prediction for input '{text}': {e}")
                top_k_chars = random.sample(
                    [char for char in char_to_idx.keys() if char != " "], k=top_k
                )
                predictions.append("".join(top_k_chars))

    return predictions


# ---- MAIN ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "preprocess"], help="Mode: train or test or preprocess")
    parser.add_argument("--lang1", type=str, default="en", help="First language for OpenSubtitles dataset")
    parser.add_argument("--lang2", type=str, default="fr", help="Second language for OpenSubtitles dataset")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
    parser.add_argument("--train_data", type=str, help="Training data file path")
    parser.add_argument("--test_data", type=str, help="Test data file path")
    parser.add_argument("--test_output", type=str, help="Test output file path")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length for training")
    parser.add_argument("--remove_spaces", action="store_true", help="Remove spaces during preprocessing")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.train_data:
            raise ValueError("Training mode requires --train_data")
        with open(args.train_data, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # Preprocess the text
        train_data = preprocess_text(raw_text, remove_spaces=args.remove_spaces)
        train_model(
            train_data=train_data,
            embed_dim=64,          # Embedding size
            hidden_dim=128,         # Hidden layer size
            num_layers=2,           # Number of RNN layers
            lr=0.0020,               # Learning rate
            epochs=10,              # Training epochs
            batch_size=128,          # Batch size
            seq_len=args.seq_len,   # Sequence length
            output_dir=args.work_dir,
        )

    elif args.mode == "test":
        print("FIRST START TIME")
        start_time = time.time()  # Start timer
        
        if not args.test_data or not args.test_output:
            raise ValueError("Testing mode requires --test_data and --test_output")

        # Step 1: Read input file
        read_start_time = time.time()
        with open(args.test_data, "r", encoding="utf-8") as f:
            input_texts = [line.strip() for line in f.readlines()]
        read_end_time = time.time()
        print(f"Time to read input file: {read_end_time - read_start_time:.2f} seconds")

        # Step 2: Generate predictions
        prediction_start_time = time.time()
        predictions = predict_next_chars(args.work_dir, input_texts)
        prediction_end_time = time.time()
        print(f"Time to generate predictions: {prediction_end_time - prediction_start_time:.2f} seconds")

        # Step 3: Write output file
        write_start_time = time.time()
        with open(args.test_output, "w", encoding="utf-8") as f:
            for prediction in predictions:
                f.write(prediction + "\n")
        write_end_time = time.time()
        print(f"Time to write output file: {write_end_time - write_start_time:.2f} seconds")

        # Final total time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        
        logging.info(f"Predictions saved to {args.test_output}")

        
    elif args.mode == "preprocess":
        preprocess_opensubtitles(args.work_dir, args.lang1, args.lang2)

    

if __name__ == "__main__":
    main()
