import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import string
import random
from transformers import TextDataset
from datasets import load_dataset
import aiohttp

logging.basicConfig(level=logging.INFO)
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


def load_dataset_from_file(tokenizer, file_path, block_size=512):
    """
    Load the dataset for GPT-2 fine-tuning.
    """
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

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
    text = "".join(ch for ch in text if ch in string.printable)

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
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out[:, -1, :])  # Only the last time step
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

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
def predict_next_chars(model_path, input_texts, top_k=3):
    logging.info("Loading model and dataset...")
    dataset = torch.load(os.path.join(model_path, "dataset.pth"))
    model = CharRNN(dataset.vocab_size, embed_dim=128, hidden_dim=256, num_layers=2)
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
    model.eval()

    predictions = []
    for text in input_texts:
        try:
            # Process each character in the input text
            encoded_input = []
            for ch in text:
                if ch in dataset.char_to_idx:
                    encoded_input.append(dataset.char_to_idx[ch])  # Encode known character
                else:
                    # Generate random predictions for unknown character
                    logging.warning(f"Unknown character '{ch}' found. Generating random predictions.")
                    random_chars = random.sample(
                        [char for char in dataset.char_to_idx.keys() if char != " "],
                        k=top_k,
                    )
                    predictions.append("".join(random_chars))
                    break  # Stop further processing for this input

            # If all characters are known, use the model for prediction
            if len(encoded_input) > 0 and len(encoded_input) == len(text):
                x = torch.tensor(encoded_input).unsqueeze(0)
                logits, _ = model(x)

                # Get logits for the next character
                top_k_ids = torch.topk(logits, k=top_k + 10).indices[0].tolist()  # Retrieve extra candidates
                top_k_chars = [dataset.idx_to_char[i] for i in top_k_ids]

                # Filter out spaces
                top_k_chars = [char for char in top_k_chars if char != " "]

                # Ensure exactly `top_k` predictions
                top_k_chars = top_k_chars[:top_k]
                predictions.append("".join(top_k_chars))

        except Exception as e:
            # Handle errors and generate fallback predictions
            logging.error(f"Error during prediction for input '{text}': {e}")
            top_k_chars = random.sample(
                [char for char in dataset.char_to_idx.keys() if char != " "], k=top_k
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
            embed_dim=128,          # Embedding size
            hidden_dim=256,         # Hidden layer size
            num_layers=2,           # Number of RNN layers
            lr=0.001,               # Learning rate
            epochs=10,              # Training epochs
            batch_size=64,          # Batch size
            seq_len=args.seq_len,   # Sequence length
            output_dir=args.work_dir,
        )

    elif args.mode == "test":
        if not args.test_data or not args.test_output:
            raise ValueError("Testing mode requires --test_data and --test_output")
        with open(args.test_data, "r", encoding="utf-8") as f:
            input_texts = [line.strip() for line in f.readlines()]
        predictions = predict_next_chars(args.work_dir, input_texts)

        with open(args.test_output, "w", encoding="utf-8") as f:
            for prediction in predictions:
                f.write(prediction + "\n")
        logging.info(f"Predictions saved to {args.test_output}")
        
    elif args.mode == "preprocess":
        preprocess_opensubtitles(args.work_dir, args.lang1, args.lang2)

    

if __name__ == "__main__":
    main()
