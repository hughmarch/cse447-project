import os
import argparse
import logging
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling,  AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)


def preprocess_opensubtitles(work_dir, lang1="en", lang2="fr", split_ratio=0.9):
    """
    Fetch and preprocess the OpenSubtitles dataset, splitting it into train/test subsets.
    """
    logging.info(f"Loading OpenSubtitles dataset for language pair {lang1}-{lang2}...")
    #download_config = DownloadConfig(timeout=1800)  # Set timeout to 30 minutes
    dataset = load_dataset("open_subtitles", lang1=lang1, lang2=lang2, split="train")

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


# def train_model(train_file, output_dir):
#     """
#     Fine-tune GPT-2 on the preprocessed dataset.
#     """
#     logging.info("Loading tokenizer and model...")
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     model = GPT2LMHeadModel.from_pretrained("gpt2")
#
#     logging.info("Loading training dataset...")
#     subset_file = "work/check.txt"
#     # preprocess_to_subset(train_file, subset_file, subset_ratio=0.)
#     #
#     # # Then pass the subset file to load_dataset_from_file
#     # train_dataset = load_dataset_from_file(tokenizer, subset_file)
#     train_dataset = load_dataset_from_file(tokenizer, train_file)
#
#     logging.info("Setting up training...")
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=False
#     )
#
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=2,  # Adjust for M2 MacBook Air memory
#         save_steps=500,
#         save_total_limit=2,
#         logging_dir=f"{output_dir}/logs",
#         logging_steps=50,
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#     )
#
#     logging.info("Starting training...")
#     trainer.train()
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     logging.info(f"Model and tokenizer saved to {output_dir}")

def train_model(train_file, output_dir):
    """
    Fine-tune GPT-2 on the preprocessed dataset for character-level next-token prediction.
    """
    logging.info("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    logging.info("Loading training dataset...")
    # Optional: Reduce the dataset size for testing purposes
    # subset_file = "work/train_subset.txt"
    # preprocess_to_subset(train_file, subset_file, subset_ratio=0.1)  # Use 10% of the file
    train_dataset = load_dataset_from_file(tokenizer, train_file)

    logging.info("Setting up training...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Ensure causal language modeling (predict next token)
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,  # Overwrite any existing checkpoints
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    logging.info("Starting training...")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model and tokenizer saved to {output_dir}")


def predict_next_chars(model_name, input_texts, top_k=3):
    """
    Generate top-k next character predictions for input texts using the base model.
    """
    logging.info(f"Loading model '{model_name}' for prediction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    predictions = []
    for text in input_texts:
        # Encode the input text and pass it to the model
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)

        # Get logits for the last token
        logits = outputs.logits[0, -1, :]

        # Get top-k predictions
        top_k_ids = torch.topk(logits, k=top_k).indices.tolist()
        top_k_chars = [tokenizer.decode([token_id]) for token_id in top_k_ids]

        # Append predictions for the current input
        predictions.append("".join(top_k_chars))

    return predictions

# def predict_next_chars(model_dir, input_texts, top_k=3):
#     """
#     Generate top-k next character predictions for input texts with a custom prompt.
#
#     Parameters:
#     - model_dir (str): Path to the model directory.
#     - input_texts (list): List of input text sequences.
#     - custom_prompt (str): Custom text to append to each input sequence.
#     - top_k (int): Number of top predictions to return.
#
#     Returns:
#     - list: List of predictions for each input text.
#     """
#     logging.info("Loading model for prediction...")
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     model = GPT2LMHeadModel.from_pretrained("gpt2")
#     model.eval()
#
#     predictions = []
#     for text in input_texts:
#         # Combine the input text with the custom prompt
#         combined_input = (
#             f"You are completing an incomplete word or phrase. Predict the next character in the sequence based on context. "
#             f"Provide exactly three possible characters that could come next. "
#             f"The word or phrase to complete is: '{text}'."
#         )
#
#         # Encode the combined input
#         inputs = tokenizer(combined_input, return_tensors="pt")
#         outputs = model(**inputs)
#
#         # Get logits for the last token
#         logits = outputs.logits[0, -1, :]
#
#         # Get top-k predictions
#         top_k_ids = torch.topk(logits, k=top_k).indices.tolist()
#
#         # Decode each predicted token ID into single characters
#         top_k_chars = []
#         for token_id in top_k_ids:
#             decoded_char = tokenizer.decode([token_id])
#
#             # Ensure the decoded output is a single character
#             if len(decoded_char) == 1 and decoded_char.isprintable():
#                 top_k_chars.append(decoded_char)
#             else:
#                 # If the decoded token is not a valid single character, append a placeholder
#                 top_k_chars.append(" ")
#
#         # Combine exactly three characters into the prediction
#         while len(top_k_chars) < top_k:
#             top_k_chars.append(" ")  # Pad with spaces if fewer than 3 characters
#         predictions.append("".join(top_k_chars[:top_k]))
#
#     return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "train", "test"], help="Mode: preprocess, train, or test")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory for checkpoints and data")
    parser.add_argument("--lang1", type=str, default="en", help="First language for OpenSubtitles dataset")
    parser.add_argument("--lang2", type=str, default="fr", help="Second language for OpenSubtitles dataset")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to testing data")
    parser.add_argument("--test_output", type=str, help="Path to save test predictions")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125M", help="Model name or path")
    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess_opensubtitles(args.work_dir, args.lang1, args.lang2)

    elif args.mode == "train":
        if not args.train_data:
            raise ValueError("Training mode requires --train_data")
        train_model(args.train_data, args.work_dir)


    elif args.mode == "test":

        if not args.test_data or not args.test_output:
            raise ValueError("Testing mode requires --test_data and --test_output")

        with open(args.test_data, "r", encoding="utf-8") as f:

            input_texts = [line.strip() for line in f.readlines()]

        predictions = predict_next_chars(args.model_name, input_texts)

        with open(args.test_output, "w", encoding="utf-8") as f:

            for prediction in predictions:
                f.write(prediction + "\n")

        logging.info(f"Predictions saved to {args.test_output}")


if __name__ == "__main__":
    main()