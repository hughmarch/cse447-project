import os
import argparse
import logging
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling,  T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

logging.basicConfig(level=logging.INFO)
#
#
# def preprocess_opensubtitles(work_dir, lang1="en", lang2="fr", split_ratio=0.9):
#     """
#     Fetch and preprocess the OpenSubtitles dataset, splitting it into train/test subsets.
#     """
#     logging.info(f"Loading OpenSubtitles dataset for language pair {lang1}-{lang2}...")
#     #download_config = DownloadConfig(timeout=1800)  # Set timeout to 30 minutes
#     dataset = load_dataset("open_subtitles", lang1=lang1, lang2=lang2, split="train")
#
#     logging.info("Processing dataset...")
#     # Extract translation text for one language (e.g., lang1)
#     all_texts = [example["translation"]["en"] for example in dataset]
#     all_texts = [text.strip() for text in all_texts if text.strip()]  # Clean empty lines
#
#     # Shuffle and split dataset
#     split_idx = int(len(all_texts) * split_ratio)
#     train_texts = all_texts[:split_idx]
#     test_texts = all_texts[split_idx:]
#
#     # Save to files
#     train_path = os.path.join(work_dir, "train.txt")
#     test_path = os.path.join(work_dir, "test.txt")
#     with open(train_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(train_texts))
#     with open(test_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(test_texts))
#
#     logging.info(f"Train dataset saved to {train_path}")
#     logging.info(f"Test dataset saved to {test_path}")
#
#     return train_path, test_path

#
# def create_character_level_dataset(file_path, tokenizer, max_length=512):
#     """
#     Create a character-level dataset for T5 fine-tuning.
#     """
#     logging.info(f"Loading dataset from {file_path}...")
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = [line.strip() for line in f.readlines() if line.strip()]
#
#     inputs, labels = [], []
#     for line in lines:
#         for i in range(1, len(line)):  # Create character-level examples
#             inputs.append(line[:i])
#             labels.append(line[i:i + 3])  # Predict next three characters
#
#     logging.info(f"Tokenizing {len(inputs)} examples...")
#     tokenized_data = tokenizer(
#         inputs,
#         max_length=max_length,
#         truncation=True,
#         padding="max_length",
#         return_tensors="pt"
#     )
#     tokenized_labels = tokenizer(
#         labels,
#         max_length=3,
#         truncation=True,
#         padding="max_length",
#         return_tensors="pt"
#     )
#
#     # Convert to Dataset
#     dataset = Dataset.from_dict({
#         "input_ids": tokenized_data["input_ids"],
#         "attention_mask": tokenized_data["attention_mask"],
#         "labels": tokenized_labels["input_ids"],
#     })
#     return dataset


# def train_model(train_file, output_dir):
#     """
#     Fine-tune T5 on the preprocessed dataset for character-level next-token prediction.
#     """
#     logging.info("Loading tokenizer and model...")
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")
#
#     logging.info("Creating character-level dataset...")
#     train_dataset = create_character_level_dataset(train_file, tokenizer)
#
#     logging.info("Setting up training...")
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=8,
#         save_steps=500,
#         save_total_limit=2,
#         logging_dir=f"{output_dir}/logs",
#         logging_steps=50,
#         fp16=False  # Mixed precision is not supported on MPS
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#     )
#
#     logging.info("Starting training...")
#     trainer.train()
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     logging.info(f"Model and tokenizer saved to {output_dir}")

# def predict_next_chars(model_dir, input_texts, top_k=3):
#     """
#     Generate top-k next character predictions for input texts, ensuring proper formatting.
#     """
#     logging.info("Loading model for prediction...")
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")
#     model.eval()
#
#     predictions = []
#     for text in input_texts:
#         # Tokenize the input text
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding="max_length",
#             max_length=512,
#         )
#
#         # Get model logits for the next token
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=2,  # Generate only 1 token (the next character)
#             num_beams=top_k,
#             num_return_sequences=top_k,
#         )
#
#         # Decode predictions into characters
#         decoded = [
#             tokenizer.decode(output, skip_special_tokens=True).strip()
#             for output in outputs
#         ]
#
#         # Ensure predictions are exactly three characters
#         top_k_chars = [char[:1] for char in decoded[:top_k]]  # Take only the first character
#         while len(top_k_chars) < top_k:
#             top_k_chars.append(" ")  # Pad with spaces if fewer than 3 characters
#
#         predictions.append("".join(top_k_chars))
#
#     return predictions

def predict_next_chars(model_dir, input_texts, top_k=3):
    """
    Generate top-k next character predictions for input texts, ensuring proper formatting.
    """
    logging.info("Loading model for prediction...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.eval()

    predictions = []
    for text in input_texts:
        # Tokenize the input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        # Generate outputs
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + 3,  # Generate 3 additional tokens
            num_beams=top_k,
            num_return_sequences=top_k,
        )

        # Decode predictions into characters
        decoded = [
            tokenizer.decode(output, skip_special_tokens=True).strip()
            for output in outputs
        ]

        # Extract the last three characters for each prediction
        top_k_chars = [decoded_output[-3:] for decoded_output in decoded[:top_k]]

        # Ensure predictions are exactly three characters
        formatted_predictions = [
            pred if len(pred) == 3 else pred.ljust(3, " ")
            for pred in top_k_chars
        ]

        # Combine predictions for the current input
        predictions.append("".join(formatted_predictions))

    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "train", "test"], help="Mode: preprocess, train, or test")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory for checkpoints and data")
    parser.add_argument("--lang1", type=str, default="en", help="Language for OpenSubtitles dataset")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to testing data")
    parser.add_argument("--test_output", type=str, help="Path to save test predictions")
    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess_opensubtitles(args.work_dir, args.lang1)

    elif args.mode == "train":
        if not args.train_data:
            raise ValueError("Training mode requires --train_data")
        train_model(args.train_data, args.work_dir)

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


if __name__ == "__main__":
    main()

