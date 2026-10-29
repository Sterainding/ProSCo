# run_ner.py (纯 PyTorch 训练，彻底告别 TrainingArguments 报错)
import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["Rest15", "Rest16", "Laptop", "Restaurant"])
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output_dir", default="../checkpoints/ner")
    return parser.parse_args()


def read_bio_file(file_path):
    words, tags = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        word_seq, tag_seq = [], []
        for line in f:
            line = line.strip()
            if line == '':
                if word_seq:
                    words.append(word_seq)
                    tags.append(tag_seq)
                    word_seq, tag_seq = [], []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    word_seq.append(parts[0])
                    tag_seq.append(parts[1])
        if word_seq:
            words.append(word_seq)
            tags.append(tag_seq)
    return words, tags


def main():
    print("============== 注意：运行的是纯 PyTorch 新版脚本 ==============")
    args = parse_args()
    label_list = ['O', 'B-ASP', 'I-ASP', 'B-OPI', 'I-OPI']
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).to(device)

    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(
            examples["words"], truncation=True, padding=False,
            is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_words, train_tags = read_bio_file(f"../data/ner/{args.dataset}/train.txt")
    dev_words, dev_tags = read_bio_file(f"../data/ner/{args.dataset}/dev.txt")

    train_dataset = Dataset.from_dict({"words": train_words, "tags": train_tags}).map(tokenize_and_align, batched=True,
                                                                                      remove_columns=["words", "tags"])
    dev_dataset = Dataset.from_dict({"words": dev_words, "tags": dev_tags}).map(tokenize_and_align, batched=True,
                                                                                remove_columns=["words", "tags"])

    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Eval Epoch {epoch + 1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()

        print(
            f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Eval Loss: {eval_loss / len(dev_loader):.4f}")

    save_path = f"{args.output_dir}/{args.dataset}/best_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[SUCCESS] Model saved to {save_path}")


if __name__ == "__main__":
    main()
