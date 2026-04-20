"""
finetune_qa.py

Fine-tunes InLegalBERT on the IndicLegalQA dataset for extractive QA.
Based on Paper 3 (Furniturewala et al., AILA 2021): joint BERT + TF-IDF
features improve legal sentence relevance classification.

Dataset: https://www.kaggle.com/datasets/kmldas/indiclegalqa-dataset
  10,000 QA pairs from 1,256 Indian Supreme Court judgments.
  Columns: question, answer, case_name, date

Task framed as: given (question, passage), predict if passage contains answer.
This trains the model to score retrieved chunks for relevance — used in
the re-ranking step of the RAG pipeline.

Download dataset first:
    kaggle datasets download -d kmldas/indiclegalqa-dataset
    unzip indiclegalqa-dataset.zip -d data/indiclegalqa/

Run:
    python training/finetune_qa.py
"""

import sys
import json
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "indiclegalqa"
MODEL_DIR  = BASE_DIR / "data" / "models" / "inlegalbert-qa"
MODEL_NAME = "law-ai/InLegalBERT"
MAX_LENGTH = 512
EPOCHS     = 3
BATCH_SIZE = 16
LR         = 2e-5


# ── Dataset ────────────────────────────────────────────────────────────────────

class QARelevanceDataset(Dataset):
    """
    Converts QA pairs into binary relevance classification:
      - Positive: (question, correct answer passage)  → label 1
      - Negative: (question, random other answer)     → label 0

    This teaches InLegalBERT to score retrieved chunks for relevance,
    which is used in the re-ranking stage (Paper 3 approach).
    """
    def __init__(self, df: pd.DataFrame, tokenizer, neg_ratio: int = 1):
        self.tokenizer = tokenizer
        self.samples   = []

        answers = df["answer"].tolist()
        for _, row in df.iterrows():
            question = str(row["question"])
            pos_ans  = str(row["answer"])

            # Positive example
            self.samples.append((question, pos_ans, 1))

            # Negative examples (random answers from other rows)
            for _ in range(neg_ratio):
                neg_ans = random.choice(answers)
                while neg_ans == pos_ans:
                    neg_ans = random.choice(answers)
                self.samples.append((question, neg_ans, 0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, passage, label = self.samples[idx]
        enc = self.tokenizer(
            question, passage,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(MAX_LENGTH, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on: {device}")

    # Load data
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV found in {DATA_DIR}. Download the dataset first:")
        print("  kaggle datasets download -d kmldas/indiclegalqa-dataset")
        print(f"  unzip indiclegalqa-dataset.zip -d {DATA_DIR}")
        return

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"Loaded {len(df)} QA pairs from {[f.name for f in csv_files]}")

    # Train / val split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split      = int(0.9 * len(df))
    train_df   = df[:split]
    val_df     = df[split:]

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds   = QARelevanceDataset(train_df, tokenizer)
    val_ds     = QARelevanceDataset(val_df, tokenizer, neg_ratio=1)

    train_dl   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl     = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Model: InLegalBERT + classification head (binary: relevant / not relevant)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch} train"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dl)

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch} val"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels         = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                preds   = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  → Saved best model (val_acc={val_acc:.4f}) to {MODEL_DIR}")

    print(f"\nFine-tuning complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    train()
