from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report


LABEL_MAP: dict[int, str] = {0: "Negative", 1: "Neutral", 2: "Positive"}
REVERSE_LABEL_MAP: dict[str, int] = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS: int = 3
BASE_MODEL_NAME: str = "distilbert-base-uncased"

SentimentLabel = Literal["Negative", "Neutral", "Positive"]


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = BASE_MODEL_NAME
    num_labels: int = NUM_LABELS
    dropout_prob: float = 0.3
    max_seq_len: int = 64
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    use_fp16: bool = True
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


class SentimentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer: AutoTokenizer,
        max_len: int,
    ) -> None:
        if labels is not None and len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must have the same length."
            )
        self._texts: list[str] = texts
        self._labels: list[int] | None = labels
        self._tokenizer = tokenizer
        self._max_len: int = max_len

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding: dict = self._tokenizer(
            self._texts[idx],
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item: dict[str, torch.Tensor] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        # DistilBERT doesn't use token_type_ids, BERT does
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        if self._labels is not None:
            item["label"] = torch.tensor(self._labels[idx], dtype=torch.long)
        return item


class BertSentimentClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config: ModelConfig = config
        self.encoder = AutoModel.from_pretrained(config.base_model)
        self._uses_token_type_ids: bool = (
            "token_type_ids" in self.encoder.config.to_dict()
            or hasattr(self.encoder.config, "type_vocab_size")
        )
        self.dropout: nn.Dropout = nn.Dropout(p=config.dropout_prob)
        self.classifier: nn.Linear = nn.Linear(
            self.encoder.config.hidden_size,
            config.num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self._uses_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        # DistilBERT returns last_hidden_state, BERT returns pooler_output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled: torch.Tensor = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        dropped: torch.Tensor = self.dropout(pooled)
        logits: torch.Tensor = self.classifier(dropped)
        return logits


@dataclass
class SentimentPipeline:
    model: BertSentimentClassifier
    tokenizer: AutoTokenizer
    config: ModelConfig


@dataclass(frozen=True)
class SentimentPrediction:
    text: str
    label: SentimentLabel
    label_id: int
    confidence: float
    probabilities: dict[str, float]


def build_pipeline(config: ModelConfig) -> SentimentPipeline:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model: BertSentimentClassifier = BertSentimentClassifier(config)
    model.to(config.device)
    return SentimentPipeline(model=model, tokenizer=tokenizer, config=config)


def save_pipeline(pipeline: SentimentPipeline, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        pipeline.model.state_dict(),
        os.path.join(save_dir, "model_weights.pt"),
    )

    pipeline.tokenizer.save_pretrained(save_dir)

    config_path: str = os.path.join(save_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "base_model": pipeline.config.base_model,
                "num_labels": pipeline.config.num_labels,
                "dropout_prob": pipeline.config.dropout_prob,
                "max_seq_len": pipeline.config.max_seq_len,
                "batch_size": pipeline.config.batch_size,
                "learning_rate": pipeline.config.learning_rate,
                "num_epochs": pipeline.config.num_epochs,
                "warmup_ratio": pipeline.config.warmup_ratio,
                "weight_decay": pipeline.config.weight_decay,
            },
            fh,
            indent=2,
        )
    print(f"[save_pipeline] Checkpoint saved → {save_dir}")


def load_pipeline(save_dir: str) -> SentimentPipeline:
    config_path: str = os.path.join(save_dir, "model_config.json")
    weights_path: str = os.path.join(save_dir, "model_weights.pt")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    config: ModelConfig = ModelConfig(
        base_model=raw["base_model"],
        num_labels=raw["num_labels"],
        dropout_prob=raw["dropout_prob"],
        max_seq_len=raw["max_seq_len"],
        batch_size=raw["batch_size"],
        learning_rate=raw["learning_rate"],
        num_epochs=raw["num_epochs"],
        warmup_ratio=raw["warmup_ratio"],
        weight_decay=raw["weight_decay"],
        device=device,
    )

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(save_dir)
    model: BertSentimentClassifier = BertSentimentClassifier(config)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"[load_pipeline] Loaded checkpoint from {save_dir} (device={device})")
    return SentimentPipeline(model=model, tokenizer=tokenizer, config=config)


def _encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    max_len: int,
    device: str,
) -> dict[str, torch.Tensor]:
    encoding: dict = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoding.items()}


def predict(pipeline: SentimentPipeline, text: str) -> SentimentPrediction:
    if not text.strip():
        raise ValueError("predict() received an empty string.")
    results: list[SentimentPrediction] = predict_batch(pipeline, [text])
    return results[0]


def predict_batch(
    pipeline: SentimentPipeline,
    texts: list[str],
) -> list[SentimentPrediction]:
    if not texts:
        raise ValueError("predict_batch() received an empty list.")

    pipeline.model.eval()
    config: ModelConfig = pipeline.config
    all_predictions: list[SentimentPrediction] = []

    for batch_start in range(0, len(texts), config.batch_size):
        batch_texts: list[str] = texts[batch_start : batch_start + config.batch_size]

        encoded: dict[str, torch.Tensor] = _encode_texts(
            batch_texts,
            pipeline.tokenizer,
            config.max_seq_len,
            config.device,
        )

        with torch.no_grad():
            logits: torch.Tensor = pipeline.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids"),
            )

        probs_tensor: torch.Tensor = torch.softmax(logits, dim=-1)
        probs_np: np.ndarray = probs_tensor.cpu().numpy()
        predicted_ids: np.ndarray = np.argmax(probs_np, axis=-1)

        for i, text in enumerate(batch_texts):
            label_id: int = int(predicted_ids[i])
            label: SentimentLabel = LABEL_MAP[label_id]  # type: ignore[assignment]
            confidence: float = float(probs_np[i, label_id])
            probabilities: dict[str, float] = {
                LABEL_MAP[j]: float(probs_np[i, j]) for j in range(NUM_LABELS)
            }
            all_predictions.append(
                SentimentPrediction(
                    text=text,
                    label=label,
                    label_id=label_id,
                    confidence=confidence,
                    probabilities=probabilities,
                )
            )

    return all_predictions


def train(
    pipeline: SentimentPipeline,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    save_dir: str,
) -> dict[str, list[float]]:
    invalid: list[int] = [l for l in train_labels + val_labels if l not in (0, 1, 2)]
    if invalid:
        raise ValueError(
            f"Label values must be 0, 1, or 2. Found invalid: {set(invalid)}"
        )

    config: ModelConfig = pipeline.config

    train_dataset = SentimentDataset(
        train_texts, train_labels, pipeline.tokenizer, config.max_seq_len
    )
    val_dataset = SentimentDataset(
        val_texts, val_labels, pipeline.tokenizer, config.max_seq_len
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(
        pipeline.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps: int = len(train_loader) * config.num_epochs
    warmup_steps: int = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp: bool = config.use_fp16 and config.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc: float = 0.0

    for epoch in range(1, config.num_epochs + 1):
        pipeline.model.train()
        total_train_loss: float = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits: torch.Tensor = pipeline.model(
                    input_ids=batch["input_ids"].to(config.device),
                    attention_mask=batch["attention_mask"].to(config.device),
                    token_type_ids=batch.get("token_type_ids", torch.tensor([])).to(config.device)
                    if "token_type_ids" in batch else None,
                )
                loss: torch.Tensor = loss_fn(logits, batch["label"].to(config.device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss: float = total_train_loss / len(train_loader)

        pipeline.model.eval()
        total_val_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = pipeline.model(
                        input_ids=batch["input_ids"].to(config.device),
                        attention_mask=batch["attention_mask"].to(config.device),
                        token_type_ids=batch.get("token_type_ids", torch.tensor([])).to(config.device)
                        if "token_type_ids" in batch else None,
                    )
                loss = loss_fn(logits, batch["label"].to(config.device))
                total_val_loss += loss.item()

                preds: torch.Tensor = torch.argmax(logits, dim=-1)
                correct += (preds == batch["label"].to(config.device)).sum().item()
                total += batch["label"].size(0)

        avg_val_loss: float = total_val_loss / len(val_loader)
        val_acc: float = correct / total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_pipeline(pipeline, save_dir)
            print(f"  → New best model saved (val_acc={val_acc:.4f})")

    return history


def evaluate(
    pipeline: SentimentPipeline,
    texts: list[str],
    true_labels: list[int],
) -> str:
    if len(texts) != len(true_labels):
        raise ValueError(
            f"texts ({len(texts)}) and true_labels ({len(true_labels)}) must have the same length."
        )

    predictions: list[SentimentPrediction] = predict_batch(pipeline, texts)
    pred_ids: list[int] = [p.label_id for p in predictions]

    report: str = classification_report(
        true_labels,
        pred_ids,
        target_names=["Negative", "Neutral", "Positive"],
    )
    print(report)
    return report
