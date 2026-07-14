from __future__ import annotations

import os
import pickle
import time
from datasets import load_dataset, DatasetDict

from preprocessing import preprocess, PreprocessingResult
from model_pipeline import (
    ModelConfig,
    SentimentPipeline,
    build_pipeline,
    train,
    evaluate,
)

CHECKPOINT_DIR: str = "checkpoints/best_model"
PREPROC_CACHE_FILE: str = "checkpoints/preprocessed_cache.pkl"


def load_tweet_eval() -> DatasetDict:
    print("[1/4] Downloading tweet_eval/sentiment dataset…")
    try:
        ds: DatasetDict = load_dataset("cardiffnlp/tweet_eval", "sentiment")
        print(
            f"      Train: {len(ds['train'])} | "
            f"Validation: {len(ds['validation'])} | "
            f"Test: {len(ds['test'])}"
        )
        return ds
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset: {exc}") from exc


def preprocess_split(texts: list[str], split_name: str) -> list[str]:
    cleaned: list[str] = []
    total: int = len(texts)
    start: float = time.perf_counter()

    for i, text in enumerate(texts):
        result: PreprocessingResult = preprocess(
            text,
            remove_stops=True,
            lemmatize=True,
            translate=False,  # tweet_eval is English only
        )
        cleaned.append(result.cleaned_text)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            elapsed: float = time.perf_counter() - start
            rate: float = (i + 1) / elapsed
            eta: float = (total - i - 1) / rate
            print(f"      [{split_name}] {i + 1}/{total}  ({rate:.0f} texts/sec  ETA {eta:.0f}s)")

    return cleaned


def load_preprocessed_cache() -> tuple[list[str], list[str], list[str]] | None:
    if not os.path.isfile(PREPROC_CACHE_FILE):
        return None
    with open(PREPROC_CACHE_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"      Loaded preprocessed cache from {PREPROC_CACHE_FILE}")
    return data["train"], data["val"], data["test"]


def save_preprocessed_cache(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
) -> None:
    os.makedirs(os.path.dirname(PREPROC_CACHE_FILE), exist_ok=True)
    with open(PREPROC_CACHE_FILE, "wb") as f:
        pickle.dump({"train": train_texts, "val": val_texts, "test": test_texts}, f)
    print(f"      Preprocessing cache saved → {PREPROC_CACHE_FILE}")


def run_training() -> None:
    ds: DatasetDict = load_tweet_eval()

    raw_train: list[str] = list(ds["train"]["text"])
    raw_val: list[str]   = list(ds["validation"]["text"])
    raw_test: list[str]  = list(ds["test"]["text"])

    train_labels: list[int] = list(ds["train"]["label"])
    val_labels: list[int]   = list(ds["validation"]["label"])
    test_labels: list[int]  = list(ds["test"]["label"])

    print("\n[2/4] Preprocessing splits…")
    cached = load_preprocessed_cache()
    if cached is not None:
        train_texts, val_texts, test_texts = cached
        print("      Skipped preprocessing (loaded from cache).")
    else:
        print("      No cache found — running preprocessing (one-time, ~4 min)…")
        train_texts = preprocess_split(raw_train, "train")
        val_texts   = preprocess_split(raw_val,   "validation")
        test_texts  = preprocess_split(raw_test,  "test")
        save_preprocessed_cache(train_texts, val_texts, test_texts)

    empty_train: int = sum(1 for t in train_texts if not t.strip())
    if empty_train > 0:
        print(f"      Warning: {empty_train} training texts became empty after preprocessing.")

    print("\n[3/4] Building DistilBERT classifier…")
    config: ModelConfig = ModelConfig(
        base_model="distilbert-base-uncased",
        num_epochs=4,
        batch_size=16,
        learning_rate=2e-5,
        max_seq_len=64,
        use_fp16=True,
    )
    pipeline: SentimentPipeline = build_pipeline(config)
    print(f"      Device: {config.device}")
    print(f"      Epochs: {config.num_epochs}  |  Batch size: {config.batch_size}")
    print(f"      Checkpoint will be saved to: {CHECKPOINT_DIR}")

    print("\n[4/4] Fine-tuning… (this takes ~10 min on GPU, ~2–3 hrs on CPU)")
    history: dict[str, list[float]] = train(
        pipeline=pipeline,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        save_dir=CHECKPOINT_DIR,
    )

    print("\n──── Test Set Evaluation ────")
    evaluate(pipeline, test_texts, test_labels)

    best_epoch: int = int(
        max(range(len(history["val_acc"])), key=lambda i: history["val_acc"][i]) + 1
    )
    best_acc: float = max(history["val_acc"])

    print("\n══════════════════════════════════════════")
    print("  Training complete!")
    print(f"  Best validation accuracy : {best_acc:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint saved to      : {CHECKPOINT_DIR}/")
    print("  Run the app with         : streamlit run app.py")
    print("══════════════════════════════════════════")


if __name__ == "__main__":
    run_training()
