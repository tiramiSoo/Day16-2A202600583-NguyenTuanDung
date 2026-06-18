"""
benchmark.py
Credit Card Fraud Detection — LightGBM CPU Benchmark

Reproduces all metrics required by step 7.6 / 7.8 of the lab:
  - Data load time
  - Training time
  - Best iteration (early stopping)
  - AUC-ROC, Accuracy, F1, Precision, Recall
  - Inference latency (single row)
  - Inference throughput (1000 rows)

Outputs:
  - Printed summary to terminal (for the required screenshot)
  - benchmark_result.json (for submission)
"""

import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = "creditcard.csv"
TARGET_COL = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str):
    start = time.perf_counter()
    df = pd.read_csv(path)
    elapsed = time.perf_counter() - start
    return df, elapsed


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    start = time.perf_counter()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=0),
        ],
    )
    elapsed = time.perf_counter() - start

    best_iteration = model.best_iteration
    return model, elapsed, best_iteration


def evaluate(model, X_test, y_test):
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    return metrics


def measure_inference_latency(model, X_test):
    # Single-row latency: average over multiple repeats for stability
    sample = X_test.iloc[[0]]
    n_repeats = 100
    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(sample, num_iteration=model.best_iteration)
    elapsed = time.perf_counter() - start
    avg_latency_ms = (elapsed / n_repeats) * 1000
    return avg_latency_ms


def measure_inference_throughput(model, X_test):
    n_rows = min(1000, len(X_test))
    sample = X_test.iloc[:n_rows]
    start = time.perf_counter()
    model.predict(sample, num_iteration=model.best_iteration)
    elapsed = time.perf_counter() - start
    throughput = n_rows / elapsed  # rows per second
    return throughput, elapsed


def main():
    print("=" * 60)
    print("Credit Card Fraud Detection — LightGBM CPU Benchmark")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df, load_time = load_data(DATA_PATH)
    print(f"  Loaded {len(df):,} rows in {load_time:.4f} seconds")

    print("\n[2/5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    print("\n[3/5] Training LightGBM model...")
    model, train_time, best_iteration = train_model(X_train, y_train, X_test, y_test)
    print(f"  Training time: {train_time:.4f} seconds")
    print(f"  Best iteration: {best_iteration}")

    print("\n[4/5] Evaluating model...")
    metrics = evaluate(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\n[5/5] Measuring inference performance...")
    latency_ms = measure_inference_latency(model, X_test)
    throughput, throughput_elapsed = measure_inference_throughput(model, X_test)
    print(f"  Single-row latency: {latency_ms:.4f} ms")
    print(f"  Throughput (1000 rows): {throughput:.2f} rows/sec "
          f"({throughput_elapsed:.4f}s total)")

    results = {
        "load_time_seconds": round(load_time, 4),
        "training_time_seconds": round(train_time, 4),
        "best_iteration": best_iteration,
        "auc_roc": round(metrics["auc_roc"], 6),
        "accuracy": round(metrics["accuracy"], 6),
        "f1_score": round(metrics["f1_score"], 6),
        "precision": round(metrics["precision"], 6),
        "recall": round(metrics["recall"], 6),
        "inference_latency_single_row_ms": round(latency_ms, 4),
        "inference_throughput_1000_rows_per_sec": round(throughput, 2),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
    }

    with open("benchmark_result.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Results saved to benchmark_result.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
