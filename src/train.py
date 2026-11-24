import os
import time
import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, CHECKPOINTS_DIR, HIDDEN_DIM, NUM_GRU_LAYERS,
    DROPOUT, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, HUBER_DELTA, SEQ_OUT
)
from src.model import WindSTGNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Computes overall, 3-hour, and 6-hour MAE and RMSE in real PM2.5 units.
    pred: (Samples, N, 6)
    target: (Samples, N, 6)
    """
    err = pred - target
    mae = torch.mean(torch.abs(err)).item()
    rmse = torch.sqrt(torch.mean(err ** 2)).item()

    # Step-specific metrics (0-indexed: 2 is +3h, 5 is +6h)
    mae_1h = torch.mean(torch.abs(err[:, :, 0])).item()
    rmse_1h = torch.sqrt(torch.mean(err[:, :, 0] ** 2)).item()

    mae_3h = torch.mean(torch.abs(err[:, :, 2])).item()
    rmse_3h = torch.sqrt(torch.mean(err[:, :, 2] ** 2)).item()

    mae_6h = torch.mean(torch.abs(err[:, :, 5])).item()
    rmse_6h = torch.sqrt(torch.mean(err[:, :, 5] ** 2)).item()

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_1h": mae_1h,
        "rmse_1h": rmse_1h,
        "mae_3h": mae_3h,
        "rmse_3h": rmse_3h,
        "mae_6h": mae_6h,
        "rmse_6h": rmse_6h,
    }

def evaluate_persistence_baseline(
    X_val_raw_pm25: torch.Tensor,
    Y_val: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluates persistence baseline: pred = last observed PM2.5 (at time T-1).
    X_val_raw_pm25: (Samples, N) last observed PM2.5
    Y_val: (Samples, N, 6)
    """
    # Expand last observed value across all 6 prediction steps
    pred_pers = X_val_raw_pm25.unsqueeze(-1).expand_as(Y_val)
    return compute_metrics(pred_pers, Y_val)

def train_model():
    device = get_device()
    logging.info(f"Target acceleration device: {device}")

    dataset_path = DATA_DIR / "dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found. Run ingest_history.py first.")

    logging.info(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path)

    X_train = dataset["X_train"]          # (Samples, 24, N, F)
    Adj_train = dataset["Adj_train"]      # (Samples, 24, N, N)
    Y_train = dataset["Y_train"]          # (Samples, N, 6) - real PM2.5 values

    X_val = dataset["X_val"]
    Adj_val = dataset["Adj_val"]
    Y_val = dataset["Y_val"]

    X_test = dataset["X_test"]
    Adj_test = dataset["Adj_test"]
    Y_test = dataset["Y_test"]

    pm25_mean = dataset["pm25_mean"]
    pm25_std = dataset["pm25_std"]

    B_samples, T_in, N_nodes, F_in = X_train.shape
    logging.info(f"Dataset stats: N={N_nodes}, T_in={T_in}, F_in={F_in}, SEQ_OUT={SEQ_OUT}")
    logging.info(f"PM2.5 Normalization: Mean={pm25_mean:.2f}, Std={pm25_std:.2f}")

    # Compute persistence baseline on Validation set:
    # Unscale the last observed PM2.5 at t = T-1 (index 0 is PM2.5 feature)
    last_val_pm25 = X_val[:, -1, :, 0] * pm25_std + pm25_mean
    pers_val_metrics = evaluate_persistence_baseline(last_val_pm25, Y_val)

    # Persistence baseline on Test set:
    last_test_pm25 = X_test[:, -1, :, 0] * pm25_std + pm25_mean
    pers_test_metrics = evaluate_persistence_baseline(last_test_pm25, Y_test)

    logging.info("================ PERSISTENCE BASELINE ================")
    logging.info(f"Val  Baseline -> MAE: {pers_val_metrics['mae']:.2f}, RMSE: {pers_val_metrics['rmse']:.2f} | 3h MAE: {pers_val_metrics['mae_3h']:.2f}, 6h MAE: {pers_val_metrics['mae_6h']:.2f} | 3h RMSE: {pers_val_metrics['rmse_3h']:.2f}, 6h RMSE: {pers_val_metrics['rmse_6h']:.2f}")
    logging.info(f"Test Baseline -> MAE: {pers_test_metrics['mae']:.2f}, RMSE: {pers_test_metrics['rmse']:.2f} | 3h MAE: {pers_test_metrics['mae_3h']:.2f}, 6h MAE: {pers_test_metrics['mae_6h']:.2f} | 3h RMSE: {pers_test_metrics['rmse_3h']:.2f}, 6h RMSE: {pers_test_metrics['rmse_6h']:.2f}")
    logging.info("======================================================")

    # Prepare DataLoaders
    train_dataset = TensorDataset(X_train, Adj_train, Y_train)
    val_dataset = TensorDataset(X_val, Adj_val, Y_val)
    test_dataset = TensorDataset(X_test, Adj_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = WindSTGNN(
        in_dim=F_in,
        hidden_dim=HIDDEN_DIM,
        seq_out=SEQ_OUT,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    best_val_rmse = float("inf")
    best_checkpoint_path = CHECKPOINTS_DIR / "best_wind_stgnn.pt"

    start_time = time.time()
    logging.info("Beginning WindSTGNN training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for b_x, b_adj, b_y in train_loader:
            b_x = b_x.to(device)
            b_adj = b_adj.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            # Predictions: (B, N, SEQ_OUT)
            pred = model(b_x, b_adj)
            loss = criterion(pred, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * len(b_x)

        train_loss /= len(train_dataset)

        # Validation evaluation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for b_x, b_adj, _ in val_loader:
                b_x = b_x.to(device)
                b_adj = b_adj.to(device)
                pred = model(b_x, b_adj)
                val_preds.append(pred.cpu())

        val_preds_tensor = torch.cat(val_preds, dim=0)
        val_metrics = compute_metrics(val_preds_tensor, Y_val)
        scheduler.step(val_metrics["rmse"])

        is_best = val_metrics["rmse"] < best_val_rmse
        if is_best:
            best_val_rmse = val_metrics["rmse"]
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "persistence_val": pers_val_metrics,
                "scaler_mean": dataset["scaler_mean"],
                "scaler_std": dataset["scaler_std"],
                "pm25_mean": pm25_mean,
                "pm25_std": pm25_std,
                "stations": dataset["station_ids"],
                "feature_names": dataset["feature_names"],
                "dist_km": dataset["dist_km"],
                "bearings": dataset["bearings"]
            }
            torch.save(checkpoint_data, best_checkpoint_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            logging.info(
                f"Epoch [{epoch:02d}/{EPOCHS}] "
                f"Loss: {train_loss:.2f} | "
                f"Val MAE: {val_metrics['mae']:.2f} µg/m³ (3h: {val_metrics['mae_3h']:.2f}, 6h: {val_metrics['mae_6h']:.2f}) | "
                f"Val RMSE: {val_metrics['rmse']:.2f} µg/m³ (3h: {val_metrics['rmse_3h']:.2f}, 6h: {val_metrics['rmse_6h']:.2f}) "
                f"{'[BEST CHECKPOINT]' if is_best else ''}"
            )

    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.1f}s ({elapsed_time/60.0:.2f} mins).")

    # Final Evaluation on Best Model Checkpoint
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    with torch.no_grad():
        for b_x, b_adj, _ in test_loader:
            b_x = b_x.to(device)
            b_adj = b_adj.to(device)
            pred = model(b_x, b_adj)
            test_preds.append(pred.cpu())

    test_preds_tensor = torch.cat(test_preds, dim=0)
    test_metrics = compute_metrics(test_preds_tensor, Y_test)

    best_val = checkpoint["val_metrics"]

    logging.info("================ FINAL VERIFICATION REPORT ================")
    logging.info(f"Best Validation Epoch: {checkpoint['epoch']}")
    logging.info(f"Persistence Val  -> Overall RMSE: {pers_val_metrics['rmse']:.2f} | 3h RMSE: {pers_val_metrics['rmse_3h']:.2f} | 6h RMSE: {pers_val_metrics['rmse_6h']:.2f}")
    logging.info(f"WindSTGNN   Val  -> Overall RMSE: {best_val['rmse']:.2f} | 3h RMSE: {best_val['rmse_3h']:.2f} | 6h RMSE: {best_val['rmse_6h']:.2f}")
    logging.info(f"Persistence Val  -> Overall MAE:  {pers_val_metrics['mae']:.2f} | 3h MAE:  {pers_val_metrics['mae_3h']:.2f} | 6h MAE:  {pers_val_metrics['mae_6h']:.2f}")
    logging.info(f"WindSTGNN   Val  -> Overall MAE:  {best_val['mae']:.2f} | 3h MAE:  {best_val['mae_3h']:.2f} | 6h MAE:  {best_val['mae_6h']:.2f}")
    logging.info("-----------------------------------------------------------")
    logging.info(f"Persistence Test -> Overall RMSE: {pers_test_metrics['rmse']:.2f} | 3h RMSE: {pers_test_metrics['rmse_3h']:.2f} | 6h RMSE: {pers_test_metrics['rmse_6h']:.2f}")
    logging.info(f"WindSTGNN   Test -> Overall RMSE: {test_metrics['rmse']:.2f} | 3h RMSE: {test_metrics['rmse_3h']:.2f} | 6h RMSE: {test_metrics['rmse_6h']:.2f}")
    logging.info(f"Persistence Test -> Overall MAE:  {pers_test_metrics['mae']:.2f} | 3h MAE:  {pers_test_metrics['mae_3h']:.2f} | 6h MAE:  {pers_test_metrics['mae_6h']:.2f}")
    logging.info(f"WindSTGNN   Test -> Overall MAE:  {test_metrics['mae']:.2f} | 3h MAE:  {test_metrics['mae_3h']:.2f} | 6h MAE:  {test_metrics['mae_6h']:.2f}")
    logging.info("===========================================================")

    # Verification checks
    rmse_beats = best_val['rmse'] < pers_val_metrics['rmse']
    mae_beats_3h = best_val['mae_3h'] < pers_val_metrics['mae_3h']
    mae_beats_6h = best_val['mae_6h'] < pers_val_metrics['mae_6h']
    rmse_beats_3h = best_val['rmse_3h'] < pers_val_metrics['rmse_3h']
    rmse_beats_6h = best_val['rmse_6h'] < pers_val_metrics['rmse_6h']

    logging.info(f"Val RMSE beats baseline: {rmse_beats} ({best_val['rmse']:.2f} vs {pers_val_metrics['rmse']:.2f})")
    logging.info(f"Val +3h MAE beats baseline: {mae_beats_3h} ({best_val['mae_3h']:.2f} vs {pers_val_metrics['mae_3h']:.2f})")
    logging.info(f"Val +6h MAE beats baseline: {mae_beats_6h} ({best_val['mae_6h']:.2f} vs {pers_val_metrics['mae_6h']:.2f})")
    logging.info(f"Val +3h RMSE beats baseline: {rmse_beats_3h} ({best_val['rmse_3h']:.2f} vs {pers_val_metrics['rmse_3h']:.2f})")
    logging.info(f"Val +6h RMSE beats baseline: {rmse_beats_6h} ({best_val['rmse_6h']:.2f} vs {pers_val_metrics['rmse_6h']:.2f})")

    return checkpoint

if __name__ == "__main__":
    train_model()
