#!/usr/bin/env python3
"""Addestra i modelli finali (predicono gap = trip_duration - osrm_eta)."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from models import (
    ResMLP, DeepResMLP, CustomModel1,
    MultiBranchModel, MultiBranchResNet,
)


FEATURES = [
    "origin_lat", "origin_lon", "destination_lat", "destination_lon",
    "osrm_eta", "osrm_distance", "svolte_totali", "svolte_sinistra",
    "incroci_totali", "incroci_complessi", "nodi_attraversati", "eucl_dist",
    "svolte_per_km", "incroci_per_km",
    "pickup_hour_sin", "pickup_hour_cos",
    "pickup_day_sin", "pickup_day_cos",
    "pickup_month_sin", "pickup_month_cos",
]


def training_minibatch(model, data_loader, opt, loss_fn, device="cpu", n_epochs=20):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(data_loader):.4f}")


def evaluate(model, data_loader, metric_fn, device="cpu"):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            all_preds.append(model(X_batch).detach().cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    return metric_fn(np.concatenate(all_targets), np.concatenate(all_preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="ultimate_train.csv")
    parser.add_argument("--output-dir", default="models/final")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)

    # Feature engineering
    df["eucl_dist"] = np.sqrt(
        (df["destination_lat"] - df["origin_lat"])**2 +
        (df["destination_lon"] - df["origin_lon"])**2
    )
    df["svolte_per_km"] = df["svolte_totali"] / df["osrm_distance"]
    df["incroci_per_km"] = df["incroci_totali"] / df["osrm_distance"]
    df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["pickup_day_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["pickup_day_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["pickup_month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["pickup_month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)
    df["gap"] = df["trip_duration"] - df["osrm_eta"]

    df = df[FEATURES + ["gap"]]
    df = df.dropna()

    # Gap filter: 1st-99th percentile
    q_low, q_high = df["gap"].quantile(0.01), df["gap"].quantile(0.99)
    df = df[(df["gap"] >= q_low) & (df["gap"] <= q_high)]

    X = df[FEATURES]
    y = df["gap"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    X_train_t = torch.tensor(X_train.to_numpy(dtype=np.float32))
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32))
    y_train_t = torch.tensor(y_train.to_numpy(dtype=np.float32)).view(-1, 1)
    y_val_t = torch.tensor(y_val.to_numpy(dtype=np.float32)).view(-1, 1)

    means = X_train_t.mean(dim=0, keepdims=True)
    stds = X_train_t.std(dim=0, keepdims=True)
    stds[stds == 0] = 1.0
    X_train_t = (X_train_t - means) / stds
    X_val_t = (X_val_t - means) / stds

    y_means = y_train_t.mean(dim=0, keepdims=True)
    y_stds = y_train_t.std(dim=0, keepdims=True)
    y_train_t = (y_train_t - y_means) / y_stds
    y_val_t = (y_val_t - y_means) / y_stds

    n_features = X_train_t.shape[1]
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size)

    archs = {
        "model_mlp_v2": nn.Sequential(
            nn.Linear(n_features, 200), nn.BatchNorm1d(200), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(200, 200), nn.BatchNorm1d(200), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(200, 1),
        ),
        "customModel1": CustomModel1(n_features),
        "model_resmlp": ResMLP(n_features),
        "model_multibranch": MultiBranchModel(n_features),
        "model_multibranch_resnet": MultiBranchResNet(n_features),
        "model_deep_resmlp": DeepResMLP(n_features),
    }

    import sklearn.metrics
    torch.manual_seed(42)
    for name, model in archs.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nTraining {name} ({n_params} params)...")
        model.to(device)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()
        training_minibatch(model, train_loader, opt, loss_fn, device, args.epochs)
        r2 = evaluate(model, val_loader, sklearn.metrics.r2_score, device)
        print(f"R²: {r2:.4f}")
        torch.save(model.state_dict(), str(out_dir / f"{name}.pth"))
        print(f"  → Saved {out_dir / name}.pth")


if __name__ == "__main__":
    main()
