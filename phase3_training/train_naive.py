#!/usr/bin/env python3
"""Addestra i modelli naive (MLP che predice osrm_eta da sole coordinate)."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


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
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(data_loader):.4f}")


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
    parser.add_argument("--data", required=True, help="ultimate_dataset_v3.csv or equivalent")
    parser.add_argument("--output-dir", default="models/naive")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df["eucl_dist"] = np.sqrt(
        (df["destination_lat"] - df["origin_lat"])**2 +
        (df["destination_lon"] - df["origin_lon"])**2
    )
    df = df[["origin_lat", "origin_lon", "destination_lat", "destination_lon", "osrm_eta", "eucl_dist"]]
    df = df.dropna()

    X = df[["origin_lat", "origin_lon", "destination_lat", "destination_lon", "eucl_dist"]]
    y = df["osrm_eta"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    X_train_t = torch.tensor(X_train.to_numpy(dtype=np.float32))
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32))
    y_train_t = torch.tensor(y_train.to_numpy(dtype=np.float32)).view(-1, 1)
    y_val_t = torch.tensor(y_val.to_numpy(dtype=np.float32)).view(-1, 1)

    means = X_train_t.mean(dim=0, keepdims=True)
    stds = X_train_t.std(dim=0, keepdims=True)
    X_train_t = (X_train_t - means) / stds
    X_val_t = (X_val_t - means) / stds

    n_features = X_train_t.shape[1]
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size)

    archs = {
        "model_mlp_v1": nn.Sequential(
            nn.Linear(n_features, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 1),
        ),
        "model_mlp_v2": nn.Sequential(
            nn.Linear(n_features, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 1),
        ),
        "model_mlp_v3": nn.Sequential(
            nn.Linear(n_features, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 1),
        ),
    }

    import sklearn.metrics
    torch.manual_seed(42)
    for name, model in archs.items():
        print(f"\nTraining {name} ({sum(p.numel() for p in model.parameters())} params)...")
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
