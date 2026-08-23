"""Lightweight LSTM direction classifier used as the second quant model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


SEQ_LEN = 15
HIDDEN_SIZE = 12
NUM_LAYERS = 1
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 3
MIN_TRAIN = 120
VALIDATION_SIZE = 180


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - mean) / std, mean, std


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < seq_len:
        return (
            np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    x = np.stack(
        [features[index : index + seq_len] for index in range(len(features) - seq_len + 1)]
    ).astype(np.float32)
    y = labels[seq_len - 1 :].astype(np.float32)
    return x, y


class LSTMDirection(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _train_model(x: np.ndarray, y: np.ndarray, epochs: int = EPOCHS) -> LSTMDirection:
    torch.manual_seed(0)
    model = LSTMDirection(input_size=x.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.train()
    best_loss = float("inf")
    patience = PATIENCE
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= max(1, len(y))
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                break
    model.eval()
    return model


def predict_proba(model: LSTMDirection, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def walk_forward_auc(
    features: np.ndarray,
    labels: np.ndarray,
    min_train: int = MIN_TRAIN,
    validation_size: int = VALIDATION_SIZE,
    seq_len: int = SEQ_LEN,
    epochs: int = EPOCHS,
) -> tuple[float, int]:
    usable_len = len(labels)
    if usable_len < min_train + validation_size:
        return 0.0, usable_len

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    end = min_train
    while end < usable_len:
        val_end = min(end + validation_size, usable_len)
        train_x, train_y = build_sequences(features[:end], labels[:end], seq_len)
        val_x, val_y = build_sequences(features[end:val_end], labels[end:val_end], seq_len)
        if len(train_x) == 0 or len(val_x) == 0:
            end = val_end
            continue
        if len(set(train_y.tolist())) < 2 or len(set(val_y.tolist())) < 2:
            end = val_end
            continue
        model = _train_model(train_x, train_y, epochs)
        predictions = predict_proba(model, val_x)
        y_true_all.extend(val_y.tolist())
        y_pred_all.extend(predictions.tolist())
        end = val_end

    if not y_true_all or len(set(y_true_all)) < 2:
        return 0.0, usable_len
    try:
        auc = float(roc_auc_score(y_true_all, y_pred_all))
    except ValueError:
        auc = 0.0
    return round(auc, 6), usable_len


def predict_latest(
    features_all: np.ndarray,
    usable_features: np.ndarray,
    usable_labels: np.ndarray,
    seq_len: int = SEQ_LEN,
    epochs: int = EPOCHS,
) -> float:
    train_x, train_y = build_sequences(usable_features, usable_labels, seq_len)
    if len(train_x) == 0:
        return 0.5
    model = _train_model(train_x, train_y, epochs)
    last = features_all[-seq_len:]
    if len(last) < seq_len:
        return 0.5
    return float(predict_proba(model, last[None, :, :])[0])
