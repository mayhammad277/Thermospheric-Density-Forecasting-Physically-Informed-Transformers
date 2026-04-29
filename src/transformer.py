# src/transformer.py
"""
TemporalDensityTransformer model and training utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


def initialize_weights(model: nn.Module):
    for name, p in model.named_parameters():
        if 'weight_ih' in name or ('weight' in name and p.dim() == 2):
            nn.init.xavier_uniform_(p)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(p)
        elif 'bias' in name:
            nn.init.zeros_(p)


class TemporalDensityTransformer(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, num_heads: int = 4,
                 num_layers: int = 3, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x)
        x = x.mean(dim=1)   # global average pooling
        return self.head(x)


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
                log_target: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    initialize_weights(model)

    def to_tensor(arr, log=False):
        t = torch.tensor(arr, dtype=torch.float32)
        return torch.log(t.clamp(min=1e-30)) if log else t

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = to_tensor(y_train, log=log_target).unsqueeze(1)
    X_v  = torch.tensor(X_val, dtype=torch.float32)
    y_v  = to_tensor(y_val, log=log_target).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val, patience, wait = float('inf'), 10, 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v.to(device))
            val_loss = criterion(val_pred, y_v.to(device)).item()

        scheduler.step()
        avg_train = train_loss / len(X_tr)
        history['train'].append(avg_train)
        history['val'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d} | Train: {avg_train:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('best_model.pt'))
    return model, history
