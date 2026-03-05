import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import cfg
from .utils import set_seed, ensure_dirs, kaggle_download
from .data import csv_to_imagefolder, make_loaders, compute_class_counts
from .models import build_model
from .losses import FocalLoss


@torch.no_grad()
def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def _mixup_batch(x, y, alpha: float):
    """Return mixed inputs, paired targets, and lambda."""
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    x_mix = lam * x + (1.0 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return x_mix, y_a, y_b, lam


def _mixup_loss(criterion, logits, y_a, y_b, lam: float):
    """Compute lam*loss(y_a) + (1-lam)*loss(y_b)."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha: float = 0.0, use_focal_loss: bool = False):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if (mixup_alpha > 0.0) and (not use_focal_loss):
            x, y_a, y_b, lam = _mixup_batch(x, y, mixup_alpha)
            logits = model(x)
            loss = _mixup_loss(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs  # ok for monitoring even with mixup
        n += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n


def build_criterion(cfg, full_train, train_subset_indices, device):
    """
    Loss setup:
    - If cfg.use_focal_loss: use FocalLoss (optionally with class weights)
    - Else: CrossEntropyLoss with label smoothing (optionally with class weights)

    Note: if use_weighted_sampler=True, class weights are usually unnecessary.
    """
    class_weights = None

    if cfg.use_class_weights and (not cfg.use_weighted_sampler):
        counts = compute_class_counts(full_train, indices=train_subset_indices, num_classes=7)
        w = (counts.sum() / counts)  # inverse frequency
        w = w / w.mean()            # normalize
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    if cfg.use_focal_loss:
        return FocalLoss(gamma=2.0, alpha=class_weights)

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=getattr(cfg, "label_smoothing", 0.1),
    )


def main():
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    fer_csv = kaggle_download(cfg)
    train_root, test_root = csv_to_imagefolder(cfg, fer_csv)

    full_train, train_ds, train_loader, val_loader, test_loader = make_loaders(cfg, train_root, test_root)

    model = build_model(cfg, num_classes=7).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = build_criterion(cfg, full_train, train_ds.indices, cfg.device)

    best_val = -1.0
    patience = 0

    best_path = Path(cfg.model_dir) / f"best_{cfg.model_name}.pt"
    history = []

    print(f"[INFO] device={cfg.device} model={cfg.model_name} epochs={cfg.epochs}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            cfg.device,
            mixup_alpha=getattr(cfg, "mixup_alpha", 0.0),
            use_focal_loss=getattr(cfg, "use_focal_loss", False),
        )

        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, cfg.device)
        scheduler.step()

        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
            "sec": dt,
            "model_name": cfg.model_name,
        }
        history.append(row)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | tr {tr_loss:.4f}/{tr_acc:.4f} | va {va_loss:.4f}/{va_acc:.4f} | {dt:.1f}s")

        if va_acc > best_val + getattr(cfg, "early_stop_min_delta", 0.0):
            best_val = va_acc
            patience = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__}, best_path)
            print(f"[INFO] saved -> {best_path} val_acc={best_val:.4f}")
        else:
            patience += 1
            if patience >= getattr(cfg, "early_stop_patience", 8):
                print(f"[INFO] Early stopping triggered (patience={patience}). Best val_acc={best_val:.4f}")
                break

    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc = eval_one_epoch(model, test_loader, criterion, cfg.device)
    print(f"[RESULT] test_loss={te_loss:.4f} test_acc={te_acc:.4f}")

    out_hist = Path(cfg.out_dir) / f"train_history_{cfg.model_name}.json"
    with open(out_hist, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] history -> {out_hist}")


if __name__ == "__main__":
    main()