import json
from pathlib import Path
import matplotlib.pyplot as plt
import os

from .config import cfg


def main():
    candidates = [
        Path(cfg.out_dir) / f"train_history_{cfg.model_name}.json",
        Path(cfg.out_dir) / "train_history.json",
    ]

    hist_path = None
    for c in candidates:
        if c.exists():
            hist_path = c
            break

    if hist_path is None:
        raise FileNotFoundError(f"No history json found in {cfg.out_dir}")

    with open(hist_path, "r") as f:
        hist = json.load(f)

    epochs = [r["epoch"] for r in hist]
    train_loss = [r["train_loss"] for r in hist]
    val_loss = [r["val_loss"] for r in hist]
    train_acc = [r["train_acc"] for r in hist]
    val_acc = [r["val_acc"] for r in hist]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curves")
    plt.legend()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/loss_curve.png", dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy curves")
    plt.legend()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()