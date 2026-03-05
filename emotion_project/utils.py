import os
import random
import shutil
from pathlib import Path

import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def ensure_dirs(cfg):
    for p in [cfg.project_dir, cfg.data_dir, cfg.raw_dir, cfg.processed_dir, cfg.model_dir, cfg.out_dir]:
        Path(p).mkdir(parents=True, exist_ok=True)


def kaggle_download(cfg):
    """Download and unzip FER2013 dataset from Kaggle into cfg.raw_dir and return fer2013.csv path."""
    raw_path = Path(cfg.raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    cmd = f'kaggle datasets download -d "{cfg.kaggle_dataset}" -p "{cfg.raw_dir}" --force'
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError("Kaggle download failed. Verify kaggle.json and 'kaggle' CLI.")

    # Unzip
    for z in raw_path.glob("*.zip"):
        shutil.unpack_archive(str(z), str(raw_path))

    # Find CSV
    candidates = list(raw_path.rglob("fer2013.csv"))
    if not candidates:
        raise FileNotFoundError("fer2013.csv not found after download/unzip.")
    return str(candidates[0])