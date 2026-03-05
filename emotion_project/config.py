from dataclasses import dataclass
import torch


@dataclass
class CFG:
    project_dir: str = "emotion_fer2013_project"
    data_dir: str = "emotion_fer2013_project/data"
    raw_dir: str = "emotion_fer2013_project/data/raw"
    processed_dir: str = "emotion_fer2013_project/data/processed"
    model_dir: str = "emotion_fer2013_project/models"
    out_dir: str = "emotion_fer2013_project/outputs"

    kaggle_dataset: str = "deadskull7/fer2013"

    seed: int = 42

    # Model input
    image_size: int = 224
    input_channels: int = 3

    # Training
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-3
    epochs: int = 50
    val_split: float = 0.1

    # Regularization / training tricks
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2          # set 0.0 to disable
    early_stop_patience: int = 8      # epochs without val_acc improvement
    early_stop_min_delta: float = 1e-4

    # Imbalance handling (choose one strategy)
    use_weighted_sampler: bool = False
    use_class_weights: bool = True
    use_focal_loss: bool = False      # if True, mixup will be disabled

    # Model choice: "simple_cnn", "resnet18", "efficientnet_b0"
    model_name: str = "efficientnet_b0"

    class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    device: str = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )


cfg = CFG()