import torch
import torch.nn as nn
from torchvision import models

class SimpleFER_CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def _replace_first_conv(model, in_channels=1):
    # Replace first conv layer to accept grayscale (1 channel)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first = module
            break
    if first.in_channels == in_channels:
        return model

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        bias=(first.bias is not None),
    )
    # If original had 3 channels, average weights across channels for a decent init
    with torch.no_grad():
        if first.weight.shape[1] == 3 and in_channels == 1:
            new_conv.weight.copy_(first.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
    # assign to common attribute names
    if hasattr(model, "conv1"):
        model.conv1 = new_conv
    else:
        # fallback: try to set first conv via state_dict replacement not handled here
        raise RuntimeError("Model first conv not found as model.conv1. Use resnet/efficientnet only.")
    return model

def build_model(cfg, num_classes=7):
    if cfg.model_name == "simple_cnn":
        return SimpleFER_CNN(num_classes=num_classes)

    if cfg.model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Only replace first conv if using 1-channel input
        if getattr(cfg, "input_channels", 3) == 1:
            m = _replace_first_conv(m, in_channels=1)

        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if cfg.model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Only replace first conv if using 1-channel input
        if getattr(cfg, "input_channels", 3) == 1:
            first = m.features[0][0]
            new_conv = nn.Conv2d(
                1,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.copy_(first.weight.mean(dim=1, keepdim=True))
            m.features[0][0] = new_conv

        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    raise ValueError(f"Unknown cfg.model_name: {cfg.model_name}")