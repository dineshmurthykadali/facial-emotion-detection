# Facial Emotion Detection (FER2013) — PyTorch

Facial emotion detection built with PyTorch on the FER2013 dataset. The source images are 48x48 grayscale. The pipeline resizes them to 224x224 for pretrained backbones.

## Emotions

Seven classes: angry, disgust, fear, happy, sad, surprise, neutral.

## Folder Structure

```
Facialemotiondetection/
  emotion_project/
    __init__.py
    config.py
    utils.py
    data.py
    models.py
    losses.py
    train.py
    infer_webcam.py
    plot_history.py

  emotion_fer2013_project/        # created on first run
    data/
      raw/                        # kaggle zip + fer2013.csv
      processed/                  # PNG folders in ImageFolder format
    models/                       # saved checkpoints
    outputs/                      # training history json
```

## Setup

Run these from the project root.

```bash
conda create --prefix "./.conda_env" python=3.11 -y
conda activate "./.conda_env"
```

Then install everything from requirements:

```bash
pip install -r requirements.txt
```

If you'd rather install by hand:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas pillow tqdm scikit-learn opencv-python kaggle matplotlib
```

## Kaggle

The dataset is pulled through the Kaggle CLI, so you need an API token.

Go to Kaggle, then Account, then API, then Create New API Token. That downloads `kaggle.json`. Put it where Kaggle expects it:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Keep `kaggle.json` out of git. It holds your API key.

Dataset: `deadskull7/fer2013`

## Training

```bash
python -m emotion_project.train
```

The first run downloads FER2013, extracts `fer2013.csv`, converts the pixel rows into PNGs in ImageFolder format, then trains. So give it a few minutes the first time.

The model and settings live in `emotion_project/config.py`. Training keeps the best checkpoint by validation accuracy and writes the history to JSON.

Outputs land here:

```
emotion_fer2013_project/models/best_<model_name>.pt
emotion_fer2013_project/outputs/train_history_<model_name>.json
```

## Switching Models and Image Size

Everything is in `emotion_project/config.py`.

ResNet18 is the quick baseline. EfficientNet-B0 hits harder but takes longer.

For the pretrained ImageNet models I run with:

```python
image_size = 224
input_channels = 3
label_smoothing = 0.1
mixup_alpha = 0.2      # 0.0 turns it off
early_stop_patience = 8
```

Just make sure the train, val, and test transforms all resize to `cfg.image_size`.

## Webcam Demo

Once you've trained a model:

```bash
python -m emotion_project.infer_webcam
```

Press `q` to quit.

If the webcam won't open on macOS, it's a permissions thing. System Settings, then Privacy & Security, then Camera, and allow access for whatever you're running from (Terminal or VS Code).

## Plotting

```bash
python -m emotion_project.plot_history
```

This reads the history JSON and saves the curves to `assets/loss_curve.png` and `assets/accuracy_curve.png`. If no window pops up, you're probably on a remote-only terminal, or an old matplotlib window is still sitting open in the background.

## What Each File Does

`train.py` runs the training loop, saves the best model, and writes the history JSON.
`data.py` converts the FER2013 CSV into ImageFolder format and sets up the dataloaders and transforms.
`models.py` holds the model definitions: a simple CNN, ResNet18, and EfficientNet.
`infer_webcam.py` runs the live webcam test off the saved model.
`plot_history.py` reads the history JSON and plots the curves.
`losses.py` has focal loss if you want it.
`utils.py` is the helper stuff: seeding, folder creation, the Kaggle download.

## Things That Tripped Me Up

DataLoader workers kept crashing on macOS, so I set `num_workers = 0` in `config.py` and left it there.

If you hit a channel mismatch (it expects 1 channel but gets 3, or the other way around), check that `input_channels` in `config.py` lines up with what your transforms actually output.

## A Recent Run

EfficientNet-B0 at 224x224, 3 channels, ImageNet normalization, got me around 70% validation accuracy. The number moves around a bit depending on settings and seed.
