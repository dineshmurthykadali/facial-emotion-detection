from pathlib import Path

import numpy as np
from PIL import Image

import torch
import cv2
from torchvision import transforms

from .config import cfg
from .models import build_model


def load_model(model_path: str):
    ckpt = torch.load(model_path, map_location=cfg.device)
    model = build_model(cfg, num_classes=7).to(cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_face(model, face_gray_uint8: np.ndarray):
    pil = Image.fromarray(face_gray_uint8, mode="L")

    in_ch = getattr(cfg, "input_channels", 3)
    print(f"[DEBUG] cfg.input_channels={in_ch} cfg.image_size={cfg.image_size} cfg.model_name={cfg.model_name}")

    if in_ch == 3:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5,)
        std = (0.5,)

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=in_ch),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    x = tfm(pil).unsqueeze(0).to(cfg.device)
    print(f"[DEBUG] x.shape={tuple(x.shape)}")

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs.max()), probs


def main():
    model_path = str(Path(cfg.model_dir) / f"best_{cfg.model_name}.pt")
    if not Path(model_path).exists():
        alt = str(Path(cfg.model_dir) / "best_fer2013_cnn.pt")
        if Path(alt).exists():
            model_path = alt
        else:
            raise FileNotFoundError(f"No model found at {model_path} or {alt}")

    model = load_model(model_path)
    print(f"[INFO] Loaded model: {model_path} on {cfg.device}")
    print("[INFO] Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            pred, conf, _ = predict_face(model, face)
            label = cfg.class_names[pred]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()