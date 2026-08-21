import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MODELS_DIR = Path(__file__).parent / "models"
CKPT_PATH = MODELS_DIR / "resnet18_pizza.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, classes


def classify_frame(model, classes, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    pred_idx = probs.argmax().item()
    return classes[pred_idx], probs[pred_idx].item(), probs.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Classify cheese vs meat pizza in a video using the fine-tuned ResNet18 model."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--skip", type=int, default=5, help="Classify every Nth frame (default: 5)")
    parser.add_argument("--output", help="Optional path to save an annotated output video (.mp4)")
    args = parser.parse_args()

    model, classes = load_model()
    print(f"Loaded model with classes: {classes}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    last_label, last_conf = None, None
    label_counts = {c: 0 for c in classes}
    prob_sums = [0.0] * len(classes)
    classified_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip == 0:
            last_label, last_conf, probs = classify_frame(model, classes, frame)
            label_counts[last_label] += 1
            for i, p in enumerate(probs):
                prob_sums[i] += p
            classified_frames += 1
            print(f"frame {frame_idx}: {last_label} ({last_conf:.2%})")

        if writer is not None and last_label is not None:
            annotated = frame.copy()
            text = f"{last_label} {last_conf:.0%}"
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"\nProcessed {frame_idx} frames, classified {classified_frames} of them.")
    if classified_frames:
        print("Label counts (sampled frames):")
        for c in classes:
            print(f"  {c}: {label_counts[c]} ({label_counts[c] / classified_frames:.1%})")

        avg_probs = [p / classified_frames for p in prob_sums]
        overall_idx = max(range(len(classes)), key=lambda i: avg_probs[i])
        print(f"\nOverall video verdict: {classes[overall_idx]} (avg confidence {avg_probs[overall_idx]:.2%})")

    if args.output:
        print(f"\nAnnotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
