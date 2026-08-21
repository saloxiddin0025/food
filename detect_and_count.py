import argparse
from collections import Counter
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "models"
CKPT_PATH = MODELS_DIR / "resnet18_pizza.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

classify_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

COCO_PIZZA_LABEL = "pizza"

LABEL_COLORS = {
    "cheese_pizza": (0, 255, 255),  # yellow, BGR
    "meat_pizza": (0, 0, 255),      # red, BGR
}


def load_classifier():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, classes


def classify_crop(model, classes, crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    tensor = classify_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], probs[idx].item()


def find_pizza_class_id(detector):
    for cls_id, name in detector.names.items():
        if name == COCO_PIZZA_LABEL:
            return cls_id
    raise SystemExit(f"'{COCO_PIZZA_LABEL}' class not found in YOLO model's classes: {detector.names}")


def draw_box(frame, box_xyxy, label, conf, text=None):
    x1, y1, x2, y2 = box_xyxy
    color = LABEL_COLORS.get(label, (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text if text is not None else f"{label} {conf:.0%}", (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_summary(frame, counts):
    y = 30
    for cls_name, count in counts.items():
        color = LABEL_COLORS.get(cls_name, (255, 255, 255))
        cv2.putText(frame, f"{cls_name}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        y += 30


def clamp_box(x1, y1, x2, y2, width, height):
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), width), min(int(y2), height)
    return x1, y1, x2, y2


def run_on_image(path, detector, pizza_class_id, classifier, classes, conf_thresh, output):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    height, width = frame.shape[:2]

    results = detector.predict(frame, conf=conf_thresh, classes=[pizza_class_id], verbose=False)[0]

    counts = {c: 0 for c in classes}
    for box in results.boxes:
        x1, y1, x2, y2 = clamp_box(*box.xyxy[0].tolist(), width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        label, conf = classify_crop(classifier, classes, crop)
        counts[label] += 1
        draw_box(frame, (x1, y1, x2, y2), label, conf)

    draw_summary(frame, counts)

    print("Detected pizzas:")
    for cls_name, count in counts.items():
        print(f"  {cls_name}: {count}")

    if output:
        cv2.imwrite(output, frame)
        print(f"\nAnnotated image saved to: {output}")


def run_on_video(path, detector, pizza_class_id, classifier, classes, conf_thresh, skip, output,
                  line_axis="horizontal", line_fraction=0.5):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    line_pos = int((height if line_axis == "horizontal" else width) * line_fraction)

    track_labels = {}    # track_id -> list of per-frame classifications seen so far
    prev_pos = {}        # track_id -> centroid position (on the line's axis) from the previous sighting
    counted = set()       # track_ids that have already crossed the line and been counted
    crossed_counts = {c: 0 for c in classes}

    frame_idx = 0
    last_drawn = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            results = detector.track(frame, conf=conf_thresh, classes=[pizza_class_id],
                                      persist=True, verbose=False, tracker="bytetrack.yaml")[0]
            annotated = frame.copy()

            if results.boxes.id is not None:
                for box, track_id in zip(results.boxes, results.boxes.id):
                    x1, y1, x2, y2 = clamp_box(*box.xyxy[0].tolist(), width, height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    label, conf = classify_crop(classifier, classes, crop)
                    tid = int(track_id.item())
                    track_labels.setdefault(tid, []).append(label)

                    cy = (y1 + y2) / 2
                    cx = (x1 + x2) / 2
                    pos = cy if line_axis == "horizontal" else cx

                    if tid in prev_pos and tid not in counted:
                        if (prev_pos[tid] < line_pos) != (pos < line_pos):
                            final_label = Counter(track_labels[tid]).most_common(1)[0][0]
                            crossed_counts[final_label] += 1
                            counted.add(tid)
                    prev_pos[tid] = pos

                    draw_box(annotated, (x1, y1, x2, y2), label, conf, text=f"#{tid} {label} {conf:.0%}")

            if line_axis == "horizontal":
                cv2.line(annotated, (0, line_pos), (width, line_pos), (255, 255, 255), 2)
            else:
                cv2.line(annotated, (line_pos, 0), (line_pos, height), (255, 255, 255), 2)

            draw_summary(annotated, crossed_counts)
            last_drawn = annotated
            print(f"frame {frame_idx}: crossed count -> " +
                  ", ".join(f"{c}={crossed_counts[c]}" for c in classes))

        if writer is not None:
            writer.write(last_drawn if last_drawn is not None else frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"\nProcessed {frame_idx} frames.")
    print(f"Final count (each pizza counted once, when it crossed the {line_axis} line):")
    for cls_name, count in crossed_counts.items():
        print(f"  {cls_name}: {count}")

    if output:
        print(f"\nAnnotated video saved to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect pizzas (pretrained YOLOv8, zero-shot) and classify each as cheese/meat "
                    "(fine-tuned ResNet18), then count."
    )
    parser.add_argument("input", help="Path to an image or video file")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO detection confidence threshold (default: 0.35)")
    parser.add_argument("--skip", type=int, default=1,
                         help="For video: process every Nth frame (default: 1). Higher values are faster but can "
                              "confuse the tracker and distort the distinct-pizza count.")
    parser.add_argument("--output", help="Optional path to save annotated image/video")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="Ultralytics YOLO checkpoint (default: yolov8n.pt)")
    parser.add_argument("--line-axis", choices=["horizontal", "vertical"], default="horizontal",
                         help="For video: orientation of the counting line (default: horizontal, i.e. pizzas "
                              "moving up/down cross it). Use 'vertical' for left/right motion.")
    parser.add_argument("--line-fraction", type=float, default=0.5,
                         help="For video: where to place the counting line, as a fraction of frame "
                              "height/width (default: 0.5 = the middle)")
    args = parser.parse_args()

    print(f"Loading YOLO detector ({args.yolo_model})...")
    detector = YOLO(args.yolo_model)
    pizza_class_id = find_pizza_class_id(detector)

    print("Loading fine-tuned ResNet18 classifier...")
    classifier, classes = load_classifier()
    print(f"Classes: {classes}")

    ext = Path(args.input).suffix.lower()
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    if ext in video_exts:
        run_on_video(args.input, detector, pizza_class_id, classifier, classes, args.conf, args.skip, args.output,
                     line_axis=args.line_axis, line_fraction=args.line_fraction)
    else:
        run_on_image(args.input, detector, pizza_class_id, classifier, classes, args.conf, args.output)


if __name__ == "__main__":
    main()
