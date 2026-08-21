# Pizza Classifier

Detects pizzas in an image or video (pretrained YOLOv8) and classifies each one as
`cheese_pizza` or `meat_pizza` (fine-tuned ResNet18), counting each instance once.

## Setup

```
pip install -r requirements.txt
```

## Dataset

1,132 images in two folders (574 cheese, 558 meat), close-up single-pizza shots.
`prepare_split.py` builds a fixed, stratified 85/15 train/val split (`split.json`)
shared by every model trained here, so results are directly comparable.

```
python prepare_split.py
```

## Training

ResNet18 is fine-tuned end-to-end on the split, reaching **98.24%** validation
accuracy.

```
python train_resnet.py
```

Saves the best checkpoint to `models/resnet18_pizza.pt`.

## Inference

`classifier.py` holds the reusable inference code (`load_classifier()` +
`classify_crop()`) that loads the checkpoint and classifies a single image.

## Usage

**Image or video, detect + classify + count** (each pizza counted once, when it
crosses a line drawn through the frame):

```
python detect_and_count.py path/to/video.mp4 --output annotated.mp4
python detect_and_count.py path/to/image.jpg --output annotated.jpg
```

Key options: `--conf` (YOLO detection threshold), `--line-axis` (`horizontal` or
`vertical`), `--line-fraction` (where the counting line sits, 0-1), `--skip`
(process every Nth video frame).

**Video, frame-by-frame classification only** (no detection/counting):

```
python predict_video.py path/to/video.mp4 --output annotated.mp4
```

## Known limitation

The classifier was trained only on close-up photos of two topping types (plain
cheese, pepperoni/meat). On out-of-distribution footage — different camera
angle, or toppings it never saw (e.g. broccoli, mushroom, olives) — its
confidence drops close to chance level. It generalizes well within its training
distribution, not automatically to arbitrary pizza video.
