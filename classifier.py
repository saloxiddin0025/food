"""Inference code for the fine-tuned ResNet18 cheese/meat pizza classifier.
Loads the checkpoint produced by train_resnet.py and classifies a single cropped image."""

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


def load_classifier():
    """Loads the fine-tuned ResNet18 checkpoint. Returns (model, classes)."""
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, classes


def classify_crop(model, classes, crop_bgr):
    """Classifies a single OpenCV BGR image (e.g. a cropped detection box).
    Returns (predicted_label, confidence)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], probs[idx].item()
