import json
from pathlib import Path
from typing import Union, Sequence

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


ImageInput = Union[str, Path, Image.Image]


class PotholeClassifier:
    """
    Binary pothole image classifier.

    Supported loading modes:
    1. TorchScript file: model_v3_traced.pt
    2. Standard PyTorch weights: weights.pth + config.json

    Example:
        clf = PotholeClassifier(r"./pothole_model_v3_export")
        result = clf.predict(r"./example.jpg")
        print(result)
    """

    def __init__(
        self,
        export_dir: Union[str, Path],
        device: Union[str, torch.device, None] = None,
        threshold: Union[float, None] = None,
        prefer_traced: bool = True,
    ):
        self.export_dir = Path(export_dir)
        if not self.export_dir.exists():
            raise FileNotFoundError(f"Export directory not found: {self.export_dir}")

        self.config_path = self.export_dir / "config.json"
        self.weights_path = self.export_dir / "weights.pth"
        self.traced_path = self.export_dir / "model_v3_traced.pt"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing config file: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.arch = self.config.get("arch", "resnet18")
        self.num_classes = int(self.config["num_classes"])
        self.class_to_idx = self.config["class_to_idx"]
        self.pothole_idx = int(self.config["pothole_idx"])
        self.threshold = float(
            threshold if threshold is not None else self.config.get("default_threshold", 0.5)
        )

        self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}

        input_size = int(self.config["input_size"])
        mean = self.config["normalize_mean"]
        std = self.config["normalize_std"]

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.using_traced = False
        self.model = self._load_model(prefer_traced)
        self.model.eval()

    def _build_model(self):
        if self.arch != "resnet18":
            raise ValueError(f"Unsupported architecture in config: {self.arch}")

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def _load_model(self, prefer_traced: bool):
        if prefer_traced and self.traced_path.exists():
            model = torch.jit.load(str(self.traced_path), map_location=self.device)
            self.using_traced = True
            return model.to(self.device)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Missing weights file: {self.weights_path}\n"
                f"Also did not find traced model: {self.traced_path}"
            )

        model = self._build_model()
        state_dict = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def _load_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            try:
                return Image.open(image_path).convert("RGB")
            except UnidentifiedImageError as e:
                raise ValueError(f"Unsupported or corrupted image file: {image_path}") from e

        raise TypeError(
            f"Unsupported input type: {type(image)}. "
            f"Expected str, Path, or PIL.Image.Image."
        )

    def _predict_batch(self, images: Sequence[ImageInput]):
        pil_images = [self._load_image(img) for img in images]
        batch = torch.stack([self.transform(img) for img in pil_images]).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            pothole_probs = probs[:, self.pothole_idx].cpu().tolist()
            pred_idx = probs.argmax(dim=1).cpu().tolist()

        results = []
        for p_prob, p_idx in zip(pothole_probs, pred_idx):
            results.append({
                "pothole_probability": float(p_prob),
                "predicted_class": self.idx_to_class[int(p_idx)],
                "is_pothole": bool(p_prob >= self.threshold),
                "threshold": self.threshold,
            })
        return results

    def predict(self, image: Union[ImageInput, Sequence[ImageInput]]):
        """
        Predict one image or a list of images.

        Returns:
            dict for single input
            list[dict] for batch input
        """
        is_single = not isinstance(image, (list, tuple))
        images = [image] if is_single else list(image)

        results = self._predict_batch(images)
        return results[0] if is_single else results

    def info(self):
        return {
            "export_dir": str(self.export_dir),
            "device": str(self.device),
            "using_traced": self.using_traced,
            "arch": self.arch,
            "num_classes": self.num_classes,
            "class_to_idx": self.class_to_idx,
            "pothole_idx": self.pothole_idx,
            "threshold": self.threshold,
        }
