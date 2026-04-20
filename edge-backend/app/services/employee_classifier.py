"""
Clasificador employee/non_employee sobre crops de personas.
"""
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackEmployeeState:
    employee_probability: float
    is_employee: bool


class EmployeeClassifier:
    """Clasificador binario que estima probabilidad de empleado en un crop."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        image_size: int = 224,
    ):
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.model = self._load_model(model_path, device)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, model_path: str, device: str) -> nn.Module:
        checkpoint = torch.load(model_path, map_location=device)
        architecture = checkpoint.get("architecture", "mobilenet_v3_small")
        if architecture != "mobilenet_v3_small":
            raise ValueError(f"Arquitectura no soportada: {architecture}")

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def predict_employee_probability(self, crop_bgr: np.ndarray) -> float:
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(1)
            prob = torch.sigmoid(logits).item()
        return float(prob)


class TrackEmployeeFilter:
    """Mantiene estado temporal por track para decidir si es empleado."""

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        employee_threshold: float = 0.75,
        vote_window: int = 8,
        min_votes: int = 5,
    ):
        self.employee_threshold = employee_threshold
        self.vote_window = vote_window
        self.min_votes = min_votes
        self.enabled = False
        self.classifier: Optional[EmployeeClassifier] = None
        self.track_probs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=vote_window))

        if model_path:
            model_file = Path(model_path)
            if model_file.exists():
                self.classifier = EmployeeClassifier(model_path=str(model_file), device=device)
                self.enabled = True
                logger.info(f"✓ Employee classifier habilitado: {model_file}")
            else:
                logger.warning(f"Modelo de empleado no encontrado en {model_file}. Se desactiva filtro.")

    def classify_tracks(self, frame: np.ndarray, detections: List[Detection]) -> Dict[int, TrackEmployeeState]:
        results: Dict[int, TrackEmployeeState] = {}
        if not self.enabled or self.classifier is None:
            return results

        frame_h, frame_w = frame.shape[:2]
        for det in detections:
            if det.track_id is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1 = max(0, min(frame_w - 1, x1))
            x2 = max(0, min(frame_w, x2))
            y1 = max(0, min(frame_h - 1, y1))
            y2 = max(0, min(frame_h, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            prob = self.classifier.predict_employee_probability(crop)
            history = self.track_probs[det.track_id]
            history.append(prob)

            if len(history) < self.min_votes:
                is_employee = False
            else:
                is_employee = (sum(history) / len(history)) >= self.employee_threshold

            results[det.track_id] = TrackEmployeeState(
                employee_probability=float(sum(history) / len(history)),
                is_employee=is_employee,
            )

        return results

