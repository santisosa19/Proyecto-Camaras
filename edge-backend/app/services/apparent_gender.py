"""
Clasificador de género aparente basado en rostro con agregación por track.
"""
from __future__ import annotations

import logging
import ssl
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


DEFAULT_GENDER_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/deploy_gender.prototxt"
)
DEFAULT_GENDER_WEIGHTS_URL = (
    "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel"
)


@dataclass
class TrackGenderState:
    label: str = "unknown"
    confidence: float = 0.0
    votes: int = 0


class ApparentGenderEstimator:
    """
    Estima género aparente por track usando detección de rostro + DNN de género.
    """

    GENDER_LABELS = ("male", "female")
    MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def __init__(
        self,
        enabled: bool = False,
        model_prototxt_path: str = "",
        model_weights_path: str = "",
        model_dir: str = "models/gender",
        auto_download: bool = True,
        sample_every_n_frames: int = 10,
        vote_window: int = 12,
        min_votes: int = 4,
        confidence_threshold: float = 0.58,
        aggregate_confidence_threshold: float = 0.62,
        female_confidence_threshold: float = 0.72,
        lock_confidence_threshold: float = 0.62,
        lock_min_votes: int = 5,
        flip_margin: float = 0.12,
        flip_min_votes: int = 5,
        stale_track_seconds: float = 25.0,
    ):
        self.enabled = bool(enabled)
        self.model_dir = Path(model_dir)
        self.auto_download = bool(auto_download)
        self.sample_every_n_frames = max(1, int(sample_every_n_frames))
        self.vote_window = max(3, int(vote_window))
        self.min_votes = max(1, int(min_votes))
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.1, 0.95))
        self.aggregate_confidence_threshold = float(
            np.clip(aggregate_confidence_threshold, 0.5, 0.99)
        )
        self.female_confidence_threshold = float(
            np.clip(max(self.confidence_threshold, female_confidence_threshold), 0.1, 0.99)
        )
        self.lock_confidence_threshold = float(np.clip(lock_confidence_threshold, 0.5, 0.99))
        self.lock_min_votes = max(self.min_votes, int(lock_min_votes))
        self.flip_margin = float(np.clip(flip_margin, 0.02, 0.5))
        self.flip_min_votes = max(1, int(flip_min_votes))
        self.stale_track_seconds = max(5.0, float(stale_track_seconds))

        self.model_prototxt_path = (
            Path(model_prototxt_path).expanduser()
            if model_prototxt_path
            else self.model_dir / "deploy_gender.prototxt"
        )
        self.model_weights_path = (
            Path(model_weights_path).expanduser()
            if model_weights_path
            else self.model_dir / "gender_net.caffemodel"
        )

        self.face_cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        self.gender_net: Optional[cv2.dnn.Net] = None
        self.track_votes: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_locked: Dict[int, TrackGenderState] = {}
        self.track_last_seen: Dict[int, float] = {}

        if not self.enabled:
            return

        if self.face_cascade.empty():
            logger.warning("No se pudo cargar Haar cascade de rostro. Se desactiva género aparente.")
            self.enabled = False
            return

        if not self._ensure_model_files():
            self.enabled = False
            return

        try:
            self.gender_net = cv2.dnn.readNetFromCaffe(
                str(self.model_prototxt_path),
                str(self.model_weights_path),
            )
            logger.info(
                (
                    "✓ Género aparente habilitado "
                    "(sample=%s, min_votes=%s, conf=%.2f, agg_conf=%.2f, female_conf=%.2f)"
                ),
                self.sample_every_n_frames,
                self.min_votes,
                self.confidence_threshold,
                self.aggregate_confidence_threshold,
                self.female_confidence_threshold,
            )
        except Exception as exc:
            logger.warning("No se pudo cargar modelo de género aparente: %s", exc)
            self.enabled = False

    def _ensure_model_files(self) -> bool:
        if self.model_prototxt_path.exists() and self.model_weights_path.exists():
            return True

        if not self.auto_download:
            logger.warning(
                "Modelos de género no encontrados y auto_download deshabilitado (%s, %s)",
                self.model_prototxt_path,
                self.model_weights_path,
            )
            return False

        self.model_dir.mkdir(parents=True, exist_ok=True)
        ssl_context = self._build_ssl_context()
        try:
            if not self.model_prototxt_path.exists():
                logger.info("Descargando modelo de género (prototxt)...")
                self._download_model_file(
                    url=DEFAULT_GENDER_PROTOTXT_URL,
                    destination=self.model_prototxt_path,
                    ssl_context=ssl_context,
                )
            if not self.model_weights_path.exists():
                logger.info("Descargando modelo de género (weights)...")
                self._download_model_file(
                    url=DEFAULT_GENDER_WEIGHTS_URL,
                    destination=self.model_weights_path,
                    ssl_context=ssl_context,
                )
            return True
        except Exception as exc:
            logger.warning(
                "Descarga de modelos de género falló; se desactiva estimador. Error: %s",
                exc,
            )
            return False

    def _build_ssl_context(self) -> ssl.SSLContext:
        try:
            import certifi  # type: ignore

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    def _download_model_file(
        self,
        url: str,
        destination: Path,
        ssl_context: ssl.SSLContext,
    ) -> None:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "traffic-analysis-system/edge-backend"},
        )
        with urllib.request.urlopen(request, context=ssl_context, timeout=60) as response:
            with open(destination, "wb") as output_file:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    output_file.write(chunk)

    def _extract_best_face(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        if person_crop is None or person_crop.size == 0:
            return None

        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(28, 28),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda face: int(face[2] * face[3]))
        x2 = min(person_crop.shape[1], x + w)
        y2 = min(person_crop.shape[0], y + h)
        if x2 <= x or y2 <= y:
            return None
        return person_crop[y:y2, x:x2]

    def _predict_face_gender(self, face_bgr: np.ndarray) -> tuple[str, float]:
        if self.gender_net is None:
            return "unknown", 0.0

        blob = cv2.dnn.blobFromImage(
            face_bgr,
            scalefactor=1.0,
            size=(227, 227),
            mean=self.MEAN_VALUES,
            swapRB=False,
            crop=False,
        )
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()
        if preds is None or preds.size < 2:
            return "unknown", 0.0

        probs = preds.flatten()
        idx = int(np.argmax(probs))
        confidence = float(np.clip(probs[idx], 0.0, 1.0))
        label = self.GENDER_LABELS[idx] if idx < len(self.GENDER_LABELS) else "unknown"

        if confidence < self.confidence_threshold:
            return "unknown", confidence
        if label == "female" and confidence < self.female_confidence_threshold:
            return "unknown", confidence
        return label, confidence

    def _aggregate_track_state(self, track_id: int) -> TrackGenderState:
        history = self.track_votes.get(track_id)
        if not history:
            return self._stabilize_track_state(
                track_id,
                current_state=TrackGenderState(),
                label_votes={"male": 0, "female": 0},
            )

        weights = {"male": 0.0, "female": 0.0, "unknown": 0.0}
        label_votes = {"male": 0, "female": 0}
        for label, confidence in history:
            safe_label = label if label in weights else "unknown"
            if safe_label == "unknown":
                weights[safe_label] += 0.12
            else:
                weights[safe_label] += max(0.05, float(confidence))
                label_votes[safe_label] += 1

        votes = len(history)
        if votes < self.min_votes:
            return self._stabilize_track_state(
                track_id,
                current_state=TrackGenderState(label="unknown", confidence=0.0, votes=votes),
                label_votes=label_votes,
            )

        gender_total = weights["male"] + weights["female"]
        if gender_total <= 1e-6:
            return self._stabilize_track_state(
                track_id,
                current_state=TrackGenderState(label="unknown", confidence=0.0, votes=votes),
                label_votes=label_votes,
            )

        label = "male" if weights["male"] >= weights["female"] else "female"
        confidence = float(max(weights["male"], weights["female"]) / gender_total)
        if confidence < self.aggregate_confidence_threshold:
            return self._stabilize_track_state(
                track_id,
                current_state=TrackGenderState(label="unknown", confidence=confidence, votes=votes),
                label_votes=label_votes,
            )

        return self._stabilize_track_state(
            track_id,
            current_state=TrackGenderState(label=label, confidence=confidence, votes=votes),
            label_votes=label_votes,
        )

    def _stabilize_track_state(
        self,
        track_id: int,
        current_state: TrackGenderState,
        label_votes: Dict[str, int],
    ) -> TrackGenderState:
        locked_state = self.track_locked.get(track_id)

        if current_state.label in ("male", "female"):
            can_lock_current = (
                current_state.confidence >= self.lock_confidence_threshold
                and current_state.votes >= self.lock_min_votes
            )

            if locked_state is None:
                if can_lock_current:
                    self.track_locked[track_id] = TrackGenderState(
                        label=current_state.label,
                        confidence=current_state.confidence,
                        votes=current_state.votes,
                    )
                return current_state

            if current_state.label == locked_state.label:
                if can_lock_current:
                    self.track_locked[track_id] = TrackGenderState(
                        label=current_state.label,
                        confidence=current_state.confidence,
                        votes=current_state.votes,
                    )
                return current_state

            # Histéresis: solo permitir flip con margen y votos acumulados suficientes.
            contender_votes = int(label_votes.get(current_state.label, 0))
            can_flip = (
                contender_votes >= self.flip_min_votes
                and (current_state.confidence - locked_state.confidence) >= self.flip_margin
            )
            if can_flip:
                self.track_locked[track_id] = TrackGenderState(
                    label=current_state.label,
                    confidence=current_state.confidence,
                    votes=current_state.votes,
                )
                return current_state

        if locked_state is not None:
            return TrackGenderState(
                label=locked_state.label,
                confidence=locked_state.confidence,
                votes=current_state.votes,
            )
        return current_state

    def _evict_stale_tracks(self, now_ts: float):
        stale_ids = [
            track_id
            for track_id, last_seen in self.track_last_seen.items()
            if (now_ts - last_seen) > self.stale_track_seconds
        ]
        for track_id in stale_ids:
            self.track_last_seen.pop(track_id, None)
            self.track_votes.pop(track_id, None)
            self.track_locked.pop(track_id, None)

    def classify_tracks(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_index: int,
        now_ts: Optional[float] = None,
    ) -> Dict[int, TrackGenderState]:
        results: Dict[int, TrackGenderState] = {}
        if not self.enabled or self.gender_net is None:
            return results
        if frame is None or frame.size == 0:
            return results

        ts = time.time() if now_ts is None else float(now_ts)
        frame_h, frame_w = frame.shape[:2]

        for det in detections:
            if det.track_id is None:
                continue

            track_id = int(det.track_id)
            self.track_last_seen[track_id] = ts
            should_sample = (
                track_id not in self.track_votes
                or (frame_index % self.sample_every_n_frames == 0)
            )

            if should_sample:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                x1 = max(0, min(frame_w - 1, x1))
                x2 = max(0, min(frame_w, x2))
                y1 = max(0, min(frame_h - 1, y1))
                y2 = max(0, min(frame_h, y2))
                if x2 > x1 and y2 > y1:
                    person_crop = frame[y1:y2, x1:x2]
                    face = self._extract_best_face(person_crop)
                    if face is not None:
                        label, confidence = self._predict_face_gender(face)
                        self.track_votes[track_id].append((label, confidence))
                    else:
                        self.track_votes[track_id].append(("unknown", 0.0))

            results[track_id] = self._aggregate_track_state(track_id)

        self._evict_stale_tracks(ts)
        return results
