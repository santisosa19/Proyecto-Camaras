"""
Acumulador de mapa de calor de ocupación basado en centroides detectados.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


class OccupancyHeatmap:
    """Construye un mapa de calor acumulado por celda a lo largo del tiempo."""

    def __init__(
        self,
        camera_id: str,
        cell_size: int = 24,
        overlay_alpha: float = 0.35,
        blur_kernel: int = 21,
        decay_per_second: float = 0.0,
        output_dir: str = "heatmaps",
        metadata: Optional[Dict] = None,
        anchor_y_ratio: float = 0.88,
        trail_enabled: bool = True,
        trail_stale_seconds: float = 2.5,
        min_sample_weight: float = 0.02,
    ):
        self.camera_id = camera_id
        self.cell_size = max(4, int(cell_size))
        self.overlay_alpha = float(np.clip(overlay_alpha, 0.05, 0.95))
        self.blur_kernel = max(1, int(blur_kernel))
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.decay_per_second = max(0.0, float(decay_per_second))
        self.output_dir = Path(output_dir)
        self.metadata = metadata or {}
        self.anchor_y_ratio = float(np.clip(anchor_y_ratio, 0.5, 0.98))
        self.trail_enabled = bool(trail_enabled)
        self.trail_stale_seconds = max(0.5, float(trail_stale_seconds))
        self.min_sample_weight = max(1e-3, float(min_sample_weight))

        self.grid: Optional[np.ndarray] = None
        self.grid_h = 0
        self.grid_w = 0
        self.frame_w = 0
        self.frame_h = 0
        self.total_samples = 0
        self.total_weight = 0.0
        self.last_update_ts: Optional[float] = None
        self.track_last_cell: Dict[int, Tuple[int, int]] = {}
        self.track_last_seen_ts: Dict[int, float] = {}

    def _evict_stale_tracks(self, now_ts: float):
        stale_ids = [
            track_id
            for track_id, seen_ts in self.track_last_seen_ts.items()
            if (now_ts - seen_ts) > self.trail_stale_seconds
        ]
        for track_id in stale_ids:
            self.track_last_seen_ts.pop(track_id, None)
            self.track_last_cell.pop(track_id, None)

    def _ensure_grid(self, frame_shape: tuple[int, int, int] | tuple[int, int]):
        frame_h, frame_w = frame_shape[:2]
        target_grid_w = max(1, int(np.ceil(frame_w / self.cell_size)))
        target_grid_h = max(1, int(np.ceil(frame_h / self.cell_size)))

        if (
            self.grid is not None
            and target_grid_w == self.grid_w
            and target_grid_h == self.grid_h
        ):
            self.frame_w = frame_w
            self.frame_h = frame_h
            return

        self.frame_w = frame_w
        self.frame_h = frame_h
        self.grid_w = target_grid_w
        self.grid_h = target_grid_h
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        logger.info(
            "Heatmap inicializado: frame=%sx%s grid=%sx%s cell=%s",
            frame_w,
            frame_h,
            self.grid_w,
            self.grid_h,
            self.cell_size,
        )

    def reset(self):
        if self.grid is not None:
            self.grid.fill(0.0)
        self.total_samples = 0
        self.total_weight = 0.0
        self.last_update_ts = None
        self.track_last_cell.clear()
        self.track_last_seen_ts.clear()

    def update(
        self,
        detections: List[Detection],
        frame_shape: tuple[int, int, int] | tuple[int, int],
        timestamp: Optional[float] = None,
    ):
        self._ensure_grid(frame_shape)
        if self.grid is None:
            return

        now = time.time() if timestamp is None else float(timestamp)
        if self.last_update_ts is None:
            delta_t = 1.0 / 15.0
        else:
            delta_t = float(np.clip(now - self.last_update_ts, 0.0, 1.0))
        self.last_update_ts = now

        if self.decay_per_second > 0 and delta_t > 0:
            decay = max(0.0, 1.0 - self.decay_per_second * delta_t)
            self.grid *= decay

        if not detections:
            self._evict_stale_tracks(now)
            return

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            anchor_x = (float(x1) + float(x2)) / 2.0
            anchor_y = float(y1) + (float(y2) - float(y1)) * self.anchor_y_ratio

            x = int(np.clip(anchor_x, 0, self.frame_w - 1))
            y = int(np.clip(anchor_y, 0, self.frame_h - 1))
            cell_x = min(self.grid_w - 1, x // self.cell_size)
            cell_y = min(self.grid_h - 1, y // self.cell_size)

            conf_weight = float(np.clip(det.confidence, 0.1, 1.0))
            weight = max(self.min_sample_weight, conf_weight * max(delta_t, 1e-3))
            self.grid[cell_y, cell_x] += weight

            if self.trail_enabled and det.track_id is not None:
                track_id = int(det.track_id)
                prev_cell = self.track_last_cell.get(track_id)
                if prev_cell is not None and prev_cell != (cell_x, cell_y):
                    cv2.line(
                        self.grid,
                        prev_cell,
                        (cell_x, cell_y),
                        color=float(weight * 0.7),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                self.track_last_cell[track_id] = (cell_x, cell_y)
                self.track_last_seen_ts[track_id] = now

            self.total_samples += 1
            self.total_weight += weight

        self._evict_stale_tracks(now)

    def _build_normalized_heat(self) -> Optional[np.ndarray]:
        if self.grid is None:
            return None

        grid_max = float(np.max(self.grid))
        if grid_max <= 0:
            return None

        heat = cv2.resize(
            self.grid,
            (self.frame_w, self.frame_h),
            interpolation=cv2.INTER_CUBIC,
        )
        non_zero = heat[heat > 0]
        if non_zero.size == 0:
            return None

        high = float(np.percentile(non_zero, 99))
        if high <= 0:
            return None

        norm = np.clip(heat / high, 0.0, 1.0)
        return (norm * 255).astype(np.uint8)

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.grid is None:
            return frame

        heat_u8 = self._build_normalized_heat()
        if heat_u8 is None:
            return frame

        if self.blur_kernel > 1:
            heat_u8 = cv2.GaussianBlur(
                heat_u8,
                (self.blur_kernel, self.blur_kernel),
                0,
            )

        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        output = frame.copy()

        mask = heat_u8 > 3
        output[mask] = cv2.addWeighted(
            frame[mask],
            1.0 - self.overlay_alpha,
            colored[mask],
            self.overlay_alpha,
            0.0,
        )
        return output

    def export_overlay_png_base64(self) -> Optional[str]:
        """
        Exportar capa de heatmap como PNG BGRA (con alpha) codificada en base64.
        Útil para render en frontend sobre imagen de cámara.
        """
        heat_u8 = self._build_normalized_heat()
        if heat_u8 is None:
            return None

        if self.blur_kernel > 1:
            heat_u8 = cv2.GaussianBlur(
                heat_u8,
                (self.blur_kernel, self.blur_kernel),
                0,
            )

        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
        alpha = np.clip((heat_u8.astype(np.float32) / 255.0) * 220.0, 0, 255).astype(np.uint8)
        bgra = np.dstack((colored, alpha))
        ok, encoded = cv2.imencode(".png", bgra)
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def get_stats(self) -> Dict:
        if self.grid is None:
            return {
                "enabled": False,
                "grid_w": 0,
                "grid_h": 0,
                "max_value": 0.0,
                "sum_value": 0.0,
                "hotspot": None,
                "samples": self.total_samples,
                "total_weight": self.total_weight,
            }

        max_value = float(np.max(self.grid))
        sum_value = float(np.sum(self.grid))
        if max_value > 0:
            max_pos = np.unravel_index(int(np.argmax(self.grid)), self.grid.shape)
            center_x = int((max_pos[1] + 0.5) * self.cell_size)
            center_y = int((max_pos[0] + 0.5) * self.cell_size)
            hotspot = {"x": center_x, "y": center_y}
        else:
            hotspot = None

        return {
            "enabled": True,
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "max_value": max_value,
            "sum_value": sum_value,
            "hotspot": hotspot,
            "samples": self.total_samples,
            "total_weight": self.total_weight,
        }

    def snapshot(self) -> Dict:
        stats = self.get_stats()
        payload = {
            "camera_id": self.camera_id,
            "generated_at": datetime.utcnow().isoformat(),
            "frame_width": self.frame_w,
            "frame_height": self.frame_h,
            "cell_size": self.cell_size,
            "grid_width": self.grid_w,
            "grid_height": self.grid_h,
            "stats": stats,
            "metadata": self.metadata,
            "grid": self.grid.tolist() if self.grid is not None else [],
        }
        return payload

    def save_snapshot(self, keep_history: bool = False) -> Optional[Path]:
        if self.grid is None:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()

        latest_path = self.output_dir / f"{self.camera_id}_heatmap_latest.json"
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        if keep_history:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            history_path = self.output_dir / f"{self.camera_id}_heatmap_{ts}.json"
            with history_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

        return latest_path
