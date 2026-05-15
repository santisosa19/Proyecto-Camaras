#!/usr/bin/env python3
"""
Script principal para ejecutar el sistema de análisis de tráfico
con cámara RTSP real

MODO CONFIGURACIÓN:
    Al iniciar, presiona 'c' para configurar líneas
    - Click para marcar puntos de las líneas
    - 's' para guardar configuración
    - 'q' para continuar sin configurar

MODO DETECCIÓN:
    - 'q' para salir
    - 'c' para reconfigurar líneas
"""
import sys
import time
import base64
import cv2
import logging
import json
import os
from collections import deque
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit
import httpx
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

from app.services.video_capture import VideoCapture
from app.services.detector import PersonDetector
from app.services.counter import PersonCounter
from app.services.heatmap import OccupancyHeatmap
from app.services.apparent_gender import ApparentGenderEstimator


# Archivo de configuración de líneas
LINES_CONFIG_FILE = Path(__file__).parent / "lines_config.json"


def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} debe ser un entero válido. Valor recibido: '{raw}'") from exc


def sanitize_rtsp_url(url: str) -> str:
    """
    Oculta credenciales en logs de URLs RTSP/HTTP.
    """
    if not url:
        return ""
    parts = urlsplit(url)
    if parts.username is None:
        return url
    auth = "***"
    if parts.password is not None:
        auth = "***:***"
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    netloc = f"{auth}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class LineConfigurator:
    """Configurador interactivo de líneas de conteo"""
    
    def __init__(self, camera_id: str, capture: VideoCapture, entry_direction: str = "positive"):
        self.camera_id = camera_id
        self.capture = capture
        self.entry_direction = entry_direction
        self.points = []
        self.lines = []
        self.current_frame = None
        self.window_name = "Line Configuration"
        
    def configure(self):
        """Modo interactivo de configuración"""
        logger.info("="*60)
        logger.info("MODO CONFIGURACIÓN DE LÍNEAS")
        logger.info("="*60)
        logger.info(f"Dirección configurada como ENTRADA: {self.entry_direction}")
        logger.info("Instrucciones:")
        logger.info("  1. Click para marcar primer punto de la línea principal")
        logger.info("  2. Click para marcar segundo punto de la línea principal")
        logger.info("  3. Presiona 'd' para cambiar dirección de ENTRADA (positive/negative)")
        logger.info("  4. Presiona 's' para GUARDAR")
        logger.info("  5. Presiona 'r' para REINICIAR")
        logger.info("  6. Presiona 'q' para SALIR sin guardar")
        logger.info("="*60)
        
        # Configurar callback del mouse en una sola ventana con título ASCII.
        # En sesiones remotas de Windows (AnyDesk/RDP), títulos con acentos
        # pueden terminar creando ventanas duplicadas y perder eventos de click.
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        try:
            cv2.destroyWindow("Configuración de Líneas")
        except cv2.error:
            pass
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        while True:
            # Capturar frame
            frame = self.capture.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            self.current_frame = frame.copy()
            display_frame = frame.copy()
            
            # Dibujar puntos temporales
            for i, point in enumerate(self.points):
                cv2.circle(display_frame, point, 5, (255, 255, 0), -1)
                cv2.putText(
                    display_frame, f"P{i+1}", 
                    (point[0]+10, point[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                )
            
            # Dibujar líneas ya configuradas
            for line in self.lines:
                color = (0, 255, 255)
                cv2.line(display_frame, tuple(line['p1']), tuple(line['p2']), color, 3)
                
                # Label
                mid_x = (line['p1'][0] + line['p2'][0]) // 2
                mid_y = (line['p1'][1] + line['p2'][1]) // 2
                cv2.putText(
                    display_frame, line['name'].upper(),
                    (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
            
            # Instrucciones en pantalla
            status = f"Líneas configuradas: {len(self.lines)}/1"
            if len(self.points) > 0:
                status += f" | Puntos: {len(self.points)}/2"
            
            cv2.putText(
                display_frame, status,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            cv2.putText(
                display_frame, "s=Guardar | r=Reiniciar | q=Salir",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                display_frame,
                f"Direccion ENTRADA: {self.entry_direction} (d=toggle)",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            cv2.imshow(self.window_name, display_frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Guardar
                if len(self.lines) >= 1:
                    self._save_configuration()
                    logger.info("✓ Configuración guardada")
                    cv2.destroyWindow(self.window_name)
                    return self.lines
                else:
                    logger.warning("⚠ Configura al menos una línea antes de guardar")
            
            elif key == ord('r'):  # Reiniciar
                self.points = []
                self.lines = []
                logger.info("Configuración reiniciada")

            elif key == ord('d'):
                self.entry_direction = "negative" if self.entry_direction == "positive" else "positive"
                logger.info(f"Dirección de ENTRADA cambiada a: {self.entry_direction}")
            
            elif key == ord('q'):  # Salir sin guardar
                logger.info("Saliendo sin guardar configuración")
                cv2.destroyWindow(self.window_name)
                return None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para clicks del mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            logger.info(f"Punto {len(self.points)} agregado: ({x}, {y})")
            
            # Si completamos 2 puntos, crear línea
            if len(self.points) == 2:
                self.lines = [{
                    "name": "main_gate",
                    "p1": list(self.points[0]),
                    "p2": list(self.points[1]),
                    "direction": "both"
                }]
                
                logger.info("✓ Línea principal creada")
                self.points = []  # Resetear puntos
    
    def _save_configuration(self):
        """Guardar configuración en archivo JSON"""
        # Cargar configuración existente
        config = {}
        if LINES_CONFIG_FILE.exists():
            with open(LINES_CONFIG_FILE, 'r') as f:
                config = json.load(f)
        
        # Actualizar con nueva configuración
        config[self.camera_id] = {
            "lines": self.lines,
            "entry_direction": self.entry_direction,
            "configured_at": datetime.now().isoformat()
        }
        
        # Guardar
        with open(LINES_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuración guardada en: {LINES_CONFIG_FILE}")


def load_lines_configuration(camera_id: str):
    """Cargar configuración de líneas desde archivo"""
    if not LINES_CONFIG_FILE.exists():
        return None
    
    try:
        with open(LINES_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        if camera_id in config:
            logger.info(f"✓ Configuración de líneas cargada para {camera_id}")
            camera_cfg = config[camera_id]
            lines = camera_cfg.get("lines", [])
            return {
                "lines": lines[:1],
                "entry_direction": camera_cfg.get("entry_direction", "positive")
            }
        else:
            return None
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        return None


class RemoteIngestClient:
    """Cliente HTTP para enviar eventos/snapshots al backend central."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 5.0,
        max_batch_size: int = 200,
        max_queue_size: int = 10000,
        crossing_flush_interval: float = 1.0,
        detection_flush_interval: float = 5.0,
        heatmap_flush_interval: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_batch_size = max_batch_size
        self.max_queue_size = max(1, max_queue_size)
        self.crossing_flush_interval = crossing_flush_interval
        self.detection_flush_interval = detection_flush_interval
        self.heatmap_flush_interval = heatmap_flush_interval
        self.pending_crossings: deque = deque()
        self.pending_detections: deque = deque()
        self.pending_heatmaps: deque = deque()
        self.last_crossing_flush = 0.0
        self.last_detection_flush = 0.0
        self.last_heatmap_flush = 0.0
        self.dropped_crossings = 0
        self.dropped_detections = 0
        self.dropped_heatmaps = 0

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        self.client = httpx.Client(
            timeout=timeout_seconds,
            headers=headers
        )

    def enqueue_crossings(self, events: list[dict]):
        for event in events:
            if len(self.pending_crossings) >= self.max_queue_size:
                self.pending_crossings.popleft()
                self.dropped_crossings += 1
                if self.dropped_crossings % 100 == 1:
                    logger.warning(
                        "Cola de crossings llena; se descartaron %s eventos",
                        self.dropped_crossings,
                    )
            self.pending_crossings.append(event)

    def enqueue_detection(self, item: dict):
        if len(self.pending_detections) >= self.max_queue_size:
            self.pending_detections.popleft()
            self.dropped_detections += 1
            if self.dropped_detections % 100 == 1:
                logger.warning(
                    "Cola de detecciones llena; se descartaron %s snapshots",
                    self.dropped_detections,
                )
        self.pending_detections.append(item)

    def enqueue_heatmap(self, item: dict):
        if len(self.pending_heatmaps) >= self.max_queue_size:
            self.pending_heatmaps.popleft()
            self.dropped_heatmaps += 1
            if self.dropped_heatmaps % 20 == 1:
                logger.warning(
                    "Cola de heatmaps llena; se descartaron %s payloads",
                    self.dropped_heatmaps,
                )
        self.pending_heatmaps.append(item)

    def _post_batch(self, path: str, key: str, batch: list[dict]) -> bool:
        payload = {key: batch}
        url = f"{self.base_url}{path}"
        try:
            response = self.client.post(url, json=payload)
            if response.status_code >= 400:
                logger.error(
                    f"Error enviando lote a {path}: HTTP {response.status_code} - {response.text}"
                )
                return False
            return True
        except Exception as exc:
            logger.error(f"Error de red enviando lote a {path}: {exc}")
            return False

    def _flush_queue(self, queue: deque, path: str, key: str, flush_all: bool = False):
        if not queue:
            return

        while queue:
            batch = []
            while queue and len(batch) < self.max_batch_size:
                batch.append(queue.popleft())

            if not self._post_batch(path=path, key=key, batch=batch):
                # Reinsertar al frente respetando orden original.
                for item in reversed(batch):
                    queue.appendleft(item)
                break

            if not flush_all:
                break

    def flush(self, force: bool = False):
        now = time.time()
        should_flush_crossings = force or (now - self.last_crossing_flush >= self.crossing_flush_interval)
        should_flush_detections = force or (now - self.last_detection_flush >= self.detection_flush_interval)
        should_flush_heatmaps = force or (now - self.last_heatmap_flush >= self.heatmap_flush_interval)

        if should_flush_crossings and self.pending_crossings:
            self._flush_queue(
                queue=self.pending_crossings,
                path="/api/v1/ingest/crossings",
                key="events",
                flush_all=force
            )
            self.last_crossing_flush = now

        if should_flush_detections and self.pending_detections:
            self._flush_queue(
                queue=self.pending_detections,
                path="/api/v1/ingest/detections",
                key="items",
                flush_all=force
            )
            self.last_detection_flush = now

        if should_flush_heatmaps and self.pending_heatmaps:
            self._flush_queue(
                queue=self.pending_heatmaps,
                path="/api/v1/ingest/heatmaps",
                key="items",
                flush_all=force
            )
            self.last_heatmap_flush = now

    def close(self):
        self.client.close()


class TrafficAnalysisSystem:
    """Sistema completo de análisis de tráfico"""
    
    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        rtsp_url: str,
        entry_direction: str = "positive",
        show_window: bool = True,
        save_to_db: bool = True,
        save_to_api: bool = False,
        remote_api_base_url: str = "",
        remote_api_key: str = "",
        max_ingest_queue_size: int = 10000,
        branch_id: str = "",
        branch_name: str = "",
    ):
        """
        Args:
            camera_id: ID único de la cámara
            camera_name: Nombre descriptivo
            rtsp_url: URL RTSP de la cámara
            entry_direction: Dirección que se considera ENTRADA (positive|negative)
            show_window: Si True, muestra ventana con video
            save_to_db: Si True, guarda datos en MySQL
            save_to_api: Si True, envía datos al backend central por API
            remote_api_base_url: URL base del backend central
            remote_api_key: API key del endpoint de ingesta
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.entry_direction = self._normalize_entry_direction(entry_direction)
        self.edge_id = env_str("EDGE_ID", camera_id)
        self.payload_version = env_str("PAYLOAD_VERSION", "1.0")
        self.show_window = show_window
        self.save_to_db = save_to_db
        self.save_to_api = save_to_api
        self.remote_api_base_url = remote_api_base_url.strip()
        self.remote_api_key = remote_api_key.strip()
        self.max_ingest_queue_size = max(1, int(max_ingest_queue_size))
        self.branch_id = branch_id.strip()
        self.branch_name = branch_name.strip()
        self.remote_ingest: RemoteIngestClient | None = None
        self.heatmap: OccupancyHeatmap | None = None
        self.hourly_heatmap: OccupancyHeatmap | None = None
        self.current_heatmap_hour_start: datetime | None = None
        self.gender_estimator: ApparentGenderEstimator | None = None
        self.latest_track_gender: dict[int, dict] = {}

        # Configuración de mapa de calor
        self.heatmap_enabled = os.getenv("HEATMAP_ENABLED", "true").lower() == "true"
        self.show_heatmap_overlay = os.getenv("SHOW_HEATMAP_OVERLAY", "true").lower() == "true"
        self.save_heatmap_snapshots = os.getenv("SAVE_HEATMAP_SNAPSHOTS", "true").lower() == "true"
        self.heatmap_keep_history = os.getenv("HEATMAP_KEEP_HISTORY", "false").lower() == "true"
        self.heatmap_snapshot_interval_seconds = float(
            os.getenv("HEATMAP_SNAPSHOT_INTERVAL_SECONDS", "60")
        )
        self.heatmap_snapshot_interval_seconds = max(5.0, self.heatmap_snapshot_interval_seconds)
        self.last_heatmap_snapshot = 0.0
        self.send_hourly_heatmap_to_api = os.getenv("SEND_HOURLY_HEATMAP_TO_API", "true").lower() == "true"
        self.hourly_heatmap_partial_flush_seconds = max(
            15.0,
            float(os.getenv("HOURLY_HEATMAP_PARTIAL_FLUSH_SECONDS", "60")),
        )
        self.last_hourly_heatmap_partial_flush = 0.0

        self.heatmap_background_max_width = max(160, int(os.getenv("HEATMAP_BACKGROUND_MAX_WIDTH", "960")))
        self.heatmap_background_jpeg_quality = int(os.getenv("HEATMAP_BACKGROUND_JPEG_QUALITY", "68"))
        self.heatmap_background_jpeg_quality = max(30, min(95, self.heatmap_background_jpeg_quality))
        self.heatmap_background_refresh_seconds = max(
            5.0,
            float(os.getenv("HEATMAP_BACKGROUND_REFRESH_SECONDS", "30"))
        )
        self.last_heatmap_background_update = 0.0
        self.latest_heatmap_background_base64: str | None = None
        self.entry_direction_force = env_bool("ENTRY_DIRECTION_FORCE", False)
        
        # Estadísticas
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'start_time': time.time()
        }
        
        logger.info(f"Inicializando sistema para cámara: {camera_name}")
        
        # Inicializar componentes
        self._init_components()

    @staticmethod
    def _normalize_entry_direction(value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in {"positive", "negative"}:
            raise ValueError("entry_direction debe ser 'positive' o 'negative'")
        return normalized
    
    def _init_components(self):
        """Inicializar todos los componentes"""
        try:
            if self.save_to_db:
                from app.database.connection import init_db
                init_db()
                logger.info("✓ Esquema de base de datos verificado")

            if self.save_to_api:
                if not self.remote_api_base_url:
                    raise ValueError("remote_api_base_url es obligatorio cuando save_to_api=True")

                self.remote_ingest = RemoteIngestClient(
                    base_url=self.remote_api_base_url,
                    api_key=self.remote_api_key,
                    timeout_seconds=5.0,
                    max_queue_size=self.max_ingest_queue_size,
                )
                logger.info(f"✓ Ingesta remota habilitada: {self.remote_api_base_url}")

            # 1. Captura de video
            logger.info(f"Conectando a cámara: {self.rtsp_url}")
            self.capture = VideoCapture(
                camera_id=self.camera_id,
                rtsp_url=self.rtsp_url,
                fps=15
            )
            
            if not self.capture.connect():
                raise Exception("No se pudo conectar a la cámara")
            
            logger.info("✓ Cámara conectada")
            
            # 2. Detector YOLO
            logger.info("Cargando detector YOLOv8...")
            yolo_model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt").strip() or "yolov8n.pt"
            yolo_confidence = float(os.getenv("YOLO_CONFIDENCE", "0.22"))
            yolo_iou = float(os.getenv("YOLO_IOU", "0.60"))
            yolo_image_size = int(os.getenv("YOLO_IMAGE_SIZE", "960"))
            yolo_max_det = int(os.getenv("YOLO_MAX_DETECTIONS", "120"))
            yolo_tracker = os.getenv("YOLO_TRACKER", "trackers/bytetrack_stable.yaml").strip() or "trackers/bytetrack_stable.yaml"
            yolo_device = os.getenv("YOLO_DEVICE", "auto").strip() or "auto"
            if not os.path.isabs(yolo_tracker):
                tracker_candidate = Path(__file__).parent / yolo_tracker
                if tracker_candidate.exists():
                    yolo_tracker = str(tracker_candidate)
            self.detector = PersonDetector(
                model_path=yolo_model_path,
                confidence_threshold=yolo_confidence,
                iou_threshold=yolo_iou,
                device=yolo_device,
                image_size=yolo_image_size,
                max_detections=yolo_max_det,
                tracker_config=yolo_tracker
            )
            logger.info("✓ Detector cargado")
            
            # 3. Contador
            logger.info("Inicializando contador...")
            self.counter = PersonCounter(camera_id=self.camera_id)
            self.counter.crossing_cooldown_seconds = float(os.getenv("CROSSING_COOLDOWN_SECONDS", "0.4"))
            self.counter.crossing_merge_distance = float(os.getenv("CROSSING_MERGE_DISTANCE", "6.0"))
            self.counter.max_track_age = float(os.getenv("MAX_TRACK_AGE_SECONDS", "8.0"))
            logger.info(
                "Counter params: cooldown=%.2fs merge_distance=%.1f max_track_age=%.1fs",
                self.counter.crossing_cooldown_seconds,
                self.counter.crossing_merge_distance,
                self.counter.max_track_age,
            )
            
            # Cargar o configurar líneas
            loaded_config = load_lines_configuration(self.camera_id)

            if loaded_config is None:
                logger.info("No hay configuración guardada")
                logger.info("Presiona 'c' en cualquier momento para configurar líneas")
                # Líneas por defecto (temporales)
                self.counter.add_line(
                    name="main_gate",
                    p1=(0, 240),
                    p2=(640, 240),
                    direction="both"
                )
            else:
                loaded_entry_direction = loaded_config.get("entry_direction", "positive")
                if self.entry_direction_force:
                    logger.info(
                        "ENTRY_DIRECTION_FORCE activo: se mantiene '%s' y se ignora '%s' de lines_config.json",
                        self.entry_direction,
                        loaded_entry_direction,
                    )
                else:
                    self.entry_direction = self._normalize_entry_direction(loaded_entry_direction)
                self.lines_config = loaded_config.get("lines", [])
                # Usar configuración guardada
                for line_config in self.lines_config:
                    self.counter.add_line(
                        name=line_config['name'],
                        p1=tuple(line_config['p1']),
                        p2=tuple(line_config['p2']),
                        direction=line_config.get('direction', 'both')
                    )
                logger.info(f"✓ {len(self.lines_config)} línea(s) configurada(s)")
            logger.info(f"Dirección de entrada configurada: {self.entry_direction}")
            
            logger.info("✓ Contador configurado")

            apparent_gender_enabled = os.getenv("APPARENT_GENDER_ENABLED", "false").lower() == "true"
            gender_model_dir = os.getenv("GENDER_MODEL_DIR", "models/gender").strip() or "models/gender"
            if not os.path.isabs(gender_model_dir):
                gender_model_dir = str(Path(__file__).parent / gender_model_dir)
            gender_model_prototxt = os.getenv("GENDER_MODEL_PROTOTXT", "").strip()
            gender_model_weights = os.getenv("GENDER_MODEL_WEIGHTS", "").strip()
            if gender_model_prototxt and not os.path.isabs(gender_model_prototxt):
                gender_model_prototxt = str(Path(__file__).parent / gender_model_prototxt)
            if gender_model_weights and not os.path.isabs(gender_model_weights):
                gender_model_weights = str(Path(__file__).parent / gender_model_weights)

            self.gender_estimator = ApparentGenderEstimator(
                enabled=apparent_gender_enabled,
                model_prototxt_path=gender_model_prototxt,
                model_weights_path=gender_model_weights,
                model_dir=gender_model_dir,
                auto_download=os.getenv("GENDER_AUTO_DOWNLOAD_MODEL", "true").lower() == "true",
                sample_every_n_frames=int(os.getenv("GENDER_SAMPLE_EVERY_N_FRAMES", "5")),
                vote_window=int(os.getenv("GENDER_VOTE_WINDOW", "12")),
                min_votes=int(os.getenv("GENDER_MIN_VOTES", "2")),
                confidence_threshold=float(os.getenv("GENDER_CONFIDENCE_THRESHOLD", "0.52")),
                stale_track_seconds=float(os.getenv("GENDER_STALE_TRACK_SECONDS", "25")),
            )
            if self.gender_estimator.enabled:
                logger.info("✓ Género aparente habilitado")
            else:
                logger.info("Género aparente deshabilitado")

            if self.heatmap_enabled:
                heatmap_output_dir = (
                    os.getenv("HEATMAP_OUTPUT_DIR", "heatmaps").strip() or "heatmaps"
                )
                if not os.path.isabs(heatmap_output_dir):
                    heatmap_output_dir = str(Path(__file__).parent / heatmap_output_dir)
                heatmap_cell_size = int(os.getenv("HEATMAP_CELL_SIZE", "24"))
                heatmap_overlay_alpha = float(os.getenv("HEATMAP_OVERLAY_ALPHA", "0.35"))
                heatmap_blur_kernel = int(os.getenv("HEATMAP_BLUR_KERNEL", "21"))
                heatmap_decay = float(os.getenv("HEATMAP_DECAY_PER_SECOND", "0.0"))
                heatmap_norm_percentile = float(os.getenv("HEATMAP_NORMALIZATION_PERCENTILE", "99"))
                heatmap_norm_rise_alpha = float(os.getenv("HEATMAP_NORMALIZATION_RISE_ALPHA", "0.08"))
                heatmap_norm_fall_alpha = float(os.getenv("HEATMAP_NORMALIZATION_FALL_ALPHA", "0.01"))
                heatmap_norm_gamma = float(os.getenv("HEATMAP_NORMALIZATION_GAMMA", "0.85"))
                heatmap_overlay_min_intensity = int(os.getenv("HEATMAP_OVERLAY_MIN_INTENSITY", "1"))

                self.heatmap = OccupancyHeatmap(
                    camera_id=self.camera_id,
                    cell_size=heatmap_cell_size,
                    overlay_alpha=heatmap_overlay_alpha,
                    blur_kernel=heatmap_blur_kernel,
                    decay_per_second=heatmap_decay,
                    output_dir=heatmap_output_dir,
                    normalization_percentile=heatmap_norm_percentile,
                    normalization_rise_alpha=heatmap_norm_rise_alpha,
                    normalization_fall_alpha=heatmap_norm_fall_alpha,
                    normalization_gamma=heatmap_norm_gamma,
                    overlay_min_intensity=heatmap_overlay_min_intensity,
                    metadata={
                        "camera_name": self.camera_name,
                        "branch_id": self.branch_id,
                        "branch_name": self.branch_name,
                    },
                )
                hourly_dir = str(Path(heatmap_output_dir) / "hourly")
                self.hourly_heatmap = OccupancyHeatmap(
                    camera_id=f"{self.camera_id}_hourly",
                    cell_size=heatmap_cell_size,
                    overlay_alpha=heatmap_overlay_alpha,
                    blur_kernel=heatmap_blur_kernel,
                    decay_per_second=0.0,
                    output_dir=hourly_dir,
                    normalization_percentile=heatmap_norm_percentile,
                    normalization_rise_alpha=heatmap_norm_rise_alpha,
                    normalization_fall_alpha=heatmap_norm_fall_alpha,
                    normalization_gamma=heatmap_norm_gamma,
                    overlay_min_intensity=heatmap_overlay_min_intensity,
                    metadata={
                        "camera_name": self.camera_name,
                        "branch_id": self.branch_id,
                        "branch_name": self.branch_name,
                    },
                )
                self.current_heatmap_hour_start = datetime.utcnow().replace(
                    minute=0, second=0, microsecond=0
                )
                logger.info(
                    "✓ Heatmap habilitado (overlay=%s, snapshots=%s, intervalo=%.0fs, hourly_api=%s, partial_flush=%.0fs)",
                    self.show_heatmap_overlay,
                    self.save_heatmap_snapshots,
                    self.heatmap_snapshot_interval_seconds,
                    self.send_hourly_heatmap_to_api,
                    self.hourly_heatmap_partial_flush_seconds,
                )
            
            logger.info("=" * 60)
            logger.info("SISTEMA LISTO")
            logger.info("  'q' = Salir")
            logger.info("  'c' = Configurar líneas")
            logger.info("  'h' = Mostrar/ocultar heatmap")
            logger.info("  'k' = Resetear heatmap")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error inicializando componentes: {e}")
            raise
    
    def run(self):
        """Ejecutar el sistema principal"""
        frame_count = 0
        
        try:
            while True:
                # Capturar frame
                frame = self.capture.get_frame()
                
                if frame is None:
                    logger.warning("Frame vacío, esperando...")
                    time.sleep(0.1)
                    continue
                
                # Detectar personas (con tracking)
                detections = self.detector.detect(frame, track=True)
                now_ts = time.time()
                now_dt = datetime.utcnow()
                if self.gender_estimator is not None and self.gender_estimator.enabled:
                    track_gender_states = self.gender_estimator.classify_tracks(
                        frame=frame,
                        detections=detections,
                        frame_index=frame_count,
                        now_ts=now_ts,
                    )
                    self._apply_track_genders(detections, track_gender_states)

                if self.hourly_heatmap is not None:
                    if now_ts - self.last_heatmap_background_update >= self.heatmap_background_refresh_seconds:
                        self.latest_heatmap_background_base64 = self._encode_background_frame(frame)
                        self.last_heatmap_background_update = now_ts

                if self.hourly_heatmap is not None and self.current_heatmap_hour_start is not None:
                    frame_hour_start = now_dt.replace(minute=0, second=0, microsecond=0)
                    if frame_hour_start > self.current_heatmap_hour_start:
                        self._flush_completed_hourly_heatmap(self.current_heatmap_hour_start)
                        self.hourly_heatmap.reset()
                        self.current_heatmap_hour_start = frame_hour_start
                        self.last_hourly_heatmap_partial_flush = now_ts

                if self.heatmap is not None:
                    self.heatmap.update(
                        detections=detections,
                        frame_shape=frame.shape,
                        timestamp=now_ts,
                    )
                if self.hourly_heatmap is not None:
                    self.hourly_heatmap.update(
                        detections=detections,
                        frame_shape=frame.shape,
                        timestamp=now_ts,
                    )
                    if (
                        self.current_heatmap_hour_start is not None
                        and self.save_to_api
                        and self.remote_ingest is not None
                        and self.send_hourly_heatmap_to_api
                        and (
                            now_ts - self.last_hourly_heatmap_partial_flush
                            >= self.hourly_heatmap_partial_flush_seconds
                        )
                    ):
                        self._flush_completed_hourly_heatmap(
                            self.current_heatmap_hour_start,
                            is_partial=True,
                        )
                        self.last_hourly_heatmap_partial_flush = now_ts
                
                # Actualizar contador
                count_stats = self.counter.update(detections)
                crossing_events = self.counter.pop_crossing_events()
                
                # Actualizar estadísticas
                self.stats['frames_processed'] += 1
                self.stats['total_detections'] += len(detections)
                frame_count += 1
                
                # Dibujar detecciones y líneas en el frame
                if self.show_window:
                    if self.heatmap is not None and self.show_heatmap_overlay:
                        frame = self.heatmap.render_overlay(frame)

                    # Dibujar bounding boxes
                    for det in detections:
                        x1, y1, x2, y2 = [int(x) for x in det.bbox]
                        
                        # Box verde
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Label con ID y confianza
                        gender_short = "?"
                        if det.apparent_gender == "male":
                            gender_short = "M"
                        elif det.apparent_gender == "female":
                            gender_short = "F"
                        track_label = det.track_id if det.track_id is not None else "-"
                        label = f"ID:{track_label} G:{gender_short} {det.confidence:.2f}"
                        cv2.putText(
                            frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                        
                        # Centroid
                        cx, cy = [int(x) for x in det.centroid]
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    
                    # Dibujar líneas de conteo
                    frame = self.counter.draw_lines(frame)
                    
                    # Info del sistema
                    fps = self.stats['frames_processed'] / (time.time() - self.stats['start_time'])
                    info_lines = [
                        f"FPS: {fps:.1f}",
                        f"Detectados: {len(detections)}",
                        f"Tracks activos: {count_stats['active_tracks']}",
                        f"ENTRADA => {self.entry_direction.upper()}",
                    ]
                    
                    # Agregar conteos de todas las líneas
                    for line_name, line_stats in count_stats.get('lines', {}).items():
                        info_lines.append(f"{line_name}: +{line_stats['positive']} -{line_stats['negative']}")

                    if self.heatmap is not None:
                        hm_stats = self.heatmap.get_stats()
                        info_lines.append(
                            f"Heat max: {hm_stats['max_value']:.2f} samples: {hm_stats['samples']}"
                        )
                    
                    # Dibujar info en pantalla
                    y_offset = 30
                    for line in info_lines:
                        cv2.putText(
                            frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                        )
                        y_offset += 30
                    
                    # Mostrar frame
                    cv2.imshow(f"Traffic Analysis - {self.camera_name}", frame)
                    
                    # Manejar teclas
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        logger.info("Saliendo...")
                        break
                    elif key == ord('c'):
                        logger.info("Entrando a modo configuración...")
                        self._configure_lines()
                        # Reiniciar contador con nuevas líneas
                        self._reload_counter()
                    elif key == ord('h') and self.heatmap is not None:
                        self.show_heatmap_overlay = not self.show_heatmap_overlay
                        logger.info(
                            "Heatmap overlay %s",
                            "habilitado" if self.show_heatmap_overlay else "deshabilitado",
                        )
                    elif key == ord('k') and self.heatmap is not None:
                        self.heatmap.reset()
                        if self.hourly_heatmap is not None:
                            self.hourly_heatmap.reset()
                            self.current_heatmap_hour_start = datetime.utcnow().replace(
                                minute=0, second=0, microsecond=0
                            )
                        logger.info("Heatmap reseteado")
                
                # Mostrar estadísticas en consola cada 30 frames
                if frame_count % 30 == 0:
                    self._print_stats(count_stats, len(detections))

                if crossing_events:
                    if self.save_to_db:
                        self._save_crossing_events(crossing_events)
                    if self.save_to_api:
                        self._queue_crossing_events(crossing_events)

                # Guardar snapshots livianos en base de datos cada ~60 frames
                if frame_count % 60 == 0:
                    if self.save_to_db:
                        logger.info(f"💾 Guardando DB: {len(detections)} personas")
                        self._save_to_database(detections)
                    if self.save_to_api:
                        self._queue_detection_snapshot(detections)

                if (
                    self.heatmap is not None
                    and self.save_heatmap_snapshots
                    and (now_ts - self.last_heatmap_snapshot >= self.heatmap_snapshot_interval_seconds)
                ):
                    self._save_heatmap_snapshot()
                    self.last_heatmap_snapshot = now_ts

                if self.save_to_api and self.remote_ingest is not None:
                    self.remote_ingest.flush()
                
        except KeyboardInterrupt:
            logger.info("Interrupción por teclado (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error en el loop principal: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def _print_stats(self, count_stats, num_detections):
        """Imprimir estadísticas en consola"""
        uptime = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / uptime if uptime > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"Cámara: {self.camera_name}")
        logger.info(f"Uptime: {uptime:.0f}s | FPS: {fps:.1f} | Frames: {self.stats['frames_processed']}")
        logger.info(f"Detectados ahora: {num_detections} | Tracks activos: {count_stats['active_tracks']}")
        
        # Mostrar conteos de todas las líneas
        for line_name, line_stats in count_stats.get('lines', {}).items():
            logger.info(
                f"{line_name} - Positive: {line_stats['positive']} | "
                f"Negative: {line_stats['negative']} | "
                f"Total: {line_stats['total']}"
            )

        if self.heatmap is not None:
            hm_stats = self.heatmap.get_stats()
            logger.info(
                "Heatmap - max: %.2f | sum: %.2f | samples: %s | hotspot: %s",
                hm_stats["max_value"],
                hm_stats["sum_value"],
                hm_stats["samples"],
                hm_stats["hotspot"],
            )
        
        logger.info("=" * 60)
    
    def _save_to_database(self, detections):
        """Guardar datos en MySQL"""
        try:
            from app.database.connection import get_db_context, DatabaseManager
            with get_db_context() as db:
                # Guardar detección solo si hay personas detectadas
                if len(detections) > 0:
                    detections_data = [det.to_dict() for det in detections]
                    DatabaseManager.save_detection(
                        db,
                        camera_id=self.camera_id,
                        person_count=len(detections),
                        detections_data=detections_data
                    )
                
                # Actualizar estado de cámara
                capture_stats = self.capture.get_stats()
                DatabaseManager.update_camera_status(
                    db,
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    rtsp_url=self.rtsp_url,
                    is_connected=capture_stats['is_connected'],
                    fps=self.stats['frames_processed'] / (time.time() - self.stats['start_time']),
                    total_frames=self.stats['frames_processed'],
                    error_count=capture_stats['error_count']
                )
                
                logger.debug("✓ Datos guardados en MySQL")
                
        except Exception as e:
            logger.error(f"Error guardando en base de datos: {e}")

    def _save_crossing_events(self, crossing_events):
        """Persistir eventos de entrada/salida."""
        try:
            from app.database.connection import get_db_context, DatabaseManager
            with get_db_context() as db:
                rows = []
                for event in crossing_events:
                    direction = (event.get('direction') or '').strip().lower()
                    if direction not in {"positive", "negative"}:
                        logger.warning("Evento de cruce con dirección inválida: %s", event)
                        continue
                    is_entry = direction == self.entry_direction
                    event_type = "entry" if is_entry else "exit"
                    event_ts = datetime.fromisoformat(event['timestamp'])
                    rows.append(
                        {
                            "camera_id": self.camera_id,
                            "line_name": event.get('line_name', 'main_gate'),
                            "direction": direction,
                            "event_type": event_type,
                            "track_id": event.get('track_id'),
                            "event_metadata": self._build_crossing_event_metadata(event),
                            "timestamp": event_ts
                        }
                    )
                if rows:
                    DatabaseManager.save_crossing_events(db=db, events=rows)
        except Exception as e:
            logger.error(f"Error guardando eventos de cruce: {e}")

    def _queue_crossing_events(self, crossing_events):
        """Encolar eventos de cruce para ingesta remota."""
        if self.remote_ingest is None:
            return

        rows = []
        for event in crossing_events:
            direction = (event.get('direction') or '').strip().lower()
            if direction not in {"positive", "negative"}:
                logger.warning("Evento de cruce con dirección inválida (ingesta remota): %s", event)
                continue
            is_entry = direction == self.entry_direction
            event_type = "entry" if is_entry else "exit"
            rows.append(
                {
                    "camera_id": self.camera_id,
                    "line_name": event.get('line_name', 'main_gate'),
                    "direction": direction,
                    "event_type": event_type,
                    "track_id": event.get('track_id'),
                    "event_metadata": self._build_crossing_event_metadata(event),
                    "timestamp": event.get('timestamp'),
                    "edge_id": self.edge_id,
                    "payload_version": self.payload_version,
                }
            )

        if rows:
            self.remote_ingest.enqueue_crossings(rows)

    def _queue_detection_snapshot(self, detections):
        """Encolar snapshot periódico para ingesta remota."""
        if self.remote_ingest is None:
            return
        if not detections:
            return

        capture_stats = self.capture.get_stats()
        payload = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "rtsp_url": self.rtsp_url,
            "branch_id": self.branch_id or None,
            "branch_name": self.branch_name or None,
            "timestamp": datetime.utcnow().isoformat(),
            "person_count": len(detections),
            "detections_data": [det.to_dict() for det in detections] if detections else [],
            "is_connected": capture_stats.get('is_connected'),
            "fps": self.stats['frames_processed'] / (time.time() - self.stats['start_time']),
            "total_frames": self.stats['frames_processed'],
            "error_count": capture_stats.get('error_count', 0),
            "edge_id": self.edge_id,
            "payload_version": self.payload_version,
        }
        self.remote_ingest.enqueue_detection(payload)

    def _apply_track_genders(self, detections, track_gender_states):
        """Aplicar estado de género aparente por track sobre detecciones del frame."""
        for det in detections:
            if det.track_id is None:
                continue
            track_id = int(det.track_id)
            state = track_gender_states.get(track_id)
            if state is None:
                continue
            det.apparent_gender = state.label
            det.apparent_gender_confidence = state.confidence
            self.latest_track_gender[track_id] = {
                "label": state.label,
                "confidence": float(state.confidence),
                "votes": int(state.votes),
                "updated_at": datetime.utcnow().isoformat(),
            }

    def _build_crossing_event_metadata(self, event: dict) -> dict:
        """Construir metadata enriquecida para un evento de cruce."""
        metadata = {
            "position": event.get("position"),
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "branch_id": self.branch_id or None,
            "branch_name": self.branch_name or None,
            "entry_direction_config": self.entry_direction,
            "edge_id": self.edge_id,
            "payload_version": self.payload_version,
        }
        track_id = event.get("track_id")
        if track_id is None:
            return metadata
        track_gender = self.latest_track_gender.get(int(track_id))
        if not track_gender:
            metadata["apparent_gender"] = "unknown"
            return metadata
        metadata["apparent_gender"] = track_gender.get("label", "unknown")
        metadata["apparent_gender_confidence"] = track_gender.get("confidence", 0.0)
        metadata["apparent_gender_votes"] = track_gender.get("votes", 0)
        return metadata

    def _save_heatmap_snapshot(self):
        """Guardar snapshot JSON del heatmap acumulado."""
        if self.heatmap is None:
            return
        try:
            snapshot_path = self.heatmap.save_snapshot(keep_history=self.heatmap_keep_history)
            if snapshot_path is not None:
                logger.debug("Heatmap snapshot guardado en %s", snapshot_path)
        except Exception as e:
            logger.error(f"Error guardando snapshot de heatmap: {e}")

    def _encode_background_frame(self, frame):
        """Codificar frame de referencia para visualización remota del heatmap."""
        try:
            if frame is None:
                return None
            h, w = frame.shape[:2]
            if w > self.heatmap_background_max_width:
                scale = self.heatmap_background_max_width / float(w)
                resized = cv2.resize(
                    frame,
                    (self.heatmap_background_max_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                resized = frame
            ok, encoded = cv2.imencode(
                ".jpg",
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.heatmap_background_jpeg_quality],
            )
            if not ok:
                return None
            return base64.b64encode(encoded.tobytes()).decode("ascii")
        except Exception as exc:
            logger.debug("No se pudo codificar background de heatmap: %s", exc)
            return None

    def _flush_completed_hourly_heatmap(self, completed_hour_start: datetime, is_partial: bool = False):
        """Enviar al backend central el heatmap de la hora cerrada."""
        if self.hourly_heatmap is None:
            return
        if self.remote_ingest is None or not self.save_to_api or not self.send_hourly_heatmap_to_api:
            return

        stats = self.hourly_heatmap.get_stats()
        if stats["sum_value"] <= 0:
            return

        snapshot = self.hourly_heatmap.snapshot()
        overlay_png_base64 = self.hourly_heatmap.export_overlay_png_base64()
        if overlay_png_base64 is None and self.heatmap is not None:
            overlay_png_base64 = self.heatmap.export_overlay_png_base64()
        payload = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "branch_id": self.branch_id or None,
            "branch_name": self.branch_name or None,
            "hour_start": completed_hour_start.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
            "frame_width": snapshot["frame_width"],
            "frame_height": snapshot["frame_height"],
            "cell_size": snapshot["cell_size"],
            "grid": snapshot["grid"],
            "stats": snapshot["stats"],
            "background_image_base64": self.latest_heatmap_background_base64,
            "overlay_png_base64": overlay_png_base64,
            "is_partial": is_partial,
            "edge_id": self.edge_id,
            "payload_version": self.payload_version,
        }
        self.remote_ingest.enqueue_heatmap(payload)
    
    def _configure_lines(self):
        """Entrar al modo de configuración de líneas"""
        configurator = LineConfigurator(
            self.camera_id,
            self.capture,
            entry_direction=self.entry_direction
        )
        new_lines = configurator.configure()
        
        if new_lines:
            self.lines_config = new_lines
            logger.info("✓ Nueva configuración de líneas aplicada")
    
    def _reload_counter(self):
        """Recargar contador con nueva configuración"""
        logger.info("Recargando contador con nueva configuración...")
        
        # Crear nuevo contador
        self.counter = PersonCounter(camera_id=self.camera_id)
        
        # Cargar líneas actualizadas
        loaded_config = load_lines_configuration(self.camera_id)
        
        if loaded_config:
            loaded_entry_direction = loaded_config.get("entry_direction", "positive")
            if self.entry_direction_force:
                logger.info(
                    "ENTRY_DIRECTION_FORCE activo tras recarga: se mantiene '%s' y se ignora '%s'",
                    self.entry_direction,
                    loaded_entry_direction,
                )
            else:
                self.entry_direction = self._normalize_entry_direction(loaded_entry_direction)
            self.lines_config = loaded_config.get("lines", [])
            for line_config in self.lines_config:
                self.counter.add_line(
                    name=line_config['name'],
                    p1=tuple(line_config['p1']),
                    p2=tuple(line_config['p2']),
                    direction=line_config.get('direction', 'both')
                )
            logger.info(f"✓ Contador recargado con {len(self.lines_config)} línea(s)")

    
    def cleanup(self):
        """Liberar recursos"""
        logger.info("Liberando recursos...")
        
        if hasattr(self, 'capture'):
            self.capture.release()

        if self.remote_ingest is not None:
            if self.hourly_heatmap is not None and self.current_heatmap_hour_start is not None:
                self._flush_completed_hourly_heatmap(
                    self.current_heatmap_hour_start,
                    is_partial=True,
                )
            self.remote_ingest.flush(force=True)
            self.remote_ingest.close()

        if self.heatmap is not None and self.save_heatmap_snapshots:
            self._save_heatmap_snapshot()
        
        if self.show_window:
            cv2.destroyAllWindows()
        
        # Imprimir resumen final
        uptime = time.time() - self.stats['start_time']
        logger.info("=" * 60)
        logger.info("RESUMEN FINAL")
        logger.info("=" * 60)
        logger.info(f"Tiempo total: {uptime:.0f} segundos")
        logger.info(f"Frames procesados: {self.stats['frames_processed']}")
        logger.info(f"FPS promedio: {self.stats['frames_processed'] / uptime:.1f}")
        logger.info(f"Detecciones totales: {self.stats['total_detections']}")
        if self.heatmap is not None:
            hm_stats = self.heatmap.get_stats()
            logger.info(
                "Heatmap final - max: %.2f | sum: %.2f | samples: %s | hotspot: %s",
                hm_stats["max_value"],
                hm_stats["sum_value"],
                hm_stats["samples"],
                hm_stats["hotspot"],
            )
        logger.info("=" * 60)
        logger.info("✓ Sistema finalizado")


def main():
    """Función principal"""
    
    # Configuración de la cámara
    CAMERA_CONFIG = {
        'camera_id': env_str('CAMERA_ID', 'camara_default'),
        'camera_name': env_str('CAMERA_NAME', 'Camara Default'),
        'rtsp_url': env_str('CAMERA_RTSP_URL', ''),
        'entry_direction': env_str('ENTRY_DIRECTION', 'positive'),
        'show_window': env_bool('SHOW_WINDOW', False),
        'save_to_db': env_bool('SAVE_TO_DB', False),
        'save_to_api': env_bool('SAVE_TO_API', False),
        'remote_api_base_url': env_str('REMOTE_API_BASE_URL', ''),
        'remote_api_key': env_str('REMOTE_API_KEY', ''),
        'max_ingest_queue_size': env_int('MAX_INGEST_QUEUE_SIZE', 10000),
        'branch_id': env_str('BRANCH_ID', ''),
        'branch_name': env_str('BRANCH_NAME', ''),
    }

    if not CAMERA_CONFIG['camera_id']:
        raise ValueError("CAMERA_ID es obligatorio")
    if not CAMERA_CONFIG['camera_name']:
        raise ValueError("CAMERA_NAME es obligatorio")
    if not CAMERA_CONFIG['rtsp_url']:
        raise ValueError("CAMERA_RTSP_URL es obligatorio")

    if CAMERA_CONFIG['save_to_api'] and not CAMERA_CONFIG['remote_api_base_url']:
        raise ValueError("REMOTE_API_BASE_URL es obligatorio cuando SAVE_TO_API=true")
    
    logger.info("=" * 60)
    logger.info("TRAFFIC ANALYSIS SYSTEM - Marathon SRL")
    logger.info("=" * 60)
    logger.info(f"Cámara: {CAMERA_CONFIG['camera_name']}")
    logger.info(f"URL: {sanitize_rtsp_url(CAMERA_CONFIG['rtsp_url'])}")
    logger.info("=" * 60)
    
    # Crear y ejecutar sistema
    system = TrafficAnalysisSystem(**CAMERA_CONFIG)
    system.run()


if __name__ == "__main__":
    main()
