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
import cv2
import logging
import json
import os
from collections import deque
from pathlib import Path
from datetime import datetime
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


# Archivo de configuración de líneas
LINES_CONFIG_FILE = Path(__file__).parent / "lines_config.json"


class LineConfigurator:
    """Configurador interactivo de líneas de conteo"""
    
    def __init__(self, camera_id: str, capture: VideoCapture, entry_direction: str = "positive"):
        self.camera_id = camera_id
        self.capture = capture
        self.entry_direction = entry_direction
        self.points = []
        self.lines = []
        self.current_frame = None
        
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
        
        # Configurar callback del mouse
        cv2.namedWindow("Configuración de Líneas")
        cv2.setMouseCallback("Configuración de Líneas", self._mouse_callback)
        
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
            
            cv2.imshow("Configuración de Líneas", display_frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Guardar
                if len(self.lines) >= 1:
                    self._save_configuration()
                    logger.info("✓ Configuración guardada")
                    cv2.destroyWindow("Configuración de Líneas")
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
                cv2.destroyWindow("Configuración de Líneas")
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
        crossing_flush_interval: float = 1.0,
        detection_flush_interval: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_batch_size = max_batch_size
        self.crossing_flush_interval = crossing_flush_interval
        self.detection_flush_interval = detection_flush_interval
        self.pending_crossings: deque = deque()
        self.pending_detections: deque = deque()
        self.last_crossing_flush = 0.0
        self.last_detection_flush = 0.0

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        self.client = httpx.Client(
            timeout=timeout_seconds,
            headers=headers
        )

    def enqueue_crossings(self, events: list[dict]):
        for event in events:
            self.pending_crossings.append(event)

    def enqueue_detection(self, item: dict):
        self.pending_detections.append(item)

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

    def _flush_queue(self, queue: deque, path: str, key: str):
        if not queue:
            return

        batch = []
        while queue and len(batch) < self.max_batch_size:
            batch.append(queue.popleft())

        if not self._post_batch(path=path, key=key, batch=batch):
            # Reinsertar al frente respetando orden original.
            for item in reversed(batch):
                queue.appendleft(item)

    def flush(self, force: bool = False):
        now = time.time()
        should_flush_crossings = force or (now - self.last_crossing_flush >= self.crossing_flush_interval)
        should_flush_detections = force or (now - self.last_detection_flush >= self.detection_flush_interval)

        if should_flush_crossings and self.pending_crossings:
            self._flush_queue(
                queue=self.pending_crossings,
                path="/api/v1/ingest/crossings",
                key="events"
            )
            self.last_crossing_flush = now

        if should_flush_detections and self.pending_detections:
            self._flush_queue(
                queue=self.pending_detections,
                path="/api/v1/ingest/detections",
                key="items"
            )
            self.last_detection_flush = now

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
        if entry_direction not in {"positive", "negative"}:
            raise ValueError("entry_direction debe ser 'positive' o 'negative'")
        self.entry_direction = entry_direction
        self.show_window = show_window
        self.save_to_db = save_to_db
        self.save_to_api = save_to_api
        self.remote_api_base_url = remote_api_base_url.strip()
        self.remote_api_key = remote_api_key.strip()
        self.remote_ingest: RemoteIngestClient | None = None
        
        # Estadísticas
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'start_time': time.time()
        }
        
        logger.info(f"Inicializando sistema para cámara: {camera_name}")
        
        # Inicializar componentes
        self._init_components()
    
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
                    timeout_seconds=5.0
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
            self.detector = PersonDetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.5,
                device="cpu"  # Cambiar a "cuda" si tienes GPU
            )
            logger.info("✓ Detector cargado")
            
            # 3. Contador
            logger.info("Inicializando contador...")
            self.counter = PersonCounter(camera_id=self.camera_id)
            
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
                self.entry_direction = loaded_config.get("entry_direction", "positive")
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
            
            logger.info("=" * 60)
            logger.info("SISTEMA LISTO")
            logger.info("  'q' = Salir")
            logger.info("  'c' = Configurar líneas")
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
                
                # Actualizar contador
                count_stats = self.counter.update(detections)
                crossing_events = self.counter.pop_crossing_events()
                
                # Actualizar estadísticas
                self.stats['frames_processed'] += 1
                self.stats['total_detections'] += len(detections)
                frame_count += 1
                
                # Dibujar detecciones y líneas en el frame
                if self.show_window:
                    # Dibujar bounding boxes
                    for det in detections:
                        x1, y1, x2, y2 = [int(x) for x in det.bbox]
                        
                        # Box verde
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Label con ID y confianza
                        label = f"ID:{det.track_id} {det.confidence:.2f}"
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
                    ]
                    
                    # Agregar conteos de todas las líneas
                    for line_name, line_stats in count_stats.get('lines', {}).items():
                        info_lines.append(f"{line_name}: +{line_stats['positive']} -{line_stats['negative']}")
                    
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
                    direction = event.get('direction')
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
                            "event_metadata": {'position': event.get('position')},
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
            direction = event.get('direction')
            is_entry = direction == self.entry_direction
            event_type = "entry" if is_entry else "exit"
            rows.append(
                {
                    "camera_id": self.camera_id,
                    "line_name": event.get('line_name', 'main_gate'),
                    "direction": direction,
                    "event_type": event_type,
                    "track_id": event.get('track_id'),
                    "event_metadata": {'position': event.get('position')},
                    "timestamp": event.get('timestamp')
                }
            )

        if rows:
            self.remote_ingest.enqueue_crossings(rows)

    def _queue_detection_snapshot(self, detections):
        """Encolar snapshot periódico para ingesta remota."""
        if self.remote_ingest is None:
            return

        capture_stats = self.capture.get_stats()
        payload = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "rtsp_url": self.rtsp_url,
            "timestamp": datetime.utcnow().isoformat(),
            "person_count": len(detections),
            "detections_data": [det.to_dict() for det in detections] if detections else [],
            "is_connected": capture_stats.get('is_connected'),
            "fps": self.stats['frames_processed'] / (time.time() - self.stats['start_time']),
            "total_frames": self.stats['frames_processed'],
            "error_count": capture_stats.get('error_count', 0),
        }
        self.remote_ingest.enqueue_detection(payload)
    
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
            self.entry_direction = loaded_config.get("entry_direction", "positive")
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
            self.remote_ingest.flush(force=True)
            self.remote_ingest.close()
        
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
        logger.info("=" * 60)
        logger.info("✓ Sistema finalizado")


def main():
    """Función principal"""
    
    # Configuración de la cámara
    CAMERA_CONFIG = {
        'camera_id': 'camara_prueba_marathon',
        'camera_name': 'Cámara de Prueba Marathon',
        'rtsp_url': os.getenv('CAMERA_RTSP_URL', 'rtsp://admin:admin@192.168.0.100:554/Streaming/Channels/101'),
        'entry_direction': 'positive',  # positive = entra, negative = sale
        'show_window': True,      # True = mostrar ventana con video
        'save_to_db': os.getenv('SAVE_TO_DB', 'false').lower() == 'true',
        'save_to_api': os.getenv('SAVE_TO_API', 'true').lower() == 'true',
        'remote_api_base_url': os.getenv('REMOTE_API_BASE_URL', ''),
        'remote_api_key': os.getenv('REMOTE_API_KEY', ''),
    }
    
    logger.info("=" * 60)
    logger.info("TRAFFIC ANALYSIS SYSTEM - Marathon SRL")
    logger.info("=" * 60)
    logger.info(f"Cámara: {CAMERA_CONFIG['camera_name']}")
    logger.info(f"URL: {CAMERA_CONFIG['rtsp_url']}")
    logger.info("=" * 60)
    
    # Crear y ejecutar sistema
    system = TrafficAnalysisSystem(**CAMERA_CONFIG)
    system.run()


if __name__ == "__main__":
    main()
