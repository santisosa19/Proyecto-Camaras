"""
Módulo de detección de personas con YOLOv8
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from ultralytics import YOLO
import cv2

logger = logging.getLogger(__name__)


class Detection:
    """Representa una detección individual"""
    
    def __init__(
        self,
        bbox: List[float],
        confidence: float,
        class_id: int = 0,
        track_id: Optional[int] = None
    ):
        """
        Args:
            bbox: [x1, y1, x2, y2] coordenadas del bounding box
            confidence: Confianza de la detección (0-1)
            class_id: ID de la clase (0 = person)
            track_id: ID de tracking (si está disponible)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.track_id = track_id
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Centro del bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Área del bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> dict:
        """Convertir a diccionario"""
        return {
            'bbox': self.bbox,
            'confidence': float(self.confidence),
            'class_id': self.class_id,
            'track_id': self.track_id,
            'centroid': self.centroid,
            'area': float(self.area)
        }


class PersonDetector:
    """
    Detector de personas basado en YOLOv8
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Args:
            model_path: Ruta al modelo YOLO (n/s/m/l/x)
            confidence_threshold: Umbral de confianza mínimo
            iou_threshold: Umbral de IoU para NMS
            device: 'cpu' o 'cuda' o número de GPU (0, 1, etc.)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Estadísticas
        self.stats = {
            'total_inferences': 0,
            'total_detections': 0,
            'avg_inference_time': 0.0
        }
        
        logger.info(f"Inicializando PersonDetector con modelo {model_path}")
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo YOLO"""
        try:
            self.model = YOLO(self.model_path)
            
            # Mover a device especificado
            if self.device != "cpu":
                self.model.to(self.device)
            
            logger.info(f"✓ Modelo YOLO cargado exitosamente en {self.device}")
            logger.info(f"  Modelo: {self.model_path}")
            logger.info(f"  Clases: {len(self.model.names)}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo YOLO: {e}")
            raise
    
    def detect(
        self,
        frame: np.ndarray,
        track: bool = False
    ) -> List[Detection]:
        """
        Detectar personas en un frame
        
        Args:
            frame: Frame BGR de OpenCV
            track: Si True, mantiene tracking entre frames
            
        Returns:
            Lista de detecciones
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Realizar inferencia
            if track:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],  # Solo personas
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            else:
                results = self.model(
                    frame,
                    classes=[0],  # Solo personas
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            
            # Procesar resultados
            detections = []
            
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Extraer datos del box
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Track ID si está disponible
                        track_id = None
                        if hasattr(boxes, 'id') and boxes.id is not None:
                            track_id = int(boxes.id[i].cpu().numpy())
                        
                        detection = Detection(
                            bbox=bbox,
                            confidence=conf,
                            class_id=0,
                            track_id=track_id
                        )
                        detections.append(detection)
            
            # Actualizar estadísticas
            self.stats['total_inferences'] += 1
            self.stats['total_detections'] += len(detections)
            
            # Calcular tiempo promedio
            if hasattr(results[0], 'speed'):
                inference_time = results[0].speed['inference']
                self.stats['avg_inference_time'] = (
                    (self.stats['avg_inference_time'] * (self.stats['total_inferences'] - 1) + inference_time)
                    / self.stats['total_inferences']
                )
            
            return detections
            
        except Exception as e:
            logger.error(f"Error en detección: {e}")
            return []
    
    def detect_and_draw(
        self,
        frame: np.ndarray,
        track: bool = False,
        show_confidence: bool = True,
        show_track_id: bool = True
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Detectar y dibujar bounding boxes en el frame
        
        Args:
            frame: Frame BGR
            track: Si True, mantiene tracking
            show_confidence: Mostrar nivel de confianza
            show_track_id: Mostrar ID de tracking
            
        Returns:
            Tupla (frame con detecciones dibujadas, lista de detecciones)
        """
        detections = self.detect(frame, track=track)
        
        if len(detections) == 0:
            return frame, detections
        
        # Copiar frame para no modificar el original
        output_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = [int(x) for x in detection.bbox]
            
            # Color del box (verde)
            color = (0, 255, 0)
            thickness = 2
            
            # Dibujar bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Preparar label
            label_parts = []
            
            if show_track_id and detection.track_id is not None:
                label_parts.append(f"ID:{detection.track_id}")
            
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Calcular tamaño del label
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Fondo del label
                cv2.rectangle(
                    output_frame,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Texto del label
                cv2.putText(
                    output_frame,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # Dibujar centroid
            cx, cy = [int(x) for x in detection.centroid]
            cv2.circle(output_frame, (cx, cy), 3, color, -1)
        
        # Info general
        info_text = f"Detecciones: {len(detections)}"
        cv2.putText(
            output_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return output_frame, detections
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del detector"""
        return {
            **self.stats,
            'avg_detections_per_frame': (
                self.stats['total_detections'] / self.stats['total_inferences']
                if self.stats['total_inferences'] > 0 else 0
            ),
            'model': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold
        }


class DetectionFilter:
    """
    Filtros para post-procesamiento de detecciones
    """
    
    @staticmethod
    def filter_by_area(
        detections: List[Detection],
        min_area: float = 1000,
        max_area: Optional[float] = None
    ) -> List[Detection]:
        """Filtrar por área del bounding box"""
        filtered = []
        for det in detections:
            area = det.area
            if area >= min_area:
                if max_area is None or area <= max_area:
                    filtered.append(det)
        return filtered
    
    @staticmethod
    def filter_by_region(
        detections: List[Detection],
        region: Tuple[int, int, int, int]
    ) -> List[Detection]:
        """
        Filtrar detecciones dentro de una región
        
        Args:
            region: (x1, y1, x2, y2) región de interés
        """
        rx1, ry1, rx2, ry2 = region
        filtered = []
        
        for det in detections:
            cx, cy = det.centroid
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                filtered.append(det)
        
        return filtered
    
    @staticmethod
    def non_maximum_suppression(
        detections: List[Detection],
        iou_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Aplicar Non-Maximum Suppression adicional
        (YOLO ya lo hace, pero útil para post-procesamiento)
        """
        if len(detections) == 0:
            return []
        
        # Ordenar por confianza
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Filtrar overlapping boxes
            detections = [
                det for det in detections
                if DetectionFilter._calculate_iou(current, det) < iou_threshold
            ]
        
        return keep
    
    @staticmethod
    def _calculate_iou(det1: Detection, det2: Detection) -> float:
        """Calcular Intersection over Union entre dos detecciones"""
        x1_1, y1_1, x2_1, y2_1 = det1.bbox
        x1_2, y1_2, x2_2, y2_2 = det2.bbox
        
        # Intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Unión
        area1 = det1.area
        area2 = det2.area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
