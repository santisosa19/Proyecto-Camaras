"""
Módulo de conteo de personas con líneas virtuales
"""
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class CountingLine:
    """Representa una línea de conteo"""
    name: str
    p1: Tuple[int, int]  # (x1, y1)
    p2: Tuple[int, int]  # (x2, y2)
    direction: str = "both"  # "up", "down", "left", "right", "both"
    
    def __post_init__(self):
        # Calcular ecuación de la línea: ax + by + c = 0
        x1, y1 = self.p1
        x2, y2 = self.p2
        
        self.a = y2 - y1
        self.b = x1 - x2
        self.c = x2 * y1 - x1 * y2
        
        # Normalizar
        norm = np.sqrt(self.a**2 + self.b**2)
        if norm > 0:
            self.a /= norm
            self.b /= norm
            self.c /= norm
    
    def signed_distance(self, point: Tuple[float, float]) -> float:
        """
        Calcular distancia signed de un punto a la línea
        Positivo = un lado, Negativo = otro lado
        """
        x, y = point
        return self.a * x + self.b * y + self.c
    
    def crosses(self, prev_point: Tuple[float, float], curr_point: Tuple[float, float]) -> Optional[str]:
        """
        Detectar si hubo un cruce de línea
        
        Returns:
            'positive' si cruzó hacia el lado positivo
            'negative' si cruzó hacia el lado negativo
            None si no hubo cruce
        """
        prev_dist = self.signed_distance(prev_point)
        curr_dist = self.signed_distance(curr_point)
        
        # Threshold para evitar falsos positivos en el borde.
        # Más bajo para no perder cruces con salto de frames.
        threshold = 3.0
        
        if prev_dist < -threshold and curr_dist > threshold:
            return 'positive'
        elif prev_dist > threshold and curr_dist < -threshold:
            return 'negative'

        # Fallback: cambio de signo con desplazamiento suficiente cerca de la línea.
        if prev_dist * curr_dist < 0 and abs(curr_dist - prev_dist) >= 2.0:
            return 'positive' if curr_dist > 0 else 'negative'
        
        return None


@dataclass
class Track:
    """Representa el seguimiento de una persona"""
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    counted: bool = False
    crossed_lines: List[str] = field(default_factory=list)
    
    def update(self, centroid: Tuple[float, float]):
        """Actualizar posición del track"""
        self.positions.append(centroid)
        self.last_seen = datetime.now()
    
    @property
    def age(self) -> float:
        """Edad del track en segundos"""
        return (datetime.now() - self.first_seen).total_seconds()
    
    @property
    def current_position(self) -> Optional[Tuple[float, float]]:
        """Posición actual"""
        return self.positions[-1] if len(self.positions) > 0 else None
    
    @property
    def previous_position(self) -> Optional[Tuple[float, float]]:
        """Posición anterior"""
        return self.positions[-2] if len(self.positions) >= 2 else None


class PersonCounter:
    """
    Contador de personas basado en líneas virtuales
    """
    
    def __init__(self, camera_id: str):
        """
        Args:
            camera_id: Identificador de la cámara
        """
        self.camera_id = camera_id
        self.tracks: Dict[int, Track] = {}
        
        # Líneas de conteo
        self.lines: Dict[str, CountingLine] = {}
        
        # Contadores
        self.counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
        
        # Configuración
        self.max_track_age = 5.0  # Segundos sin actualización antes de eliminar track
        self.crossing_cooldown_seconds = 2.0
        self.crossing_merge_distance = 25.0
        self.recent_crossings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.crossing_events: deque = deque(maxlen=1000)
        
        logger.info(f"PersonCounter inicializado para cámara {camera_id}")
    
    def add_line(
        self,
        name: str,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        direction: str = "both"
    ):
        """
        Agregar línea de conteo
        
        Args:
            name: Nombre identificador de la línea
            p1: Punto inicial (x, y)
            p2: Punto final (x, y)
            direction: Dirección a contar ("both", "up", "down", "left", "right")
        """
        line = CountingLine(name, p1, p2, direction)
        self.lines[name] = line
        logger.info(f"Línea '{name}' agregada: {p1} -> {p2}")
    
    def update(self, detections: List[Detection]) -> Dict:
        """
        Actualizar contador con nuevas detecciones
        
        Args:
            detections: Lista de detecciones del frame actual
            
        Returns:
            Diccionario con estadísticas actualizadas
        """
        current_time = datetime.now()
        
        # Mapear detecciones por track_id
        detection_by_track = {}
        for det in detections:
            if det.track_id is not None:
                detection_by_track[det.track_id] = det
        
        # Actualizar tracks existentes
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if track_id in detection_by_track:
                # Track activo - actualizar posición
                det = detection_by_track[track_id]
                prev_pos = track.current_position
                curr_pos = det.centroid
                
                track.update(curr_pos)
                
                # Verificar cruces de línea
                if prev_pos is not None:
                    self._check_line_crossings(track, prev_pos, curr_pos)
            
            else:
                # Track sin detección - verificar edad
                age = (current_time - track.last_seen).total_seconds()
                if age > self.max_track_age:
                    tracks_to_remove.append(track_id)
        
        # Eliminar tracks viejos
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Agregar nuevos tracks
        for track_id, det in detection_by_track.items():
            if track_id not in self.tracks:
                track = Track(track_id=track_id)
                track.update(det.centroid)
                self.tracks[track_id] = track
        
        # Retornar estadísticas
        return self.get_stats()
    
    def _check_line_crossings(
        self,
        track: Track,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float]
    ):
        """Verificar si el track cruzó alguna línea"""
        for line_name, line in self.lines.items():
            crossing = line.crosses(prev_pos, curr_pos)
            
            if crossing is not None:
                # Evitar contar el mismo cruce múltiples veces
                crossing_key = f"{line_name}_{crossing}"
                
                if crossing_key not in track.crossed_lines:
                    if self._is_duplicate_crossing(line_name, crossing, curr_pos):
                        logger.debug(
                            f"Cruce ignorado por cooldown/reidentificación: "
                            f"track={track.track_id}, line={line_name}, dir={crossing}"
                        )
                        continue

                    track.crossed_lines.append(crossing_key)
                    self.counts[line_name][crossing] += 1
                    self._register_crossing(line_name, crossing, curr_pos)
                    self.crossing_events.append(
                        {
                            'timestamp': datetime.now().isoformat(),
                            'line_name': line_name,
                            'direction': crossing,
                            'track_id': track.track_id,
                            'position': {'x': float(curr_pos[0]), 'y': float(curr_pos[1])}
                        }
                    )
                    
                    logger.info(
                        f"Track {track.track_id} cruzó línea '{line_name}' "
                        f"en dirección {crossing}"
                    )

    def _crossing_cache_key(self, line_name: str, direction: str) -> str:
        return f"{line_name}:{direction}"

    def _is_duplicate_crossing(
        self,
        line_name: str,
        direction: str,
        curr_pos: Tuple[float, float]
    ) -> bool:
        key = self._crossing_cache_key(line_name, direction)
        now = datetime.now()
        entries = self.recent_crossings[key]
        kept = deque(maxlen=entries.maxlen)

        duplicate = False
        for entry_time, entry_pos in entries:
            if (now - entry_time).total_seconds() <= self.crossing_cooldown_seconds:
                kept.append((entry_time, entry_pos))
                distance = np.hypot(curr_pos[0] - entry_pos[0], curr_pos[1] - entry_pos[1])
                if distance <= self.crossing_merge_distance:
                    duplicate = True

        self.recent_crossings[key] = kept
        return duplicate

    def _register_crossing(
        self,
        line_name: str,
        direction: str,
        curr_pos: Tuple[float, float]
    ):
        key = self._crossing_cache_key(line_name, direction)
        self.recent_crossings[key].append((datetime.now(), curr_pos))
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas actuales"""
        stats = {
            'camera_id': self.camera_id,
            'timestamp': datetime.now().isoformat(),
            'active_tracks': len(self.tracks),
            'lines': {}
        }
        
        for line_name, counts in self.counts.items():
            stats['lines'][line_name] = {
                'positive': counts['positive'],
                'negative': counts['negative'],
                'total': counts['positive'] + counts['negative']
            }
        
        return stats
    
    def reset_counts(self):
        """Resetear contadores (mantener tracks activos)"""
        self.counts.clear()
        logger.info(f"Contadores reseteados para cámara {self.camera_id}")

    def pop_crossing_events(self) -> List[Dict]:
        """Obtener y vaciar eventos nuevos de cruce."""
        events = list(self.crossing_events)
        self.crossing_events.clear()
        return events
    
    def draw_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibujar líneas de conteo en el frame
        
        Args:
            frame: Frame BGR
            
        Returns:
            Frame con líneas dibujadas
        """
        import cv2
        
        output = frame.copy()
        
        for line_name, line in self.lines.items():
            # Dibujar línea
            cv2.line(
                output,
                line.p1,
                line.p2,
                (0, 0, 255),  # Rojo
                2
            )
            
            # Label con conteos
            counts = self.counts[line_name]
            label = f"{line_name}: +{counts['positive']} -{counts['negative']}"
            
            # Posición del label (punto medio de la línea)
            mid_x = (line.p1[0] + line.p2[0]) // 2
            mid_y = (line.p1[1] + line.p2[1]) // 2
            
            cv2.putText(
                output,
                label,
                (mid_x, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        return output


class ZoneCounter:
    """
    Contador basado en zonas (áreas rectangulares o polígonos)
    """
    
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.zones: Dict[str, Dict] = {}
        self.zone_counts: Dict[str, int] = defaultdict(int)
        
        logger.info(f"ZoneCounter inicializado para cámara {camera_id}")
    
    def add_zone(
        self,
        name: str,
        points: List[Tuple[int, int]]
    ):
        """
        Agregar zona de conteo
        
        Args:
            name: Nombre de la zona
            points: Lista de puntos [(x1,y1), (x2,y2), ...] que definen la zona
                   Para rectángulo: 2 puntos (esquinas opuestas)
                   Para polígono: N puntos
        """
        self.zones[name] = {
            'points': points,
            'type': 'rect' if len(points) == 2 else 'polygon'
        }
        logger.info(f"Zona '{name}' agregada con {len(points)} puntos")
    
    def update(self, detections: List[Detection]) -> Dict:
        """
        Contar personas en cada zona
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Conteos por zona
        """
        # Resetear conteos
        self.zone_counts.clear()
        
        for det in detections:
            cx, cy = det.centroid
            
            for zone_name, zone in self.zones.items():
                if self._point_in_zone(cx, cy, zone):
                    self.zone_counts[zone_name] += 1
        
        return dict(self.zone_counts)
    
    def _point_in_zone(self, x: float, y: float, zone: Dict) -> bool:
        """Verificar si un punto está dentro de una zona"""
        if zone['type'] == 'rect':
            # Rectángulo
            x1, y1 = zone['points'][0]
            x2, y2 = zone['points'][1]
            
            return (min(x1, x2) <= x <= max(x1, x2) and
                    min(y1, y2) <= y <= max(y1, y2))
        
        else:
            # Polígono (algoritmo ray casting)
            points = zone['points']
            n = len(points)
            inside = False
            
            p1x, p1y = points[0]
            for i in range(1, n + 1):
                p2x, p2y = points[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            
            return inside
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Dibujar zonas en el frame"""
        import cv2
        
        output = frame.copy()
        
        for zone_name, zone in self.zones.items():
            points = zone['points']
            
            if zone['type'] == 'rect':
                # Rectángulo
                cv2.rectangle(
                    output,
                    points[0],
                    points[1],
                    (255, 0, 0),  # Azul
                    2
                )
            else:
                # Polígono
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(output, [pts], True, (255, 0, 0), 2)
            
            # Label
            count = self.zone_counts.get(zone_name, 0)
            label = f"{zone_name}: {count}"
            
            # Posición del label
            label_pos = points[0]
            cv2.putText(
                output,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
        
        return output
