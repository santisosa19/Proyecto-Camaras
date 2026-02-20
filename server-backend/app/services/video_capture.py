"""
Módulo de captura de video desde cámaras RTSP
"""
import cv2
import time
import logging
from typing import Optional, Tuple
import numpy as np
from threading import Lock

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Captura de video desde fuente RTSP con reconexión automática
    """
    
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        fps: int = 15,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 10
    ):
        """
        Args:
            camera_id: Identificador único de la cámara
            rtsp_url: URL RTSP de la cámara
            fps: Frames por segundo deseados
            reconnect_delay: Segundos de espera entre intentos de reconexión
            max_reconnect_attempts: Máximo de intentos de reconexión
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.frame_count = 0
        self.error_count = 0
        self.lock = Lock()
        
        # Estadísticas
        self.stats = {
            'total_frames': 0,
            'successful_reads': 0,
            'failed_reads': 0,
            'reconnections': 0
        }
        
        logger.info(f"VideoCapture inicializado para cámara {camera_id}")
    
    def connect(self) -> bool:
        """
        Conectar a la fuente de video
        
        Returns:
            True si la conexión fue exitosa
        """
        try:
            if self.cap is not None:
                self.cap.release()
            
            logger.info(f"Conectando a cámara {self.camera_id}: {self.rtsp_url}")
            
            # Configurar captura con parámetros optimizados
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Configurar propiedades
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para latencia baja
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                raise Exception("No se pudo abrir la fuente de video")
            
            # Verificar que podemos leer un frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise Exception("No se pudo leer frame de prueba")
            
            self.is_connected = True
            self.error_count = 0
            self.stats['reconnections'] += 1
            
            logger.info(f"✓ Cámara {self.camera_id} conectada exitosamente")
            logger.info(f"  Resolución: {frame.shape[1]}x{frame.shape[0]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error conectando a cámara {self.camera_id}: {e}")
            self.is_connected = False
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Obtener siguiente frame
        
        Returns:
            Frame BGR o None si hay error
        """
        with self.lock:
            if not self.is_connected or self.cap is None:
                logger.warning(f"Cámara {self.camera_id} no conectada")
                return None
            
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.error_count += 1
                    self.stats['failed_reads'] += 1
                    
                    logger.warning(
                        f"Error leyendo frame de {self.camera_id} "
                        f"(errores consecutivos: {self.error_count})"
                    )
                    
                    # Intentar reconectar después de varios errores
                    if self.error_count >= 3:
                        logger.warning(f"Intentando reconectar {self.camera_id}...")
                        self.is_connected = False
                        self._try_reconnect()
                    
                    return None
                
                # Frame exitoso
                self.error_count = 0
                self.frame_count += 1
                self.stats['total_frames'] += 1
                self.stats['successful_reads'] += 1
                
                return frame
                
            except Exception as e:
                logger.error(f"Excepción leyendo frame de {self.camera_id}: {e}")
                self.stats['failed_reads'] += 1
                self.is_connected = False
                return None
    
    def _try_reconnect(self) -> bool:
        """
        Intentar reconectar a la cámara
        
        Returns:
            True si reconexión exitosa
        """
        for attempt in range(1, self.max_reconnect_attempts + 1):
            logger.info(
                f"Intento de reconexión {attempt}/{self.max_reconnect_attempts} "
                f"para {self.camera_id}"
            )
            
            time.sleep(self.reconnect_delay)
            
            if self.connect():
                return True
        
        logger.error(
            f"No se pudo reconectar a {self.camera_id} después de "
            f"{self.max_reconnect_attempts} intentos"
        )
        return False
    
    def release(self):
        """Liberar recursos"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.is_connected = False
            logger.info(f"Cámara {self.camera_id} liberada")
    
    def get_stats(self) -> dict:
        """Obtener estadísticas de captura"""
        return {
            **self.stats,
            'is_connected': self.is_connected,
            'error_count': self.error_count,
            'frame_count': self.frame_count,
            'success_rate': (
                self.stats['successful_reads'] / self.stats['total_frames'] * 100
                if self.stats['total_frames'] > 0 else 0
            )
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


# Clase auxiliar para gestionar múltiples cámaras
class MultiCameraCapture:
    """
    Gestor de múltiples cámaras
    """
    
    def __init__(self):
        self.cameras: dict[str, VideoCapture] = {}
        logger.info("MultiCameraCapture inicializado")
    
    def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        fps: int = 15
    ) -> bool:
        """
        Agregar una cámara
        
        Args:
            camera_id: Identificador único
            rtsp_url: URL RTSP
            fps: Frames por segundo
            
        Returns:
            True si se agregó exitosamente
        """
        if camera_id in self.cameras:
            logger.warning(f"Cámara {camera_id} ya existe")
            return False
        
        camera = VideoCapture(camera_id, rtsp_url, fps)
        
        if camera.connect():
            self.cameras[camera_id] = camera
            logger.info(f"Cámara {camera_id} agregada exitosamente")
            return True
        else:
            logger.error(f"No se pudo agregar cámara {camera_id}")
            return False
    
    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Obtener frame de una cámara específica"""
        if camera_id not in self.cameras:
            logger.warning(f"Cámara {camera_id} no encontrada")
            return None
        
        return self.cameras[camera_id].get_frame()
    
    def get_all_frames(self) -> dict[str, np.ndarray]:
        """Obtener frames de todas las cámaras"""
        frames = {}
        for camera_id, camera in self.cameras.items():
            frame = camera.get_frame()
            if frame is not None:
                frames[camera_id] = frame
        return frames
    
    def remove_camera(self, camera_id: str):
        """Remover una cámara"""
        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
            logger.info(f"Cámara {camera_id} removida")
    
    def release_all(self):
        """Liberar todas las cámaras"""
        for camera in self.cameras.values():
            camera.release()
        self.cameras.clear()
        logger.info("Todas las cámaras liberadas")
    
    def get_stats(self) -> dict:
        """Obtener estadísticas de todas las cámaras"""
        return {
            camera_id: camera.get_stats()
            for camera_id, camera in self.cameras.items()
        }
    
    def __del__(self):
        """Destructor"""
        self.release_all()
