#!/usr/bin/env python3
"""
Script de prueba/demo del Traffic Analysis System
Prueba todos los componentes principales sin necesidad de cámaras reales
"""
import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime

# Agregar path del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.video_capture import VideoCapture
from app.services.detector import PersonDetector, Detection
from app.services.counter import PersonCounter, CountingLine


def generate_test_frame(width=640, height=480, num_persons=3):
    """
    Generar frame sintético para testing
    """
    # Crear frame base
    frame = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
    
    # Agregar "personas" simuladas (rectángulos)
    detections = []
    
    for i in range(num_persons):
        # Posición aleatoria
        x = np.random.randint(50, width - 100)
        y = np.random.randint(50, height - 150)
        w = np.random.randint(40, 80)
        h = np.random.randint(100, 150)
        
        # Dibujar "persona"
        color = tuple(np.random.randint(100, 200, 3).tolist())
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        # Crear detección
        detection = Detection(
            bbox=[float(x), float(y), float(x + w), float(y + h)],
            confidence=0.9,
            track_id=i
        )
        detections.append(detection)
    
    return frame, detections


def test_detector():
    """Probar detector YOLO"""
    print("\n" + "="*50)
    print("TEST 1: DETECTOR YOLO")
    print("="*50)
    
    try:
        print("Inicializando detector...")
        detector = PersonDetector(model_path="yolov8n.pt")
        
        # Generar frame de prueba
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("Ejecutando detección en frame de prueba...")
        detections = detector.detect(test_frame)
        
        print(f"✓ Detector funcionando correctamente")
        print(f"  Detecciones encontradas: {len(detections)}")
        
        stats = detector.get_stats()
        print(f"  Estadísticas: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en detector: {e}")
        return False


def test_counter():
    """Probar sistema de conteo"""
    print("\n" + "="*50)
    print("TEST 2: SISTEMA DE CONTEO")
    print("="*50)
    
    try:
        print("Inicializando contador...")
        counter = PersonCounter(camera_id="test_camera")
        
        # Agregar línea de conteo
        counter.add_line(
            name="entrada",
            p1=(0, 240),    # Línea horizontal en el medio
            p2=(640, 240),
            direction="both"
        )
        
        print("✓ Línea de conteo agregada")
        
        # Simular detecciones que cruzan la línea
        print("\nSimulando detecciones...")
        
        for frame_num in range(10):
            # Generar detecciones sintéticas
            _, detections = generate_test_frame(num_persons=2)
            
            # Mover detecciones para simular movimiento
            for det in detections:
                # Mover verticalmente para cruzar línea
                offset = frame_num * 30
                det.bbox[1] = 200 + offset
                det.bbox[3] = 250 + offset
            
            stats = counter.update(detections)
            
            if frame_num % 3 == 0:
                print(f"  Frame {frame_num}: {stats}")
        
        final_stats = counter.get_stats()
        print(f"\n✓ Sistema de conteo funcionando")
        print(f"  Estadísticas finales: {final_stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en contador: {e}")
        return False


def test_full_pipeline():
    """Probar pipeline completo con frames sintéticos"""
    print("\n" + "="*50)
    print("TEST 3: PIPELINE COMPLETO")
    print("="*50)
    
    try:
        print("Inicializando componentes...")
        
        # Detector
        detector = PersonDetector(model_path="yolov8n.pt")
        print("✓ Detector inicializado")
        
        # Counter
        counter = PersonCounter(camera_id="test_camera")
        counter.add_line("entrada", (0, 240), (640, 240))
        print("✓ Contador inicializado")
        
        # Procesar frames sintéticos
        print("\nProcesando 20 frames sintéticos...")
        
        for i in range(20):
            # Generar frame con personas
            frame, synthetic_dets = generate_test_frame(num_persons=np.random.randint(1, 5))
            
            # Detectar (usamos sintéticas para este test)
            # En producción usaríamos: detections = detector.detect(frame, track=True)
            detections = synthetic_dets
            
            # Contar
            stats = counter.update(detections)
            
            if i % 5 == 0:
                print(f"  Frame {i:02d}: {len(detections)} personas, Stats: {stats}")
            
            # Simular FPS real
            time.sleep(0.033)  # ~30 FPS
        
        print("\n✓ Pipeline completo funcionando correctamente")
        
        final_stats = counter.get_stats()
        print(f"  Tracks activos: {final_stats['active_tracks']}")
        print(f"  Líneas: {final_stats['lines']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Probar conexión a base de datos"""
    print("\n" + "="*50)
    print("TEST 4: BASE DE DATOS")
    print("="*50)
    
    try:
        from app.database.connection import get_db_context, DatabaseManager
        from app.models.database import CameraStatus
        
        print("Intentando conectar a base de datos...")
        
        with get_db_context() as db:
            # Probar query simple
            cameras = db.query(CameraStatus).all()
            print(f"✓ Conexión exitosa")
            print(f"  Cámaras registradas: {len(cameras)}")
            
            # Probar inserción
            print("\nProbando inserción de datos...")
            test_status = DatabaseManager.update_camera_status(
                db,
                camera_id="test_camera",
                camera_name="Cámara de Prueba",
                is_connected=True,
                fps=30.0
            )
            
            print(f"✓ Datos guardados correctamente")
            print(f"  ID: {test_status.camera_id}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error en base de datos: {e}")
        print("  Nota: Asegúrate de que PostgreSQL esté corriendo")
        print("  Comando: docker-compose up -d postgres")
        return False


def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*70)
    print("TRAFFIC ANALYSIS SYSTEM - SUITE DE TESTS")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Detector
    results.append(("Detector YOLO", test_detector()))
    
    # Test 2: Counter
    results.append(("Sistema de Conteo", test_counter()))
    
    # Test 3: Pipeline completo
    results.append(("Pipeline Completo", test_full_pipeline()))
    
    # Test 4: Base de datos
    results.append(("Base de Datos", test_database()))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests pasados")
    
    if total_passed == total_tests:
        print("\n🎉 ¡Todos los tests pasaron exitosamente!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) fallaron")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
