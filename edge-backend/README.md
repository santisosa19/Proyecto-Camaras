# Edge Backend (Sucursal)

Este agente corre en cada sucursal. Procesa RTSP localmente y envía eventos al servidor central.

## Ejecutar local

1. Preparar entorno:

```bash
cp .env.example .env
```

2. Completar variables mínimas en `.env`:

```bash
CAMERA_ID=camara_local_001
CAMERA_NAME=Ingreso Local 001
CAMERA_RTSP_URL=rtsp://user:pass@ip_dvr:554/Streaming/Channels/101
SAVE_TO_API=true
REMOTE_API_BASE_URL=http://ip-servidor-central:8000
REMOTE_API_KEY=tu_ingest_api_key
```

Para piloto (1-2 locales), recomendado:

```bash
PILOT_MODE=true
DETECTION_SNAPSHOT_INTERVAL_SECONDS=8
```

3. Levantar:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_camera.py
```

## Ajuste de precisión/tracking (opcional)

Para mejorar estabilidad de boxes cuando una persona está quieta o parcialmente ocluida:

```bash
YOLO_MODEL_PATH=yolov8n.pt
YOLO_DEVICE=auto
YOLO_CONFIDENCE=0.22
YOLO_IOU=0.60
YOLO_IMAGE_SIZE=1280
YOLO_TRACKER=trackers/bytetrack_stable.yaml
MAX_TRACK_AGE_SECONDS=8
```

## Heatmap de ocupación (nuevo)

El edge ahora acumula un mapa de calor por cámara usando centroides de detección.

Controles en ventana:
- `h`: mostrar/ocultar overlay del heatmap
- `k`: resetear acumulación del heatmap

Variables recomendadas:

```bash
HEATMAP_ENABLED=true
SHOW_HEATMAP_OVERLAY=true
HEATMAP_CELL_SIZE=24
HEATMAP_OVERLAY_ALPHA=0.35
HEATMAP_BLUR_KERNEL=21
HEATMAP_DECAY_PER_SECOND=0.0
SAVE_HEATMAP_SNAPSHOTS=true
HEATMAP_SNAPSHOT_INTERVAL_SECONDS=60
HEATMAP_OUTPUT_DIR=heatmaps
HEATMAP_KEEP_HISTORY=false
SEND_HOURLY_HEATMAP_TO_API=true
HEATMAP_BACKGROUND_MAX_WIDTH=960
HEATMAP_BACKGROUND_JPEG_QUALITY=68
HEATMAP_BACKGROUND_REFRESH_SECONDS=30
```

Snapshots generados:
- archivo `latest`: `heatmaps/<camera_id>_heatmap_latest.json`
- opcional histórico por timestamp si `HEATMAP_KEEP_HISTORY=true`
- ingesta central por hora (si `SAVE_TO_API=true`): endpoint `/api/v1/ingest/heatmaps`

## Docker

```bash
docker build -t traffic-edge-backend .
docker run --env-file .env traffic-edge-backend
```

## Excluir empleados del conteo

El sistema puede clasificar tracks como `employee` o `non_employee` y excluir empleados del conteo.

### 1) Dataset para entrenamiento

```bash
dataset/
  train/
    employee/
    non_employee/
  val/
    employee/
    non_employee/
```

### 2) Entrenar modelo

```bash
python scripts/train_employee_classifier.py \
  --dataset-dir ./dataset \
  --output ./models/employee_classifier.pt \
  --epochs 8 \
  --batch-size 32
```

### 3) Activar en runtime

```bash
EMPLOYEE_FILTER_ENABLED=true
EMPLOYEE_MODEL_PATH=./models/employee_classifier.pt
EMPLOYEE_DEVICE=cpu
EMPLOYEE_THRESHOLD=0.75
EMPLOYEE_VOTE_WINDOW=8
EMPLOYEE_MIN_VOTES=5
```

Tip: en pruebas de oficina dejalo en `false`; activalo cuando montes cámara en local.
