# Edge Backend (Sucursal)

Este agente corre en cada sucursal. Procesa RTSP localmente y envía eventos al servidor central.

## Ejecutar local

1. Preparar entorno:

```bash
cp .env.example .env
```

2. Completar variables mínimas en `.env`:

```bash
CAMERA_RTSP_URL=rtsp://user:pass@ip_dvr:554/Streaming/Channels/101
SAVE_TO_API=true
REMOTE_API_BASE_URL=http://ip-servidor-central:8000
REMOTE_API_KEY=tu_ingest_api_key
```

3. Levantar:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_camera.py
```

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
EMPLOYEE_MODEL_PATH=./models/employee_classifier.pt
EMPLOYEE_THRESHOLD=0.75
EMPLOYEE_VOTE_WINDOW=8
EMPLOYEE_MIN_VOTES=5
```
