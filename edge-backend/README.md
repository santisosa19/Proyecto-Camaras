# Edge Backend (Sucursal)

Este backend corre en cada sucursal. Procesa RTSP localmente y envía eventos al servidor central.

## Ejecutar local

1. Copiar variables:

```bash
cp .env.example .env
```

2. Definir variables necesarias en terminal:

```bash
export CAMERA_RTSP_URL='rtsp://user:pass@ip_dvr:554/Streaming/Channels/101'
export SAVE_TO_DB=false
export SAVE_TO_API=true
export REMOTE_API_BASE_URL='http://ip-servidor-central:8000'
export REMOTE_API_KEY='tu_api_key'
```

3. Iniciar agente:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_camera.py
```

## Docker

```bash
docker build -t traffic-edge-backend .
docker run --env-file .env traffic-edge-backend
```

## Excluir empleados del conteo

El sistema puede clasificar cada track como `employee` o `non_employee` y excluir empleados del conteo de cruces.

### 1) Dataset para entrenamiento

Estructura esperada:

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

Variables de entorno:

```bash
export EMPLOYEE_MODEL_PATH='./models/employee_classifier.pt'
export EMPLOYEE_THRESHOLD='0.75'
export EMPLOYEE_VOTE_WINDOW='8'
export EMPLOYEE_MIN_VOTES='5'
```

Con estas variables, los tracks clasificados como empleados no se contabilizan en entradas/salidas.
