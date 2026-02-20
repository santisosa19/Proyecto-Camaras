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
