# Traffic Analysis System

Sistema distribuido para conteo de personas en sucursales (edge) y consolidación de métricas en servidor central.

## Arquitectura

- `edge-backend/`: agente en sucursal que consume RTSP, detecta personas y envía eventos/snapshots al central.
- `server-backend/`: API FastAPI central para ingestión, métricas y dashboard.
- `frontend/`: dashboard React (Vite).

## Requisitos

- Docker + Docker Compose v2
- Python 3.11 (recomendado para `edge-backend`)
- Node.js 18+ (si corrés frontend local)

## Configuración

1. Copiar configuración base:

```bash
cp .env.example .env
```

2. Ajustar como mínimo en `.env`:

- `JWT_SECRET_KEY`
- `INGEST_API_KEY`
- `ADMIN_API_KEY`
- `CAMERA_RTSP_URL` (si vas a levantar edge)

## Ejecución con Docker

### Stack central (recomendado)

```bash
docker compose up -d postgres redis server-backend
```

Servicios:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Workers (opcional)

```bash
docker compose --profile workers up -d celery-worker celery-beat flower
```

### Edge agent (opcional)

```bash
docker compose --profile edge up -d edge-backend
```

## Ejecución local (sin Docker)

### Server central

```bash
cd server-backend
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Edge agent

```bash
cd edge-backend
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_camera.py
```

## Seguridad operativa incorporada

- Validación estricta de configuración en producción (`ENVIRONMENT=production`).
- `INGEST_API_KEY` obligatoria en producción para `/api/v1/ingest/*`.
- `ADMIN_API_KEY` para endpoints administrativos de cámaras/procesamiento.
- Health checks reales de DB y Redis (`/health`, `/health/ready`).
- Middleware de request-id y headers de seguridad básicos.

## Comandos útiles

```bash
make help
make docker-config
make docker-up
make docker-up-workers
make docker-up-edge
make health
```
