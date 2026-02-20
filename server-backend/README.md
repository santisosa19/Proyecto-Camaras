# Server Backend (Central)

Este backend corre en el servidor central y expone la API para:
- gestión de cámaras
- consultas de métricas
- ingestión remota desde sucursales (`/api/v1/ingest/*`)

## Ejecutar local

1. Copiar variables:

```bash
cp .env.example .env
```

2. Levantar API:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t traffic-server-backend .
docker run --env-file .env -p 8000:8000 traffic-server-backend
```
