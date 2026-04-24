# Server Backend (Central)

Backend central FastAPI para:

- gestión de cámaras
- métricas y dashboard
- ingestión remota desde sucursales (`/api/v1/ingest/*`)

## Ejecutar local

1. Configurar variables:

```bash
cp .env.example .env
```

2. Levantar API:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. Verificar:

```bash
curl http://localhost:8000/health
```

## Docker

```bash
docker build -t traffic-server-backend .
docker run --env-file .env -p 8000:8000 traffic-server-backend
```

## Seguridad

- `INGEST_API_KEY`: protege endpoints de ingesta remota.
- `ADMIN_API_KEY`: protege endpoints administrativos (alta/baja/activación de cámaras y control de procesamiento).
- En `ENVIRONMENT=production`, ambas claves son obligatorias.

El login del dashboard usa usuarios en tabla `app_users`.

Bootstrap inicial (una sola vez cuando no hay usuarios):

```bash
curl -X POST http://localhost:8000/api/v1/auth/bootstrap-admin \
  -H "Content-Type: application/json" \
  -H "X-Admin-API-Key: <ADMIN_API_KEY>" \
  -d '{"username":"admin","password":"ChangeThisNow123"}'
```

Con token de superadmin podés crear más usuarios con `POST /api/v1/auth/users`.
