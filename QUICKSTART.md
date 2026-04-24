# Quick Start

## 1) Preparar entorno

```bash
cp .env.example .env
```

Editá `.env` y definí al menos:

- `JWT_SECRET_KEY`
- `INGEST_API_KEY`
- `ADMIN_API_KEY`

Luego creá el primer usuario admin en base de datos:

```bash
curl -X POST http://localhost:8000/api/v1/auth/bootstrap-admin \
  -H "Content-Type: application/json" \
  -H "X-Admin-API-Key: change-this-admin-key" \
  -d '{"username":"admin","password":"ChangeThisNow123"}'
```

Si vas a correr edge:

- `CAMERA_ID` y `CAMERA_NAME`
- `CAMERA_RTSP_URL`

## 2) Levantar backend central

```bash
docker compose up -d postgres redis server-backend
```

Validar:

```bash
curl http://localhost:8000/health
```

## 3) (Opcional) levantar worker/beat/flower

```bash
docker compose --profile workers up -d celery-worker celery-beat flower
```

## 4) (Opcional) levantar edge

```bash
docker compose --profile edge up -d edge-backend
```

## 5) Accesos

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Flower (si perfil workers): `http://localhost:5555`

## 6) Apagar stack

```bash
docker compose down
```

## Troubleshooting rápido

- `docker compose config` falla: revisar `.env` y sintaxis YAML.
- `/health` devuelve 503: verificar conexión a Postgres/Redis y credenciales.
- Edge no inicia: confirmar `CAMERA_RTSP_URL` válido y accesible desde el host/contenedor.
- Install de `edge-backend` falla con Python 3.14: usar Python 3.11.
