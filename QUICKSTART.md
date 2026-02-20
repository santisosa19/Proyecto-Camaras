# 🚀 Guía de Inicio Rápido - Traffic Analysis System

## ✅ Pre-requisitos

Antes de empezar, asegúrate de tener instalado:

- **Docker** y **Docker Compose**
- **Python 3.10+** (para desarrollo local)
- **Git**
- **Cámara Hikvision** o cualquier cámara compatible con RTSP

## 📦 Instalación

### 1. Descomprimir el proyecto

```bash
tar -xzf traffic-analysis-system.tar.gz
cd traffic-analysis-system
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
nano .env  # Editar con tus configuraciones
```

**Importante:** Actualiza las siguientes variables en `.env`:
- `CAMERA_LOCAL_1_URL`: URL RTSP de tu cámara Hikvision
- `DATABASE_URL`: Si usas Docker, déjalo como está
- `JWT_SECRET_KEY`: Cambia a un valor seguro en producción

### 3. Iniciar servicios con Docker

```bash
# Construir imágenes
docker-compose build

# Iniciar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f
```

### 4. Verificar que todo funciona

```bash
# Check API
curl http://localhost:8000/health

# Debería responder: {"status":"healthy",...}
```

## 🎯 Acceso a los Servicios

Una vez iniciado, tendrás acceso a:

- **API REST**: http://localhost:8000
- **Documentación Interactiva (Swagger)**: http://localhost:8000/docs
- **Base de Datos**: localhost:5432 (user: traffic_user, pass: secure_password_123)
- **Redis**: localhost:6379
- **Flower (Monitor Celery)**: http://localhost:5555

## 🧪 Probar el Sistema

### Opción A: Test Sin Cámara Real

```bash
cd backend
python test_system.py
```

Este script prueba todos los componentes sin necesidad de cámara real.

### Opción B: Test con Cámara Real

1. Asegúrate de tener tu cámara configurada en `.env`
2. Ejecuta el siguiente script:

```python
# test_camera.py
import cv2

# Reemplaza con tu URL RTSP
rtsp_url = "rtsp://admin:Password123@192.168.1.100:554/Streaming/Channels/102"

cap = cv2.VideoCapture(rtsp_url)

if cap.isOpened():
    print("✓ Conexión exitosa a la cámara")
    ret, frame = cap.read()
    if ret:
        print(f"✓ Frame capturado: {frame.shape}")
    else:
        print("✗ Error leyendo frame")
else:
    print("✗ No se pudo conectar a la cámara")

cap.release()
```

## 📝 Uso Básico de la API

### 1. Registrar una cámara

```bash
curl -X POST "http://localhost:8000/api/v1/cameras/" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "local_centro",
    "camera_name": "Local Centro",
    "rtsp_url": "rtsp://admin:pass@192.168.1.100:554/stream",
    "fps": 15
  }'
```

### 2. Listar cámaras

```bash
curl http://localhost:8000/api/v1/cameras/
```

### 3. Obtener métricas en tiempo real

```bash
curl http://localhost:8000/api/v1/metrics/realtime/local_centro
```

### 4. Obtener métricas por hora

```bash
curl "http://localhost:8000/api/v1/metrics/hourly/local_centro?target_date=2025-02-14"
```

### 5. Obtener flujo de entradas/salidas por hora

```bash
curl "http://localhost:8000/api/v1/metrics/flow/local_centro?target_date=2025-02-14&initial_occupancy=0"
```

## 🔧 Desarrollo Local (Sin Docker)

Si prefieres desarrollar sin Docker:

### 1. Crear entorno virtual

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar PostgreSQL local

```bash
# Instala PostgreSQL si no lo tienes
# Ubuntu/Debian:
sudo apt-get install postgresql

# Crear base de datos
sudo -u postgres psql
CREATE DATABASE traffic_db;
CREATE USER traffic_user WITH PASSWORD 'secure_password_123';
GRANT ALL PRIVILEGES ON DATABASE traffic_db TO traffic_user;
\q
```

### 4. Iniciar servicios

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: PostgreSQL (ya corriendo como servicio)

# Terminal 3: API
cd backend
python -m uvicorn app.main:app --reload

# Terminal 4: Celery Worker (opcional)
celery -A app.celery_app worker --loglevel=info
```

## 🐛 Troubleshooting

### Problema: No se conecta a la cámara

**Solución:**
1. Verifica que la cámara esté en la misma red
2. Prueba la URL RTSP con VLC primero
3. Verifica credenciales y puerto (554 por defecto)

### Problema: Error de permisos en Docker

**Solución:**
```bash
sudo usermod -aG docker $USER
# Logout y login de nuevo
```

### Problema: Puerto 8000 ya en uso

**Solución:**
```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8001:8000"  # Usar 8001 en vez de 8000
```

### Problema: Base de datos no inicializa

**Solución:**
```bash
# Eliminar volumen y recrear
docker-compose down -v
docker-compose up -d postgres
# Esperar 10 segundos
docker-compose up -d
```

## 📚 Próximos Pasos

1. **Configura tu cámara**: Edita `.env` con tus URLs RTSP
2. **Explora la API**: Ve a http://localhost:8000/docs
3. **Revisa los logs**: `docker-compose logs -f backend`
4. **Lee la documentación completa**: Ver carpeta `docs/`
5. **Configura líneas de conteo**: Modifica el código en `app/services/counter.py`

## 📞 Soporte

Para problemas o preguntas:
- Email: santiago.sosa@marathon.com
- Ver documentación en `docs/`
- Revisar logs: `docker-compose logs`

## 🎉 ¡Listo!

Tu sistema de análisis de tráfico está funcionando. Ahora puedes:
- Ver detecciones en tiempo real
- Configurar líneas de conteo
- Integrar con Cegid
- Generar reportes y heatmaps

**¡A desarrollar!** 🚀
