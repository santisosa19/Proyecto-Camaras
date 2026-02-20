# Sistema de Análisis de Tráfico y Comportamiento de Clientes
## Marathon SRL - Traffic Analysis System

Sistema de visión computacional para análisis de tráfico de clientes en locales de retail.

### 🎯 Características

- **Detección de personas en tiempo real** con YOLOv8
- **Conteo automático** de entradas y salidas
- **Integración con CRM Cegid** para cálculo de conversión
- **Heatmaps** de zonas más transitadas
- **Dashboard web** en tiempo real
- **API REST** para integraciones

### 📁 Estructura del Proyecto

```
traffic-analysis-system/
├── backend/                 # API y servicios de procesamiento
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── api/            # Endpoints REST
│   │   ├── models/         # Modelos de datos
│   │   ├── services/       # Lógica de negocio
│   │   │   ├── video_capture.py
│   │   │   ├── detector.py
│   │   │   ├── counter.py
│   │   │   └── heatmap.py
│   │   └── database/       # Acceso a datos
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
├── frontend/               # Dashboard React
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

### 🚀 Quick Start

#### 1. Clonar y configurar
```bash
git clone [repo-url]
cd traffic-analysis-system
cp .env.example .env
# Editar .env con tus configuraciones
```

#### 2. Iniciar con Docker
```bash
docker-compose up -d
```

#### 3. Acceder
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Dashboard: http://localhost:3000

### 📋 Requisitos

#### Hardware
- CPU: 4+ cores
- RAM: 8GB mínimo
- Storage: 100GB
- GPU: Opcional (mejora 3-5x el rendimiento)

#### Software
- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 14+
- Redis 7+
- Node.js 18+

### 🔧 Configuración

#### Variables de Entorno
```bash
# Base de datos
DATABASE_URL=postgresql://user:pass@localhost:5432/traffic_db

# Redis
REDIS_URL=redis://localhost:6379/0

# Cámaras RTSP
CAMERA_1_URL=rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/102

# Cegid
CEGID_DB_URL=postgresql://readonly:pass@cegid-server:5432/cegid

# JWT
JWT_SECRET=your-secret-key-here
```

### 📊 Stack Tecnológico

**Backend:**
- Python 3.10+
- FastAPI (API REST)
- YOLOv8 (Detección)
- OpenCV (Procesamiento de video)
- PostgreSQL (Base de datos)
- Redis + Celery (Colas y tareas async)

**Frontend:**
- React 18
- Next.js 14
- TailwindCSS
- Recharts (Visualización)

**DevOps:**
- Docker & Docker Compose
- Nginx (Reverse proxy)

### 🎯 Fases del Proyecto

#### ✅ Fase 1: POC (2.5 semanas)
- [x] Captura RTSP
- [x] Detección con YOLO
- [x] Conteo básico
- [x] API REST
- [x] Dashboard simple

#### 🔄 Fase 2: MVP (3 semanas)
- [ ] Procesamiento 24/7
- [ ] Integración Cegid
- [ ] Dashboard completo
- [ ] Deploy en locales

#### 📈 Fase 3: Avanzado (2.5 semanas)
- [ ] Heatmaps
- [ ] Análisis de comportamiento
- [ ] Reportes automáticos
- [ ] API completa

### 👨‍💻 Desarrollo

#### Instalar dependencias backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Ejecutar tests
```bash
pytest tests/
```

#### Desarrollo frontend
```bash
cd frontend
npm install
npm run dev
```

### 📝 Documentación

- [Propuesta Ejecutiva](docs/01_Propuesta_Ejecutiva.docx)
- [Plan de Proyecto](docs/02_Plan_Proyecto.docx)
- [Arquitectura Técnica](docs/03_Arquitectura_Tecnica.docx)
- [Manual de Implementación](docs/04_Manual_Implementacion.docx)

### 🤝 Contribución

Desarrollado por **Santiago Sosa** para Marathon SRL.

### 📄 Licencia

Propietario - Marathon SRL © 2025
