-- Inicialización de base de datos
-- Traffic Analysis System

-- Extensiones útiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Función para actualizar timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers para updated_at
-- (se crearán automáticamente con SQLAlchemy)

-- Índices adicionales para optimizar queries
-- (también se crean con SQLAlchemy, pero aquí como referencia)

-- Índices opcionales (solo si las tablas existen)
DO $$
BEGIN
    IF to_regclass('public.detections') IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS idx_detections_camera_timestamp
            ON detections(camera_id, timestamp DESC);
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('public.hourly_metrics') IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS idx_hourly_metrics_lookup
            ON hourly_metrics(camera_id, date, hour);
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('public.crossing_events') IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS idx_crossing_events_lookup
            ON crossing_events(camera_id, event_type, timestamp);
    END IF;
END $$;

-- Insertar datos de ejemplo (opcional)
-- INSERT INTO camera_status (camera_id, camera_name, rtsp_url, is_active)
-- VALUES ('local_1', 'Local Centro', 'rtsp://admin:pass@192.168.1.100:554/stream', true);

COMMIT;
