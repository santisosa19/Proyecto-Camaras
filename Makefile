.PHONY: help install run test clean docker-build docker-up docker-down db-init db-migrate

help:
	@echo "Traffic Analysis System - Comandos disponibles:"
	@echo ""
	@echo "  make install       - Instalar dependencias"
	@echo "  make run           - Ejecutar servidor de desarrollo"
	@echo "  make test          - Ejecutar tests"
	@echo "  make test-system   - Ejecutar test completo del sistema"
	@echo "  make clean         - Limpiar archivos temporales"
	@echo "  make docker-build  - Construir imágenes Docker"
	@echo "  make docker-up     - Iniciar servicios Docker"
	@echo "  make docker-down   - Detener servicios Docker"
	@echo "  make db-init       - Inicializar base de datos"
	@echo "  make logs          - Ver logs de Docker"
	@echo ""

install:
	cd backend && pip install -r requirements.txt

run:
	cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	cd backend && pytest tests/ -v

test-system:
	cd backend && python test_system.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Servicios iniciados:"
	@echo "  - API:      http://localhost:8000"
	@echo "  - Docs:     http://localhost:8000/docs"
	@echo "  - Frontend: http://localhost:3000"
	@echo "  - Flower:   http://localhost:5555"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

db-init:
	cd backend && python -c "from app.database.connection import init_db; init_db(); print('✓ Base de datos inicializada')"

db-shell:
	docker-compose exec postgres psql -U traffic_user -d traffic_db

format:
	cd backend && black app/
	cd backend && isort app/

lint:
	cd backend && flake8 app/
	cd backend && mypy app/

dev-setup:
	@echo "Configurando entorno de desarrollo..."
	cp .env.example .env
	@echo "✓ Archivo .env creado"
	@echo "  Edita .env con tus configuraciones"
	make install
	@echo "✓ Dependencias instaladas"
	make docker-up
	@echo "✓ Servicios Docker iniciados"
	sleep 5
	make db-init
	@echo "✓ Base de datos inicializada"
	@echo ""
	@echo "🎉 ¡Entorno listo!"
	@echo "   Accede a http://localhost:8000/docs para ver la API"
