.PHONY: help install-server install-edge install-frontend run-server run-edge \
\tdocker-config docker-build docker-up docker-up-workers docker-up-edge \
\tdocker-down docker-logs health check-server check-edge check-frontend verify clean

help:
	@echo "Traffic Analysis System - comandos disponibles:"
	@echo ""
	@echo "  make install-server    - Instalar dependencias de server-backend"
	@echo "  make install-edge      - Instalar dependencias de edge-backend"
	@echo "  make install-frontend  - Instalar dependencias de frontend"
	@echo "  make run-server        - Ejecutar API central en local"
	@echo "  make run-edge          - Ejecutar agente de cámara en local"
	@echo "  make docker-config     - Validar docker compose"
	@echo "  make docker-build      - Construir imágenes"
	@echo "  make docker-up         - Levantar postgres + redis + server-backend"
	@echo "  make docker-up-workers - Levantar worker/beat/flower (perfil workers)"
	@echo "  make docker-up-edge    - Levantar edge-backend (perfil edge)"
	@echo "  make docker-down       - Detener stack"
	@echo "  make docker-logs       - Ver logs"
	@echo "  make health            - Consultar health de la API"
	@echo "  make check-server      - Validación sintáctica del backend central"
	@echo "  make check-edge        - Validación sintáctica del backend edge"
	@echo "  make check-frontend    - Build de frontend"
	@echo "  make verify            - Ejecuta check-server + check-edge + check-frontend"
	@echo ""

install-server:
	cd server-backend && python3 -m pip install -r requirements.txt

install-edge:
	cd edge-backend && python3 -m pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

run-server:
	cd server-backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

run-edge:
	cd edge-backend && python run_camera.py

docker-config:
	docker compose config

docker-build:
	docker compose build

docker-up:
	docker compose up -d postgres redis server-backend

docker-up-workers:
	docker compose --profile workers up -d celery-worker celery-beat flower

docker-up-edge:
	docker compose --profile edge up -d edge-backend

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f --tail=200

health:
	curl -fsS http://localhost:8000/health | python3 -m json.tool

check-server:
	python3 -m compileall server-backend/app

check-edge:
	python3 -m compileall edge-backend/app edge-backend/run_camera.py

check-frontend:
	cd frontend && npm run build

verify: check-server check-edge check-frontend

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
