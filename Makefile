# EdgeOps AI Platform: Automation Makefile

PYTHON = python
UVICORN = uvicorn
NPM = npm

.PHONY: all setup test lint run-api run-frontend run-stream docker-up docker-down clean

all: setup test

setup:
	$(PYTHON) -m pip install -r requirements.txt
	cd edgeops-dashboard && $(NPM) install

test:
	$(PYTHON) -m pytest tests/ -v

run-api:
	$(UVICORN) api:app --host 127.0.0.1 --port 8000 --reload

run-frontend:
	cd edgeops-dashboard && $(NPM) run dev

run-stream:
	$(PYTHON) data_streamer.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ .pytest_cache stream_data/*.csv temp_plots/*.png
