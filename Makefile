PYTHON ?= $(if $(wildcard venv/bin/python),$(CURDIR)/venv/bin/python,python3)
COMPOSE ?= docker compose
export PYTHONPATH := $(CURDIR)
export UV_CACHE_DIR ?= $(CURDIR)/.cache/uv

.PHONY: test test-unit test-integration test-db test-down check generate-api check-api lint evaluation-smoke docker-check

test-db:
	$(COMPOSE) -f docker-compose.test.yml up -d --wait

test-unit:
	$(PYTHON) -m pytest tests/unit -q

test-integration:
	$(PYTHON) -m pytest tests/backend -q

test: test-unit test-integration

test-down:
	$(COMPOSE) -f docker-compose.test.yml down -v

lint:
	node scripts/check_lint.mjs

generate-api:
	$(PYTHON) -m scripts.export_openapi
	cd theseus-ui && npm run generate:api

check-api:
	$(PYTHON) -m scripts.export_openapi --check
	node scripts/check_api_types.mjs

check: test lint check-api evaluation-smoke
	cd theseus-ui && npm test && npm run build
	uv lock --check

# Requires Docker; separate from the fast local gate.
docker-check:
	docker build -t theseus-insight:validation .
	$(PYTHON) -m scripts.validate_container

evaluation-smoke:
	$(PYTHON) -m scripts.evaluate_research --smoke
