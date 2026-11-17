.PHONY: check test lint format help format-fix install-hooks docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make check        - Run all CI checks (format check, lint, test)"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Run ruff format checker (check only)"
	@echo "  make test         - Run pytest"
	@echo "  make format-fix   - Auto-fix formatting and fixable lint issues"
	@echo "  make docker-build - Build the Docker image"
	@echo "  make docker-run   - Run the MCP server in Docker (stdio)"

check: format lint test

format:
	ruff format --check .

lint:
	ruff check .

test:
	pytest

format-fix:
	ruff format .
	ruff check --fix .

install-hooks:
	pip install pre-commit
	pre-commit install

docker-build:
	docker build -t sktime-mcp .

docker-run: docker-build
	docker run -i --rm sktime-mcp
