.PHONY: check test lint format help format-fix install-hooks

help:
	@echo "Available commands:"
	@echo "  make check      - Run all CI checks (format check, lint, test)"
	@echo "  make lint       - Run ruff linter"
	@echo "  make format     - Run ruff format checker (check only)"
	@echo "  make test       - Run pytest"
	@echo "  make format-fix - Auto-fix formatting and fixable lint issues"

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
