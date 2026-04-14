.PHONY: check test lint format help format-fix

help:
	@echo "Available commands:"
	@echo "  make check      - Run all CI checks (format check, lint, test)"
	@echo "  make lint       - Run ruff linter"
	@echo "  make format     - Run black formatter (check only)"
	@echo "  make test       - Run pytest"
	@echo "  make format-fix - Auto-fix formatting and fixable lint issues"

check: format lint test

format:
	black --check .

lint:
	ruff check .

test:
	pytest

format-fix:
	black .
	ruff check --fix .
