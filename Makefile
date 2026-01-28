.PHONY: install run dev deploy lint format clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make run        - Run the sandbox agent"
	@echo "  make dev        - Start LangGraph dev server"
	@echo "  make deploy     - Deploy to LangGraph Cloud"
	@echo "  make lint       - Run linting with ruff"
	@echo "  make format     - Format code with ruff"
	@echo "  make clean      - Clean up cache files"

# Install dependencies
install:
	uv sync

# Run the agent directly
run:
	uv run python main.py

# Start LangGraph dev server (requires langgraph-cli)
dev:
	uv run langgraph dev

# Deploy to LangGraph Cloud
deploy:
	uv run langgraph deploy

# Lint code
lint:
	uv run ruff check .

# Format code
format:
	uv run ruff format .
	uv run ruff check --fix .

# Clean up
clean:
	rm -rf __pycache__ .ruff_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
