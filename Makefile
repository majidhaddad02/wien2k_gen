# ==============================================================================
# Wien2kGen Production Makefile
# Modern, HPC-ready, and aligned with pyproject.toml & offline installation
# ==============================================================================

PYTHON ?= python3
PIP ?= pip3
VENV ?= .venv
APP_NAME := wien2k_gen
SRC_DIR := src
PKG_DIR := $(SRC_DIR)/$(APP_NAME)
COMPLETIONS_DIR := completions
OFFLINE_DIR := offline_packages

.PHONY: all install dev minimal test lint format clean build \
        install-offline download-offline install-completions \
        run tui wizard docker singularity help

all: install

# ==============================================================================
# Environment & Setup
# ==============================================================================
install: $(VENV)/bin/activate
	@$(VENV)/bin/python -m pip install --upgrade pip setuptools wheel
	@$(VENV)/bin/python -m pip install -e .
	@echo "✅ Core dependencies installed."

dev: $(VENV)/bin/activate
	@$(VENV)/bin/python -m pip install -e ".[dev]"
	@$(VENV)/bin/pre-commit install
	@echo "🛠️  Dev environment & pre-commit hooks ready."

minimal: $(VENV)/bin/activate
	@$(VENV)/bin/python -m pip install --upgrade pip setuptools
	@$(VENV)/bin/python -m pip install --no-deps -e .
	@$(VENV)/bin/python -m pip install rich pyyaml numpy psutil
	@echo "📦 Minimal install: only essential deps (no textual TUI)."

$(VENV)/bin/activate: pyproject.toml
	@$(PYTHON) -m venv $(VENV)
	@$(VENV)/bin/python -m pip install --upgrade pip setuptools wheel
	@echo "📦 Virtual environment created at $(VENV)"

# ==============================================================================
# Testing & Quality Assurance
# ==============================================================================
test:
	@$(VENV)/bin/pytest tests/ -v -n auto --cov=$(APP_NAME) --cov-report=term-missing

test-html:
	@$(VENV)/bin/pytest tests/ -v --cov=$(APP_NAME) --cov-report=html:coverage_html
	@echo "🌐 Coverage report generated at coverage_html/index.html"

lint:
	@$(VENV)/bin/ruff check $(SRC_DIR)/ tests/
	@$(VENV)/bin/mypy $(PKG_DIR)/ --ignore-missing-imports

format:
	@$(VENV)/bin/ruff format $(SRC_DIR)/ tests/
	@echo "✨ Code formatted successfully."

# ==============================================================================
# Offline Support (HPC/Cluster Friendly)
# ==============================================================================
download-offline: $(VENV)/bin/activate
	@mkdir -p $(OFFLINE_DIR)
	@$(VENV)/bin/pip download -r $(OFFLINE_DIR)/requirements-offline.txt -d $(OFFLINE_DIR)/ \
		--only-binary=:all: --python-version 3.9 --platform manylinux_2_17_x86_64
	@echo "💾 Offline packages downloaded to $(OFFLINE_DIR)/"

install-offline: $(VENV)/bin/activate
	@if [ -d "$(OFFLINE_DIR)" ] && [ "$(shell ls -A $(OFFLINE_DIR) 2>/dev/null)" ]; then \
		$(VENV)/bin/pip install --no-index --find-links=$(OFFLINE_DIR) -e ".[dev]"; \
		echo "✅ Offline installation complete."; \
	else \
		echo "❌ Offline directory missing or empty. Run 'make download-offline' first."; \
		exit 1; \
	fi

# ==============================================================================
# Packaging & Deployment
# ==============================================================================
build: $(VENV)/bin/activate
	@$(VENV)/bin/python -m build --sdist --wheel
	@echo "📦 Package built in dist/"

install-completions:
	@mkdir -p ~/.local/share/bash-completion/completions
	@mkdir -p ~/.local/share/zsh/site-functions
	@cp $(COMPLETIONS_DIR)/wien2k_gen.bash ~/.local/share/bash-completion/completions/wien2k_gen 2>/dev/null || true
	@cp $(COMPLETIONS_DIR)/wien2k_sbatch.bash ~/.local/share/bash-completion/completions/wien2k_sbatch 2>/dev/null || true
	@cp $(COMPLETIONS_DIR)/wien2k_wizard.bash ~/.local/share/bash-completion/completions/wien2k_wizard 2>/dev/null || true
	@cp $(COMPLETIONS_DIR)/wien2k_gen.zsh ~/.local/share/zsh/site-functions/_wien2k_gen 2>/dev/null || true
	@cp $(COMPLETIONS_DIR)/wien2k_sbatch.zsh ~/.local/share/zsh/site-functions/_wien2k_sbatch 2>/dev/null || true
	@cp $(COMPLETIONS_DIR)/wien2k_wizard.zsh ~/.local/share/zsh/site-functions/_wien2k_wizard 2>/dev/null || true
	@echo "🔗 Shell completions installed. Restart your shell or run: source ~/.bashrc"

docker:
	@docker build --no-cache -t $(APP_NAME):latest -f Dockerfile .
	@echo "🐳 Docker image built: $(APP_NAME):latest"

singularity:
	@apptainer build --force $(APP_NAME).sif Singularity.def
	@echo "📦 Singularity image built: $(APP_NAME).sif"

# ==============================================================================
# Developer Shortcuts
# ==============================================================================
run:
	@$(VENV)/bin/$(APP_NAME) --help

tui:
	@$(VENV)/bin/$(APP_NAME) tui

wizard:
	@$(VENV)/bin/wien2k_wizard

# ==============================================================================
# Cleanup
# ==============================================================================
clean:
	rm -rf $(VENV) dist/ build/ *.egg-info coverage.xml .pytest_cache .mypy_cache .ruff_cache
	rm -rf coverage_html/ $(OFFLINE_DIR)/ $(APP_NAME).sif
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@docker rmi -f $(APP_NAME):latest 2>/dev/null || true
	@echo "🧹 Cleaned build artifacts, caches, and containers."

help:
	@echo "📖 Wien2kGen Makefile Targets:"
	@echo "  make install          → Create venv & install core deps"
	@echo "  make dev              → Install dev deps + pre-commit hooks"
	@echo "  make minimal          → Install only essential deps (no TUI)"
	@echo "  make test             → Run pytest with terminal coverage"
	@echo "  make test-html        → Run pytest with HTML coverage report"
	@echo "  make lint             → Ruff + MyPy static analysis"
	@echo "  make format           → Auto-format code with Ruff"
	@echo "  make download-offline → Download wheels for HPC cluster install"
	@echo "  make install-offline  → Install from local $(OFFLINE_DIR)/"
	@echo "  make build            → Build sdist & wheel"
	@echo "  make install-completions → Install shell auto-completions"
	@echo "  make docker           → Build Docker image"
	@echo "  make singularity      → Build Apptainer/Singularity image"
	@echo "  make tui              → Launch interactive TUI"
	@echo "  make wizard           → Launch CLI wizard"
	@echo "  make clean            → Remove venv, caches, builds, and containers"