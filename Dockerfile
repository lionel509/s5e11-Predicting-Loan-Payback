# syntax=docker/dockerfile:1.6
# Production Dockerfile for running loan_meta_optimized.ipynb non-interactively
# Multi-stage build keeps final image lean.

ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} AS base

# Prevent interactive prompts & set UTF-8 locale
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# System deps (build + runtime) for scientific stack & LightGBM/CatBoost/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libgomp1 \
        libopenblas-dev \
        liblapack-dev \
        libx11-6 \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements early to leverage Docker layer caching
COPY requirements.txt ./requirements.txt

# Add additional production dependencies
# - jupyter & papermill: notebook run automation
# - catboost, lightgbm, xgboost: gradient boosters used in notebook
# - optuna (optional) left commented; enable if tuning added
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt \
 && pip install jupyter papermill catboost lightgbm xgboost \
 && pip check || true

# Create a non-root user for security
RUN useradd -m -u 1000 runner && chown -R runner:runner /app
USER runner

# Copy project (only what's needed for execution)
COPY --chown=runner:runner . .

# Default environment variables for notebook execution
ENV NOTEBOOK_PATH=loan_meta_optimized.ipynb \
    OUTPUT_NOTEBOOK=executed_loan_meta_optimized.ipynb \
    PARAMETERS='{}'

# Entrypoint script executes notebook via papermill
COPY --chown=runner:runner run_notebook.sh /usr/local/bin/run_notebook.sh
RUN chmod +x /usr/local/bin/run_notebook.sh

ENTRYPOINT ["/usr/local/bin/run_notebook.sh"]
CMD []
