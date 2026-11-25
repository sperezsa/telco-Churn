# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.12.21 AS uv

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock ./
# This model runs on CPU; its Linux XGBoost dependency does not need the NCCL GPU runtime.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --no-install-project --no-install-package nvidia-nccl-cu13

COPY README.md ./
COPY src ./src
COPY api ./api
COPY monitoring ./monitoring
COPY config ./config
COPY feature_store ./feature_store

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --no-install-package nvidia-nccl-cu13

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
