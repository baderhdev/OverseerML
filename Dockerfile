FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
WORKDIR /app

# Serving dependencies only: the train/dev groups (CryptoGuard, SHAP, pytest) stay out.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-cache

# The artifact is built on the host by `uv run overseer-train` (see README).
COPY models/v1 ./models/v1

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "overseerml.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
