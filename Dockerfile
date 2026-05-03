FROM python:3.14-alpine

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY src/ src/

# Run application
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "src/main.py"]

