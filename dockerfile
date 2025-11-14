# -----------------------------------------
#  Base image with uv (Python 3.11)
# -----------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

WORKDIR /app

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock README.md ./
COPY credit_risk_prediction ./credit_risk_prediction

# Install dependencies
RUN uv sync --locked

# Copy project code

# Expose ports for API + Streamlit
EXPOSE 8000
EXPOSE 8501

# Default command (overridden by docker-compose)
CMD ["bash"]
