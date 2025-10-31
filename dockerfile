FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

# Copy dependencies
COPY pyproject.toml uv.lock ./

RUN uv sync

# Copy project
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Run Streamlit (or Flask)
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
