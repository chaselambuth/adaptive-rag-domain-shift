FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src HF_HOME=/models
WORKDIR /app
COPY requirements.txt .
# CPU wheels avoid downloading CUDA libraries for this local demonstration.
RUN python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY config ./config
COPY run.py .
RUN useradd --create-home rag && mkdir -p /models && chown rag:rag /models
USER rag
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "300", "app.web:create_app()"]
