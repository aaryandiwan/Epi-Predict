FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Expose ports
EXPOSE 8000 8501

# Create supervisor configuration to run both services
RUN echo "[supervisord]" > /etc/supervisor/conf.d/supervisord.conf && \
    echo "nodaemon=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:api]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=uvicorn api.main:app --host 0.0.0.0 --port 8000" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:dashboard]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0" >> /etc/supervisor/conf.d/supervisord.conf

# Run both via supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
