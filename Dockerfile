FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN mkdir -p logs results experiments data

# Use template config.ini
COPY config.ini /app/config.ini.template

# Create a script to initialize the container
RUN echo '#!/bin/bash\n\
if [ ! -f /app/config.ini ]; then\n\
  echo "Using template config.ini"\n\
  cp /app/config.ini.template /app/config.ini\n\
fi\n\
exec "$@"\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "src/api.py"]
