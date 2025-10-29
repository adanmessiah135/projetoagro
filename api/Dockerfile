# Cana ML API - Dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependências de sistema
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    default-libmysqlclient-dev \
    libglib2.0-0 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instala dependências Python a partir do arquivo correto
COPY api/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . /app

EXPOSE 5000

CMD ["python", "api/app.py"]
