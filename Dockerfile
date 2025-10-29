# ----------------------------------------------------
# Cana ML API - Dockerfile
# ----------------------------------------------------
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instala dependências do sistema (para TensorFlow e MySQL)
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    default-libmysqlclient-dev \
    libglib2.0-0 \
    libgl1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia os requirements corretos (da pasta api/)
COPY api/requirements.txt /app/requirements.txt

# Instala dependências Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do projeto
COPY . /app

# Torna o script de boot executável
RUN chmod +x /app/api/boot.sh

# Expõe a porta do Flask
EXPOSE 5000

# Comando padrão
CMD ["python", "api/app.py"]