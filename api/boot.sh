#!/bin/sh

# Aborta em caso de erro
set -e

# 1. Espera o banco de dados ficar pronto
echo "⏳ Esperando o banco de dados ficar disponível..."
python api/wait_for_db.py

# 2. Inicializa o banco de dados (cria tabelas)
echo "🛠️  Inicializando o banco de dados (criando tabelas)..."
python -c "from api.database import init_db; init_db()"

# 3. Inicia o servidor Gunicorn
echo "🚀 Iniciando o servidor Gunicorn..."
gunicorn --bind 0.0.0.0:5000 api.app:app