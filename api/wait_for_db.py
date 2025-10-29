import time
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

def wait_for_db():
    """Espera o banco de dados ficar disponível."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        db_host = os.getenv("DB_HOST", "db")
        db_user = os.getenv("DB_USER", "root")
        db_password = os.getenv("DB_PASS", "root")
        db_name = os.getenv("DB_NAME", "cana_ml")
        db_port = os.getenv("DB_PORT", 3306)
        db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    engine = create_engine(db_url)
    
    retries = 30
    delay = 2  # segundos

    for i in range(retries):
        try:
            connection = engine.connect()
            connection.close()
            print("✅ Conexão com o banco de dados estabelecida com sucesso!")
            return
        except OperationalError as e:
            print(f"⏳ Banco de dados indisponível, tentando novamente em {delay}s... ({i+1}/{retries})")
            print(f"   (Erro: {e})")
            time.sleep(delay)

    print("❌ Não foi possível conectar ao banco de dados após várias tentativas.")
    exit(1)

if __name__ == "__main__":
    wait_for_db()