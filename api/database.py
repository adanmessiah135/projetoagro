import os
import pymysql
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# =====================================================
# CONFIGURAÇÃO DO BANCO
# =====================================================
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "root")
DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "cana_ml")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# =====================================================
# MODELO (TABELA)
# =====================================================
class Analise(Base):
    __tablename__ = "analises"

    id = Column(Integer, primary_key=True, autoincrement=True)
    classe = Column(String(100))
    confianca = Column(Float)
    nome_arquivo = Column(String(255))
    data = Column(DateTime, default=datetime.now)
    latitude = Column(String(50))
    longitude = Column(String(50))

    def to_dict(self):
        """Converte o objeto para um dicionário."""
        return {
            "id": self.id,
            "classe": self.classe,
            "confianca": self.confianca,
            "nome_arquivo": self.nome_arquivo,
            "data": self.data.isoformat() if self.data else None,
            "latitude": self.latitude,
            "longitude": self.longitude
        }

# =====================================================
# FUNÇÕES DE ACESSO E INICIALIZAÇÃO
# =====================================================
def get_session():
    """Cria e retorna uma nova sessão do banco de dados."""
    return SessionLocal()

def ensure_columns():
    """Garante que as colunas latitude e longitude existam na tabela analises."""
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = conn.cursor()

        cursor.execute("SHOW COLUMNS FROM analises LIKE 'latitude';")
        has_lat = cursor.fetchone()
        cursor.execute("SHOW COLUMNS FROM analises LIKE 'longitude';")
        has_lon = cursor.fetchone()

        if not has_lat:
            cursor.execute("ALTER TABLE analises ADD COLUMN latitude VARCHAR(50) NULL;")
            print("✅ Coluna 'latitude' adicionada com sucesso.")
        if not has_lon:
            cursor.execute("ALTER TABLE analises ADD COLUMN longitude VARCHAR(50) NULL;")
            print("✅ Coluna 'longitude' adicionada com sucesso.")

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"⚠️ Erro ao verificar/adicionar colunas latitude/longitude: {e}")

def init_db():
    """Cria a tabela 'analises' se não existir e garante colunas atualizadas."""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tabela 'analises' verificada/criada com sucesso!")
        ensure_columns()
    except Exception as e:
        print(f"❌ Erro ao inicializar o banco: {e}")

# =====================================================
# EXECUÇÃO DIRETA
# =====================================================
if __name__ == "__main__":
    init_db()



