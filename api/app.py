"""
app.py
-------------------------------------
API Flask do projeto Cana ML.
Permite upload de imagem e retorna diagnóstico de saúde da cana.
Autor: Adão
"""

import os
import glob
import json
import numpy as np
from datetime import datetime

from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =====================================================
# IMPORTS INTERNOS (compatíveis local + Docker)
# =====================================================
try:
    from api.database import Analise, get_session
except ModuleNotFoundError:
    from database import Analise, get_session

# =====================================================
# CONFIGURAÇÕES GERAIS
# =====================================================
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
IMG_SIZE = (224, 224)

print("🚀 Inicializando API Cana ML...")


def _find_latest_model(models_dir: str) -> str:
    """Retorna o caminho do modelo mais recente (.keras ou .h5) dentro de models_dir."""
    pattern_keras = os.path.join(models_dir, "*.keras")
    pattern_h5 = os.path.join(models_dir, "*.h5")
    candidates = glob.glob(pattern_keras) + glob.glob(pattern_h5)
    if not candidates:
        raise FileNotFoundError(f"❌ Nenhum modelo encontrado em {models_dir} (.keras ou .h5).")
    # mais recente por mtime
    latest = max(candidates, key=os.path.getmtime)
    return latest


# Carrega modelo mais recente automaticamente
MODEL_PATH = _find_latest_model(MODELS_DIR)
model = load_model(MODEL_PATH)
print(f"✅ Modelo carregado com sucesso: {os.path.basename(MODEL_PATH)}")

# =====================================================
# CLASSES (ordem do treino) + nomes de exibição
#  -> Keras usa ordem alfabética das pastas no treino
# =====================================================
MODEL_CLASSES = [
    "amarelecimento",
    "falsa_mancha",
    "ferrugem",
    "helmintosporiose",
    "mancha_vermelha",
    "mosaic",
    "sadia",
]

DISPLAY_MAP = {
    "amarelecimento": "Amarelecimento",
    "falsa_mancha": "Falsa Manhã",
    "ferrugem": "Ferrugem da Cana",
    "helmintosporiose": "Helmintosporiose",
    "mancha_vermelha": "Mancha Vermelha",
    "mosaic": "Mosaico",
    "sadia": "Sadia",
}

# Lista útil para /classes
CLASSES_DISPLAY = [DISPLAY_MAP[c] for c in MODEL_CLASSES]


# =====================================================
# FUNÇÃO AUXILIAR DE INTERPRETAÇÃO (calibrada)
# =====================================================
def interpretar_resultado(preds):
    """
    Recebe 'preds' (saída do modelo, shape [1, num_classes] com probabilidades)
    Retorna: (classe_display, confiança_em_% , faixa_confianca)
    Regras:
      - Só confirma doença se a confiança ultrapassar o limiar da classe.
      - Caso contrário, classifica como 'Sadia' com confiança complementar.
    """
    probs = preds[0].astype(float)
    idx_max = int(np.argmax(probs))
    classe_raw = MODEL_CLASSES[idx_max]           # nome interno (do treino)
    classe_display = DISPLAY_MAP[classe_raw]      # nome bonito para a UI
    conf_max = float(probs[idx_max])              # 0..1

    # Limiar mínimo por classe (ajuste fino depois com seus dados)
    THRESHOLDS = {
        "ferrugem": 0.80,
        "mancha_vermelha": 0.78,
        "helmintosporiose": 0.78,
        "falsa_mancha": 0.75,
        "mosaic": 0.78,
        "amarelecimento": 0.78,
        "sadia": 0.0,  # fallback
    }

    # Se a top já for Sadia, mantém
    if classe_raw == "sadia":
        predicted_display = DISPLAY_MAP["sadia"]
        confidence = conf_max
    else:
        limiar = THRESHOLDS.get(classe_raw, 0.80)
        if conf_max >= limiar:
            predicted_display = classe_display
            confidence = conf_max
        else:
            predicted_display = DISPLAY_MAP["sadia"]
            confidence = 1.0 - conf_max  # confiança complementar

    # Faixas
    conf_pct = confidence * 100.0
    if conf_pct >= 85:
        faixa = "Alta"
    elif conf_pct >= 60:
        faixa = "Média"
    else:
        faixa = "Baixa"

    return predicted_display, confidence, faixa


# =====================================================
# ROTAS
# =====================================================
@app.route("/", methods=["GET"])
def login_page():
    """Renderiza a tela de login"""
    return render_template("login.html")


@app.route("/auth", methods=["POST"])
def autenticar():
    """Simula autenticação simples"""
    username = request.form.get("username")
    password = request.form.get("password")
    if username == "admin" and password == "1234":
        return redirect(url_for("dashboard"))
    else:
        return jsonify({"error": "Credenciais inválidas"}), 401


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Renderiza o dashboard"""
    return render_template("dashboard.html")


@app.route("/health", methods=["GET"])
def health():
    """Verifica se o servidor está ativo"""
    return jsonify({
        "status": "ok",
        "message": "API Cana ML está online.",
        "model": os.path.basename(MODEL_PATH)
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """Recebe imagem, executa o modelo e retorna diagnóstico com faixa de confiança e geolocalização."""
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nome de arquivo inválido."}), 400

    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

    session = get_session()
    try:
        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", file.filename)
        file.save(temp_path)

        # Pré-processa (consistente com treino: rescale 1/255)
        img = image.load_img(temp_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predição
        preds = model.predict(img_array)
        predicted_class, confidence, faixa = interpretar_resultado(preds)

        # Salva no banco
        nova_analise = Analise(
            classe=predicted_class,
            confianca=confidence * 100,
            nome_arquivo=file.filename,
            latitude=latitude,
            longitude=longitude
        )
        session.add(nova_analise)
        session.commit()

        return jsonify({
            "classe": predicted_class,
            "confianca": round(confidence * 100, 2),
            "faixa_confianca": faixa,
            "latitude": latitude,
            "longitude": longitude
        })

    except Exception as e:
        session.rollback()
        print("❌ Erro na análise:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Permite acessar imagens enviadas"""
    return send_from_directory("uploads", filename)


@app.route("/results", methods=["GET"])
def results():
    """Retorna histórico de análises"""
    session = get_session()
    try:
        analises = session.query(Analise).order_by(Analise.data.desc()).all()
        return jsonify([analise.to_dict() for analise in analises]), 200
    except Exception as e:
        print("❌ Erro ao carregar histórico:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.route("/classes", methods=["GET"])
def get_classes():
    """Retorna lista de classes exibidas"""
    return jsonify(CLASSES_DISPLAY)


# =====================================================
# INICIALIZAÇÃO
# =====================================================
if __name__ == "__main__":
    # Porta interna segue 5000; no docker-compose você mapeia 5001:5000
    app.run(host="0.0.0.0", port=5000, debug=True)



