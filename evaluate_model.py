"""
evaluate_model.py
-------------------------------------
Avalia o modelo treinado (model.h5) do projeto Cana ML.
Gera métricas e gráficos de desempenho no diretório /reports.
Autor: Adão (MVP Cana ML)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================================================
# CONFIGURAÇÕES
# =====================================================
DATASET_DIR = os.path.join(os.getcwd(), "dataset")
MODEL_PATH = os.path.join(os.getcwd(), "api", "ml", "models", "model.h5")
REPORT_DIR = os.path.join(os.getcwd(), "reports")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# Cria pasta de relatórios
os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================================
# FUNÇÃO: carregar dataset para teste
# =====================================================
def create_test_generator(dataset_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    test_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    return test_gen

# =====================================================
# FUNÇÃO: gerar relatório e gráficos
# =====================================================
def evaluate_model(model, test_gen):
    print("📊 Avaliando modelo...")

    # Predições
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Acurácia: {acc * 100:.2f}%\n")

    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Distribuição das previsões
    plt.figure(figsize=(6, 4))
    sns.countplot(x=[class_labels[i] for i in y_pred], palette="viridis")
    plt.title("Distribuição das Previsões")
    plt.tight_layout()
    dist_path = os.path.join(REPORT_DIR, "predictions_distribution.png")
    plt.savefig(dist_path)
    plt.close()

    # Salva relatório JSON
    report_path = os.path.join(REPORT_DIR, "metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Relatórios salvos em: {REPORT_DIR}")
    print(f"- Acurácia geral: {acc * 100:.2f}%")
    print(f"- Matriz de confusão: {cm_path}")
    print(f"- Distribuição de classes: {dist_path}")
    print(f"- Relatório JSON: {report_path}")

# =====================================================
# EXECUÇÃO PRINCIPAL
# =====================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Modelo não encontrado em {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    test_gen = create_test_generator(DATASET_DIR)
    evaluate_model(model, test_gen)