import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 🔧 CONFIGURAÇÕES GERAIS
# ============================================================
BASE_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# --- CAMINHOS DE SAÍDA ---
OUTPUT_DIR = "ml"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cana_model_v5_mobilenet.keras")
ACCURACY_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_accuracy_mobilenet.png")
CM_PLOT_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_mobilenet.png")

# ✨ AJUSTE: Cria os diretórios de saída se não existirem ✨
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\n🚀 Iniciando fine-tuning com MobileNetV2 (ImageNet)...")
print(f"Salvando modelo em: {MODEL_PATH}")
print(f"Salvando gráficos em: {OUTPUT_DIR}/")

# ============================================================
# 📊 GERADORES DE DADOS
# ============================================================
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Gerador para TREINO (com Data Augmentation e pré-processamento)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # <-- ESTA É A CORREÇÃO
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  # Separa 20% para validação
)

# 2. Gerador para VALIDAÇÃO (APENAS pré-processamento, SEM Augmentation)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # <-- CORREÇÃO
    validation_split=0.2  # Deve ser o mesmo split
)

# 3. Cria os geradores a partir das pastas
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"  # Pega a fatia de treino
)

val_generator = val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",  # Pega a fatia de validação
    shuffle=False
)

# ============================================================
# ‼️ ‼️ BLOCO DE VERIFICAÇÃO ‼️ ‼️
# ============================================================
# ✨ ADICIONADO: Verifica se as pastas foram lidas corretamente
print("\n" + "="*50)
print("VERIFICANDO PASTAS E CLASSES:")
print("Classes encontradas:", train_generator.class_indices)
print(f"Total de imagens de treino: {train_generator.samples}")
print(f"Total de imagens de validação: {val_generator.samples}")
print(f"Total de classes detectadas: {train_generator.num_classes}")
print("="*50 + "\n")

# Se "Total de classes detectadas" não for 5, PARE E ARRUME AS PASTAS.
# ============================================================

# ============================================================
# 🧠 MODELO BASE (MobileNetV2 PRÉ-TREINADO)
# ============================================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congela as camadas iniciais

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)  # Regularização forte
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)  # Mais regularização
predictions = layers.Dense(train_generator.num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4), # Volte para 1e-4
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 📈 CALLBACKS
# ============================================================
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,  # Para 5 épocas sem melhoria na perda de validação
    restore_best_weights=True
)

# ============================================================
# 🚀 TREINAMENTO (APENAS ETAPA 1)
# ============================================================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ============================================================
# 📈 RELATÓRIO DE RESULTADOS
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Treino")
plt.plot(history.history["val_accuracy"], label="Validação")
plt.title("Acurácia do Modelo (MobileNetV2 Fine-Tuning)")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.tight_layout()
plt.savefig(ACCURACY_PLOT_PATH) # Salva no caminho definido
plt.show()

# ============================================================
# 🧠 RELATÓRIO AUTOMÁTICO DE CONFIANÇA
# ============================================================
# Encontra a melhor acurácia de validação alcançada
final_val_acc = max(history.history["val_accuracy"]) * 100
if final_val_acc >= 80:
    interpretacao = "Alta confiança ✅ — modelo muito consistente."
elif final_val_acc >= 65:
    interpretacao = "Confiança média 🔶 — modelo razoável, pode melhorar com mais dados."
else:
    interpretacao = "Confiança baixa ⚠️ — modelo precisa de mais exemplos ou ajuste de classes."

print(f"\n📊 Acurácia final de validação: {final_val_acc:.2f}%")
print(f"💬 Interpretação: {interpretacao}")

# ============================================================
# 📉 MATRIZ DE CONFUSÃO E RELATÓRIO POR CLASSE
# ============================================================
print("\n📊 Gerando matriz de confusão e relatório por classe...")

# Predições no conjunto de validação
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Matriz de confusão normalizada
cm = confusion_matrix(y_true, y_pred_classes, normalize="true")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Matriz de Confusão Normalizada (Validação)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(CM_PLOT_PATH) # Salva no caminho definido
plt.show()

# Relatório detalhado
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print("\n📋 Relatório de Classificação:\n")
print(report)

print(f"\n✅ Fine-tuning concluído e salvo em: {MODEL_PATH}")
print("🧠 Classes detectadas:", class_labels)