# train_model_finetune_v3.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# 🔧 CONFIGURAÇÕES GERAIS
# ============================================================
BASE_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = "ml/models/cana_model_v5_mobilenet.keras"

print("\n🚀 Iniciando fine-tuning com MobileNetV2 (ImageNet)...")

# ============================================================
# 📊 GERADORES DE DADOS
# ============================================================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"
)

# ============================================================
# 🧠 MODELO BASE (MobileNetV2 PRÉ-TREINADO)
# ============================================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
predictions = layers.Dense(train_generator.num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
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
    patience=5,
    restore_best_weights=True
)

# ============================================================
# 🚀 TREINAMENTO
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
plt.savefig("ml/training_accuracy_mobilenet.png")
plt.show()

print("\n✅ Fine-tuning concluído com sucesso!")
print(f"📁 Modelo salvo em: {MODEL_PATH}")
print("🧠 Classes detectadas:", list(train_generator.class_indices.keys()))

final_val_acc = max(history.history["val_accuracy"]) * 100

if final_val_acc >= 80:
    interpretacao = "Alta confiança ✅ — modelo muito consistente."
elif final_val_acc >= 65:
    interpretacao = "Confiança média 🔶 — modelo razoável, pode melhorar com mais dados."
else:
    interpretacao = "Confiança baixa ⚠️ — modelo precisa de mais exemplos ou ajuste de classes."

print(f"\n📊 Acurácia final de validação: {final_val_acc:.2f}%")
print(f"💬 Interpretação: {interpretacao}")



