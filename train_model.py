import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# 🔧 CONFIGURAÇÕES GERAIS
# ============================================================
BASE_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = "ml/models/cana_model_v3_finetuned.h5"

# ============================================================
# 📊 GERADORES DE DADOS
# ============================================================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
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
# 🧠 MODELO BASE - EfficientNetB0
# ============================================================
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# 🔓 Descongela as últimas 30 camadas da EfficientNet para fine-tuning
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

print(f"🔧 Camadas treináveis: {sum([l.trainable for l in base_model.layers])} de {len(base_model.layers)}")

# ============================================================
# 🧩 MONTAGEM DO MODELO
# ============================================================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dense(train_generator.num_classes, activation="softmax")
])

# ============================================================
# ⚙️ COMPILAÇÃO
# ============================================================
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),  # menor LR para fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 🧩 CALLBACKS
# ============================================================
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

# ============================================================
# 🚀 TREINAMENTO
# ============================================================
print("\n🚀 Iniciando Fine-Tuning...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ============================================================
# 📈 RELATÓRIO DE RESULTADOS
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(history.history["accuracy"], label="Treino")
plt.plot(history.history["val_accuracy"], label="Validação")
plt.title("Acurácia do Modelo (Fine-Tuning)")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.tight_layout()
plt.savefig("ml/training_accuracy_finetuned.png")
plt.show()

print("\n✅ Fine-tuning concluído com sucesso!")
print(f"📁 Modelo salvo em: {MODEL_PATH}")
print("🧠 Classes detectadas:", list(train_generator.class_indices.keys()))



