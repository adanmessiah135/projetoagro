"""
train_model.py
Treina o modelo de detecção de doenças da cana-de-açúcar.
Autor: Adão
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# ===========================================================
# CONFIGURAÇÕES
# ===========================================================
DATASET_DIR = "dataset"
MODEL_PATH = "api/ml/models/model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15  # Pode aumentar se quiser mais precisão (20~30)

# ===========================================================
# CARREGAMENTO DO DATASET
# ===========================================================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation'
)

# ===========================================================
# CONSTRUÇÃO DO MODELO (TRANSFER LEARNING)
# ===========================================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # Congela a base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===========================================================
# TREINAMENTO
# ===========================================================
print("\n🚀 Iniciando treinamento...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ===========================================================
# SALVAR MODELO
# ===========================================================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"\n✅ Modelo salvo em: {MODEL_PATH}")

# ===========================================================
# EXIBIR INFORMAÇÕES
# ===========================================================
print("\n📘 Classes detectadas:")
for cls, idx in train_gen.class_indices.items():
    print(f" - {cls} → índice {idx}")

# ===========================================================
# GRÁFICOS DE DESEMPENHO
# ===========================================================
plt.figure(figsize=(12, 5))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Evolução da Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()

