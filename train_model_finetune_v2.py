import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ✨ IMPORTAÇÃO CORRIGIDA ✨
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 🎯 CONFIGURAÇÕES PARA GPU (ADAPTAÇÃO ESPECÍFICA)
# ============================================================
# Verifica se GPU está disponível e configura
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configura crescimento de memória para evitar alocação total da GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detectada e configurada: {len(gpus)} dispositivo(s)")
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"GPUs lógicas disponíveis: {len(logical_gpus)}")
    except RuntimeError as e:
        print(f"⚠️ Erro ao configurar GPU: {e}")
else:
    print("⚠️ Nenhuma GPU detectada. Usando CPU.")

# Opcional: Limita ao uso de 1 GPU se houver múltiplas
# tf.config.set_visible_devices(gpus[0], 'GPU')  # Descomente se quiser forçar 1 GPU

# Ativa mixed precision para acelerar treinamento em GPU (TensorFlow 2.4+)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')  # Usa float16 para aceleração, mas mantém estabilidade
print("🔥 Mixed precision ativado para aceleração em GPU.")

# ============================================================
# 🔧 CONFIGURAÇÕES GERAIS
# ============================================================
BASE_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Aumente se sua GPU tiver mais VRAM (ex: 64 para RTX 30xx+)
EPOCHS_ETAPA_1 = 30  # Épocas para a primeira fase (Transfer Learning)
EPOCHS_ETAPA_2 = 20  # Épocas extras para a segunda fase (Fine-Tuning)

# --- CAMINHOS DE SAÍDA (ETAPA 1) ---
OUTPUT_DIR = "ml"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cana_model_v5_base.keras") # Salva o melhor modelo base

# --- CAMINHOS DE SAÍDA (ETAPA 2) ---
MODEL_PATH_FT = os.path.join(MODEL_DIR, "cana_model_v5_FINETUNED.keras")
ACCURACY_PLOT_PATH_FT = os.path.join(OUTPUT_DIR, "training_accuracy_FINETUNED.png")
CM_PLOT_PATH_FT = os.path.join(OUTPUT_DIR, "confusion_matrix_FINETUNED.png")

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\n🚀 Iniciando Treinamento em Duas Etapas (com GPU otimizado)...")
print(f"Salvando modelo base em: {MODEL_PATH}")
print(f"Salvando modelo final em: {MODEL_PATH_FT}")

# ============================================================
# 📊 GERADORES DE DADOS (VERSÃO CORRIGIDA)
# ============================================================

# 1. Gerador para TREINO (com Data Augmentation e pré-processamento)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # <-- ✨ CORREÇÃO CRÍTICA
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

# 2. Gerador para VALIDAÇÃO (APENAS pré-processamento, SEM Augmentation)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # <-- ✨ CORREÇÃO CRÍTICA
    validation_split=0.2
)

# 3. Cria os geradores a partir das pastas
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"
)

val_generator = val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ============================================================
# ‼️ ‼️ BLOCO DE VERIFICAÇÃO ‼️ ‼️
# ============================================================
print("\n" + "="*50)
print("VERIFICANDO PASTAS E CLASSES:")
print("Classes encontradas:", train_generator.class_indices)
print(f"Total de imagens de treino: {train_generator.samples}")
print(f"Total de imagens de validação: {val_generator.samples}")
print(f"Total de classes detectadas: {train_generator.num_classes}")
print("="*50 + "\n")

# ============================================================
# 🧠 ETAPA 1: CRIAÇÃO E TREINO DO MODELO BASE
# ============================================================
print("Iniciando Etapa 1: Treinamento do 'cabeçote' (Top Layers)")

# ✨✨ ESTE É O CÓDIGO CORRETO PARA CRIAR O MODELO ✨✨
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3), name="mobilenet_base")
base_model.trainable = False  # Congela o modelo base

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
predictions = layers.Dense(train_generator.num_classes, activation="softmax", dtype='float32')(x)  # dtype para mixed precision

model = models.Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4), # LR segura para Transfer Learning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# Callbacks para a Etapa 1
checkpoint = ModelCheckpoint(
    MODEL_PATH, # Salva o melhor modelo DESTA etapa
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True # Garante que o 'model' final é o melhor
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history_stage1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_ETAPA_1,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Opcional: Recalcula métricas pós-restore_best_weights para gráfico suave
print("Recalculando métricas pós-restore para gráfico contínuo...")
train_loss, train_acc = model.evaluate(train_generator, verbose=0)
val_loss, val_acc = model.evaluate(val_generator, verbose=0)

# Atualiza o último ponto do histórico (evita 'pulo' no plot)
history_stage1.history['loss'][-1] = train_loss
history_stage1.history['accuracy'][-1] = train_acc
history_stage1.history['val_loss'][-1] = val_loss
history_stage1.history['val_accuracy'][-1] = val_acc
print(f"Métricas atualizadas: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")


# ============================================================
# 🔓 ETAPA 2: FINE-TUNING (AJUSTE FINO)
# ============================================================
print("\n" + "="*50)
print(f"Iniciando Etapa 2: Fine-Tuning (continuando do melhor da Etapa 1)")
print("Descongelando camadas superiores do base_model...")
print("="*50 + "\n")

# Opcional: Se quiser carregar o modelo salvo (garante o melhor, mas reinicia otimizador)
# model = models.load_model(MODEL_PATH)
# print(f"Modelo carregado de: {MODEL_PATH}")
# Nesse caso, mude o freeze para: for layer in model.layers[:fine_tune_at]: layer.trainable = False
# E set epochs=EPOCHS_ETAPA_2, initial_epoch=0

# Usa a referência original do base_model (já em escopo, sem get_layer!)
base_model.trainable = True  # Descongela TODO o base_model

# Congela as primeiras 100 camadas do base_model (early layers)
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
print(f"Camadas congeladas: 0 a {fine_tune_at-1} (base_model tem {len(base_model.layers)} camadas)")

# Recompila com LR baixa
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5), # 0.00001
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks para a Etapa 2 (salva em caminhos diferentes)
checkpoint_ft = ModelCheckpoint(
    MODEL_PATH_FT, # Salva o melhor modelo de FINE-TUNING
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop_ft = EarlyStopping(
    monitor="val_loss",
    patience=5, # 5 épocas de paciência para o fine-tuning
    restore_best_weights=True
)

# Continua o treinamento (fine-tuning) - CORRIGIDO: épocas exatas
num_epochs_stage1 = len(history_stage1.history['loss'])
history_stage2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=num_epochs_stage1 + EPOCHS_ETAPA_2, # Corrige: rodadas na 1 + 20 extras
    initial_epoch=num_epochs_stage1,
    callbacks=[checkpoint_ft, early_stop_ft, reduce_lr] # Re-usa o reduce_lr
)

# ============================================================
# 📈 RELATÓRIO DE RESULTADOS (COMBINADO)
# ============================================================
# Combina o histórico das duas etapas para um gráfico completo
acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
loss = history_stage1.history['loss'] + history_stage2.history['loss']
val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']

# Plota a acurácia combinada
plt.figure(figsize=(12, 6))
plt.plot(acc, label="Treino")
plt.plot(val_acc, label="Validação")
# Marca o início do Fine-Tuning - CORRIGIDO
plt.axvline(x=num_epochs_stage1 - 1, color='red', linestyle='--', label='Início do Fine-Tuning')
plt.title("Acurácia do Modelo (Completo: Transfer + Fine-Tuning)")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.tight_layout()
plt.savefig(ACCURACY_PLOT_PATH_FT) # Salva o novo gráfico
plt.show()

# ============================================================
# 🧠 RELATÓRIO AUTOMÁTICO DE CONFIANÇA (PÓS-ETAPA 2)
# ============================================================
# Pega o melhor resultado da Etapa 2 (ou Etapa 1 se a 2 não melhorar)
final_val_acc = max(val_acc) * 100
if final_val_acc >= 80:
    interpretacao = "Alta confiança ✅ — modelo muito consistente."
elif final_val_acc >= 65:
    interpretacao = "Confiança média 🔶 — modelo razoável."
else:
    interpretacao = "Confiança baixa ⚠️ — modelo precisa de mais dados."

print(f"\n📊 Acurácia final de validação: {final_val_acc:.2f}%")
print(f"💬 Interpretação: {interpretacao}")

# ============================================================
# 📉 MATRIZ DE CONFUSÃO E RELATÓRIO POR CLASSE (PÓS-ETAPA 2)
# ============================================================
print("\n📊 Gerando matriz de confusão e relatório (Modelo Final Pós-Fine-Tuning)...")

# O 'model' já foi restaurado para o melhor da Etapa 2 pelo EarlyStopping
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Matriz de confusão normalizada
cm = confusion_matrix(y_true, y_pred_classes, normalize="true")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Matriz de Confusão Normalizada (Modelo Final Pós-Fine-Tuning)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(CM_PLOT_PATH_FT) # ✨ CORREÇÃO: Salva no caminho correto
plt.show()

# Relatório detalhado
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print("\n📋 Relatório de Classificação (Modelo Final Pós-Fine-Tuning):\n")
print(report)

print(f"\n✅ Fine-tuning concluído e salvo em: {MODEL_PATH_FT}") # ✨ CORREÇÃO: Mostra o caminho correto
print("🧠 Classes detectadas:", class_labels)

# ============================================================
# 🧪 TESTE RÁPIDO EM IMAGEM NOVA (OPCIONAL - CORRIGIDO)
# ============================================================
# Descomente e ajuste o path para testar uma imagem nova
# from tensorflow.keras.preprocessing import image
# img_path = "path/to/sua_imagem_teste.jpg"  # Troque pelo path real
# img = image.load_img(img_path, target_size=IMG_SIZE)
# img_array = image.img_to_array(img)
# img_array_expanded = np.expand_dims(img_array, axis=0)
# img_preprocessed = preprocess_input(img_array_expanded) # ✨ CORREÇÃO: Usa preprocess_input
#
# pred = model.predict(img_preprocessed)
# class_idx = np.argmax(pred)
# confidence = pred[0][class_idx]
# print(f"\n🧪 Predição em imagem de teste: {class_labels[class_idx]} (confiança: {confidence:.2f})")

