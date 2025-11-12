# ⚙️ Konfigurasi & Tweaking Training

Panduan lengkap untuk mengatur parameter training dan optimasi model.

---

## 📊 Parameter Utama

Semua parameter bisa diubah di **`train.py`** (baris 25-30) atau di **cell konfigurasi** di `train.ipynb`.

```python
# === BASIC CONFIGURATION ===
IMG_SIZE = (224, 224)      # Ukuran input gambar (jangan diubah untuk MobileNetV2)
BATCH_SIZE = 16            # Jumlah gambar per batch
EPOCHS = 200               # Jumlah iterasi training
DATASET_DIR = 'dataset'    # Lokasi folder dataset

# === OPTIMIZER SETTINGS ===
LEARNING_RATE = 0.001      # Learning rate untuk Adam optimizer

# === MODEL ARCHITECTURE ===
DROPOUT_RATE = 0.3         # Dropout untuk regularisasi (0.0 - 0.5)
DENSE_UNITS = 128          # Jumlah neuron di layer Dense
```

---

## 🎛️ Panduan Tweaking

### 1. BATCH_SIZE - Jumlah gambar per batch

```python
BATCH_SIZE = 32    # Default (balance antara speed & memory)
BATCH_SIZE = 16    # Jika GPU RAM terbatas atau OOM error
BATCH_SIZE = 8     # Untuk GPU kecil (<4GB VRAM)
BATCH_SIZE = 64    # Untuk GPU besar (>10GB VRAM) - training lebih cepat
```

**Efek:**
- ⬆️ Lebih besar = Training lebih cepat, butuh RAM lebih banyak
- ⬇️ Lebih kecil = Training lebih lambat, RAM lebih hemat
- 💡 Tip: Gunakan kelipatan 8 (8, 16, 32, 64) untuk performa optimal GPU

---

### 2. EPOCHS - Jumlah iterasi training

```python
EPOCHS = 50      # Quick testing (5-10 menit)
EPOCHS = 100     # Standard training
EPOCHS = 200     # Deep training (rekomendasi untuk production)
EPOCHS = 500     # Over-training (hati-hati overfitting!)
```

**Efek:**
- ⬆️ Lebih banyak = Accuracy lebih tinggi (sampai titik jenuh)
- ⬇️ Lebih sedikit = Training lebih cepat, accuracy mungkin kurang
- 💡 Tip: Gunakan **Early Stopping** (sudah aktif) untuk stop otomatis

---

### 3. LEARNING_RATE - Kecepatan learning

```python
LEARNING_RATE = 0.01     # Fast learning (berisiko overshoot)
LEARNING_RATE = 0.001    # Default (recommended)
LEARNING_RATE = 0.0001   # Slow learning (lebih stabil)
LEARNING_RATE = 0.00001  # Very slow (untuk fine-tuning)
```

**Efek:**
- ⬆️ Lebih besar = Convergence cepat, tapi bisa unstable
- ⬇️ Lebih kecil = Training lebih stabil, tapi lambat
- 💡 Tip: Mulai dengan 0.001, turunkan jika loss berfluktuasi

---

### 4. DROPOUT_RATE - Regularisasi untuk cegah overfitting

```python
DROPOUT_RATE = 0.0     # No dropout (berisiko overfitting)
DROPOUT_RATE = 0.2     # Light regularization
DROPOUT_RATE = 0.3     # Default (recommended)
DROPOUT_RATE = 0.5     # Heavy regularization (cegah overfitting kuat)
DROPOUT_RATE = 0.7     # Too much (underfitting)
```

**Kapan diubah:**
- ⬆️ Naikan jika: Training acc >> Validation acc (overfitting)
- ⬇️ Turunkan jika: Validation acc stagnan rendah (underfitting)
- 💡 Tip: Sweet spot biasanya 0.3 - 0.5

---

### 5. DENSE_UNITS - Kompleksitas model

```python
DENSE_UNITS = 64       # Simple model (lebih cepat)
DENSE_UNITS = 128      # Default (balance)
DENSE_UNITS = 256      # Complex model (lebih akurat, lambat)
DENSE_UNITS = 512      # Very complex (butuh data lebih banyak)
```

**Efek:**
- ⬆️ Lebih besar = Model lebih powerful, butuh data lebih banyak
- ⬇️ Lebih kecil = Model lebih cepat, simpel
- 💡 Tip: 128 cocok untuk dataset 500-1000 gambar

---

## 🎯 Callback Settings (Advanced)

Edit di `train.py` sekitar baris 140-165:

```python
# 1. EARLY STOPPING - Stop otomatis jika tidak membaik
EarlyStopping(
    monitor='val_loss',     # Metric yang dimonitor
    patience=15,            # Tunggu 15 epoch sebelum stop
    restore_best_weights=True
)

# 2. LEARNING RATE REDUCTION - Turunkan LR otomatis
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,             # LR dikurangi 50%
    patience=7,             # Tunggu 7 epoch
    min_lr=1e-7            # LR minimum
)

# 3. MODEL CHECKPOINT - Simpan model terbaik
ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',  # Berdasarkan val_accuracy
    save_best_only=True     # Hanya simpan yang terbaik
)
```

### Tweaking Callbacks

| Parameter | Default | Jika Overfitting | Jika Underfitting |
|-----------|---------|------------------|-------------------|
| `patience` (EarlyStopping) | 15 | 10 | 20 |
| `patience` (ReduceLR) | 7 | 5 | 10 |
| `factor` (ReduceLR) | 0.5 | 0.3 | 0.7 |

---

## 📈 Data Augmentation (Advanced)

Edit di `train.py` sekitar baris 90-98:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        # Rotasi ±20° (0-180)
    width_shift_range=0.2,    # Geser horizontal 20% (0.0-1.0)
    height_shift_range=0.2,   # Geser vertical 20%
    horizontal_flip=True,     # Flip horizontal random
    zoom_range=0.2,           # Zoom in/out 20%
    shear_range=0.15,         # Shear transformation
    fill_mode='nearest'       # Fill untuk pixel kosong
)
```

### Tweaking Augmentation

**LIGHT AUGMENTATION** (untuk dataset besar > 1000 images):
```python
rotation_range=10
width_shift_range=0.1
height_shift_range=0.1
zoom_range=0.1
```

**HEAVY AUGMENTATION** (untuk dataset kecil < 500 images):
```python
rotation_range=30
width_shift_range=0.3
height_shift_range=0.3
zoom_range=0.3
brightness_range=[0.8, 1.2]  # Tambahan: variasi brightness
```

---

## 🔧 Problem-Solution Matrix

### Problem 1: Training Accuracy tinggi, Validation Accuracy rendah (OVERFITTING)

**Solusi:**
```python
DROPOUT_RATE = 0.5          # Naikan dari 0.3
BATCH_SIZE = 32             # Perbesar dari 16
# Tambah augmentation (lihat di atas)
```

---

### Problem 2: Training & Validation Accuracy sama-sama rendah (UNDERFITTING)

**Solusi:**
```python
EPOCHS = 300                # Tambah epochs
DENSE_UNITS = 256           # Perbesar dari 128
DROPOUT_RATE = 0.2          # Turunkan dari 0.3
LEARNING_RATE = 0.001       # Pastikan tidak terlalu kecil
```

---

### Problem 3: Loss tidak turun atau fluktuatif

**Solusi:**
```python
LEARNING_RATE = 0.0001      # Turunkan dari 0.001
BATCH_SIZE = 32             # Perbesar untuk stabilitas
```

---

### Problem 4: Out of Memory (OOM)

**Solusi:**
```python
BATCH_SIZE = 8              # Turunkan dari 16
# Atau tambahkan di awal script:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
```

---

### Problem 5: Training terlalu lambat

**Solusi:**
```python
BATCH_SIZE = 32             # Naikan dari 16
DENSE_UNITS = 64            # Turunkan dari 128
EPOCHS = 100                # Kurangi dari 200
```

---

## 📊 Interpretasi Hasil

### Good Training Results
```
Final Training Accuracy:    0.9523 (95.23%)
Final Validation Accuracy:  0.9176 (91.76%)
Gap: 3.47% → ✅ Good generalization
```

### Overfitting (Perlu tweaking)
```
Final Training Accuracy:    0.9850 (98.50%)
Final Validation Accuracy:  0.8235 (82.35%)
Gap: 16.15% → ⚠️ Overfitting! (Naikan DROPOUT_RATE)
```

### Underfitting (Perlu lebih banyak epochs/complexity)
```
Final Training Accuracy:    0.7821 (78.21%)
Final Validation Accuracy:  0.7512 (75.12%)
Gap: 3.09% → ⚠️ Underfitting! (Tambah EPOCHS atau DENSE_UNITS)
```

---

## 🎯 Target Accuracy

- **≥ 90%** → Excellent (production ready) ✅
- **85-90%** → Good (acceptable)
- **80-85%** → Fair (perlu improvement)
- **< 80%** → Poor (perlu retrain dengan lebih banyak data)

**Gap Training vs Validation:**
- **< 5%** → Perfect generalization ✅
- **5-10%** → Good (slight overfitting)
- **10-15%** → Moderate overfitting (naikan dropout)
- **> 15%** → Severe overfitting (tambah data + naikan dropout)

---

## 📝 Model Architecture

- **Base Model:** MobileNetV2 (pre-trained ImageNet)
- **Custom Head:** GlobalAveragePooling → Dense(128) → Dropout(0.3) → Dense(1)
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Crossentropy

---

Kembali ke [README.md](README.md) | Lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
