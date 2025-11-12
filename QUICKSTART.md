# 🚀 Quick Start Guide

Panduan super cepat untuk mulai training dalam 5 menit!

---

## ⚡ 5-Minute Quick Start

### Step 1: Setup Environment (2 menit)

```powershell
# 1. Buka PowerShell/Terminal di folder project
cd "d:\Flutter Interesting Thing\SolDef_AI PCB dataset for defect detection\SolDef_AI"

# 2. Jalankan setup otomatis
.\setup.bat

# Setup akan:
# ✅ Create virtual environment (.venv)
# ✅ Install TensorFlow 2.10.0
# ✅ Install dependencies
# ✅ Verify GPU
```

**Atau setup manual:**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 2: Cek Dataset (30 detik)

Dataset sudah tersedia di folder `dataset/`:
```
dataset/
├── def_front/    # 453 images (cacat produksi)
└── ok_front/     # 563 images (lulus QC)
```

**Verifikasi:**
```powershell
dir dataset\def_front | Measure-Object -Line
dir dataset\ok_front | Measure-Object -Line
```

---

### Step 3: Training! (20-25 menit)

**Cara Paling Mudah:**
```powershell
.\train.bat
```

**Atau manual:**
```powershell
.venv\Scripts\activate
python train.py
```

**Atau pakai Jupyter Notebook:**
```powershell
jupyter notebook train.ipynb
```

---

### Step 4: Monitor Progress (real-time)

Saat training berjalan, Anda akan lihat:
```
Epoch 1/200
342/342 [==============================] - 8s 23ms/step
loss: 0.3456 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.7891

Epoch 2/200
342/342 [==============================] - 6s 18ms/step
loss: 0.2345 - accuracy: 0.8912 - val_loss: 0.3234 - val_accuracy: 0.8456
...
```

**Metrics to watch:**
- 📈 `accuracy` → Training accuracy (target: ≥ 95%)
- 📊 `val_accuracy` → Validation accuracy (target: ≥ 90%)
- 📉 `loss` → Training loss (makin kecil makin bagus)
- 📉 `val_loss` → Validation loss (makin kecil makin bagus)

---

### Step 5: Check Results (1 menit)

Setelah training selesai:
```
✅ Training Complete!
Final Training Accuracy:    95.23%
Final Validation Accuracy:  91.76%
Model saved: qc_inspector_model.h5
```

**Output Files:**
```
✅ qc_inspector_model.h5        # Model final
✅ best_model.h5                # Backup terbaik
✅ training_history.json        # Metrics lengkap
✅ logs/                        # TensorBoard logs
```

---

## 🎯 What's Next?

### Test Model with New Image

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('qc_inspector_model.h5')

# Load image
img_path = 'test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
result = "CACAT ❌" if prediction[0][0] < 0.5 else "LULUS ✅"
confidence = (1 - prediction[0][0]) * 100 if prediction[0][0] < 0.5 else prediction[0][0] * 100

print(f"Hasil: {result}")
print(f"Confidence: {confidence:.2f}%")
```

---

## 🔧 Common Commands

### Activate Virtual Environment
```powershell
.venv\Scripts\activate
```

### Deactivate Virtual Environment
```powershell
deactivate
```

### Check GPU
```powershell
nvidia-smi
```

### Verify TensorFlow & GPU
```python
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### Re-train Model
```powershell
.\train.bat
# Atau:
python train.py
```

---

## ⚙️ Quick Configuration

Edit di `train.py` baris 25-30:

### For Faster Training (Lower Accuracy)
```python
EPOCHS = 50          # Quick test
BATCH_SIZE = 32      # Larger batch
DENSE_UNITS = 64     # Simpler model
```

### For Better Accuracy (Slower)
```python
EPOCHS = 300         # More iterations
BATCH_SIZE = 16      # Default
DENSE_UNITS = 256    # More complex model
DROPOUT_RATE = 0.5   # Prevent overfitting
```

### For GPU with Low VRAM (<4GB)
```python
BATCH_SIZE = 8       # Smaller batch
DENSE_UNITS = 64     # Simpler model
```

**Detail lengkap:** [CONFIGURATION.md](CONFIGURATION.md)

---

## 🐛 Quick Troubleshooting

### Problem: GPU tidak terdeteksi
```bash
# Cek GPU
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Fix: Install CUDA 11.2 + cuDNN 8.1
# Lihat: WINDOWS_SETUP.md
```

### Problem: Out of Memory
```python
# Turunkan batch size di train.py
BATCH_SIZE = 8  # dari 16
```

### Problem: Training terlalu lambat
```python
# Naikan batch size di train.py
BATCH_SIZE = 32  # dari 16

# Atau kurangi epochs
EPOCHS = 100  # dari 200
```

### Problem: Accuracy rendah
```python
# Tambah epochs
EPOCHS = 300  # dari 200

# Atau perbesar model
DENSE_UNITS = 256  # dari 128
```

**Solusi lengkap:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📊 Understanding Results

### Good Results ✅
```
Training Accuracy:    95.23%
Validation Accuracy:  91.76%
Gap: 3.47%
```
→ Model siap deploy!

### Overfitting ⚠️
```
Training Accuracy:    98.50%
Validation Accuracy:  82.35%
Gap: 16.15%
```
→ Naikan `DROPOUT_RATE` atau tambah data

### Underfitting ⚠️
```
Training Accuracy:    78.21%
Validation Accuracy:  75.12%
Gap: 3.09%
```
→ Tambah `EPOCHS` atau `DENSE_UNITS`

---

## 🎓 Learning Resources

**Beginner:**
- [README.md](README.md) - Overview & installation
- [DATASET.md](DATASET.md) - Dataset info

**Intermediate:**
- [CONFIGURATION.md](CONFIGURATION.md) - Tweaking parameters
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Solve common issues

**Advanced:**
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deploy to production
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - CUDA setup

---

## 💡 Pro Tips

1. **Always use GPU** - Training 10-20x faster
2. **Monitor val_accuracy** - Lebih penting dari train_accuracy
3. **Use Early Stopping** - Sudah aktif, stop otomatis
4. **Save checkpoints** - Sudah otomatis per epoch
5. **Visualize with TensorBoard** - `tensorboard --logdir=logs`

---

## 🎯 Training Checklist

- [ ] ✅ Virtual environment aktif (`.venv`)
- [ ] ✅ TensorFlow installed (`pip list | findstr tensorflow`)
- [ ] ✅ GPU terdeteksi (`nvidia-smi`)
- [ ] ✅ Dataset ready (`dataset/def_front` & `dataset/ok_front`)
- [ ] ✅ Configuration tweaked (opsional)
- [ ] ✅ Run training (`.\train.bat` atau `python train.py`)
- [ ] ✅ Monitor progress (lihat terminal output)
- [ ] ✅ Check results (accuracy ≥ 90%)
- [ ] ✅ Test model dengan gambar baru

---

## 🚀 Ready?

```powershell
# 1. Setup
.\setup.bat

# 2. Train
.\train.bat

# 3. Profit! 🎉
```

**Selamat training! 🔥**

---

Kembali ke [README.md](README.md) untuk dokumentasi lengkap.
