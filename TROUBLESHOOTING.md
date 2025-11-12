# 🔧 Troubleshooting & FAQ

Panduan solusi untuk masalah umum saat training Casting Product Defect Detection model (submersible pump impeller inspection).

---

## ❌ GPU Issues

### Problem: GPU tidak terdeteksi

**Symptoms:**
```
GPU: []  # List kosong
```

**Diagnosis:**
```bash
# 1. Cek CUDA terinstall
nvcc --version

# 2. Cek cuDNN
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# 3. Cek GPU visibility
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Solutions:**
1. Install CUDA 11.2 + cuDNN 8.1 (lihat [WINDOWS_SETUP.md](WINDOWS_SETUP.md))
2. Restart komputer setelah install
3. Pastikan NVIDIA driver up-to-date
4. Reinstall TensorFlow:
   ```bash
   pip uninstall tensorflow
   pip install tensorflow==2.10.0
   ```

---

### Problem: Training hang/freeze setelah "Epoch 1/200"

**Symptoms:**
```
Epoch 1/200
[HANG - No progress]
```

**Solutions:**
1. **Sudah diset di script** - `workers=0` dan `max_queue_size=1`
2. Restart komputer (clear GPU memory)
3. Cek antivirus memblock GPU access
4. Turunkan batch size:
   ```python
   BATCH_SIZE = 8  # dari 16
   ```
5. Disable TensorBoard sementara (comment di script)

---

### Problem: Out of Memory (OOM)

**Symptoms:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
```python
# Solusi 1: Turunkan batch size
BATCH_SIZE = 8  # atau 4

# Solusi 2: Enable memory growth (sudah ada di script)
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Solusi 3: Limit GPU memory
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
)
```

---

## ❌ Import & Environment Issues

### Problem: Import Error / ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solutions:**
```bash
# 1. Pastikan virtual environment aktif
.venv\Scripts\activate

# 2. Reinstall dependencies
pip install -r requirements.txt

# 3. Verifikasi installation
pip list | findstr tensorflow
```

---

### Problem: DLL Load Failed (Windows)

**Symptoms:**
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

**Solutions:**
1. Install Visual C++ Redistributable:
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install dan restart PC

2. Cek PATH environment variable (harus ada):
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
   ```

3. Restart terminal/VS Code setelah install CUDA

---

### Problem: CUDA Version Mismatch

**Symptoms:**
```
Could not load dynamic library 'cudart64_110.dll'
```

**Solution:**
Pastikan menggunakan TensorFlow 2.10.0 (compatible dengan CUDA 11.2):
```bash
pip uninstall tensorflow -y
pip install tensorflow==2.10.0
```

---

## ❌ Training Issues

### Problem: Accuracy tidak naik / stagnan

**Symptoms:**
```
Epoch 50/200
loss: 0.6931 - accuracy: 0.5234 - val_loss: 0.6928 - val_accuracy: 0.5123
[Tidak berubah dari epoch 1]
```

**Solutions:**

**1. Learning rate terlalu kecil:**
```python
LEARNING_RATE = 0.001  # Naikan dari 0.0001
```

**2. Model terlalu simpel:**
```python
DENSE_UNITS = 256  # Naikan dari 128
```

**3. Data tidak bervariasi:**
- Tambah data augmentation (lihat [CONFIGURATION.md](CONFIGURATION.md))
- Tambah lebih banyak gambar training

**4. Data tidak balance:**
```python
# Jika cacat_produksi << lulus_qc, tambah class_weight
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
# Di model.fit() tambahkan: class_weight=dict(enumerate(class_weights))
```

---

### Problem: Training terlalu lambat

**Symptoms:**
```
Epoch 1/200
342/342 [====] - 450s 1s/step  # Terlalu lama!
```

**Solutions:**
```python
# 1. Naikan batch size
BATCH_SIZE = 32  # dari 16

# 2. Simplify model
DENSE_UNITS = 64  # dari 128

# 3. Reduce augmentation
# Comment beberapa augmentation di ImageDataGenerator

# 4. Pastikan GPU digunakan
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Harus ada GPU
```

---

### Problem: Overfitting (Train acc >> Val acc)

**Symptoms:**
```
Training Accuracy:   0.9850 (98.50%)
Validation Accuracy: 0.8235 (82.35%)
Gap: 16.15% → Overfitting!
```

**Solutions:**
```python
# 1. Naikan dropout
DROPOUT_RATE = 0.5  # dari 0.3

# 2. Tambah data augmentation
rotation_range=30
width_shift_range=0.3
height_shift_range=0.3

# 3. Reduce model complexity
DENSE_UNITS = 64  # dari 128

# 4. Add more training data (recommended)
```

---

### Problem: Underfitting (Train & Val acc rendah)

**Symptoms:**
```
Training Accuracy:   0.7821 (78.21%)
Validation Accuracy: 0.7512 (75.12%)
```

**Solutions:**
```python
# 1. Tambah epochs
EPOCHS = 300  # dari 200

# 2. Increase model complexity
DENSE_UNITS = 256  # dari 128

# 3. Reduce dropout
DROPOUT_RATE = 0.2  # dari 0.3

# 4. Increase learning rate (hati-hati)
LEARNING_RATE = 0.001  # pastikan tidak terlalu kecil
```

---

## ❓ FAQ

### Berapa lama training biasanya?

**Dengan GPU (RTX 3080 Ti):**
- 50 epochs: ~5-7 menit
- 100 epochs: ~10-15 menit
- 200 epochs: ~20-25 menit

**Dengan CPU:**
- 50 epochs: ~2-3 jam
- 200 epochs: ~8-12 jam ⚠️

**Dengan Google Colab (T4 GPU):**
- 50 epochs: ~8-10 menit
- 200 epochs: ~30-40 menit

---

### Accuracy berapa yang bagus?

**Target Accuracy:**
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

### Kapan harus stop training manual?

**Stop jika:**
1. ✅ **Early Stopping triggered** (sudah otomatis di script)
2. ✅ **Validation accuracy tidak naik 10+ epochs**
3. ✅ **Loss mulai naik** (sign of overfitting)
4. ✅ **Sudah mencapai target accuracy** (misal 95%)

**Jangan stop jika:**
1. ❌ Baru 5-10 epochs (terlalu awal)
2. ❌ Validation loss masih turun
3. ❌ Belum ada warning dari callback

---

### Bagaimana cara menambah data training?

**Opsi 1: Data Augmentation (Recommended)**
- Sudah aktif di script (rotation, zoom, flip)
- Increase augmentation parameters (lihat [CONFIGURATION.md](CONFIGURATION.md))

**Opsi 2: Download Full Dataset (7,348 images)**
1. Kunjungi [Kaggle Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
2. Download "casting_data.zip" (300x300 with augmentation)
3. Gunakan folder train (6,633 images total)

**Opsi 3: Tambah gambar impeller baru**
1. Ambil foto impeller casting (top view)
2. Pastikan lighting stabil dan konsisten
3. Masukkan ke folder `dataset/def_front/` atau `dataset/ok_front/`
4. Minimal 100 gambar per class

**Opsi 4: Download dataset casting tambahan**
- Cari "casting defect" di Kaggle atau Roboflow
- Pastikan format gambar sama (JPG/PNG)

---

### Bagaimana cara test model dengan gambar impeller baru?

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('qc_inspector_model.h5')

# Load & preprocess gambar impeller (top view)
img_path = 'test_impeller.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
prediction = model.predict(img_array)
print(f"Prediction Score: {prediction[0][0]:.4f}")

# Interpretasi
if prediction[0][0] > 0.5:
    print("Hasil: OK Casting ✅ (Pass Quality Control)")
    print(f"Confidence: {prediction[0][0] * 100:.2f}%")
else:
    print("Hasil: Defective Casting ❌ (Reject)")
    print(f"Confidence: {(1 - prediction[0][0]) * 100:.2f}%")
    print("Possible defects: blow holes, pinholes, burr, shrinkage, etc.")
```

---

### Bagaimana cara visualisasi training history?

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training')
plt.plot(history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150)
plt.show()
```

---

### Bisa training tanpa GPU?

**Ya, tapi sangat lambat:**
- GPU: ~20-25 menit untuk 200 epochs ✅
- CPU: ~8-12 jam untuk 200 epochs ⚠️

**Alternatif tanpa GPU lokal:**
1. **Google Colab** (gratis) - GPU T4 ~15GB VRAM
2. **Kaggle Notebooks** (gratis) - GPU P100
3. **Cloud VM** (berbayar) - AWS, GCP, Azure

---

### File apa yang penting setelah training?

**File Output:**
```
✅ qc_inspector_model.h5        # Model final (PENTING!)
✅ best_model.h5                # Backup terbaik
✅ training_history.json        # Metrics per epoch
✅ logs/                        # TensorBoard logs
```

**Untuk deployment:**
- Simpan `qc_inspector_model.h5` (atau convert ke TFLite)
- Simpan `training_history.json` untuk dokumentasi
- TensorBoard logs untuk analisis (opsional)

---

## 📚 Resources Tambahan

- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - Setup CUDA & cuDNN
- [CONFIGURATION.md](CONFIGURATION.md) - Tweaking parameters
- [DATASET.md](DATASET.md) - Info dataset
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deploy ke Flutter/API
- [README.md](README.md) - Quick start

---

**Masih ada masalah?**
- Cek [TensorFlow Forum](https://discuss.tensorflow.org/)
- Buka issue di GitHub repository ini
- Stack Overflow dengan tag `tensorflow` + `keras`
