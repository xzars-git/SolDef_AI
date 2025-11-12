# PCB Defect Detection - AI Training

Model klasifikasi gambar untuk deteksi defect pada PCB (Printed Circuit Board) menggunakan Transfer Learning dengan MobileNetV2.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.2-green)](https://developer.nvidia.com/cuda-toolkit)
[![cuDNN](https://img.shields.io/badge/cuDNN-8.1-green)](https://developer.nvidia.com/cudnn)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-blue)](https://www.microsoft.com/windows)

> **Dataset:** [SolDef_AI PCB Dataset](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) by Maurizio Calabrese on Kaggle

---

## 📚 Dokumentasi Lengkap

| Dokumen | Deskripsi |
|---------|-----------|
| **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** | Setup CUDA 11.2 + cuDNN 8.1 untuk Windows |
| **[DATASET.md](DATASET.md)** | Info dataset, download, dan statistics |
| **[CONFIGURATION.md](CONFIGURATION.md)** | Tweaking parameters & callbacks |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Troubleshooting & FAQ |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Evaluasi, TFLite, Flutter integration |

---

## ⚡ Quick Start (3 Langkah)

```bash
# 1. Setup environment
conda create -n pcb python=3.9 -y
conda activate pcb
pip install -r requirements.txt

# 2. Verifikasi GPU (CUDA 11.2 + cuDNN 8.1)
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# 3. Jalankan training
python train.py
```

**Hasil:** Model `qc_inspector_model.h5` siap digunakan! 🎉

---

## 🗂️ Struktur Project

```
SolDef_AI/
├── dataset/                    # Dataset gambar PCB (428 images)
│   ├── lulus_qc/              # ✅ PCB lolos QC (312 images)
│   └── cacat_produksi/        # ❌ PCB dengan defect (116 images)
├── train.py                    # Training script untuk VS Code/Terminal
├── train.bat                   # Batch file untuk Windows command line
├── train.ipynb                 # Jupyter Notebook untuk Local/Colab
├── requirements.txt            # Dependencies Python
├── README.md                   # Quick start guide (file ini)
├── WINDOWS_SETUP.md            # Setup CUDA & cuDNN
├── DATASET.md                  # Info dataset
├── CONFIGURATION.md            # Tweaking parameters
├── TROUBLESHOOTING.md          # Troubleshooting & FAQ
└── DEPLOYMENT.md               # Deployment guide
```

---

## 🚀 Cara Training - 3 Metode

### 📍 Metode 1: VS Code (Recommended)

**Setup pertama kali:**
```powershell
# Di Anaconda Prompt atau PowerShell
conda create -n pcb python=3.9 -y
conda activate pcb
pip install -r requirements.txt
```

**Cara training:**
1. Buka `train.py` di VS Code
2. Pilih Python Interpreter: `Python 3.9 ('pcb')`
3. Run: `Ctrl + F5` atau klik kanan → "Run Python File"
4. Monitor progress di terminal

**Kelebihan:**
- ✅ Debugging mudah dengan breakpoints
- ✅ Bisa edit code sambil training
- ✅ Git integration
- ✅ IntelliSense & autocomplete

**Estimasi waktu (RTX 3080 Ti):**
- 50 epochs: ~5-7 menit
- 100 epochs: ~10-15 menit
- 200 epochs: ~20-25 menit

---

### 📍 Metode 2: Jupyter Notebook (Local)

**Setup pertama kali:**
```powershell
conda activate pcb
pip install jupyter ipykernel
python -m ipykernel install --user --name=pcb --display-name="PCB Training"
jupyter notebook
```

**Cara training:**
1. Browser akan otomatis terbuka
2. Navigate ke folder project → klik `train.ipynb`
3. Pilih kernel: `PCB Training`
4. Jalankan cell dengan `Shift + Enter` atau `Cell → Run All`
5. Monitor training dengan visualisasi real-time

**Kelebihan:**
- ✅ Visualisasi interaktif (grafik langsung muncul)
- ✅ Bisa jalankan per cell (iterative development)
- ✅ Dokumentasi inline dengan Markdown
- ✅ Mudah eksperimen dengan hyperparameters

---

### 📍 Metode 3: Google Colab (Cloud GPU)

**Cara training:**
1. Buka https://colab.research.google.com
2. File → Upload notebook → Pilih `train.ipynb`
3. Runtime → Change runtime type → GPU (T4)
4. Upload dataset atau mount Google Drive
5. Runtime → Run all

**Kelebihan:**
- ✅ Tidak perlu install CUDA/cuDNN
- ✅ GPU gratis (T4 ~15 GB VRAM)
- ✅ Akses dari mana saja
- ✅ Solusi alternatif jika Windows GPU bermasalah

**Upload dataset:**
```python
# Opsi 1: Upload ZIP
from google.colab import files
uploaded = files.upload()

# Opsi 2: Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/PCB_Dataset/dataset ./
```

---

## ⚙️ Konfigurasi Training

Edit parameter di `train.py` (baris 25-30):

```python
IMG_SIZE = (224, 224)      # Ukuran input gambar
BATCH_SIZE = 16            # Jumlah gambar per batch
EPOCHS = 200               # Jumlah iterasi training
LEARNING_RATE = 0.001      # Learning rate
DROPOUT_RATE = 0.3         # Dropout untuk regularisasi
DENSE_UNITS = 128          # Jumlah neuron di Dense layer
```

**Panduan lengkap:** Lihat [CONFIGURATION.md](CONFIGURATION.md)

---

## 📦 Output Training

Setelah training selesai:

```
✅ qc_inspector_model.h5        # Model final (gunakan ini!)
✅ best_model.h5                # Backup model terbaik
✅ training_history.json        # Metrics per epoch
✅ logs/                        # TensorBoard logs
```

**Cara test model:**
```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('qc_inspector_model.h5')

img = image.load_img('test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
print("Lulus QC ✅" if prediction > 0.5 else "Cacat Produksi ❌")
```

---

## 🎯 Target Accuracy

- **≥ 90%** → Excellent (production ready) ✅
- **85-90%** → Good (acceptable)
- **80-85%** → Fair (perlu improvement)
- **< 80%** → Poor (perlu retrain)

**Gap Training vs Validation:**
- **< 5%** → Perfect generalization ✅
- **5-10%** → Good (slight overfitting)
- **> 15%** → Overfitting (naikan dropout atau tambah data)

---

## 🔧 Troubleshooting

### GPU tidak terdeteksi?
```bash
# Verifikasi GPU
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
**Solusi lengkap:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Training hang setelah "Epoch 1/200"?
- Sudah diset `workers=0` di script (fix Windows issue)
- Restart komputer
- Turunkan `BATCH_SIZE`

### Out of Memory?
```python
BATCH_SIZE = 8  # Turunkan dari 16
```

**Masalah lainnya:** Lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 🚀 Next Steps

Setelah training selesai dan accuracy ≥ 90%:

1. **Evaluasi Model** - Test dengan gambar baru ([DEPLOYMENT.md](DEPLOYMENT.md))
2. **Convert ke TFLite** - Untuk Flutter mobile app ([DEPLOYMENT.md](DEPLOYMENT.md))
3. **Deploy ke Production** - API server atau mobile app ([DEPLOYMENT.md](DEPLOYMENT.md))

---

## 📖 Dokumentasi Lengkap

| Dokumen | Konten |
|---------|--------|
| **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** | ⚙️ Setup CUDA 11.2, cuDNN 8.1, Anaconda, Environment Variables |
| **[DATASET.md](DATASET.md)** | 📊 Download dataset, struktur, statistics, tips |
| **[CONFIGURATION.md](CONFIGURATION.md)** | 🎛️ Tweaking parameters (batch size, epochs, dropout, dll) |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | 🔧 Solusi GPU, OOM, Import Error, Training Issues, FAQ |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | 🚀 Evaluasi, TFLite conversion, Flutter integration, API server |

---

## 📝 Model Info

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Framework:** TensorFlow 2.10.0 / Keras
- **Input Size:** 224x224x3
- **Classes:** 2 (Lulus QC, Cacat Produksi)
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Crossentropy

---

## 📄 License & Credits

**Dataset:** [SolDef_AI PCB Dataset](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) by Maurizio Calabrese (Kaggle)

**License:** MIT

---

**Created for PCB Quality Control Inspection** 🔍

*Untuk pertanyaan atau bantuan, lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md) atau buka issue di repository.*
