# PCB Defect Detection - AI Training

Model klasifikasi gambar untuk deteksi defect pada PCB (Printed Circuit Board) menggunakan Transfer Learning dengan MobileNetV2.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.2-green)](https://developer.nvidia.com/cuda-toolkit)
[![cuDNN](https://img.shields.io/badge/cuDNN-8.1-green)](https://developer.nvidia.com/cudnn)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-blue)](https://www.microsoft.com/windows)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Dataset:** [SolDef_AI PCB Dataset](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) by Maurizio Calabrese on Kaggle

---

## � Quick Start (Windows - 3 Langkah)

```bash
# 1. Setup environment
conda create -n pcb python=3.9 -y
conda activate pcb
pip install -r requirements.txt

# 2. Verifikasi GPU (CUDA 11.2 + cuDNN 8.1)
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# 3. Jalankan training
python train.py
# atau di VS Code: Ctrl+F5
# atau di Jupyter: jupyter notebook → buka train.ipynb
```

**Hasil:** Model `qc_inspector_model.h5` siap digunakan! 🎉

---

## 💻 Windows Setup Guide (CUDA 11.2 + cuDNN 8.1)

### Prerequisites untuk Windows 10/11

| Component | Version | Required | Download |
|-----------|---------|----------|----------|
| **Windows** | 10/11 64-bit | ✅ | - |
| **NVIDIA GPU** | RTX 3080 Ti (atau GPU lain) | ✅ | - |
| **NVIDIA Driver** | ≥ 452.39 | ✅ | [Download](https://www.nvidia.com/Download/index.aspx) |
| **CUDA Toolkit** | 11.2 | ✅ | [Download](https://developer.nvidia.com/cuda-11.2.0-download-archive) |
| **cuDNN** | 8.1 for CUDA 11.x | ✅ | [Download](https://developer.nvidia.com/rdp/cudnn-archive) |
| **Anaconda/Miniconda** | Latest | ✅ | [Download](https://www.anaconda.com/download) |
| **Visual Studio** | 2019/2022 (Build Tools) | ⚠️ | [Download](https://visualstudio.microsoft.com/downloads/) |

### Step-by-Step Installation (Windows)

#### 1️⃣ Install NVIDIA Driver

```bash
# Cek current driver version
nvidia-smi
```

Jika belum terinstall atau versi lama, download driver terbaru dari [NVIDIA](https://www.nvidia.com/Download/index.aspx).

---

#### 2️⃣ Install CUDA Toolkit 11.2

**Download:**
- Kunjungi: https://developer.nvidia.com/cuda-11.2.0-download-archive
- Pilih: Windows → x86_64 → 10 → exe (local)

**Install:**
```powershell
# Run installer (cuda_11.2.0_460.89_win10.exe)
# Pilih "Express Installation"
# Lokasi default: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
```

**Verifikasi:**
```bash
nvcc --version
# Output: Cuda compilation tools, release 11.2, V11.2.67
```

**Set Environment Variables (Manual):**
```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
CUDA_PATH_V11_2 = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2

# Tambahkan ke PATH:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
```

---

#### 3️⃣ Install cuDNN 8.1

**Download:**
- Kunjungi: https://developer.nvidia.com/rdp/cudnn-archive
- Pilih: **cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0, 11.1 and 11.2**
- Download: `cudnn-11.2-windows-x64-v8.1.0.77.zip`

**Install:**
```powershell
# 1. Extract ZIP file
# 2. Copy files ke folder CUDA:

# Copy bin files
Copy-Item "cudnn-11.2-windows-x64-v8.1.0.77\bin\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\" -Force

# Copy include files
Copy-Item "cudnn-11.2-windows-x64-v8.1.0.77\include\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\" -Force

# Copy lib files
Copy-Item "cudnn-11.2-windows-x64-v8.1.0.77\lib\x64\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64\" -Force
```

**Verifikasi:**
```bash
# Cek files di CUDA bin folder
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cudnn*.dll"

# Harus ada:
# - cudnn64_8.dll
# - cudnn_adv_infer64_8.dll
# - cudnn_adv_train64_8.dll
# - cudnn_cnn_infer64_8.dll
# - cudnn_cnn_train64_8.dll
# - cudnn_ops_infer64_8.dll
# - cudnn_ops_train64_8.dll
```

---

#### 4️⃣ Install Anaconda (jika belum)

**Download & Install:**
- Download: https://www.anaconda.com/download
- Install untuk "Just Me" (tidak perlu admin)
- Lokasi default: `C:\Users\<YourName>\anaconda3`

**Verifikasi:**
```powershell
conda --version
# Output: conda 23.x.x
```

---

#### 5️⃣ Create Conda Environment & Install Dependencies

```powershell
# 1. Buat environment baru
conda create -n pcb python=3.9 -y

# 2. Aktivasi environment
conda activate pcb

# 3. Install TensorFlow 2.10.0 (compatible dengan CUDA 11.2 + cuDNN 8.1)
pip install tensorflow==2.10.0

# 4. Install dependencies lainnya
pip install -r requirements.txt

# 5. Verifikasi instalasi
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

**Expected Output:**
```
TensorFlow: 2.10.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

#### 6️⃣ Troubleshooting Windows-Specific Issues

**Problem: DLL Load Failed**
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

**Solution:**
1. Install Visual C++ Redistributable:
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install dan restart PC

2. Cek PATH environment variable (harus ada):
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
   ```

3. Restart terminal/VS Code setelah install CUDA

---

**Problem: GPU Not Detected**
```
GPU: []
```

**Solution:**
```powershell
# 1. Cek NVIDIA driver
nvidia-smi

# 2. Cek CUDA version
nvcc --version

# 3. Cek cuDNN files
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cudnn*.dll"

# 4. Reinstall TensorFlow
pip uninstall tensorflow -y
pip install tensorflow==2.10.0

# 5. Restart PC (important!)
```

---

**Problem: CUDA Version Mismatch**
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

### Windows-Specific Configuration

**PowerShell Execution Policy (jika error):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Anaconda Prompt (Recommended):**
- Buka "Anaconda Prompt" dari Start Menu
- Lebih stabil untuk conda commands

**VS Code Terminal Settings:**
```json
// settings.json
{
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "python.condaPath": "C:\\Users\\<YourName>\\anaconda3\\Scripts\\conda.exe"
}
```

---

```
SolDef_AI/
├── dataset/                    # Dataset gambar PCB
│   ├── lulus_qc/              # Gambar PCB yang lolos QC (312 images)
│   └── cacat_produksi/        # Gambar PCB dengan defect (116 images)
├── train.py                    # Training script untuk local/VS Code
├── train.bat                   # Batch file untuk command line
├── train.ipynb                 # Jupyter Notebook untuk Colab/Local
├── requirements.txt            # Dependencies Python
├── README.md                   # Dokumentasi (file ini)
└── .gitignore                  # File yang diabaikan git
```

---

## 🚀 Cara Training - 3 Metode (Windows)

### 📍 Metode 1: VS Code (Recommended untuk Windows Development)

**Prerequisites:**
- ✅ Windows 10/11 64-bit
- ✅ NVIDIA GPU dengan driver terbaru
- ✅ CUDA Toolkit 11.2 installed
- ✅ cuDNN 8.1 installed
- ✅ Anaconda/Miniconda installed
- ✅ VS Code dengan Python Extension

**Setup Pertama Kali:**

```powershell
# Di Anaconda Prompt atau PowerShell

# 1. Buat conda environment
conda create -n pcb python=3.9 -y
conda activate pcb

# 2. Install TensorFlow 2.10.0 (CUDA 11.2 compatible)
pip install tensorflow==2.10.0

# 3. Install semua dependencies
pip install -r requirements.txt

# 4. Verifikasi GPU terdeteksi
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

**Expected Output:**
```
TensorFlow: 2.10.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Cara Training di VS Code:**

1. **Buka file `train.py`** di VS Code
2. **Pilih Python Interpreter:**
   - Tekan `Ctrl + Shift + P`
   - Ketik "Python: Select Interpreter"
   - Pilih `Python 3.9.x ('pcb')` conda environment
3. **Jalankan script:**
   - **Cara 1:** Klik kanan di editor → `Run Python File in Terminal`
   - **Cara 2:** Tekan `Ctrl + F5` (Run without debugging)
   - **Cara 3:** Di terminal: `python train.py`
4. **Monitor progress** di terminal VS Code
5. **Stop training:** Tekan `Ctrl + C` jika perlu

**Kelebihan untuk Windows:**
- ✅ Debugging mudah dengan breakpoints
- ✅ Bisa edit code sambil training
- ✅ Terminal terintegrasi dengan conda
- ✅ Git integration untuk version control
- ✅ IntelliSense & autocomplete
- ✅ Error detection real-time

---

### 📍 Metode 2: Jupyter Notebook (Local Windows)

**Setup Pertama Kali:**

```powershell
# Di Anaconda Prompt

# 1. Aktivasi environment
conda activate pcb

# 2. Install Jupyter (jika belum)
pip install jupyter ipykernel

# 3. Register kernel untuk Jupyter
python -m ipykernel install --user --name=pcb --display-name="PCB Training (Python 3.9)"

# 4. Jalankan Jupyter Notebook
jupyter notebook
```

**Cara Training:**

1. **Browser akan otomatis terbuka**
2. **Navigate ke folder project** dan klik `train.ipynb`
3. **Pilih kernel:** 
   - Kernel → Change kernel → `PCB Training (Python 3.9)`
4. **Verifikasi GPU** di cell pertama:
   ```python
   import tensorflow as tf
   print('TensorFlow:', tf.__version__)
   print('GPU:', tf.config.list_physical_devices('GPU'))
   ```
5. **Jalankan cell secara berurutan:**
   - Tekan `Shift + Enter` untuk run cell
   - Atau klik `Cell → Run All`
6. **Monitor training** dengan visualisasi real-time
7. **Stop training:** Kernel → Interrupt

**Kelebihan untuk Windows:**
- ✅ Visualisasi interaktif (grafik langsung muncul)
- ✅ Bisa jalankan per cell (iterative development)
- ✅ Dokumentasi inline dengan Markdown
- ✅ Save checkpoints per cell
- ✅ Mudah eksperimen dengan hyperparameters
- ✅ Browser-based (akses dari browser apapun)

**Windows-Specific Tips:**
```powershell
# Jika Jupyter tidak bisa akses conda environment
conda install -n pcb ipykernel -y
python -m ipykernel install --user --name=pcb

# Jika port 8888 sudah digunakan
jupyter notebook --port=8889
```

---

### 📍 Metode 3: Google Colab (Cloud - Alternatif jika CUDA bermasalah)

**Cara Training di Google Colab:**

1. **Upload Notebook:**
   - Buka https://colab.research.google.com
   - File → Upload notebook → Pilih `train.ipynb`

2. **Aktifkan GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator: `GPU`
   - GPU type: `T4` (gratis)
   - Save

3. **Upload Dataset:**
   
   **Opsi A: Upload ZIP dari PC Windows**
   ```python
   from google.colab import files
   import zipfile
   
   # Upload dataset.zip
   uploaded = files.upload()
   
   # Extract
   for filename in uploaded.keys():
       if filename.endswith('.zip'):
           with zipfile.ZipFile(filename, 'r') as zip_ref:
               zip_ref.extractall('.')
   
   # Rename folder
   !mv Dataset dataset
   ```

   **Opsi B: Google Drive (Recommended)**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Copy dataset dari Drive
   !cp -r /content/drive/MyDrive/PCB_Dataset/dataset ./
   ```

4. **Verifikasi Setup:**
   ```python
   # Check GPU
   import tensorflow as tf
   print('GPU:', tf.config.list_physical_devices('GPU'))
   
   # Check dataset
   !ls dataset/
   !ls dataset/lulus_qc | wc -l
   !ls dataset/cacat_produksi | wc -l
   ```

5. **Jalankan Training:**
   - Runtime → Run all
   - Atau jalankan cell per cell dengan `Shift + Enter`

6. **Download Model:**
   ```python
   from google.colab import files
   files.download('qc_inspector_model.h5')
   files.download('best_model.h5')
   files.download('training_history.json')
   ```

**Kelebihan:**
- ✅ Tidak perlu install CUDA/cuDNN di Windows
- ✅ GPU gratis (T4 ~15 GB VRAM)
- ✅ Akses dari mana saja
- ✅ Pre-installed TensorFlow
- ✅ Cloud storage integration
- ✅ Solusi alternatif jika Windows GPU bermasalah

**Keterbatasan:**
- ⚠️ Session timeout setelah 12 jam
- ⚠️ Disconnect jika idle 90 menit
- ⚠️ Perlu upload dataset setiap session
- ⚠️ Perlu download model setelah training

---

## 📊 Dataset

### Dataset Source & Credit

**Original Dataset:** [SolDef_AI PCB Dataset for Defect Detection](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection)

**Author:** Maurizio Calabrese  
**Platform:** Kaggle  
**License:** Database Contents License (DbCL)  

**Citation:**
```
Calabrese, Maurizio. (2022). SolDef_AI PCB Dataset for Defect Detection.
Kaggle. https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection
```

**Dataset Description:**
Dataset ini berisi gambar PCB (Printed Circuit Board) untuk quality control inspection dengan 2 kategori:
- ✅ **Lulus QC** - PCB yang lolos quality control
- ❌ **Cacat Produksi** - PCB dengan defect/cacat

---

### Struktur Dataset

```
dataset/
├── lulus_qc/          # ✅ Gambar PCB yang lolos QC
│   ├── WIN_20220329_14_30_32_Pro.jpg
│   ├── WIN_20220329_14_30_42_Pro.jpg
│   └── ... (312 images total)
└── cacat_produksi/    # ❌ Gambar PCB dengan defect
    ├── WIN_20220329_14_31_21_Pro.jpg
    ├── WIN_20220329_14_31_29_Pro.jpg
    └── ... (116 images total)
```

**Format gambar yang didukung:** JPG, JPEG, PNG

### Dataset Statistics

- **Total images:** 428
- **Class distribution:**
  - Lulus QC: 312 images (72.9%) ✅
  - Cacat Produksi: 116 images (27.1%) ❌
- **Train/Val split:** 80/20
  - Training: 343 images
  - Validation: 85 images

### Cara Download Dataset (dari Kaggle)

#### Opsi 1: Manual Download (Recommended untuk Windows)

1. **Buka link dataset:**
   ```
   https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection
   ```

2. **Login ke Kaggle** (buat account jika belum punya)

3. **Download dataset:**
   - Klik tombol "Download" di halaman dataset
   - File akan terdownload sebagai `archive.zip` atau `soldef-ai-pcb-dataset-for-defect-detection.zip`

4. **Extract & Setup:**
   ```powershell
   # Extract ZIP file
   Expand-Archive -Path "soldef-ai-pcb-dataset-for-defect-detection.zip" -DestinationPath "."
   
   # Rename folder menjadi "dataset"
   Rename-Item -Path "Dataset" -NewName "dataset"
   
   # Verifikasi struktur
   dir dataset\
   ```

---

#### Opsi 2: Kaggle API (Command Line)

```powershell
# 1. Install Kaggle CLI
pip install kaggle

# 2. Setup Kaggle credentials
# - Login ke Kaggle
# - Go to Account → Create New API Token
# - Download kaggle.json
# - Copy ke: C:\Users\<YourName>\.kaggle\kaggle.json

# 3. Download dataset
kaggle datasets download -d mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection

# 4. Extract
Expand-Archive -Path "soldef-ai-pcb-dataset-for-defect-detection.zip" -DestinationPath "."
Rename-Item -Path "Dataset" -NewName "dataset"
```

---

### Persiapan Dataset

**Untuk Local/VS Code:**
```bash
# Dataset sudah ada di folder project
# Pastikan struktur folder benar
ls dataset/lulus_qc | wc -l        # Should show 312
ls dataset/cacat_produksi | wc -l  # Should show 116
```

**Untuk Jupyter Notebook Local:**
```python
# Cell untuk verifikasi dataset
import os
from pathlib import Path

dataset_dir = 'dataset'
classes = ['lulus_qc', 'cacat_produksi']

for class_name in classes:
    path = Path(dataset_dir) / class_name
    count = len(list(path.glob('*.jpg')))
    print(f"{class_name}: {count} images")
```

**Untuk Google Colab:**

**Opsi 1: Upload ZIP**
```python
# Cell 1: Upload dataset.zip
from google.colab import files
import zipfile

print("Pilih file dataset.zip...")
uploaded = files.upload()

# Cell 2: Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("✅ Done!")

# Cell 3: Verify
!ls dataset/
!echo "Lulus QC:" && ls dataset/lulus_qc | wc -l
!echo "Cacat:" && ls dataset/cacat_produksi | wc -l
```

**Opsi 2: Google Drive (Recommended untuk dataset besar)**
```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Copy dari Drive
!cp -r /content/drive/MyDrive/PCB_Dataset/dataset ./

# Cell 3: Verify
!ls dataset/
!echo "Lulus QC:" && ls dataset/lulus_qc | wc -l
!echo "Cacat:" && ls dataset/cacat_produksi | wc -l
```

**Opsi 3: Download dari URL**
```python
# Cell 1: Download
!wget https://your-server.com/dataset.zip

# Cell 2: Extract
!unzip -q dataset.zip

# Cell 3: Verify
!ls dataset/
```

### Tips Dataset

1. **Minimum recommended:** 100 images per class
2. **Ideal:** 500-1000 images per class
3. **Image quality:** Consistent lighting, angle, resolution
4. **Balance classes:** Usahakan jumlah gambar per class seimbang
   - Jika tidak seimbang, gunakan `class_weight` (lihat FAQ)
5. **Remove duplicates:** Hindari gambar duplikat
6. **Consistent naming:** Gunakan naming convention yang jelas

## ⚙️ Konfigurasi Training (Tweaking Parameters)

Semua parameter bisa diubah di **`train.py`** (baris 25-30) atau di **cell konfigurasi** di `train.ipynb`.

### 📊 Parameter Utama

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

### 🎛️ Panduan Tweaking

#### 1. **BATCH_SIZE** - Jumlah gambar per batch

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

#### 2. **EPOCHS** - Jumlah iterasi training

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

#### 3. **LEARNING_RATE** - Kecepatan learning

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

#### 4. **DROPOUT_RATE** - Regularisasi untuk cegah overfitting

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

#### 5. **DENSE_UNITS** - Kompleksitas model

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

### 🎯 Callback Settings (Advanced)

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

**Tweaking Callbacks:**

| Parameter | Default | Jika Overfitting | Jika Underfitting |
|-----------|---------|------------------|-------------------|
| `patience` (EarlyStopping) | 15 | 10 | 20 |
| `patience` (ReduceLR) | 7 | 5 | 10 |
| `factor` (ReduceLR) | 0.5 | 0.3 | 0.7 |

---

### 📈 Data Augmentation (Advanced)

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

**Tweaking Augmentation:**

```python
# LIGHT AUGMENTATION (untuk dataset besar > 1000 images)
rotation_range=10
width_shift_range=0.1
height_shift_range=0.1
zoom_range=0.1

# HEAVY AUGMENTATION (untuk dataset kecil < 500 images)
rotation_range=30
width_shift_range=0.3
height_shift_range=0.3
zoom_range=0.3
brightness_range=[0.8, 1.2]  # Tambahan: variasi brightness
```

---

### 🔧 Troubleshooting by Tweaking

#### Problem 1: Training Accuracy tinggi, Validation Accuracy rendah (OVERFITTING)

**Solusi:**
```python
DROPOUT_RATE = 0.5          # Naikan dari 0.3
BATCH_SIZE = 32             # Perbesar dari 16
# Tambah augmentation (lihat di atas)
```

#### Problem 2: Training & Validation Accuracy sama-sama rendah (UNDERFITTING)

**Solusi:**
```python
EPOCHS = 300                # Tambah epochs
DENSE_UNITS = 256           # Perbesar dari 128
DROPOUT_RATE = 0.2          # Turunkan dari 0.3
LEARNING_RATE = 0.001       # Pastikan tidak terlalu kecil
```

#### Problem 3: Loss tidak turun atau fluktuatif

**Solusi:**
```python
LEARNING_RATE = 0.0001      # Turunkan dari 0.001
BATCH_SIZE = 32             # Perbesar untuk stabilitas
```

#### Problem 4: Out of Memory (OOM)

**Solusi:**
```python
BATCH_SIZE = 8              # Turunkan dari 16
# Atau tambahkan di awal script:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### Problem 5: Training terlalu lambat

**Solusi:**
```python
BATCH_SIZE = 32             # Naikan dari 16
DENSE_UNITS = 64            # Turunkan dari 128
EPOCHS = 100                # Kurangi dari 200
```

---

## 📦 Output Training

Setelah training selesai, akan menghasilkan file-file berikut:

### File Output

```
SolDef_AI/
├── qc_inspector_model.h5        # ⭐ Model final (gunakan ini untuk deployment)
├── best_model.h5                # Model checkpoint terbaik (backup)
├── training_history.json        # History metrics (loss, accuracy per epoch)
└── logs/                        # TensorBoard logs untuk visualisasi
    └── 20251112-143052/         # Folder per training session
```

### Cara Menggunakan Output

#### 1. Load Model untuk Testing

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('qc_inspector_model.h5')

# Test dengan gambar baru
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
prediction = model.predict(img_array)
print(f"Prediction: {prediction[0][0]:.4f}")

# Interpretasi (sesuaikan dengan class_indices Anda)
if prediction[0][0] > 0.5:
    print("Hasil: Lulus QC ✅")
else:
    print("Hasil: Cacat Produksi ❌")
```

#### 2. Visualisasi Training History

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Plot accuracy
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

#### 3. TensorBoard untuk Monitoring Real-time

```bash
# Jalankan TensorBoard
tensorboard --logdir=logs

# Buka browser: http://localhost:6006
```

### Interpretasi Hasil

**Good Training Results:**
```
Final Training Accuracy:    0.9523 (95.23%)
Final Validation Accuracy:  0.9176 (91.76%)
Gap: 3.47% → ✅ Good generalization
```

**Overfitting (Perlu tweaking):**
```
Final Training Accuracy:    0.9850 (98.50%)
Final Validation Accuracy:  0.8235 (82.35%)
Gap: 16.15% → ⚠️ Overfitting! (Naikan DROPOUT_RATE)
```

**Underfitting (Perlu lebih banyak epochs/complexity):**
```
Final Training Accuracy:    0.7821 (78.21%)
Final Validation Accuracy:  0.7512 (75.12%)
Gap: 3.09% → ⚠️ Underfitting! (Tambah EPOCHS atau DENSE_UNITS)
```

## 🔧 Troubleshooting & FAQ

### ❌ Problem: GPU tidak terdeteksi

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
1. Install CUDA 11.2 + cuDNN 8.1 sesuai dokumentasi NVIDIA
2. Restart komputer setelah install
3. Pastikan NVIDIA driver up-to-date
4. Reinstall TensorFlow:
   ```bash
   pip uninstall tensorflow
   pip install tensorflow==2.10.0
   ```

---

### ❌ Problem: Training hang/freeze setelah "Epoch 1/200"

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

### ❌ Problem: Out of Memory (OOM)

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

### ❌ Problem: Import Error / ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solutions:**
```bash
# 1. Pastikan environment aktif
conda activate pcb

# 2. Reinstall dependencies
pip install -r requirements.txt

# 3. Verifikasi installation
pip list | grep tensorflow
```

---

### ❌ Problem: Accuracy tidak naik / stagnan

**Symptoms:**
```
Epoch 50/200
loss: 0.6931 - accuracy: 0.5234 - val_loss: 0.6928 - val_accuracy: 0.5123
[Tidak berubah dari epoch 1]
```

**Solutions:**
1. **Learning rate terlalu kecil:**
   ```python
   LEARNING_RATE = 0.001  # Naikan dari 0.0001
   ```

2. **Model terlalu simpel:**
   ```python
   DENSE_UNITS = 256  # Naikan dari 128
   ```

3. **Data tidak bervariasi:**
   - Tambah data augmentation (lihat section Tweaking)
   - Tambah lebih banyak gambar training

4. **Data tidak balance:**
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

### ❌ Problem: Training terlalu lambat

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

### ❓ FAQ: Berapa lama training biasanya?

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

### ❓ FAQ: Accuracy berapa yang bagus?

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

### ❓ FAQ: Kapan harus stop training manual?

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

## 📝 Model Architecture

- **Base Model:** MobileNetV2 (pre-trained ImageNet)
- **Custom Head:** GlobalAveragePooling → Dense(128) → Dropout(0.3) → Dense(1)
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Crossentropy

## 📈 Training Tips

- Minimal 100-200 gambar per kelas untuk hasil yang baik
- Data augmentation sudah aktif (rotasi, zoom, flip)
- Early stopping aktif (patience=15 epochs)
- Learning rate reduction otomatis (patience=7 epochs)

## 🎯 Next Steps & Deployment

Setelah training selesai dan model memiliki accuracy bagus:

### 1. **Evaluasi Model dengan Test Data**

```python
# evaluate.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('qc_inspector_model.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'test_dataset',  # Buat folder terpisah untuk test
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# Evaluate
results = model.evaluate(test_gen)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

predictions = model.predict(test_gen)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_gen.classes

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))
```

---

### 2. **Konversi ke TFLite untuk Mobile (Flutter)**

```python
# convert_tflite.py
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('qc_inspector_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization (opsional)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save
with open('qc_inspector_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite!")
print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
```

**TFLite dengan Quantization (smaller size, faster inference):**
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # FP16 quantization
```

---

### 3. **Integrasi dengan Flutter**

**Tambahkan dependency di `pubspec.yaml`:**
```yaml
dependencies:
  tflite_flutter: ^0.10.0
  image_picker: ^0.8.7+5
  image: ^4.0.17
```

**Load dan predict di Flutter:**
```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class PCBInspector {
  Interpreter? _interpreter;
  
  // Load model
  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/qc_inspector_model.tflite');
  }
  
  // Predict
  Future<String> predict(String imagePath) async {
    // Load & preprocess image
    final imageData = File(imagePath).readAsBytesSync();
    img.Image? image = img.decodeImage(imageData);
    img.Image resized = img.copyResize(image!, width: 224, height: 224);
    
    // Convert to float32 array
    var input = List.generate(
      224,
      (y) => List.generate(
        224,
        (x) {
          var pixel = resized.getPixel(x, y);
          return [
            img.getRed(pixel) / 255.0,
            img.getGreen(pixel) / 255.0,
            img.getBlue(pixel) / 255.0,
          ];
        },
      ),
    );
    
    // Run inference
    var output = List.filled(1, 0.0).reshape([1, 1]);
    _interpreter!.run([input], output);
    
    double prediction = output[0][0];
    
    // Interpret (sesuaikan dengan class_indices)
    if (prediction > 0.5) {
      return "Lulus QC ✅ (${(prediction * 100).toStringAsFixed(1)}%)";
    } else {
      return "Cacat Produksi ❌ (${((1 - prediction) * 100).toStringAsFixed(1)}%)";
    }
  }
}
```

---

### 4. **Deploy ke Production**

**API Server (Python Flask):**
```python
# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('qc_inspector_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    # Load image
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    result = {
        'prediction': float(prediction),
        'class': 'lulus_qc' if prediction > 0.5 else 'cacat_produksi',
        'confidence': float(prediction if prediction > 0.5 else 1 - prediction)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

### 5. **Model Versioning & Monitoring**

**Save metadata:**
```python
import json
from datetime import datetime

metadata = {
    'model_version': '1.0.0',
    'training_date': datetime.now().isoformat(),
    'dataset_size': 428,
    'train_accuracy': 0.9523,
    'val_accuracy': 0.9176,
    'epochs_trained': 200,
    'class_indices': {'cacat_produksi': 0, 'lulus_qc': 1},
    'hyperparameters': {
        'batch_size': 16,
        'learning_rate': 0.001,
        'dropout': 0.3
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## 📚 Referensi & Resources

### Official Documentation
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [TFLite Converter](https://www.tensorflow.org/lite/convert)

### Tutorials
- [Image Classification with Transfer Learning](https://www.tensorflow.org/tutorials/images/classification)
- [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)
- [TensorBoard Visualization](https://www.tensorflow.org/tensorboard/get_started)

### Community
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Stack Overflow - TensorFlow](https://stackoverflow.com/questions/tagged/tensorflow)
- [GitHub - TensorFlow Issues](https://github.com/tensorflow/tensorflow/issues)

---

## 📝 Changelog

### Version 1.0.0 (2025-11-12)
- ✅ Initial release
- ✅ MobileNetV2 transfer learning
- ✅ Windows support (CUDA 11.2 + cuDNN 8.1)
- ✅ Support untuk 3 metode training (VS Code, Jupyter, Colab)
- ✅ Comprehensive configuration guide
- ✅ Troubleshooting & FAQ
- ✅ Dataset: 428 images (312 lulus_qc, 116 cacat_produksi)
- ✅ Target accuracy: 90%+

---

## 🙏 Credits & Acknowledgments

### Dataset
**SolDef_AI PCB Dataset for Defect Detection**
- **Author:** Maurizio Calabrese
- **Platform:** Kaggle
- **Link:** [kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection)
- **License:** Database Contents License (DbCL)

### Technologies
- **TensorFlow** - Deep learning framework
- **MobileNetV2** - Pre-trained model architecture
- **Keras** - High-level neural networks API
- **CUDA** - NVIDIA GPU acceleration
- **cuDNN** - NVIDIA deep learning primitives library

### Development Environment
- **Windows 10/11** - Operating system
- **CUDA 11.2** - GPU computing toolkit
- **cuDNN 8.1** - Deep neural network library
- **Anaconda** - Python distribution
- **VS Code** - Code editor
- **Jupyter Notebook** - Interactive development
- **Google Colab** - Cloud GPU platform

---

## 📄 License

MIT License

---

**Created for PCB Quality Control Inspection** 🔍
