# PCB Defect Detection - AI Training

Model klasifikasi gambar untuk deteksi defect pada PCB (Printed Circuit Board).

## 📋 Struktur Project

```
SolDef_AI/
├── dataset/                    # Dataset gambar PCB
│   ├── lulus_qc/              # Gambar PCB yang lolos QC
│   └── cacat_produksi/        # Gambar PCB dengan defect
├── train.py                    # Training script (lokal Windows)
├── train.bat                   # Batch file untuk menjalankan training
├── train.ipynb                 # Jupyter Notebook (Colab/Local)
├── requirements.txt            # Dependencies Python
└── .gitignore                  # File yang diabaikan git
```

## 🚀 Cara Menggunakan

### Opsi 1: Google Colab (Gratis GPU)

1. Upload `train.ipynb` ke Google Colab
2. Ubah runtime ke **T4 GPU**: `Runtime > Change runtime type > T4 GPU`
3. Upload dataset atau mount Google Drive
4. Jalankan semua cell

### Opsi 2: Local Windows (CUDA 11.2 + cuDNN 8.1)

**Prerequisites:**
- Windows 10/11
- NVIDIA GPU (RTX 3080 Ti atau yang lain)
- CUDA Toolkit 11.2
- cuDNN 8.1
- Anaconda/Miniconda

**Setup:**

```bash
# 1. Create conda environment
conda create -n pcb python=3.9 -y
conda activate pcb

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verifikasi GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Training:**

```bash
# Cara 1: Gunakan batch file (Recommended)
train.bat

# Cara 2: Langsung dengan Python
python train.py
```

## 📊 Dataset

**Struktur yang diharapkan:**
```
dataset/
├── lulus_qc/          # Gambar PCB yang lolos QC
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── cacat_produksi/    # Gambar PCB dengan defect
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Format gambar yang didukung:** JPG, JPEG, PNG

## ⚙️ Configuration

Edit di `train.py`:

```python
IMG_SIZE = (224, 224)   # Ukuran input gambar
BATCH_SIZE = 16         # Batch size untuk training
EPOCHS = 200            # Jumlah epoch
```

## 📦 Output

Setelah training selesai, akan menghasilkan:

- `qc_inspector_model.h5` - Model final
- `best_model.h5` - Model checkpoint terbaik
- `training_history.json` - History training (loss, accuracy)
- `logs/` - TensorBoard logs

## 🔧 Troubleshooting

### GPU tidak terdeteksi

```bash
# Cek instalasi CUDA
nvcc --version

# Cek TensorFlow bisa akses GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Training hang/freeze

- Pastikan menggunakan `workers=0` di `model.fit()` (sudah diset di script)
- Restart komputer
- Cek apakah antivirus memblock akses GPU

### Out of Memory (OOM)

Kurangi `BATCH_SIZE` di `train.py`:
```python
BATCH_SIZE = 8  # atau 4
```

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

## 🎯 Next Steps

Setelah training selesai:

1. **Evaluasi model** dengan test data
2. **Konversi ke TFLite** untuk mobile deployment
3. **Integrasi** dengan aplikasi Flutter

## 📄 License

MIT License

---

**Created for PCB Quality Control Inspection** 🔍
