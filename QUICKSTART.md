# 🚀 Quick Start Guide

Panduan singkat untuk memulai training PCB defect detection model dengan cepat.

---

## ⚡ Quick Start (3 Langkah)

```bash
# 1. Setup environment
conda create -n pcb python=3.9 -y
conda activate pcb
pip install -r requirements.txt

# 2. Verifikasi GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# 3. Jalankan training
python train.py
```

**Hasil:** Model `qc_inspector_model.h5` siap digunakan! 🎉

---

## 🎯 3 Metode Training

### 1️⃣ VS Code (Recommended)

**Kelebihan:** Debugging, Git integration, code completion
**Waktu Training:** 20-25 menit (200 epochs)

```bash
# Terminal di VS Code
conda activate pcb
python train.py
# atau Ctrl+F5
```

---

### 2️⃣ Jupyter Notebook

**Kelebihan:** Visualisasi interaktif, run per cell
**Waktu Training:** 20-25 menit (200 epochs)

```bash
conda activate pcb
jupyter notebook
# Buka train.ipynb → Run all
```

---

### 3️⃣ Google Colab (Cloud GPU)

**Kelebihan:** Gratis T4 GPU, tidak perlu install
**Waktu Training:** 30-40 menit (200 epochs)

```
1. Buka https://colab.research.google.com
2. Upload train.ipynb
3. Runtime → GPU (T4)
4. Upload dataset atau mount Drive
5. Run all cells
```

---

## ⚡ Expected Results

**Training Time (RTX 3080 Ti):**
- 50 epochs: ~5-7 menit
- 100 epochs: ~10-15 menit  
- 200 epochs: ~20-25 menit

**Target Accuracy:**
- Training: ≥ 90% ✅
- Validation: ≥ 90% ✅

**Output Files:**
```
✅ qc_inspector_model.h5        # Model final
✅ best_model.h5                # Best checkpoint
✅ training_history.json        # Training metrics
✅ logs/                        # TensorBoard logs
```

---

## 🔧 Common Issues

| Problem | Quick Fix | Detail |
|---------|-----------|--------|
| GPU not detected | Install CUDA 11.2 + cuDNN 8.1 | [WINDOWS_SETUP.md](WINDOWS_SETUP.md) |
| Import Error | `conda activate pcb` | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| OOM Error | `BATCH_SIZE = 8` | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Training hang | Already fixed (workers=0) | - |
| Low accuracy | Tambah epochs atau data | [CONFIGURATION.md](CONFIGURATION.md) |

---

## � Dokumentasi Lengkap

| Dokumen | Konten |
|---------|--------|
| **[README.md](README.md)** | Quick start & cara running |
| **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** | Setup CUDA & cuDNN |
| **[DATASET.md](DATASET.md)** | Info dataset & download |
| **[CONFIGURATION.md](CONFIGURATION.md)** | Tweaking parameters |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Troubleshooting & FAQ |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Deploy ke production |

---

**Ready to start?** Run `python train.py` dan monitor progress! 🚀
