# PCB Defect Detection - Quick Start Guide

## 🎯 Pilih Metode Training Anda

### 1️⃣ VS Code (Untuk Developer - Recommended)

**Kelebihan:** Debugging, Git integration, code completion
**Waktu Setup:** 10 menit
**Waktu Training:** 20-25 menit (200 epochs)

```bash
# Terminal di VS Code
conda activate pcb
python train.py
```

**Tips:**
- Set breakpoint untuk debugging
- Gunakan Ctrl+C untuk stop training
- Lihat output real-time di terminal

---

### 2️⃣ Jupyter Notebook (Untuk Eksperimen)

**Kelebihan:** Visualisasi interaktif, run per cell, dokumentasi inline
**Waktu Setup:** 5 menit
**Waktu Training:** 20-25 menit (200 epochs)

```bash
# Terminal
conda activate pcb
jupyter notebook
# Browser akan terbuka → klik train.ipynb
```

**Tips:**
- Jalankan cell satu per satu dengan Shift+Enter
- Visualisasi langsung muncul
- Bisa edit parameter tanpa restart

---

### 3️⃣ Google Colab (Untuk Cloud/Gratis GPU)

**Kelebihan:** Gratis T4 GPU, tidak perlu install apapun
**Waktu Setup:** 2 menit
**Waktu Training:** 30-40 menit (200 epochs)

```
1. Buka https://colab.research.google.com
2. Upload train.ipynb
3. Runtime → Change runtime type → GPU (T4)
4. Upload dataset (atau mount Google Drive)
5. Run all cells
```

**Tips:**
- Keep tab terbuka (jangan close browser)
- Download model setelah selesai
- Session timeout 12 jam

---

## 📊 Expected Results

### Training Progress (Example)

```
Epoch 1/200
343/343 [==============================] - 8s 21ms/step
loss: 0.4521 - accuracy: 0.7821 - val_loss: 0.3912 - val_accuracy: 0.8235

Epoch 50/200
343/343 [==============================] - 6s 18ms/step
loss: 0.1234 - accuracy: 0.9456 - val_loss: 0.1567 - val_accuracy: 0.9176

Epoch 100/200
343/343 [==============================] - 6s 18ms/step
loss: 0.0821 - accuracy: 0.9678 - val_loss: 0.1245 - val_accuracy: 0.9294

Epoch 150/200
343/343 [==============================] - 6s 18ms/step
loss: 0.0567 - accuracy: 0.9789 - val_loss: 0.1123 - val_accuracy: 0.9412

Early Stopping: Restoring best weights from epoch 145
```

### Final Results (Good Model)

```
✅ TRAINING COMPLETED!

Final Results:
Training Accuracy: 0.9678
Validation Accuracy: 0.9412
Gap: 2.66% → Excellent generalization!

Files saved:
  - qc_inspector_model.h5 (9.2 MB)
  - best_model.h5 (9.2 MB)
  - training_history.json (15 KB)
```

---

## 🔧 Common Issues & Quick Fix

| Problem | Quick Fix |
|---------|-----------|
| GPU not detected | Restart PC, check CUDA 11.2 installed |
| Training hang | Already fixed with `workers=0` |
| Out of Memory | Change `BATCH_SIZE = 8` in train.py |
| Too slow | Check GPU is being used, not CPU |
| Low accuracy | Train longer (300 epochs) or add more data |

---

## 📞 Need Help?

1. **Check README.md** - Comprehensive troubleshooting guide
2. **Check logs/** - TensorBoard untuk visualisasi
3. **Check training_history.json** - Metrics per epoch

---

## ✅ Next Steps After Training

1. ✅ **Test model** → Load model & predict test images
2. ✅ **Convert to TFLite** → For Flutter mobile app
3. ✅ **Deploy** → Integrate with your application

**See README.md section "Next Steps & Deployment" for complete guide!**
