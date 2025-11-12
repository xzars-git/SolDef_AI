# 🎯 Demo App - Quick Start

## 🚀 Cara Paling Mudah:

### Double-click: `DEMO_LAUNCHER.bat`

Menu akan muncul dengan 2 pilihan:
1. **Upload Image** - Upload dan analisis gambar impeller
2. **Real-time Camera** - Deteksi live via webcam

---

## 📋 Atau Manual:

### Upload Image Version:
```powershell
# Double-click run_upload.bat
# Atau:
python app_upload.py
```

### Real-time Camera Version:
```powershell
# Double-click run_realtime.bat
# Atau:
python app_realtime.py
```

---

## 📦 First-time Setup:

```powershell
# 1. Activate virtual environment
cd ..
.venv\Scripts\activate

# 2. Install dependencies
cd "Demo App"
pip install -r requirements.txt

# 3. Run demo
python app_upload.py
```

---

## ✅ Checklist:

- [ ] Virtual environment activated
- [ ] Dependencies installed (`tensorflow`, `opencv-python`, `pillow`)
- [ ] Model file exists (`qc_inspector_model.h5`)
- [ ] Camera connected (untuk real-time version)

---

## 🎯 Features:

### Upload Image App:
- Beautiful GUI with Tkinter
- Drag & drop support (via file dialog)
- Detailed inspection report
- Confidence score visualization

### Real-time Camera App:
- Live video feed with overlay
- FPS counter & statistics
- Capture frame with prediction
- Real-time confidence meter

---

**Ready to demo!** 🎉
