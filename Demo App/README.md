# 🎯 Demo App - Casting Defect Detection

Demo aplikasi Python untuk deteksi cacat casting (submersible pump impeller) dengan 2 mode:
1. **Upload Image** - Upload dan analisis gambar
2. **Real-time Detection** - Deteksi via webcam/camera

---

## 📦 Installation

### 1. Pastikan Virtual Environment Aktif
```powershell
cd "d:\Flutter Interesting Thing\SolDef_AI PCB dataset for defect detection\SolDef_AI"
.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
cd "Demo App"
pip install -r requirements.txt
```

**Dependencies:**
- `tensorflow==2.10.0` - Deep learning framework
- `numpy` - Numerical operations
- `pillow` - Image processing
- `opencv-python` - Camera & video processing
- `tkinter` - GUI (built-in with Python)

---

## 🚀 Running the Apps

### Option 1: Upload Image App
```powershell
python app_upload.py
```

**Features:**
- ✅ Upload gambar impeller (JPG, PNG, BMP)
- ✅ Analisis defect dengan 1 klik
- ✅ Tampilan hasil detail (DEFECTIVE/OK)
- ✅ Confidence score & inspection report
- ✅ Beautiful GUI dengan Tkinter

**Usage:**
1. Click "📁 Upload Image"
2. Pilih gambar impeller (top view)
3. Click "🔍 Analyze Defect"
4. Lihat hasil analisis

---

### Option 2: Real-time Camera App
```powershell
python app_realtime.py
```

**Features:**
- ✅ Real-time detection via webcam
- ✅ Live preview dengan overlay result
- ✅ FPS counter & statistics
- ✅ Capture frame dengan hasil prediksi
- ✅ Confidence bar real-time

**Usage:**
1. Click "▶️ Start Camera"
2. Arahkan camera ke impeller casting
3. Lihat hasil detection real-time
4. Click "📸 Capture Frame" untuk simpan

**Camera Requirements:**
- Webcam atau USB camera
- Lighting: Cukup terang dan stabil
- Position: Top view impeller (seperti dataset)
- Distance: ~20-30 cm dari impeller

---

## 📁 File Structure

```
Demo App/
├── app_upload.py           # Upload image version
├── app_realtime.py         # Real-time camera version
├── requirements.txt        # Python dependencies
├── README.md              # Documentation (this file)
└── captures/              # Captured frames (auto-created)
    └── capture_*.jpg
```

---

## 🎨 UI Preview

### Upload Image App:
```
┌─────────────────────────────────────────────┐
│  🔍 Casting Defect Detection                │
│  Submersible Pump Impeller Inspection       │
├─────────────────────────────────────────────┤
│                          │                  │
│   📸 Impeller Image     │  📁 Upload Image  │
│                          │  🔍 Analyze       │
│   [Image Preview]        │                  │
│                          │  📊 Result:       │
│                          │  ┌──────────────┐│
│                          │  │ DEFECTIVE ❌ ││
│                          │  └──────────────┘│
│                          │  Confidence: 95% │
│                          │                  │
│                          │  📋 Details:     │
│                          │  • Status: REJECT│
│                          │  • Product: Imp. │
└─────────────────────────────────────────────┘
```

### Real-time Camera App:
```
┌─────────────────────────────────────────────┐
│  📹 Real-time Casting Defect Detection      │
│  Live Inspection via Camera                 │
├─────────────────────────────────────────────┤
│                          │                  │
│   📹 Live Camera Feed   │  ▶️ Start Camera │
│                          │  📸 Capture      │
│   ┌──────────────────┐  │                  │
│   │  REJECT          │  │  🎯 Result:      │
│   │  DEFECTIVE (92%) │  │  ┌──────────────┐│
│   │  [Live Video]    │  │  │ DEFECTIVE ❌ ││
│   └──────────────────┘  │  └──────────────┘│
│                          │  Confidence: 92% │
│                          │  [Progress Bar]  │
│                          │                  │
│                          │  📊 Statistics:  │
│                          │  FPS: 15        │
│                          │  Frames: 1234   │
└─────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

### Model Path
Kedua app akan mencari model di:
1. `../qc_inspector_model.h5` (relative ke Demo App folder)
2. `qc_inspector_model.h5` (di Demo App folder)

**Jika model tidak ditemukan:**
```powershell
# Copy model ke Demo App folder
cd "d:\Flutter Interesting Thing\SolDef_AI PCB dataset for defect detection\SolDef_AI"
copy qc_inspector_model.h5 "Demo App\"
```

### Camera Settings (app_realtime.py)
Edit di line ~265:
```python
self.camera = cv2.VideoCapture(0)  # 0 = default camera
# Ganti ke 1, 2, dst jika punya multiple cameras
```

---

## 🎯 Prediction Results

### DEFECTIVE (Cacat)
- **Score:** < 0.5
- **Display:** Red background, "DEFECTIVE ❌"
- **Recommendation:** REJECT - Send for rework or scrap
- **Possible defects:**
  - Blow holes (air pockets)
  - Pinholes (small holes)
  - Burr (unwanted protrusions)
  - Shrinkage defects
  - Surface defects

### OK (Pass)
- **Score:** ≥ 0.5
- **Display:** Green background, "OK ✅"
- **Recommendation:** PASS - Proceed to next stage
- **Quality:** Surface good, shape integrity good

---

## 🐛 Troubleshooting

### Problem: Model not found
```
Error: Failed to load model
```

**Solution:**
```powershell
# Copy model dari parent folder
copy ..\qc_inspector_model.h5 .
```

---

### Problem: Camera not detected
```
Error: Could not access camera
```

**Solutions:**
1. Pastikan webcam terhubung
2. Check permissions (Windows Camera settings)
3. Coba camera lain (ganti `cv2.VideoCapture(1)`)
4. Restart aplikasi

---

### Problem: Tkinter not found
```
ModuleNotFoundError: No module named 'tkinter'
```

**Solution:**
```
Tkinter sudah built-in dengan Python.
Jika error, reinstall Python dengan centang "tcl/tk and IDLE"
```

---

### Problem: Import Error
```
ImportError: DLL load failed
```

**Solution:**
```powershell
# Install Visual C++ Redistributable
# Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Atau reinstall dependencies
pip uninstall tensorflow opencv-python pillow -y
pip install tensorflow==2.10.0 opencv-python pillow
```

---

## 📸 Captured Images

Real-time app akan menyimpan captured frames di:
```
Demo App/captures/
├── capture_20251112_143052.jpg
├── capture_20251112_143145.jpg
└── ...
```

**Format filename:** `capture_YYYYMMDD_HHMMSS.jpg`

---

## 🎯 Performance Tips

### Upload Image App:
- Fast prediction (~0.5-1 detik)
- No special requirements
- Works offline

### Real-time Camera App:
- **FPS:** 10-20 FPS (dengan GPU)
- **FPS:** 3-8 FPS (dengan CPU only)
- **Resolution:** 640x480 (display), 224x224 (prediction)
- **Latency:** ~50-100ms per frame

**Untuk FPS lebih tinggi:**
```python
# Reduce camera resolution (edit app_realtime.py)
self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Skip frames (predict setiap N frames)
if self.frame_count % 3 == 0:  # Predict setiap 3 frames
    prediction = self.model.predict(...)
```

---

## 🚀 Next Steps

### Deploy ke Production:
1. **Web App:** Convert ke Flask/FastAPI (lihat [DEPLOYMENT.md](../DEPLOYMENT.md))
2. **Desktop App:** Package dengan PyInstaller
3. **Mobile App:** Deploy ke Flutter dengan TFLite
4. **Industrial:** Integrate dengan conveyor belt system

### Improvements:
- [ ] Multi-threading untuk faster inference
- [ ] Batch processing untuk multiple images
- [ ] History logging (save all predictions)
- [ ] Export report (PDF/Excel)
- [ ] ROI selection untuk focus area
- [ ] Multiple camera support

---

## 📞 Support

**Issues?** Kembali ke main README atau TROUBLESHOOTING.md

**Model accuracy rendah?** Check:
1. Lighting consistency (stable & bright)
2. Camera position (top view, centered)
3. Image quality (not blurry)
4. Retrain model dengan data lebih banyak

---

**Ready to demo!** 🎉

Jalankan `python app_upload.py` atau `python app_realtime.py`
