# 📊 Dataset Information

Informasi lengkap tentang dataset PCB defect detection dan cara penggunaannya.

---

## 📦 Dataset Source & Credit

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

## 🗂️ Struktur Dataset

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

---

## 📈 Dataset Statistics

- **Total images:** 428
- **Class distribution:**
  - Lulus QC: 312 images (72.9%) ✅
  - Cacat Produksi: 116 images (27.1%) ❌
- **Train/Val split:** 80/20
  - Training: 343 images
  - Validation: 85 images

---

## 📥 Cara Download Dataset

### Opsi 1: Manual Download (Recommended untuk Windows)

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

### Opsi 2: Kaggle API (Command Line)

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

## ✅ Verifikasi Dataset

### Untuk Local/VS Code:
```bash
# Dataset sudah ada di folder project
# Pastikan struktur folder benar
ls dataset/lulus_qc | wc -l        # Should show 312
ls dataset/cacat_produksi | wc -l  # Should show 116
```

### Untuk Jupyter Notebook Local:
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

### Untuk Google Colab:

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

---

## 💡 Tips Dataset

### Minimum Requirements
- **Minimum recommended:** 100 images per class
- **Ideal:** 500-1000 images per class
- **Current dataset:** 312 + 116 = 428 images (Good untuk start)

### Data Quality
- ✅ Consistent lighting
- ✅ Same angle/perspective
- ✅ Similar resolution
- ✅ Clear focus
- ❌ Avoid duplicates
- ❌ Avoid blurry images

### Class Balance
- **Current ratio:** 72.9% vs 27.1% (slightly imbalanced)
- **Jika imbalance parah (>80% vs <20%):** Gunakan `class_weight` parameter (lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md))

### Naming Convention
- Gunakan nama yang konsisten
- Hindari karakter spesial
- Contoh: `pcb_001.jpg`, `pcb_002.jpg`, dst.

---

## 📁 Menambah Data Training

### Opsi 1: Foto Manual
1. Ambil foto PCB dengan kamera/smartphone
2. Pastikan lighting konsisten
3. Resize ke ukuran optimal (224x224 atau 512x512)
4. Copy ke folder `dataset/lulus_qc/` atau `dataset/cacat_produksi/`

### Opsi 2: Data Augmentation (Otomatis)
- Sudah aktif di training script
- Rotation, zoom, flip, shift otomatis
- Tidak perlu action tambahan

### Opsi 3: Download Dataset Tambahan
**Kaggle Datasets:**
- Search: "PCB defect detection"
- Filter: Datasets dengan lisensi open
- Download & merge dengan dataset existing

**Roboflow Universe:**
- https://universe.roboflow.com
- Search: "PCB inspection" atau "PCB defect"
- Export dalam format "Folder Structure"

---

## 🔍 Eksplorasi Dataset (Advanced)

### Visualisasi Sample Images
```python
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Get one batch
images, labels = next(generator)

# Plot
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    label = "Lulus QC" if labels[i] == 1 else "Cacat"
    ax.set_title(f"{label}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('dataset_sample.png', dpi=150)
plt.show()
```

### Analisis Class Distribution
```python
import os
import matplotlib.pyplot as plt

classes = ['lulus_qc', 'cacat_produksi']
counts = []

for cls in classes:
    count = len(os.listdir(f'dataset/{cls}'))
    counts.append(count)
    print(f"{cls}: {count} images ({count/sum(counts)*100:.1f}%)")

# Plot
plt.figure(figsize=(8, 6))
plt.bar(classes, counts, color=['green', 'red'])
plt.title('Class Distribution')
plt.ylabel('Number of Images')
plt.xlabel('Class')
for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)
plt.savefig('class_distribution.png', dpi=150)
plt.show()
```

### Cek Gambar Corrupt
```python
from PIL import Image
import os

def check_corrupted_images(dataset_dir):
    corrupted = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, file)
                try:
                    img = Image.open(filepath)
                    img.verify()  # Verify it's a valid image
                except Exception as e:
                    corrupted.append(filepath)
                    print(f"❌ Corrupted: {filepath}")
    
    if not corrupted:
        print("✅ All images are valid!")
    return corrupted

corrupted_images = check_corrupted_images('dataset')
```

---

## 📚 Dataset Best Practices

### DO's ✅
- Gunakan gambar dengan kualitas tinggi
- Pastikan lighting konsisten
- Balance class distribution (50:50 ideal)
- Tambah data augmentation jika dataset kecil
- Split data: 80% train, 20% validation
- Reserve 10% untuk testing (belum di-train)

### DON'Ts ❌
- Jangan duplikasi gambar
- Jangan mix dataset berbeda format
- Jangan training dengan test data
- Jangan ignore imbalanced classes
- Jangan gunakan gambar blur/rusak

---

Kembali ke [README.md](README.md) | Lihat [CONFIGURATION.md](CONFIGURATION.md)
