# 📊 Dataset Information

Informasi lengkap tentang dataset Casting Product Image Data yang digunakan untuk training.

---

## 📁 Dataset Structure

```
dataset/
├── def_front/          # Gambar produk dengan cacat (defect)
│   ├── cast_def_0_0.jpeg
│   ├── cast_def_0_1000.jpeg
│   ├── cast_def_0_1001.jpeg
│   └── ... (453 images)
│
└── ok_front/           # Gambar produk lulus QC (OK)
    ├── cast_ok_0_0.jpeg
    ├── cast_ok_0_1.jpeg
    ├── cast_ok_0_10.jpeg
    └── ... (563 images)
```

---

## 📈 Dataset Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **def_front** (Cacat) | 453 images | 44.6% |
| **ok_front** (Lulus QC) | 563 images | 55.4% |
| **TOTAL** | **1,016 images** | 100% |

**Status:** ✅ Relatif balance (rasio 44:56)

---

## 🖼️ Image Specifications

| Property | Value |
|----------|-------|
| **Original Size** | 512 x 512 pixels |
| **Resized to** | 224 x 224 pixels (for MobileNetV2) |
| **Format** | JPEG |
| **Color Space** | RGB (3 channels) |
| **Bit Depth** | 8-bit per channel |
| **File Size** | ~30-80 KB per image |

---

## 🔍 Class Definitions

### Class 0: def_front (Cacat Produksi)
**Karakteristik:**
- Produk casting dengan defect/cacat
- Pola tidak sempurna
- Ada lubang, retak, atau deformasi
- Warna tidak merata
- Permukaan tidak smooth

**Sample Filenames:**
```
cast_def_0_0.jpeg
cast_def_0_1000.jpeg
cast_def_0_1001.jpeg
...
```

### Class 1: ok_front (Lulus QC)
**Karakteristik:**
- Produk casting sempurna
- Pola konsisten dan simetris
- Tidak ada lubang atau retak
- Warna merata
- Permukaan smooth

**Sample Filenames:**
```
cast_ok_0_0.jpeg
cast_ok_0_1.jpeg
cast_ok_0_10.jpeg
...
```

---

## 📊 Data Split

Training script akan otomatis split data:

```python
# Automatic split dengan ImageDataGenerator
validation_split = 0.2  # 20% untuk validation

# Hasil split:
Training Set:   ~813 images (80%)
Validation Set: ~203 images (20%)
```

**Distribusi per Class:**
- **def_front:**
  - Training: ~362 images
  - Validation: ~91 images
- **ok_front:**
  - Training: ~450 images
  - Validation: ~113 images

---

## 🔄 Data Augmentation

Untuk meningkatkan variasi data dan cegah overfitting:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        # Rotasi random ±20°
    width_shift_range=0.2,    # Geser horizontal 20%
    height_shift_range=0.2,   # Geser vertical 20%
    horizontal_flip=True,     # Flip horizontal random
    zoom_range=0.2,           # Zoom in/out 20%
    shear_range=0.15,         # Shear transformation
    fill_mode='nearest',      # Fill pixel kosong
    validation_split=0.2      # 20% untuk validation
)
```

**Efek Augmentation:**
- Dataset "efektif" menjadi ~3x lipat
- Model lebih robust terhadap variasi posisi & orientasi
- Reduce overfitting

---

## 📥 Dataset Source

**Original Dataset:**
- **Name:** Casting Product Image Data for Quality Inspection
- **Source:** Kaggle / Public Dataset
- **License:** CC BY 4.0 (Free to use)
- **Format:** casting_512x512

**Lokasi Dataset:**
```
d:\Bapenda New\explore\Data Set\Casting Product Image Data For QA\
└── casting_512x512\
    └── casting_512x512\
        ├── def_front\
        └── ok_front\
```

**Dataset sudah di-copy ke:**
```
d:\Flutter Interesting Thing\SolDef_AI PCB dataset for defect detection\SolDef_AI\
└── dataset\
    ├── def_front\
    └── ok_front\
```

---

## 🎯 Dataset Quality

### ✅ Kualitas Baik:
- Image resolution tinggi (512x512)
- Lighting konsisten
- Background uniform (hitam)
- Objek terpusat
- Kontras bagus

### ⚠️ Perhatian:
- Jumlah data relatif kecil (~1000 images)
- Perlu data augmentation (sudah aktif)
- Variasi defect terbatas
- Single angle (frontal view only)

---

## 📈 Dataset Recommendations

### Untuk Production-Ready Model:

**Minimal Requirements:**
- **≥ 2,000 images total** (1000 per class)
- Multiple angles (front, side, top)
- Various lighting conditions
- Different backgrounds

**Current Dataset (1,016 images):**
- ✅ Cukup untuk POC (Proof of Concept)
- ✅ Good untuk training awal
- ⚠️ Perlu lebih banyak data untuk production
- ⚠️ Consider data augmentation (sudah aktif)

### Cara Menambah Data:

**Opsi 1: Foto Lebih Banyak Produk**
```
1. Ambil foto produk casting dengan berbagai angle
2. Pastikan lighting konsisten
3. Simpan ke folder dataset/def_front/ atau dataset/ok_front/
4. Format: JPEG, minimal 224x224 pixels
```

**Opsi 2: Download Dataset Tambahan**
- Kaggle: Casting defect datasets
- Roboflow: Manufacturing defect datasets
- Public datasets: Quality inspection images

**Opsi 3: Synthetic Data (Advanced)**
- Gunakan Blender/3D rendering
- Image compositing
- GAN-generated images

---

## 🔍 Sample Inspection

### Cek Distribusi Dataset:
```python
import os

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.endswith(('.jpeg', '.jpg', '.png'))])

def_count = count_images('dataset/def_front')
ok_count = count_images('dataset/ok_front')
total = def_count + ok_count

print(f"Cacat (def_front): {def_count} images ({def_count/total*100:.1f}%)")
print(f"Lulus (ok_front):  {ok_count} images ({ok_count/total*100:.1f}%)")
print(f"Total: {total} images")
```

### Visualisasi Sample Images:
```python
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

def show_samples(class_name, num_samples=5):
    folder = f'dataset/{class_name}'
    files = os.listdir(folder)[:num_samples]
    
    plt.figure(figsize=(15, 3))
    for i, file in enumerate(files):
        img = image.load_img(f'{folder}/{file}', target_size=(224, 224))
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(f'{class_name}\n{file}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show samples
show_samples('def_front', 5)
show_samples('ok_front', 5)
```

---

## 📊 Data Analysis

### Image Size Distribution:
```python
from PIL import Image
import os

def analyze_image_sizes(folder):
    sizes = []
    for file in os.listdir(folder):
        if file.endswith(('.jpeg', '.jpg', '.png')):
            img = Image.open(f'{folder}/{file}')
            sizes.append(img.size)
    
    unique_sizes = set(sizes)
    print(f"Unique sizes in {folder}:")
    for size in unique_sizes:
        count = sizes.count(size)
        print(f"  {size[0]}x{size[1]}: {count} images")

analyze_image_sizes('dataset/def_front')
analyze_image_sizes('dataset/ok_front')
```

---

## 🎯 Expected Model Performance

Dengan dataset ini (1,016 images):

| Metric | Expected Range | Target |
|--------|---------------|--------|
| **Training Accuracy** | 90-98% | ≥ 95% |
| **Validation Accuracy** | 85-95% | ≥ 90% |
| **Gap (Train - Val)** | 3-8% | < 5% |
| **Training Time** | 20-25 min | - |

**Faktor yang Mempengaruhi:**
- ✅ Transfer Learning (MobileNetV2 pre-trained)
- ✅ Data Augmentation aktif
- ✅ Early Stopping & LR Reduction
- ⚠️ Dataset relatif kecil

---

## 🔄 Dataset Updates

**Current Version:** v1.0 (1,016 images)

**Planned Updates:**
- [ ] v1.1 - Add 500+ images per class
- [ ] v1.2 - Multiple angles (side, top views)
- [ ] v1.3 - Various lighting conditions
- [ ] v2.0 - Multi-class defect types

---

## 📝 Dataset Citation

```
@dataset{casting_product_qa,
  title={Casting Product Image Data for Quality Inspection},
  year={2024},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/...}
}
```

---

## 🔗 Related Files

- [README.md](README.md) - Main documentation
- [CONFIGURATION.md](CONFIGURATION.md) - Training parameters
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

---

**Dataset ready! Start training:** `.\train.bat`
