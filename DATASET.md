# 📊 Dataset Information

Informasi lengkap tentang **Real-life Industrial Dataset of Casting Product** yang digunakan untuk training model deteksi cacat produksi casting.

---

## 📖 About the Dataset

### Context
Dataset ini berisi gambar produk manuf### Opsi 1: Gunakan Full Dataset dari Kaggle
```
Download full dataset (7,348 images with augmentation):
1. Kunjungi: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
2. Download "casting_data.zip" (300x300 dataset dengan augmentation)
3. Extract dan gunakan train folder
4. Training accuracy akan lebih tinggi dengan data lebih banyak
```

**Opsi 2: Foto Lebih Banyak Produk Impeller**
```
1. Ambil foto impeller casting dengan top view yang konsisten
2. Pastikan lighting stabil (gunakan special arrangement)
3. Simpan ke folder dataset/def_front/ atau dataset/ok_front/
4. Format: JPEG, minimal 224x224 pixels, grayscale atau RGB
5. Label dengan jelas (defective vs OK)
```

**Opsi 3: Download Dataset Casting Tambahan**
- Kaggle: Casting defect datasets (search "casting defect")
- Roboflow: Manufacturing defect datasets
- Public datasets: Metal casting quality inspection

**Opsi 4: Synthetic Data (Advanced)**
- Gunakan Blender/3D rendering untuk impeller models
- Image compositing dengan defect simulation
- GAN-generated defect imagesbmersible pump impeller).

**Casting** adalah proses manufaktur di mana material cair dituangkan ke dalam cetakan yang berisi rongga dengan bentuk yang diinginkan, kemudian dibiarkan mengeras.

### Problem Statement
**Casting defects** adalah ketidakteraturan yang tidak diinginkan dalam proses metal casting. 

**Jenis-jenis defect:**
- Blow holes (lubang udara)
- Pinholes (lubang kecil)
- Burr (bram/tonjolan)
- Shrinkage defects (cacat penyusutan)
- Mould material defects (cacat material cetakan)
- Pouring metal defects (cacat tuangan logam)
- Metallurgical defects (cacat metalurgi)

**Masalah di Industri:**
- ❌ Inspeksi manual sangat memakan waktu
- ❌ Akurasi manusia tidak 100% konsisten
- ❌ Human error dapat menyebabkan penolakan seluruh order
- ❌ Kerugian besar bagi perusahaan

**Solusi:**
✅ Automatic inspection menggunakan Deep Learning classification model

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
| **Product** | Submersible pump impeller |
| **View Angle** | Top view |
| **Original Size** | 512 x 512 pixels |
| **Resized to** | 224 x 224 pixels (for MobileNetV2) |
| **Format** | JPEG |
| **Color Space** | Grayscale (converted to RGB for model) |
| **Bit Depth** | 8-bit per channel |
| **File Size** | ~30-80 KB per image |
| **Lighting** | Stable lighting with special arrangement |
| **Augmentation** | Not applied (original 512x512 dataset)

---

## 🔍 Class Definitions

### Class 0: def_front (Defective Casting)
**Karakteristik:**
- Impeller casting dengan defect/cacat
- **Blow holes:** Lubang udara di permukaan
- **Pinholes:** Lubang kecil di casting
- **Burr:** Tonjolan atau bram yang tidak diinginkan
- **Shrinkage defects:** Cacat penyusutan material
- **Deformasi:** Bentuk tidak sempurna/bengkok
- **Surface defects:** Permukaan tidak smooth/kasar
- **Mould defects:** Cacat dari cetakan

**Sample Filenames:**
```
cast_def_0_0.jpeg
cast_def_0_1000.jpeg
cast_def_0_1001.jpeg
...
```

**Total:** 453 images (defective impellers)

---

### Class 1: ok_front (OK/Pass Casting)
**Karakteristik:**
- Impeller casting sempurna tanpa cacat
- Pola konsisten dan simetris
- Tidak ada lubang (blow holes/pinholes)
- Tidak ada burr atau tonjolan
- Permukaan smooth dan rata
- Bentuk sesuai spesifikasi
- Lulus quality control inspection

**Sample Filenames:**
```
cast_ok_0_0.jpeg
cast_ok_0_1.jpeg
cast_ok_0_10.jpeg
...
```

**Total:** 563 images (OK impellers)

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
- **Name:** Real-life Industrial Dataset of Casting Product
- **Author:** Ravirajsinh Dabhi
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
- **Company:** PILOT TECHNOCAST, Shapar, Rajkot, India
- **License:** CC BY 4.0 (Free to use)
- **Format:** 512x512 grayscale images (without augmentation)
- **Product:** Submersible pump impeller

**Full Dataset Available:**
- **Large dataset:** 7,348 images (300x300 pixels, grayscale, with augmentation)
  - Train: def_front (3,758) + ok_front (2,875) = 6,633 images
  - Test: def_front (453) + ok_front (262) = 715 images
- **Current dataset:** 1,016 images (512x512 pixels, grayscale, without augmentation)
  - def_front: 453 images
  - ok_front: 563 images

**Lokasi Dataset:**
```
d:\Bapenda New\explore\Data Set\Casting Product Image Data For QA\
└── casting_512x512\
    └── casting_512x512\
        ├── def_front\  (453 images)
        └── ok_front\   (563 images)
```

**Dataset sudah di-copy ke:**
```
d:\Flutter Interesting Thing\SolDef_AI PCB dataset for defect detection\SolDef_AI\
└── dataset\
    ├── def_front\  (453 images - defective impellers)
    └── ok_front\   (563 images - OK impellers)
```

---

## 🎯 Dataset Quality

### ✅ Kualitas Baik:
- **Real-life industrial data** dari pabrik aktual
- Image resolution tinggi (512x512)
- **Stable lighting** dengan special arrangement
- Background uniform (hitam)
- Objek terpusat (top-view impeller)
- Kontras bagus
- **No pre-augmentation** (original images)
- Captured with controlled environment

### ⚠️ Perhatian:
- Jumlah data relatif kecil (~1,000 images)
- Perlu data augmentation (sudah aktif di training)
- Variasi defect types bervariasi
- Single angle (top view only)
- Grayscale images (converted to RGB for model)

### 🏭 Industrial Context:
- Dataset diambil dari **PILOT TECHNOCAST**, perusahaan casting di Rajkot, India
- Produk: Submersible pump impeller
- Use case: Menggantikan manual inspection dengan AI-powered automatic inspection
- Target: Mengurangi rejection rate dan meningkatkan quality control efficiency

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
@dataset{casting_product_defect_detection,
  title={Real-life Industrial Dataset of Casting Product},
  author={Ravirajsinh Dabhi},
  year={2020},
  publisher={Kaggle},
  organization={PILOT TECHNOCAST, Shapar, Rajkot},
  url={https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product}
}
```

**Contact:**
- Email: ravirajsinhdabhi86@gmail.com
- LinkedIn: [Ravirajsinh Dabhi](https://www.linkedin.com/in/ravirajsinh-dabhi/)

**Working Prototype:**
The original author has also created a working prototype using this dataset.
Check the Kaggle page for more details.

---

## 🔗 Related Files

- [README.md](README.md) - Main documentation
- [CONFIGURATION.md](CONFIGURATION.md) - Training parameters
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

---

**Dataset ready! Start training:** `.\train.bat`
