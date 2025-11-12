"""
📦 SCRIPT KONVERSI DATASET - SolDef_AI PCB Defect Detection
============================================================

TUJUAN:
    Mengkonversi dataset dari format LabelMe (JSON anotasi) 
    menjadi format yang siap untuk training (folder-based classification)

FORMAT DATASET ANDA SAAT INI:
    Labeled/
    ├── WIN_20220329_14_30_32_Pro.jpg      # Gambar PCB
    ├── WIN_20220329_14_30_32_Pro.json     # Anotasi (label "no_good", etc)
    └── ...

    Dataset/
    ├── CS1/, CS2/, CS3/, ...              # Komponen PCB per kategori
    
FORMAT YANG DIBUTUHKAN UNTUK TRAINING:
    dataset/
    ├── lulus_qc/                          # PCB yang bagus (tidak ada defect)
    │   ├── good_001.jpg
    │   └── ...
    └── cacat_produksi/                    # PCB yang cacat
        ├── defect_001.jpg
        └── ...

CARA KERJA SCRIPT INI:
    1. Baca semua file JSON di folder "Labeled/"
    2. Cek label di JSON (contoh: "no_good" = cacat)
    3. Copy gambar ke folder yang sesuai:
       - Jika ada label "no_good" → copy ke cacat_produksi/
       - Jika tidak ada label atau label "good" → copy ke lulus_qc/
    4. Tampilkan summary hasil konversi

CARA MENJALANKAN:
    python prepare_dataset.py

Author: AI Assistant
Date: November 2025
"""

import os
import json
import shutil
from pathlib import Path
from collections import Counter

print("="*70)
print("📦 KONVERSI DATASET - SolDef_AI PCB Defect Detection")
print("="*70)
print()


# ============================================================================
# KONFIGURASI
# ============================================================================
class Config:
    """Konfigurasi path dan label"""
    
    # Input paths
    LABELED_DIR = 'Labeled'          # Folder berisi gambar + JSON anotasi
    
    # Output paths
    OUTPUT_DIR = 'dataset'           # Folder output untuk training
    GOOD_DIR = 'dataset/lulus_qc'    # Subfolder untuk PCB bagus
    DEFECT_DIR = 'dataset/cacat_produksi'  # Subfolder untuk PCB cacat
    
    # Label mapping (sesuaikan dengan label di JSON Anda!)
    DEFECT_LABELS = [
        'no_good',       # Label untuk defect di JSON Anda
        'defect',
        'cacat',
        'rusak',
        'bad',
        'ng'             # No Good
    ]
    
    GOOD_LABELS = [
        'good',
        'ok',
        'pass',
        'lulus'
    ]


# ============================================================================
# FUNGSI UTILITY
# ============================================================================
def create_output_folders(config):
    """
    Buat folder output untuk dataset training
    """
    print("📁 [1/4] Membuat folder output...")
    print("-" * 70)
    
    # Buat folder utama
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"   ✅ Folder '{config.OUTPUT_DIR}/' dibuat")
    
    # Buat subfolder
    Path(config.GOOD_DIR).mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Folder '{config.GOOD_DIR}/' dibuat")
    
    Path(config.DEFECT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Folder '{config.DEFECT_DIR}/' dibuat")
    
    print()


def read_json_annotation(json_path):
    """
    Baca file JSON anotasi dan extract label
    
    Parameter:
        json_path (str): Path ke file JSON
    
    Return:
        list: List of labels found in JSON
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract labels dari shapes
        labels = []
        if 'shapes' in data:
            for shape in data['shapes']:
                if 'label' in shape:
                    labels.append(shape['label'].lower())
        
        return labels
    
    except Exception as e:
        print(f"   ⚠️  Error reading {json_path}: {e}")
        return []


def classify_image(labels, config):
    """
    Klasifikasikan gambar berdasarkan label
    
    Parameter:
        labels (list): List of labels from JSON
        config: Config object
    
    Return:
        str: 'good' atau 'defect'
    """
    # Jika tidak ada label, anggap bagus (good)
    if not labels:
        return 'good'
    
    # Cek apakah ada label defect
    for label in labels:
        for defect_label in config.DEFECT_LABELS:
            if defect_label in label:
                return 'defect'
    
    # Jika ada label good explicitly
    for label in labels:
        for good_label in config.GOOD_LABELS:
            if good_label in label:
                return 'good'
    
    # Default: good (jika tidak match dengan defect labels)
    return 'good'


def process_labeled_dataset(config):
    """
    Process semua file di folder Labeled/
    
    Return:
        dict: Statistics hasil proses
    """
    print("🔄 [2/4] Memproses dataset berlabel...")
    print("-" * 70)
    
    # Statistics
    stats = {
        'total_files': 0,
        'good_count': 0,
        'defect_count': 0,
        'error_count': 0,
        'labels_found': Counter()
    }
    
    # Cek apakah folder Labeled exists
    if not os.path.exists(config.LABELED_DIR):
        print(f"   ❌ ERROR: Folder '{config.LABELED_DIR}/' tidak ditemukan!")
        return stats
    
    # Get semua file JSON
    json_files = [f for f in os.listdir(config.LABELED_DIR) 
                  if f.endswith('.json')]
    
    print(f"   📊 Ditemukan {len(json_files)} file JSON anotasi")
    print()
    
    # Process setiap file
    for i, json_file in enumerate(json_files, 1):
        json_path = os.path.join(config.LABELED_DIR, json_file)
        
        # Extract image filename from JSON filename
        image_file = json_file.replace('.json', '.jpg')
        image_path = os.path.join(config.LABELED_DIR, image_file)
        
        # Cek apakah gambar exists
        if not os.path.exists(image_path):
            print(f"   ⚠️  [{i}/{len(json_files)}] Gambar tidak ditemukan: {image_file}")
            stats['error_count'] += 1
            continue
        
        # Read JSON dan extract labels
        labels = read_json_annotation(json_path)
        
        # Count label statistics
        for label in labels:
            stats['labels_found'][label] += 1
        
        # Classify image
        classification = classify_image(labels, config)
        
        # Determine destination folder
        if classification == 'defect':
            dest_dir = config.DEFECT_DIR
            stats['defect_count'] += 1
            icon = '❌'
        else:
            dest_dir = config.GOOD_DIR
            stats['good_count'] += 1
            icon = '✅'
        
        # Copy image ke destination
        dest_path = os.path.join(dest_dir, image_file)
        
        try:
            shutil.copy2(image_path, dest_path)
            stats['total_files'] += 1
            
            # Print progress setiap 10 file
            if i % 10 == 0 or i == len(json_files):
                print(f"   {icon} [{i}/{len(json_files)}] Processed: {image_file} → {classification}")
        
        except Exception as e:
            print(f"   ⚠️  Error copying {image_file}: {e}")
            stats['error_count'] += 1
    
    print()
    return stats


def print_statistics(stats, config):
    """
    Tampilkan statistik hasil konversi
    """
    print("📊 [3/4] Statistik Konversi")
    print("-" * 70)
    
    print(f"\n   📁 Total file diproses: {stats['total_files']}")
    print(f"   ✅ PCB Lulus QC (Bagus): {stats['good_count']}")
    print(f"   ❌ PCB Cacat Produksi: {stats['defect_count']}")
    
    if stats['error_count'] > 0:
        print(f"   ⚠️  Error: {stats['error_count']}")
    
    print(f"\n   📋 Label yang ditemukan:")
    if stats['labels_found']:
        for label, count in stats['labels_found'].most_common():
            print(f"      - {label}: {count} kali")
    else:
        print(f"      (Tidak ada label ditemukan)")
    
    # Balance check
    print(f"\n   ⚖️  Balance Dataset:")
    if stats['good_count'] > 0 and stats['defect_count'] > 0:
        ratio = stats['good_count'] / stats['defect_count']
        print(f"      Ratio (Good:Defect) = {ratio:.2f}:1")
        
        if 0.5 <= ratio <= 2.0:
            print(f"      ✅ Dataset relatif balance (bagus!)")
        elif ratio > 2.0:
            print(f"      ⚠️  Lebih banyak gambar 'good' ({stats['good_count']}) dibanding 'defect' ({stats['defect_count']})")
            print(f"         Pertimbangkan tambah data 'defect' atau gunakan augmentasi")
        else:
            print(f"      ⚠️  Lebih banyak gambar 'defect' ({stats['defect_count']}) dibanding 'good' ({stats['good_count']})")
            print(f"         Pertimbangkan tambah data 'good' atau gunakan augmentasi")
    
    print()


def verify_output(config):
    """
    Verifikasi hasil konversi
    """
    print("✅ [4/4] Verifikasi Hasil")
    print("-" * 70)
    
    print(f"\n   📂 Struktur Output:")
    print(f"   {config.OUTPUT_DIR}/")
    
    # Count files in each folder
    good_files = len([f for f in os.listdir(config.GOOD_DIR) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"   ├── lulus_qc/ ({good_files} gambar)")
    
    defect_files = len([f for f in os.listdir(config.DEFECT_DIR) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"   └── cacat_produksi/ ({defect_files} gambar)")
    
    total = good_files + defect_files
    print(f"\n   📊 Total: {total} gambar siap untuk training!")
    
    # Recommendation
    print(f"\n   💡 Rekomendasi:")
    if total < 400:
        print(f"      ⚠️  Dataset relatif kecil ({total} gambar)")
        print(f"         - Minimum recommended: 400 gambar (200 per kelas)")
        print(f"         - Gunakan data augmentasi saat training")
    elif total < 1000:
        print(f"      ✓ Dataset cukup untuk training ({total} gambar)")
        print(f"         - Gunakan data augmentasi untuk hasil lebih baik")
    else:
        print(f"      ✅ Dataset bagus! ({total} gambar)")
        print(f"         - Cukup untuk training model yang robust")
    
    print()


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """
    Main function untuk konversi dataset
    """
    
    config = Config()
    
    print("📋 Konfigurasi:")
    print(f"   - Input: {config.LABELED_DIR}/")
    print(f"   - Output: {config.OUTPUT_DIR}/")
    print(f"   - Defect labels: {', '.join(config.DEFECT_LABELS)}")
    print(f"   - Good labels: {', '.join(config.GOOD_LABELS)}")
    print()
    
    # Step 1: Buat folder output
    create_output_folders(config)
    
    # Step 2: Process dataset
    stats = process_labeled_dataset(config)
    
    # Step 3: Print statistics
    print_statistics(stats, config)
    
    # Step 4: Verify output
    verify_output(config)
    
    # Final message
    print("="*70)
    print("🎉 KONVERSI DATASET SELESAI!")
    print("="*70)
    print()
    print("✅ Dataset sudah siap untuk training!")
    print(f"✅ Folder output: {config.OUTPUT_DIR}/")
    print()
    print("🚀 Langkah Selanjutnya:")
    print("   1. Verifikasi gambar di folder 'dataset/'")
    print("   2. Jalankan training script:")
    print("      python Phase1_Training_Script.py")
    print("   3. Atau gunakan Jupyter Notebook:")
    print("      jupyter notebook Phase1_Training_Model.ipynb")
    print()
    print("="*70)
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Konversi dibatalkan oleh user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
