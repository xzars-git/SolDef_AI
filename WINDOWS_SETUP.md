# 💻 Windows Setup Guide (CUDA 11.2 + cuDNN 8.1)

Panduan lengkap instalasi environment untuk training PCB defect detection di Windows.

---

## Prerequisites untuk Windows 10/11

| Component | Version | Required | Download |
|-----------|---------|----------|----------|
| **Windows** | 10/11 64-bit | ✅ | - |
| **NVIDIA GPU** | RTX 3080 Ti (atau GPU lain) | ✅ | - |
| **NVIDIA Driver** | ≥ 452.39 | ✅ | [Download](https://www.nvidia.com/Download/index.aspx) |
| **CUDA Toolkit** | 11.2 | ✅ | [Download](https://developer.nvidia.com/cuda-11.2.0-download-archive) |
| **cuDNN** | 8.1 for CUDA 11.x | ✅ | [Download](https://developer.nvidia.com/rdp/cudnn-archive) |
| **Python** | 3.9.x | ✅ | [Download](https://www.python.org/downloads/) |
| **Visual Studio** | 2019/2022 (Build Tools) | ⚠️ | [Download](https://visualstudio.microsoft.com/downloads/) |

---

## Step-by-Step Installation

### 1️⃣ Install NVIDIA Driver

```bash
# Cek current driver version
nvidia-smi
```

Jika belum terinstall atau versi lama, download driver terbaru dari [NVIDIA](https://www.nvidia.com/Download/index.aspx).

---

### 2️⃣ Install CUDA Toolkit 11.2

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

### 3️⃣ Install cuDNN 8.1

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

### 4️⃣ Install Python 3.9 (jika belum)

**Download & Install:**
- Download: https://www.python.org/downloads/
- Pilih Python 3.9.x (recommended: 3.9.13)
- **PENTING:** Centang "Add Python to PATH" saat install
- Install untuk "All Users" atau "Just Me"

**Verifikasi:**
```powershell
python --version
# Output: Python 3.9.x
```

---

### 5️⃣ Create Virtual Environment & Install Dependencies

```powershell
# 1. Buat virtual environment dengan Python 3.9
python -m venv .venv

# 2. Aktivasi virtual environment (PowerShell)
.venv\Scripts\activate

# Atau jika pakai Command Prompt (CMD):
# .venv\Scripts\activate.bat

# 3. Upgrade pip (recommended)
python -m pip install --upgrade pip

# 4. Install TensorFlow 2.10.0 (compatible dengan CUDA 11.2 + cuDNN 8.1)
pip install tensorflow==2.10.0

# 5. Install dependencies lainnya
pip install -r requirements.txt

# 6. Verifikasi instalasi
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

**Expected Output:**
```
TensorFlow: 2.10.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Catatan:**
- Virtual environment akan dibuat di folder `.venv` di project directory
- Setiap kali buka terminal baru, jalankan `.venv\Scripts\activate` dulu
- Untuk deactivate: ketik `deactivate`

---

## Windows-Specific Troubleshooting

### Problem: DLL Load Failed

**Symptoms:**
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

### Problem: GPU Not Detected

**Symptoms:**
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

### Problem: CUDA Version Mismatch

**Symptoms:**
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

## Windows-Specific Configuration

### PowerShell Execution Policy (jika error)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Aktivasi Virtual Environment di VS Code
**Buka VS Code:**
1. Tekan `Ctrl + Shift + P`
2. Ketik "Python: Select Interpreter"
3. Pilih `.venv` (Python 3.9.x)

**Terminal Settings (settings.json):**
```json
{
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}
```

### Troubleshooting Virtual Environment

**Problem: `.venv\Scripts\activate` tidak jalan**

**Solution:**
```powershell
# Gunakan full path
& "d:\Flutter Interesting Thing\SolDef_AI PCB dataset for defect detection\SolDef_AI\.venv\Scripts\Activate.ps1"

# Atau ubah execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Problem: Virtual environment tidak terdeteksi di VS Code**

**Solution:**
1. Reload VS Code (`Ctrl + Shift + P` → "Reload Window")
2. Atau restart VS Code
3. Atau manually pilih interpreter (Ctrl + Shift + P → "Python: Select Interpreter")

---

## Next Steps

Setelah setup selesai, kembali ke [README.md](README.md) untuk mulai training!
