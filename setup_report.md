# Vehicle Detector & OSNet Vehicle Re-ID Setup Report

**Date:** 2026-10-24  
**Project:** Indian-Traffic Vehicle Detector & Grid-Pooled OSNet Vehicle Re-ID  
**GitHub Repository:** [https://github.com/prashantsingh85699-gif/pipe](https://github.com/prashantsingh85699-gif/pipe)  
**Privacy & Compliance:** Strictly enforced. No number-plate reading, no face recognition, no personal-identifier extraction. Zero plate annotations used.

---

## 1. Hardware & System Diagnostics

| Metric | Value | Verification Status |
|---|---|---|
| **OS** | Windows 11 Home Single Language (Build 26300) | `systeminfo` verified |
| **GPU** | Intel(R) Iris(R) Xe Graphics (integrated) | Verified |
| **NVIDIA GPU** | **None** (`nvidia-smi` not found / failed) | Verified |
| **VRAM** | **0.0 GB dedicated (under 4 GB)** | `torch.cuda.is_available() == False` |
| **System RAM** | 7,863 MB (~8.0 GB total, ~822 MB free) | Verified |
| **Disk Space** | C: Drive: 475.6 GB total, **227.4 GB free** (well above 15 GB threshold) | Verified |

> **VRAM Notice:** Dedicated VRAM is **under 4 GB** (0.0 GB dedicated, Intel Iris Xe shared system memory only). Local training is not feasible on this laptop; training must run on Google Colab or cloud GPU.

---

## 2. Environment & Python

- **Python Version:** 3.11.9 (`venv311`)
- **PyTorch:** `2.14.1+cpu` (matching available CPU hardware)
- **CUDA Availability:** `False`
- **GPU Name:** `None`

---

## 3. Packages & Windows Cython / C++ / NumPy 2 Build Issues

### Installed Packages:
`torchreid` (1.4.0 develop), `scipy` (1.17.1), `opencv-python` (5.0.0), `matplotlib` (3.11.2), `tqdm` (4.70.1), `pyyaml` (6.0.3), `gdown` (6.4.2), `kaggle` (2.2.5), `h5py` (3.16.0), `yacs` (0.1.8), `ultralytics` (8.4.186), `pycocotools` (2.0.11), `numpy` (1.26.4).

### Exact Windows Build & Compatibility Errors Encountered & Resolved:

1. **Missing MSVC C++ Build Tools:**
   ```text
   building 'torchreid.metrics.rank_cylib.rank_cy' extension
   error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```
   **Fix:** Patched `deep-person-reid/setup.py` with `NO_CYTHON=1` fallback. TorchReID seamlessly switches to its native Python evaluation (`torchreid.metrics.rank.py`), allowing develop mode installation without requiring MSVC build tools.

2. **NumPy 2.x C-API & Dependency Incompatibility:**
   ```text
   error: numpy 1.26.4 is installed but numpy>=2; python_version >= "3.9" is required by {'opencv-python'}
   ```
   **Fix:** Installed `numpy<2` (`numpy==1.26.4`) to support TorchReID Cython/data structures, and ran `setup.py develop --no-deps` to eliminate setuptools easy_install dependency conflicts. Clean import test verified.

---

## 4. Pretrained Weights & OSNet Feature Map Extraction

- **Model:** `osnet_x1_0(pretrained=True)`
- **Cached Weights:** `C:\Users\chtus\.cache\torch\checkpoints\osnet_x1_0_imagenet.pth` (10.9 MB, downloaded via Google Drive fallback)
- **Input Tensor:** `torch.randn(2, 3, 224, 224)`
- **Extracted Feature Map Shape (before global pooling):** `torch.Size([2, 512, 14, 14])`
  - Batch size (B): 2
  - Feature Channels (C): 512
  - Spatial Height (H): 14
  - Spatial Width (W): 14
- **Spatial Resolution for 3x3 Grid Pooling:** **14 x 14** (spatial dimensions 14x14 > 3x3, allowing 3x3 grid pooling across the 512 feature channels).

---

## 5. Dataset Status: VeRi-776 (Kaggle Mirror)

- **Target Dataset:** `abhyudaya12/veri-vehicle-re-identification-dataset`
- **Credential Verification:**
  - `C:\Users\chtus\.kaggle\kaggle.json`: **NOT FOUND**
  - `KAGGLE_USERNAME` / `KAGGLE_KEY`: **NOT SET**
- **Action Taken (Per Mandatory Instruction):**
  - **STOPPED Step 5.** Awaiting user Kaggle API credentials.

---

## 6. Offline Verification Checks

Two standalone verification processes with outbound network calls strictly blocked:
1. `offline_check.py`: Loaded `yolo11s.pt` (18.4 MB) and `UVH-26-MV-YOLOv11-S.pt` (54.5 MB) from local disk. **PASSED**.
2. `offline_osnet_check.py`: Loaded `osnet_x1_0(pretrained=True)` from local cache `~/.cache/torch/checkpoints/` and executed forward feature map extraction. **PASSED**.
