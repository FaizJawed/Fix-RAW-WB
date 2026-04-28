# AI-Powered White Balance Correction Tool

A command-line tool for Windows to perform white balance correction on RAW images (.CR2, .NEF) using either Traditional Auto WB (AWB), Camera As-Shot WB (ASH), or a Deep Learning PyTorch model (DWB). It integrates directly with DigiKam's Batch Queue Manager.

## Features
- **ASH**: As-Shot white balance directly from the camera metadata.
- **AWB**: Fast traditional Auto White Balance via `rawpy`.
- **DWB**: Deep White-Balance Editing using a Convolutional Neural Network.
- **Non-Destructive**: Generates Adobe-standard `.xmp` sidecar files containing `<crs:Temperature>` and `<crs:Tint>`.
- **Configurable**: Define presets (`true_neutral`, `warm`, `cool`) and limits in `config.json`.

---

## 1. Installation

### Requirements
- Python 3.10+
- A Windows machine

### Setup Python Environment
1. Clone or download this repository.
2. Open a Command Prompt or PowerShell in the folder.
3. Install the required Python packages:
   ```cmd
   pip install numpy Pillow rawpy colour-science
   ```
4. Install PyTorch and Torchvision. If you want CUDA (GPU) support, use the command from the [PyTorch website](https://pytorch.org/get-started/locally/). For CPU only:
   ```cmd
   pip install torch torchvision
   ```

---

## 2. Obtaining the Deep White-Balance Model (For DWB method)

If you plan to use the `DWB` method, you must download the pre-trained weights from the original authors.

1. Go to the [Deep White-Balance Editing repository](https://github.com/mahmoudnafifi/Deep_White_Balance).
2. Look for the PyTorch pre-trained model link (usually provided via Google Drive or OneDrive).
3. Download the `.pth` file (e.g., `net_awb.pth`).
4. Place the `.pth` file in the `models/` directory inside this project folder.
5. Ensure `config.json` points to the correct `model_path`:
   ```json
   "model_path": "models/net_awb.pth"
   ```

---

## 3. Creating a Windows Executable (PyInstaller)

Instead of a single massive `.exe` (which unpacks slowly every run), we compile this into a standard folder distribution for maximum performance during batch processing.

1. Install PyInstaller:
   ```cmd
   pip install pyinstaller
   ```
2. Build the executable as a folder directory (`--onedir` is the default when omitting `--onefile`):
   ```cmd
   pyinstaller --onedir --name wb_ai wb_ai.py
   ```
3. Once the build is complete, you will find a `dist/wb_ai/` folder. This folder contains `wb_ai.exe` and all its dependencies.
4. **Important**: You must manually copy the `arch` folder, `models` folder (containing your `.pth`), and `config.json` into the `dist/wb_ai/` folder so the executable can find them.

---

## 4. Usage (Command Line)

You can run the Python script directly or use the compiled `.exe`.

**Process a single image using Deep White Balance:**
```cmd
python wb_ai.py C:\Images\photo.CR2 --method DWB
```

**Process a folder using As-Shot White Balance:**
```cmd
python wb_ai.py C:\Images\ --method ASH
```

**Process a folder using traditional Auto White Balance:**
```cmd
python wb_ai.py C:\Images\ --method AWB
```

---

## 5. DigiKam Batch Queue Manager Integration

You can integrate this tool into DigiKam to automatically process RAWs and generate XMP sidecars during import or batch operations.

1. In DigiKam, open the **Batch Queue Manager** (BQM).
2. Add your RAW files to the queue.
3. In the right panel, find **Custom Script** and add it to the workflow.
4. Set the script path to point to the `wb_wrapper.bat` file provided in this project (e.g., `C:\path\to\wb_wrapper.bat`).
5. Ensure the parameters passed to the script in DigiKam's interface are exactly:
   ```text
   "%INPUT%" "%OUTPUT%"
   ```
6. **Note on methods**: By default, `wb_wrapper.bat` is set to use the `--method DWB`. You can open `wb_wrapper.bat` in a text editor and change it to `ASH` or `AWB` if desired.
7. Run the queue. The script will create XMP files alongside your RAW files, containing the computed Temperature and Tint.

---

## 6. Troubleshooting

- **PyTorch/CUDA Errors**: Ensure your NVIDIA drivers are up to date. If `wb_ai.py` crashes with CUDA memory errors, edit `config.json` and change `"device": "cuda"` to `"device": "cpu"`. It will be slower but will not run out of VRAM.
- **Colour-Science converge errors**: Sometimes chromaticity conversion fails for extreme tints. The script has a fallback to a secondary algorithm, and ultimately a hard fallback to 5500K/0 tint to prevent breaking the batch queue.
- **Executable size is huge**: This is normal. PyTorch and its CUDA libraries are massive (~2-3 GB). Using the folder distribution (`--onedir`) ensures it loads instantly instead of unzipping the 3GB payload every single time it runs.
