# Offline Face Recognition Script for Windows (DirectML)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **completely offline**, real‑time face recognition application for Windows that uses  
**Haar cascades** for face detection and **ArcFace** (ONNX) for face embeddings,  
accelerated on Intel iGPUs via **DirectML** (DirectX 12).

Built with `CustomTkinter`, `OpenCV`, `SQLite`, and `ONNX Runtime` –  
no cloud, no telemetry, no internet required after the initial model download.


---

## ✨ Features

- **Live camera feed** with bounding boxes and estimated facial landmarks.
- **Face registration** – capture a single face and assign a name.
- **Real‑time recognition** – displays a green **“MATCH FOUND: [name]”** banner and a closable pop‑up when a registered person appears.
- **Manage registered faces** – open a dedicated window to view all names and **delete** any entry.
- **100% offline** – all processing is local; no internet connection is needed (except to download the embedding model once).
- **Hardware acceleration** – the ArcFace embedding model runs on your Intel iGPU (or any DirectML‑capable GPU) for fast inference.
- **Lightweight** – detection uses the built‑in Haar cascade (no extra model downloads needed).

---

## 🧰 Requirements

- **Operating System**: Windows 10 or 11 (x86_64)
- **Python**: 3.10 or newer
- **Hardware**: A webcam and an Intel iGPU (or any GPU with DirectX 12 support) for DirectML acceleration
- **Disk space**: ~200 MB for the Python environment and the ArcFace model

---

## 📥 Installation

### 1. Clone the repository
```bash
git clone https://github.com/richard-pius/Face-Recognition-Script.git
cd Face-Recognition-Script
```

### 2. (Optional but recommended) Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install the Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the ArcFace embedding model
- Go to the [InsightFace buffalo_l release page](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) and download the zip file (~166 MB).
- Extract the archive and copy **only** `w600k_r50.onnx` into the `models/` folder of this project.

> **Note**: The face detection model (Haar cascade) is already included in OpenCV – no separate download required.

---

## 🚀 Usage

Launch the application:
```bash
python face_recognition_app.py
```

### Graphical Controls
| Button | Action |
|--------|--------|
| **Register Face** | Detects the most prominent face, asks for a name, and saves the face embedding. |
| **Manage Faces** | Opens a window showing all registered names. Click **Delete** to remove a face permanently. |
| **Quit** | Stops the camera and closes the application. |

When a registered face is detected, you will see:
- A **green rectangle** and a `"MATCH: [name]"` label on the video feed.
- A **green banner** at the bottom of the main window.
- A pop‑up message stating the match (click **OK** to close).

Unknown faces are shown with a **red rectangle** and labelled `"Unknown"`.

---

## 📁 File Structure

```
Face-Recognition-Script/
├── face_recognition_app.py    # Main application code
├── requirements.txt           # Python dependencies
├── models/                    # Place the ONNX model here
│   └── w600k_r50.onnx
└── README.md
```

---

## ⚖️ Legal & Privacy Notice

### Data Storage and Privacy
- **All data is stored locally** on your computer inside `%USERPROFILE%\FaceRecognitionDML\faces.db`.
- Only **mathematical face embeddings** (512‑dimensional vectors) are saved – **no images, no video, no raw biometric data**.
- **No internet connection is used** by the application after the initial model download. No telemetry, no analytics, no network calls are made.

### Consent
- This software is intended for **personal or educational use** where you have obtained **explicit consent** from every individual whose face you register.
- **You are solely responsible** for complying with all applicable privacy laws (e.g., GDPR, CCPA, BIPA) in your jurisdiction.
- **Do not register faces of individuals without their permission.**

### Third‑Party Models and Libraries
| Component | License |
|-----------|---------|
| ArcFace model (`w600k_r50.onnx`) – [InsightFace](https://github.com/deepinsight/insightface) | **MIT** |
| Haar cascade (OpenCV) – [opencv/opencv](https://github.com/opencv/opencv) | **Apache 2.0** |
| ONNX Runtime – [Microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | **MIT** |
| CustomTkinter, NumPy, Pillow, etc. | Various permissive (MIT, BSD, Apache 2.0) |

All components are open‑source and freely usable, even in commercial products, subject to their respective licenses.

### Disclaimer
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 🔧 Troubleshooting

- **`ModuleNotFoundError: No module named 'distutils'`**  
  → Upgrade CustomTkinter to ≥5.2.1: `pip install --upgrade customtkinter`

- **Embedding error: `Required inputs (['input.1']) are missing...`**  
  → The script automatically detects the model’s input name. Verify that you placed the correct `w600k_r50.onnx` file (from the `buffalo_l` pack) in the `models/` folder.

- **Camera does not open**  
  → Ensure no other application is using the webcam. If you have multiple cameras, change the camera index in `capture_thread()` (line 146) from `0` to `1` or higher.

- **Slow detection / lag**  
  → The Haar cascade detector runs on the CPU. For better performance, you can later replace it with an ONNX face detector (not included to keep the project dependency‑free).

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute the code, provided you retain the original copyright and license notice.  
See the [LICENSE](LICENSE) file for the full text.

---

*Built with ❤️ for offline, privacy‑respecting computer vision on Windows.*
