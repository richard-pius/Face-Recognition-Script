import cv2
import numpy as np
import sqlite3
import pickle
import threading
import queue
import time
import sys
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

import onnxruntime as ort

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "models"
RECOGNITION_MODEL = MODEL_DIR / "w600k_r50.onnx"   # ArcFace

DB_PATH = Path.home() / "FaceRecognitionDML" / "faces.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

MATCH_THRESHOLD = 0.4
DET_THRESHOLD = 0.5

# ArcFace canonical landmarks
REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655]
], dtype=np.float32)

# ----------------------------------------------------------------------
# Database functions
# ----------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE,
                  embedding BLOB)''')
    conn.commit()
    conn.close()

def save_face(name, embedding):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO faces (name, embedding) VALUES (?, ?)",
              (name, pickle.dumps(embedding)))
    conn.commit()
    conn.close()

def load_all_faces():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM faces")
    rows = c.fetchall()
    conn.close()
    names, embeddings = [], []
    for name, blob in rows:
        emb = pickle.loads(blob)
        emb = emb / np.linalg.norm(emb)
        names.append(name)
        embeddings.append(emb)
    return names, embeddings

def delete_face_by_name(name):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("DELETE FROM faces WHERE name = ?", (name,))
    conn.commit()
    conn.close()

def get_all_names():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT name FROM faces")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

# ----------------------------------------------------------------------
# ONNX session for embedding (DirectML) – auto input name
# ----------------------------------------------------------------------
def create_session(model_path):
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    print(f"[INFO] Loaded {model_path.name} on {sess.get_providers()[0]}")
    input_name = sess.get_inputs()[0].name
    print(f"[INFO] ArcFace input name: '{input_name}'")
    return sess, input_name

rec_session, rec_input_name = create_session(RECOGNITION_MODEL)

# ----------------------------------------------------------------------
# Haar Cascade Face Detector (built into OpenCV)
# ----------------------------------------------------------------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("ERROR: Could not load Haar cascade. Reinstall opencv-python.")
    sys.exit(1)
print("[INFO] Face detector: Haar cascade (CPU)")

def detect_faces_haar(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    result = []
    for (x, y, w, h) in faces:
        # Estimate 5 landmarks from bounding box
        lx = x + w * 0.3; ly = y + h * 0.35
        rx = x + w * 0.7; ry = ly
        nx = x + w * 0.5; ny = y + h * 0.5
        mlx = x + w * 0.3; mly = y + h * 0.65
        mrx = x + w * 0.7; mry = mly
        landmarks = np.array([[lx, ly], [rx, ry], [nx, ny], [mlx, mly], [mrx, mry]], dtype=np.float32)
        result.append((x, y, x+w, y+h, 1.0, landmarks))
    return result

# ----------------------------------------------------------------------
# Face alignment and embedding
# ----------------------------------------------------------------------
def align_face(frame_bgr, landmarks, output_size=112):
    M, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_POINTS, method=cv2.LMEDS)
    if M is None:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        d = np.linalg.norm(right_eye - left_eye)
        if d == 0:
            return cv2.resize(frame_bgr, (output_size, output_size))
        scale = output_size * 0.35 / d
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        center = tuple(((left_eye + right_eye) / 2).astype(int))
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += output_size / 2 - center[0]
        M[1, 2] += output_size * 0.4 - center[1]
    return cv2.warpAffine(frame_bgr, M, (output_size, output_size), borderMode=cv2.BORDER_CONSTANT)

def preprocess_face_for_recognition(aligned_bgr):
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32)
    img_rgb = (img_rgb - 127.5) / 128.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_chw = np.expand_dims(img_chw, axis=0)
    return img_chw

def get_embedding(aligned_bgr):
    input_tensor = preprocess_face_for_recognition(aligned_bgr)
    outputs = rec_session.run(None, {rec_input_name: input_tensor})
    emb = outputs[0][0]
    emb = emb / np.linalg.norm(emb)
    return emb

def cosine_distance(a, b):
    return 1.0 - np.dot(a, b)

# ----------------------------------------------------------------------
# Threading and globals
# ----------------------------------------------------------------------
capture_queue = queue.Queue(maxsize=2)
inference_queue = queue.Queue()
gui_queue = queue.Queue(maxsize=1)
popup_queue = queue.Queue()          # for match notifications

known_names, known_embeddings = [], []
db_lock = threading.Lock()
stop_event = threading.Event()

def capture_thread(cam_index=0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        try:
            capture_queue.put_nowait(frame.copy())
        except queue.Full:
            pass
        inference_queue.put(frame.copy())
    cap.release()

def recognition_thread():
    global known_names, known_embeddings
    last_popup_time = 0
    while not stop_event.is_set():
        try:
            frame = inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            faces = detect_faces_haar(frame)
        except Exception as e:
            print(f"Detection error: {e}")
            continue

        annotated = frame.copy()
        match_found = False
        matched_name = ""

        for (x1, y1, x2, y2, score, landmarks) in faces:
            if x2 <= x1 or y2 <= y1:
                continue
            aligned = align_face(frame, landmarks, 112)
            try:
                emb = get_embedding(aligned)
            except Exception as e:
                print(f"Embedding error: {e}")
                continue

            with db_lock:
                names_snap = known_names[:]
                embs_snap = known_embeddings[:]

            best_dist = 1.0
            best_idx = -1
            for i, db_emb in enumerate(embs_snap):
                dist = cosine_distance(emb, db_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0 and best_dist < MATCH_THRESHOLD:
                match_found = True
                matched_name = names_snap[best_idx]
                color = (0, 255, 0)
                label = f"MATCH: {matched_name}"
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            for (lx, ly) in landmarks.astype(int):
                cv2.circle(annotated, (lx, ly), 2, (255, 255, 0), -1)

        # Popup only for matched faces, throttled to once per second
        if match_found and (time.time() - last_popup_time) > 1.0:
            popup_queue.put(("match", matched_name))
            last_popup_time = time.time()

        try:
            gui_queue.put_nowait((annotated, match_found, matched_name))
        except queue.Full:
            pass

# ----------------------------------------------------------------------
# GUI Application
# ----------------------------------------------------------------------
class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition – DirectML (ArcFace + Haar)")
        self.geometry("1024x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Video frame
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Initializing camera...", corner_radius=8)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Match banner
        self.match_label = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white", fg_color="green", corner_radius=10
        )
        self.match_label.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        # Button bar
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=20, pady=(5, 15), sticky="ew")
        btn_frame.grid_columnconfigure((0,1,2), weight=1)

        self.reg_btn = ctk.CTkButton(btn_frame, text="Register Face", command=self.register_face)
        self.reg_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.manage_btn = ctk.CTkButton(btn_frame, text="Manage Faces", command=self.open_manage_window)
        self.manage_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.quit_btn = ctk.CTkButton(btn_frame, text="Quit", command=self.quit_app, fg_color="red")
        self.quit_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.match_clear_job = None
        self.protocol("WM_DELETE_WINDOW", self.quit_app)

        # Start threads
        self.cap_thread = threading.Thread(target=capture_thread, daemon=True)
        self.rec_thread = threading.Thread(target=recognition_thread, daemon=True)
        self.cap_thread.start()
        self.rec_thread.start()

        # Periodic UI updates
        self.update_gui()
        self.check_popup()

    def update_gui(self):
        try:
            annotated, match, name = gui_queue.get_nowait()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Resize to fit label while keeping aspect ratio
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            if label_width > 1 and label_height > 1:
                pil_img.thumbnail((label_width, label_height), Image.LANCZOS)
            else:
                pil_img.thumbnail((800, 600), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)
            self.video_label.configure(image=ctk_img, text="")

            if match:
                self.match_label.configure(text=f"MATCH FOUND: {name}", fg_color="green")
                if self.match_clear_job:
                    self.after_cancel(self.match_clear_job)
                self.match_clear_job = self.after(
                    3000, lambda: self.match_label.configure(text="", fg_color="transparent")
                )
        except queue.Empty:
            pass
        self.after(30, self.update_gui)

    def check_popup(self):
        try:
            msg_type, name = popup_queue.get_nowait()
            if msg_type == "match":
                messagebox.showinfo("Match Found", f"A registered face was recognised: {name}")
        except queue.Empty:
            pass
        self.after(100, self.check_popup)

    def register_face(self):
        try:
            frame = capture_queue.get_nowait()
        except queue.Empty:
            messagebox.showerror("Error", "No frame available")
            return

        faces = detect_faces_haar(frame)
        if len(faces) == 0:
            messagebox.showwarning("No Face", "No face detected.")
            return
        if len(faces) > 1:
            messagebox.showwarning("Multiple Faces", "Only one person allowed.")
            return

        x1, y1, x2, y2, score, landmarks = faces[0]
        aligned = align_face(frame, landmarks, 112)
        emb = get_embedding(aligned)

        # Ask for name
        dialog = ctk.CTkInputDialog(text="Enter name for this face:", title="Register Face")
        name = dialog.get_input()
        if not name:
            return

        with db_lock:
            save_face(name, emb)
            known_names.append(name)
            known_embeddings.append(emb)

        self.match_label.configure(text=f"Registered: {name}", fg_color="blue")
        if self.match_clear_job:
            self.after_cancel(self.match_clear_job)
        self.match_clear_job = self.after(
            3000, lambda: self.match_label.configure(text="", fg_color="transparent")
        )

    def open_manage_window(self):
        """Opens a modal window to view and delete registered faces."""
        self.manage_win = ctk.CTkToplevel(self)
        self.manage_win.title("Manage Registered Faces")
        self.manage_win.geometry("400x400")
        self.manage_win.grab_set()  # Make it modal

        names = get_all_names()
        if not names:
            ctk.CTkLabel(self.manage_win, text="No faces registered.").pack(pady=20)
            return

        scroll_frame = ctk.CTkScrollableFrame(self.manage_win, width=350, height=300)
        scroll_frame.pack(pady=10, padx=10, fill="both", expand=True)

        for name in names:
            row_frame = ctk.CTkFrame(scroll_frame)
            row_frame.pack(fill="x", pady=2, padx=5)

            name_label = ctk.CTkLabel(row_frame, text=name, anchor="w", width=200)
            name_label.pack(side="left", padx=5)

            delete_btn = ctk.CTkButton(
                row_frame, text="Delete", width=60, fg_color="red",
                command=lambda n=name: self.delete_face(n)
            )
            delete_btn.pack(side="right", padx=5)

    def delete_face(self, name):
        """Deletes a face from database and updates in‑memory lists."""
        confirm = messagebox.askyesno("Confirm Delete", f"Delete '{name}'?")
        if not confirm:
            return

        with db_lock:
            delete_face_by_name(name)
            global known_names, known_embeddings
            known_names, known_embeddings = load_all_faces()

        # Refresh the manage window
        self.manage_win.destroy()
        self.open_manage_window()

    def quit_app(self):
        stop_event.set()
        self.destroy()

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    known_names, known_embeddings = load_all_faces()
    app = FaceRecognitionApp()
    app.mainloop()