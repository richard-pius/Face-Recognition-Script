import face_recognition
import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QDialog
)
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap

class TimedMessageBox(QDialog):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Detected!")
        layout = QVBoxLayout(self)
        label = QLabel(message)
        layout.addWidget(label)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.close)
        self.timer.setSingleShot(True)
        self.timer.start(5000)

class VideoStreamWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.app = app

        self.video_label = QLabel()
        self.video_label.setScaledContents(True)
        self.layout.addWidget(self.video_label)

        self.start_button = QPushButton("Start Recognition")
        self.start_button.clicked.connect(self.start_recognition)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recognition")
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)
        self.setFixedSize(640, 480)

        self.stop_recognition = False
        self.video_capture = None
        self.last_popup_time = QDateTime.currentDateTime()
        self.TOLERANCE = 0.6
        self.popup_open = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.known_encodings = None
        self.known_names = None
        self.load_known_faces()

    def load_known_faces(self):
        try:
            self.known_encodings = np.load("known_face_encodings.npy")
            self.known_names = np.load("known_face_names.npy")
            self.status_label.setText("Loaded known faces.")
            self.start_button.setEnabled(True)
        except FileNotFoundError:
            self.status_label.setText("Error: Run encode_faces.py first.")
            self.start_button.setEnabled(False)

    def show_popup(self, name):
        if not self.popup_open:
            self.popup_open = True
            now = QDateTime.currentDateTime()
            elapsed_time = self.last_popup_time.msecsTo(now)
            if elapsed_time > 2000:
                self.last_popup_time = now
                popup = TimedMessageBox(f"Object Detected: {name}", self)
                popup.finished.connect(self.popup_closed)
                popup.open()

    def popup_closed(self):
        self.popup_open = False

    def update_frame(self):
        if self.stop_recognition:
            self.release_resources()
            self.status_label.setText("Recognition stopped.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.timer.stop()
            return

        if self.video_capture is None or not self.video_capture.isOpened():
            self.status_label.setText("Webcam disconnected or not found.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.timer.stop()
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.release_resources()
            self.status_label.setText("Webcam disconnected.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.timer.stop()
            return

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)

        if self.known_encodings is None or self.known_names is None:
            return

        for (top, right, bottom, left) in face_locations:
            name = "Human_Face"
            try:
                face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=[(top, right, bottom, left)])
                if face_encodings:
                    face_encoding = face_encodings[0]
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.TOLERANCE)
                    if True in matches:
                        best_match_index = matches.index(True)
                        name = self.known_names[best_match_index]
                        self.show_popup(name)
                        break
            except IndexError as e:
                print(f"IndexError during face encoding: {e}")
            except Exception as e:
                print(f"Error during face encoding: {e}")

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.video_label.setPixmap(pixmap)
        self.timer.start(10)

    def start_recognition(self):
        if self.known_encodings is None or self.known_names is None:
            self.status_label.setText("Please load face data first.")
            return

        self.stop_recognition = False
        self.last_popup_time = QDateTime.currentDateTime()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Recognition started...")
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.status_label.setText("Error: Could not open webcam.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            return
        self.timer.start(10)

    def stop_recognition(self):
        self.stop_recognition = True
        self.status_label.setText("Stopping recognition...")
        self.timer.stop()
        self.release_resources()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def release_resources(self):
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()
        self.video_capture = None
        self.video_label.clear()
        # DO NOT release known_encodings or known_names

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoStreamWidget()
    window.setWindowTitle("Facial Recognition")
    window.show()
    sys.exit(app.exec_())