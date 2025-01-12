import face_recognition
import cv2
import numpy as np
import os

known_face_encodings = []
known_face_names = []

image_dir = "images"

for name in os.listdir(image_dir):
    person_dir = os.path.join(image_dir, name)
    if os.path.isdir(person_dir):
        for filename in os.listdir(person_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        encoding = encodings[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                    else:
                        print(f"No face detected in {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

np.save("known_face_encodings.npy", np.array(known_face_encodings))  # Save as NumPy array
np.save("known_face_names.npy", np.array(known_face_names))      # Save as NumPy array

print("Face encodings saved.")