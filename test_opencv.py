import cv2
import os
import numpy as np

print(f"CV2 Version: {cv2.__version__}")
try:
    print(f"CV2 Data Path: {cv2.data.haarcascades}")
    path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Cascade Path: {path}")
    
    if os.path.exists(path):
        print("Cascade file exists.")
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            print("Error: CascadeClassifier loaded but is empty/invalid.")
        else:
            print("CascadeClassifier loaded successfully.")
    else:
        print("Error: Cascade file does NOT exist at that path.")

except Exception as e:
    print(f"Error accessing cv2.data: {e}")

# Test detection on a dummy black image (should find nothing, but not crash)
img = np.zeros((100, 100, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = clf.detectMultiScale(gray, 1.1, 4)
print(f"Detection run. Faces found: {len(faces)}")
