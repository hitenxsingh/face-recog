import cv2
import os
import numpy as np
import csv
import shutil
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
TRAINER_DIR = os.path.join(SCRIPT_DIR, "trainer")
TRAINER_FILE = "trainer.yml"
ATTENDANCE_FILE = os.path.join(SCRIPT_DIR, "attendance.csv")

# Initialize Face Detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_dataset_dir(name):
    """
    Creates a directory for a user in the dataset folder.
    Returns the user ID (integer) and the directory path.
    """
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    existing_ids = []
    for folder in os.listdir(DATASET_DIR):
        try:
            eid = int(folder.split('.')[0])
            existing_ids.append(eid)
        except:
            pass
            
    new_id = 1
    if existing_ids:
        new_id = max(existing_ids) + 1
        
    user_folder = f"{new_id}.{name}"
    user_path = os.path.join(DATASET_DIR, user_folder)
    
    if not os.path.exists(user_path):
        os.makedirs(user_path)
        
    return new_id, user_path

def save_training_image(image, user_id, count, save_path):
    """
    Detects face in the image and saves the cropped face to the save_path.
    Returns True if a face was saved, False otherwise.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            file_path = f"{save_path}/User.{user_id}.{count}.jpg"
            cv2.imwrite(file_path, gray[y:y+h, x:x+w])
            return True
        return False
    except Exception:
        return False

def train_recognizer():
    """
    Trains the LBPH recognizer on all images in the dataset directory.
    Saves the model to trainer/trainer.yml.
    Returns the number of users trained.
    """
    if not os.path.exists(DATASET_DIR):
        return 0

    faces = []
    ids = []
    
    for user_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, user_folder)
        if not os.path.isdir(folder_path):
            continue
            
        try:
            user_id = int(user_folder.split('.')[0])
        except:
            continue
            
        for image_name in os.listdir(folder_path):
            if image_name.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, image_name)
                img_numpy = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), 'uint8')
                faces.append(img_numpy)
                ids.append(user_id)
                
    if not faces:
        return 0

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    
    if not os.path.exists(TRAINER_DIR):
        os.makedirs(TRAINER_DIR)
        
    recognizer.write(os.path.join(TRAINER_DIR, TRAINER_FILE))
    return len(np.unique(ids))

def load_recognizer():
    fname = os.path.join(TRAINER_DIR, TRAINER_FILE)
    if not os.path.exists(fname):
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(fname)
    return recognizer

def get_user_map():
    if not os.path.exists(DATASET_DIR):
        return {}
    user_map = {}
    for folder in os.listdir(DATASET_DIR):
        try:
            parts = folder.split('.')
            uid = int(parts[0])
            name = ".".join(parts[1:])
            user_map[uid] = name
        except:
            pass
    return user_map

def get_registered_users():
    if not os.path.exists(DATASET_DIR):
        return []
    names = []
    for folder in os.listdir(DATASET_DIR):
        try:
            parts = folder.split('.')
            name = ".".join(parts[1:])
            names.append(name)
        except:
            pass
    return names

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%I:%M:%S %p")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time", "Exit Time"])

    with open(ATTENDANCE_FILE, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    for line in lines:
        if line and len(line) >= 2:
            existing_name = line[0]
            existing_date = line[1]
            if existing_name == name and existing_date == date_str:
                return False

    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str, ""])
    
    return True

def delete_user(user_id):
    if not os.path.exists(DATASET_DIR):
        return False
        
    deleted = False
    for folder in os.listdir(DATASET_DIR):
        try:
            uid = int(folder.split('.')[0])
            if uid == user_id:
                folder_path = os.path.join(DATASET_DIR, folder)
                shutil.rmtree(folder_path)
                deleted = True
                break
        except:
            pass
            
    if deleted:
        if os.path.exists(os.path.join(TRAINER_DIR, TRAINER_FILE)):
            os.remove(os.path.join(TRAINER_DIR, TRAINER_FILE))
            
    return deleted
