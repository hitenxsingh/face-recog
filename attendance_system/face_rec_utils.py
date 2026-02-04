import shutil

# ... existing imports ...

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ... existing code ...

def mark_attendance(name):
    """
    Marks attendance for the given name if not already marked for today.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%I:%M:%S %p")

    if not os.path.exists(ATTENDANCE_FILE):
# ... existing code ...
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str, ""])
    
    return True

def delete_user(user_id):
    """
    Deletes the dataset folder for the given user ID and the trainer file.
    Returns True if successful, False otherwise.
    """
    if not os.path.exists(DATASET_DIR):
        return False
        
    deleted = False
    for folder in os.listdir(DATASET_DIR):
        try:
            uid = int(folder.split('.')[0])
            if uid == user_id:
                folder_path = os.path.join(DATASET_DIR, folder)
                shutil.rmtree(folder_path) # Robust deletion
                deleted = True
                break
        except:
            pass
            
    if deleted:
        # If we deleted a user, the model is now outdated.
        # We should delete the trainer file so the user knows to retrain.
        if os.path.exists(os.path.join(TRAINER_DIR, TRAINER_FILE)):
            os.remove(os.path.join(TRAINER_DIR, TRAINER_FILE))
            
    return deleted
