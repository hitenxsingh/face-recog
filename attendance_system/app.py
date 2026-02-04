import streamlit as st
import cv2
import numpy as np
import face_rec_utils as utils
import pandas as pd
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Page config
st.set_page_config(
    page_title="Face Attendance",
    page_icon="👤",
    layout="centered"
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .stButton > button { width: 100%; border-radius: 5px; }
    h1 { color: #1E88E5; }
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("👤 Face Attendance")
choice = st.sidebar.radio("", ["🏠 Home", "📷 Mark Attendance", "⚙️ Manage Users"], label_visibility="collapsed")

# RTC Configuration
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- Processors ---
class RegistrationProcessor(VideoProcessorBase):
    def __init__(self):
        self.count = 0
        self.user_id = None
        self.save_path = None
        self.capturing = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def update_config(self, user_id, save_path, capturing):
        self.user_id = user_id
        self.save_path = save_path
        self.capturing = capturing

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if img is None:
            return av.VideoFrame.from_ndarray(np.zeros((1, 1, 3), dtype=np.uint8), format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (30, 144, 255), 2)

        if self.capturing and self.user_id is not None and self.count < 30:
            saved = utils.save_training_image(img, self.user_id, self.count, self.save_path)
            if saved:
                self.count += 1
        
        cv2.putText(img, f"{self.count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 144, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.recognizer = utils.load_recognizer()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.user_map = utils.get_user_map()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if img is None:
            return av.VideoFrame.from_ndarray(np.zeros((1, 1, 3), dtype=np.uint8), format="bgr24")
             
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            if self.recognizer:
                try:
                    id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    if confidence < 100:
                        name = self.user_map.get(id, f"User {id}")
                        utils.mark_attendance(name)
                        color = (0, 200, 0)
                        label = name
                    else:
                        color = (0, 0, 200)
                        label = "Unknown"
                except:
                    color = (0, 0, 200)
                    label = "Error"
            else:
                color = (0, 0, 200)
                label = "No Model"

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Pages ---

if "🏠 Home" in choice:
    st.title("Face Attendance System")
    
    # Status Metrics
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(os.path.join(utils.TRAINER_DIR, utils.TRAINER_FILE)):
            st.metric("System Status", "Online", delta="Model Ready", delta_color="normal")
        else:
             st.metric("System Status", "Offline", delta="Needs Training", delta_color="inverse")
    with col2:
        users = utils.get_registered_users()
        st.metric("Registered Users", len(users))

    st.markdown("---")
    
    # Attendance Table with Date Filter
    st.subheader("📋 Attendance Log")
    
    if os.path.exists(utils.ATTENDANCE_FILE):
        df = pd.read_csv(utils.ATTENDANCE_FILE)
        st.dataframe(df, use_container_width=True)
        
        # Admin Controls for Demo
        with st.expander("Admin Controls (Demo Only)"):
            if st.button("Clear Attendance Log"):
                try:
                    os.remove(utils.ATTENDANCE_FILE)
                    st.toast("Attendance log cleared!", icon="🗑️")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("No attendance records found.")

elif "⚙️ Manage Users" in choice:
    st.title("Manage Users")
    
    tab1, tab2 = st.tabs(["Add User", "Delete User"])
    
    with tab1:
        st.subheader("Register New User")
        name_input = st.text_input("Name", placeholder="Enter full name")
        
        # Center the camera
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            ctx = webrtc_streamer(
                key="registration", 
                video_processor_factory=RegistrationProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False}
            )
        
        if ctx.video_processor:
            if st.button("📸 Start Capture", type="primary", use_container_width=True):
                if not name_input:
                    st.toast("Please enter a name first!", icon="⚠️")
                else:
                    user_id, save_path = utils.create_dataset_dir(name_input)
                    ctx.video_processor.update_config(user_id, save_path, True)
                    st.toast(f"Capturing face data for {name_input}...", icon="📸")
        
        st.divider()
        if st.button("🧠 Train Model", type="primary"):
            with st.spinner("Training recognition model..."):
                n_faces = utils.train_recognizer()
            if n_faces > 0:
                st.toast(f"Model trained successfully with {n_faces} users!", icon="✅")
                st.balloons()
            else:
                st.toast("No training data found!", icon="⚠️")

    with tab2:
        st.subheader("Existing Users")
        users_map = utils.get_user_map() # Returns dict {id: name}
        if users_map:
            for uid, name in users_map.items():
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(f"**{name}** (ID: {uid})")
                with col_btn:
                    if st.button("Delete", key=f"del_{uid}"):
                        if utils.delete_user(uid):
                            st.toast(f"User {name} deleted!", icon="🗑️")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.toast("Failed to delete user.", icon="❌")
        else:
            st.info("No users registered.")

elif "📷 Mark" in choice:
    st.title("Mark Attendance")
    
    if utils.load_recognizer() is None:
        st.warning("⚠️ The recognition model is not trained. Please go to 'Manage Users' to train it.")
    else:
        st.write("Look at the camera to mark your attendance automatically.")
        
        # Center the camera
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
             webrtc_streamer(
                key="attendance",
                video_processor_factory=AttendanceProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False}
            )
