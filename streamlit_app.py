import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import io
from utils.sam_clip import run_sam_clip_pipeline
from utils.pond_env import PondCleaningEnvFromDetections
from utils.agent import load_agent, decide_action
from utils.visuals import display_segments
from datetime import datetime

st.set_page_config(page_title="Pond Cleaner AI", layout="wide")

# Sidebar
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to", ["📸 Camera Feed", "🧠 SAM + CLIP Detection", "🤖 RL Agent Interaction", "📊 Logs & Settings"])
esp32_url = st.sidebar.text_input("ESP32-CAM URL", "http://192.168.1.100:5000")
confidence_thresh = st.sidebar.slider("CLIP Confidence Threshold", 0.3, 1.0, 0.65)

# Cache agent and environment
@st.cache_resource
def load_env_and_agent():
    env = PondCleaningEnvFromDetections()
    agent = load_agent()
    return env, agent

env, agent = load_env_and_agent()
logs = []

# ========================== 📸 Camera Feed ==========================
if section == "📸 Camera Feed":
    st.title("📸 ESP32-CAM Live Feed")

    if st.button("Capture Image"):
        try:
            response = requests.get(f"{esp32_url}/capture", timeout=5)
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption="Captured from ESP32-CAM", use_column_width=True)
            st.session_state["last_image"] = image
        except Exception as e:
            st.error(f"❌ Could not capture image: {e}")

# ========================== 🧠 SAM + CLIP Detection ==========================
elif section == "🧠 SAM + CLIP Detection":
    st.title("🧠 SAM + CLIP Object Detection")

    image_source = st.radio("Choose image source", ["📸 Last Captured", "📂 Upload Image"])
    img = None

    if image_source == "📂 Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
    elif "last_image" in st.session_state:
        img = st.session_state["last_image"]

    if img:
        st.image(img, caption="Selected Image", use_column_width=True)
        if st.button("Run SAM + CLIP"):
            segments = run_sam_clip_pipeline(img, conf_thresh=confidence_thresh)
            st.success(f"Detected {len(segments)} objects!")
            display_segments(segments, img)
            st.session_state["segments"] = segments
    else:
        st.warning("⚠️ No image selected.")

# ========================== 🤖 RL Agent Interaction ==========================
elif section == "🤖 RL Agent Interaction":
    st.title("🤖 RL Agent Interaction")

    if "segments" not in st.session_state:
        st.warning("⚠️ Run SAM + CLIP detection first.")
    else:
        segments = st.session_state["segments"]
        obs = env.reset_from_segments(segments)
        st.image(env.render(), caption="Environment State", use_column_width=True)

        if st.button("Agent Decide Action"):
            action = decide_action(agent, obs)
            reward, done, info = env.step(action)
            st.success(f"🎯 Agent chose action: {action}")
            st.info(f"Reward: {reward}, Done: {done}")
            st.image(env.render(), caption="Post-action State")

            logs.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "action": action,
                "reward": reward,
                "labels": [seg["label"] for seg in segments]
            })

# ========================== 📊 Logs & Settings ==========================
elif section == "📊 Logs & Settings":
    st.title("📊 Logs & Agent Decisions")

    if logs:
        for log in logs[-10:][::-1]:
            st.write(f"[{log['time']}] Action: {log['action']} | Reward: {log['reward']} | Labels: {log['labels']}")
    else:
        st.write("No logs yet.")

    if st.button("Clear Logs"):
        logs.clear()
        st.success("Logs cleared.")

