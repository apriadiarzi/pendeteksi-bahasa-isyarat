import os
import av
import threading
import streamlit as st
# from streamlit_option_menu import option_menu
import streamlit_nested_layout
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('model/SIBI_ASL_Keras2.h5')

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

x1, y1, x2, y2 = 100, 100, 300, 300
color = (0, 255, 0)
thickness = 2
    
# Streamlit Components
st.set_page_config(
    page_title="Pendeteksi Bahasa Isyarat",
    page_icon="https://cdn-icons-png.flaticon.com/512/6268/6268080.png",
    layout="wide",  # centered, wide
    initial_sidebar_state="expanded",
)

col1, col2 = st.columns(spec=[6, 2], gap="medium")


def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="bgr24")  # Decode and convert frame to RGB
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cropped = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    reshaped = resized.reshape(1, 28, 28, 1)

    normalized = reshaped / 255.0

    prediction = model.predict(normalized)
    label = np.argmax(prediction)
    output = class_names[label]

    cv2.putText(frame, output, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # cv2.imshow('Deteksi Bahasa Isyarat', frame)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")  # Encode and return BGR frame


with col1:
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_frame_callback=video_frame_callback,
        # audio_frame_callback=audio_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this to config for cloud deployment.
        media_stream_constraints={"video": {"height": {"ideal": 480}}, "audio": False},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True),
    )
