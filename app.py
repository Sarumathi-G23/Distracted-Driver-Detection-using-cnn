import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    load_model_fn = tf.keras.models.load_model
except Exception:
    from keras.models import load_model as load_model_fn

MODEL_PATH = "my_model.keras"  # path to your model

# Load model
model = None
load_error = None
try:
    model = load_model_fn(MODEL_PATH, compile=False)
except Exception as e:
    load_error = e

# Streamlit UI
st.set_page_config(page_title="Distracted Driver Detection", layout="centered")
st.markdown("<h1 style='text-align: center; color: #03dac6;'>🚗 Distracted Driver Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

if model is None:
    st.error("❌ Could not load model.")
    st.write(f"Model path: `{MODEL_PATH}`")
    if load_error:
        st.subheader("Error details")
        st.code(str(load_error))
    st.stop()

st.success("✅ Model loaded successfully.")

# Class labels
classes = [
    'Safe Driving', 'Texting - Right', 'Talking on the Phone - Right',
    'Texting - Left', 'Talking on the Phone - Left', 'Operating the Radio',
    'Drinking', 'Reaching Behind', 'Hair and Makeup', 'Talking to Passenger'
]

# Preprocessing
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Input choice
option = st.radio("Choose input type:", ["Image", "Video"])

# Image Input
if option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        try:
            pil_img = Image.open(uploaded_image)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            frame = np.array(pil_img)[:, :, ::-1]  # RGB to BGR
            inp = preprocess(frame)

            if inp.shape != (1, 224, 224, 3):
                raise ValueError(f"Invalid input shape: {inp.shape}")

            preds = model.predict(inp)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx]) * 100.0
            label = f"{classes[idx]} ({conf:.2f}%)"

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_container_width=True)

            st.markdown(
    f"<div style='text-align:center; font-size:32px; font-weight:bold; margin-top:25px; color:#4CAF50;'>"
    f"Prediction: {label}</div>",
    unsafe_allow_html=True
)
        except Exception as e:
            st.error("Prediction failed: " + str(e))

# Video Input
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.markdown("### 🔍 Live Predictions (sampled frames)")
        pred_display = st.empty()

        frame_count = 0
        skip_frame_rate = 10  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frame_rate != 0:
                continue
            try:
                inp = preprocess(frame)
                if inp.shape != (1, 224, 224, 3):
                    raise ValueError(f"Invalid input shape: {inp.shape}")

                preds = model.predict(inp)[0]
                idx = int(np.argmax(preds))
                conf = float(preds[idx]) * 100.0
                label = f"{classes[idx]} ({conf:.2f}%)"
            except Exception as e:
                label = "Prediction error"
                st.warning("Prediction error: " + str(e))

            # Convert to RGB and show without text overlay
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            # Show label below frame
            pred_display.markdown(
    f"<div style='text-align:center; font-size:32px; font-weight:bold; margin-top:15px; color:#4CAF50;'>"
    f"Prediction: {label}</div>",
    unsafe_allow_html=True
)
        cap.release()
