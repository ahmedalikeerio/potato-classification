import os
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Potato Disease Detection", page_icon="🥔")
st.title("Potato Disease Detection")
st.caption("Upload a potato leaf image → prediction + confidence")

# Health check
try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    if r.ok:
        st.success(f"Backend connected  {API_URL}  | input: {r.json().get('input_size')}")
    else:
        st.warning("Backend responded but not OK")
except Exception:
    st.error(f"Backend not reachable: {API_URL}")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            files = {"file": (file.name, file.getvalue(), file.type)}
            resp = requests.post(f"{API_URL}/predict", files=files, timeout=60)

        if resp.ok:
            data = resp.json()
            st.subheader("Result")
            st.write(f"**Prediction:** {data['predicted_class']}")
            st.write(f"**Confidence:** {data['confidence']:.4f}")

            st.subheader("Probabilities")
            st.bar_chart(data["probabilities"])
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
