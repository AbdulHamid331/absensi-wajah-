
import streamlit as st
import cv2
import numpy as np
from utils import get_embedding, save_knowledge_base, load_knowledge_base

st.title("📝 Daftarkan Wajah Pegawai Baru")
name = st.text_input("Nama Pegawai")
img_file = st.camera_input("Ambil foto wajah")

if img_file is not None and name:
    bytes_data = img_file.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR")

    embedding = get_embedding(image)
    if embedding is not None:
        kb = load_knowledge_base()
        kb.append({
            "user": name,
            "embedding": embedding.tolist(),
            "threshold": 0.6
        })
        save_knowledge_base(kb)
        st.success(f"✅ Pegawai '{name}' berhasil didaftarkan.")
    else:
        st.warning("Wajah tidak terdeteksi.")
