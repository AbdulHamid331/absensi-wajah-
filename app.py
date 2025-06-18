
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from utils import get_embedding, cosine_similarity, load_knowledge_base

st.title("📸 Absensi Wajah Pegawai")
img_file = st.camera_input("Ambil foto wajah untuk absensi")

if img_file is not None:
    bytes_data = img_file.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR")

    embedding = get_embedding(image)
    if embedding is not None:
        kb = load_knowledge_base()
        user_id = None
        for user in kb:
            sim = cosine_similarity(embedding, user["embedding"])
            if sim >= user["threshold"]:
                user_id = user["user"]
                confidence = sim
                break

        if user_id:
            st.success(f"✅ Absensi Berhasil: {user_id} (Confidence: {confidence:.2f})")
            with open("logs/absensi_log.csv", "a") as f:
                f.write(f"{datetime.now()},{user_id},{confidence:.2f}\n")
        else:
            st.error("❌ Wajah tidak dikenali.")
    else:
        st.warning("Tidak ada wajah terdeteksi.")

st.subheader("📊 Riwayat Absensi")
try:
    df = pd.read_csv("logs/absensi_log.csv", names=["Waktu", "User", "Confidence"])
    st.dataframe(df.tail(10))
except FileNotFoundError:
    st.info("Belum ada data absensi.")
