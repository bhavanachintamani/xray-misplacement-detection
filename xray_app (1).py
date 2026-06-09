
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="X-Ray Misplacement Detector", page_icon="🩻", layout="wide")

st.title("🩻 X-Ray Misplacement Detector")
st.caption("Detect misplaced chest X-rays using PCA + Isolation Forest | Canadian Healthcare")

@st.cache_resource
def load_models():
    kmeans     = pickle.load(open('kmeans_model.pkl','rb'))
    iso_forest = pickle.load(open('isolation_forest.pkl','rb'))
    pca        = pickle.load(open('pca_model.pkl','rb'))
    scaler     = pickle.load(open('scaler.pkl','rb'))
    return kmeans, iso_forest, pca, scaler

@st.cache_data
def load_results():
    df = pd.read_csv('xray_results.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    return df

kmeans, iso_forest, pca, scaler = load_models()
results_df = load_results()
IMG_SIZE = 128

tab1, tab2 = st.tabs(["Upload X-Ray", "Dataset Overview"])

with tab1:
    st.subheader("Upload a Chest X-Ray Image")
    uploaded_file = st.file_uploader("Upload JPEG/PNG", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        file_bytes  = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img         = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_flat    = (img_resized / 255.0).flatten().reshape(1, -1)
        img_scaled  = scaler.transform(img_flat)
        img_pca     = pca.transform(img_scaled)
        score       = iso_forest.decision_function(img_pca)[0]
        pred        = iso_forest.predict(img_pca)[0]
        cluster     = kmeans.predict(img_pca)[0]
        is_bad      = pred == -1
        col1, col2  = st.columns(2)
        with col1:
            st.image(img_resized, caption="Uploaded X-Ray", clamp=True)
        with col2:
            if is_bad:
                st.error(f"HIGH RISK — Anomaly Score: {score:.4f}")
            else:
                st.success(f"LOW RISK — Anomaly Score: {score:.4f}")
            st.metric("Cluster", f"Cluster {cluster}")
            st.metric("Anomaly Score", f"{score:.4f}")

with tab2:
    st.subheader("Dataset Overview")
    st.dataframe(results_df.head(20), use_container_width=True)

st.caption("Built by Bhavana Aswin | B.Tech CS (AI) | PCA + UMAP + Isolation Forest")
