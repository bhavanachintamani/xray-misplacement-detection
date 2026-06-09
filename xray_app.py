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

st.set_page_config(
    page_title="X-Ray Misplacement Detector",
    page_icon="🩻",
    layout="wide"
)

st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:700;color:#185FA5;}
.subtitle{font-size:1rem;color:#5F5E5A;margin-bottom:2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🩻 X-Ray Misplacement Detector</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect misplaced or anomalous chest X-rays using PCA + UMAP + Isolation Forest | Built for Canadian healthcare sector</p>',
            unsafe_allow_html=True)

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

has_misplaced = 'is_misplaced' in results_df.columns
has_cluster   = 'kmeans_cluster' in results_df.columns
has_umap      = 'umap_x' in results_df.columns and 'umap_y' in results_df.columns
has_label     = 'true_label' in results_df.columns
has_risk      = 'risk_score' in results_df.columns

total     = len(results_df)
high_risk = int(results_df['is_misplaced'].sum()) if has_misplaced else 0
n_clust   = int(results_df['kmeans_cluster'].nunique()) if has_cluster else 0
rate      = f"{results_df['is_misplaced'].mean()*100:.1f}%" if has_misplaced else "0%"

tab1, tab2, tab3 = st.tabs(["Upload X-Ray", "Dataset Overview", "About"])

with tab1:
    st.subheader("Upload a Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray (JPEG/PNG)",
        type=['jpg','jpeg','png']
    )

    if uploaded_file is not None:
        file_bytes  = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img         = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        # Get correct image size from scaler
        n_features  = scaler.n_features_in_
        img_size    = int(n_features ** 0.5)
        img_resized = cv2.resize(img, (img_size, img_size))

        col1, col2  = st.columns(2)
        with col1:
            st.image(img_resized, caption="Uploaded X-Ray",
                     use_column_width=True, clamp=True)

        img_flat   = (img_resized / 255.0).flatten().reshape(1, -1)
        img_scaled = scaler.transform(img_flat)
        img_pca    = pca.transform(img_scaled)

        anomaly_score = iso_forest.decision_function(img_pca)[0]
        anomaly_pred  = iso_forest.predict(img_pca)[0]
        cluster       = kmeans.predict(img_pca)[0]

        is_anomaly = anomaly_pred == -1
        risk_level = "HIGH RISK ⚠️" if is_anomaly else "LOW RISK ✅"
        color      = "#A32D2D" if is_anomaly else "#27500A"
        bg         = "#FCEBEB" if is_anomaly else "#EAF3DE"

        with col2:
            st.markdown(f"""
            <div style="background:{bg};border-radius:10px;padding:16px;text-align:center;margin-bottom:16px">
                <h2 style="color:{color};margin:0">{risk_level}</h2>
                <p style="color:{color};margin:4px 0 0 0">Anomaly Score: {anomaly_score:.4f}</p>
            </div>""", unsafe_allow_html=True)

            st.metric("Risk Level",    risk_level)
            st.metric("Cluster",       f"Cluster {cluster}")
            st.metric("Anomaly Score", f"{anomaly_score:.4f}")
            st.metric("Image Size",    f"{img_size}x{img_size}")

            if is_anomaly:
                st.error("⚠️ MISPLACEMENT DETECTED — Recommend radiologist review.")
            else:
                st.success("✅ NORMAL PLACEMENT — X-ray appears correctly placed.")

with tab2:
    st.subheader("Dataset Analysis Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images",   total)
    col2.metric("High Risk",      high_risk)
    col3.metric("Clusters Found", n_clust)
    col4.metric("Anomaly Rate",   rate)

    st.markdown("---")

    if has_label and has_umap:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Label Distribution")
            label_counts = results_df['true_label'].value_counts()
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(label_counts.index, label_counts.values,
                   color=['#378ADD','#D85A30'], width=0.4)
            ax.set_title('Normal vs Pneumonia', fontweight='bold')
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("UMAP Cluster Visualization")
            fig, ax = plt.subplots(figsize=(6,4))
            colors = ['#378ADD' if l=='NORMAL' else '#D85A30'
                      for l in results_df['true_label']]
            ax.scatter(results_df['umap_x'], results_df['umap_y'],
                       c=colors, alpha=0.5, s=10)
            ax.set_title('UMAP Projection', fontweight='bold')
            ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor='#378ADD', label='NORMAL'),
                Patch(facecolor='#D85A30', label='PNEUMONIA')
            ])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    if has_risk:
        st.subheader("Risk Score Distribution")
        risk_counts = results_df['risk_score'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(risk_counts.index.astype(str), risk_counts.values,
               color=['#27500A','#D4A017','#A32D2D'], width=0.4)
        ax.set_title('Risk Score Distribution', fontweight='bold')
        ax.set_xlabel('Risk Score (0=Safe, 3=High Risk)')
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Results Data")
    st.dataframe(results_df.head(20), use_container_width=True)

with tab3:
    st.subheader("About This Project")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Problem:** Misplaced X-rays in hospital systems cause diagnostic
        errors and delays. Manual review of thousands of images is not feasible.

        **Solution:** Automated misplacement detection using unsupervised ML —
        no labeled training data required.

        **Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle — 5,863 images
        """)
    with col2:
        st.markdown("""
        **Pipeline:**
        1. Load and preprocess X-ray images (OpenCV)
        2. Normalize with StandardScaler
        3. Reduce dimensions with PCA (50 components)
        4. Visualize with UMAP (2D projection)
        5. Cluster with K-Means (K=3)
        6. Detect anomalies with Isolation Forest
        7. Flag high-risk images for review

        **Tech Stack:** Python · OpenCV · Scikit-learn · UMAP · Streamlit
        """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Images Processed", str(total))
    col2.metric("Anomalies Detected", str(high_risk))
    col3.metric("PCA Components", "50")

st.markdown("---")
st.caption("Built by Bhavana Aswin | B.Tech CS (AI) | PCA + UMAP + Isolation Forest + Streamlit")
