import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.model import get_model
from src.gradcam import get_gradcam, preprocess_for_gradcam


st.set_page_config(
    page_title="Authentik",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f7f8fa !important;
    color: #0a0a0a !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem; max-width: 800px; }

/* ── Hero ── */
.hero {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 16px;
    padding: 2rem 2rem 1.6rem;
    margin-bottom: 1.5rem;
}
.logo-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.logo-mark {
    width: 40px; height: 40px;
    background: #0a0a0a;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    font-size: 18px;
    font-weight: 700;
    flex-shrink: 0;
}
.logo-name {
    font-size: 1.9rem;
    font-weight: 700;
    color: #0a0a0a;
    letter-spacing: -1px;
    line-height: 1;
}
.logo-name span { color: #2563eb; }
.tagline {
    font-size: 13px;
    color: #888;
    margin-bottom: 1.2rem;
    margin-left: 52px;
}
.stats-row {
    display: flex;
    gap: 8px;
    margin-left: 52px;
    flex-wrap: wrap;
}
.stat-chip {
    background: #f1f5ff;
    color: #2563eb;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #dbe4ff;
}

/* ── Section card ── */
.section-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 12px;
    font-weight: 600;
    color: #999;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.badge-danger  { background: #fff0f0; color: #cc2200; border: 1px solid #ffd5cc; }
.badge-warning { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
.badge-success { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

/* ── Metric card ── */
.metric-card {
    background: #f7f8fa;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
}
.metric-label {
    font-size: 10px;
    font-weight: 600;
    color: #aaa;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0a0a0a;
    line-height: 1;
    letter-spacing: -1px;
}
.metric-track {
    margin-top: 10px;
    height: 5px;
    background: #e8eaed;
    border-radius: 3px;
    overflow: hidden;
}
.metric-fill-danger  { height: 100%; background: linear-gradient(90deg, #ff6b6b, #cc2200); border-radius: 3px; }
.metric-fill-warning { height: 100%; background: linear-gradient(90deg, #fcd34d, #b45309); border-radius: 3px; }
.metric-fill-success { height: 100%; background: linear-gradient(90deg, #4ade80, #166534); border-radius: 3px; }

/* ── Info rows ── */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 13px;
}
.info-row:last-child { border-bottom: none; }
.info-key   { color: #aaa; font-weight: 400; }
.info-value { color: #0a0a0a; font-weight: 500; }
.info-value.danger  { color: #cc2200; }
.info-value.success { color: #166534; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #e8eaed; margin: 1.5rem 0; }

/* ── Streamlit overrides ── */
p, .stMarkdown p { color: #666 !important; font-size: 0.88rem !important; line-height: 1.6 !important; }
h2, h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #0a0a0a !important; margin-top: 1.5rem !important; }

.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
.stFileUploader > div {
    background: #ffffff !important;
    border: 2px dashed #dbe4ff !important;
    border-radius: 12px !important;
}
img { border-radius: 10px !important; border: 1px solid #e8eaed !important; }
.stAlert { border-radius: 10px !important; font-size: 13px !important; }
.stSpinner > div { border-top-color: #2563eb !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 4px; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 11px;
    color: #ccc;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #f0f0f0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(
        torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE, weights_only=True)
    )
    model.eval()
    return model


# ── Hero Section ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="logo-row">
        <div class="logo-mark">A</div>
        <div class="logo-name">Authent<span>ik</span></div>
    </div>
    <div class="tagline">AI-powered identity document authenticity verifier</div>
    <div class="stats-row">
        <span class="stat-chip">EfficientNet-B0</span>
        <span class="stat-chip">90.1% Accuracy</span>
        <span class="stat-chip">96.4% ROC-AUC</span>
        <span class="stat-chip">Real-time analysis</span>
        <span class="stat-chip">v1.0.0</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Upload Section ────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">Verify document</div>', unsafe_allow_html=True)

doc_type = st.selectbox(
    "Document type",
    ["Passport", "Driver's License", "National ID", "Aadhar Card", "Other"]
)

uploaded_file = st.file_uploader(
    "Upload ID photo to analyze",
    type=["jpg", "jpeg", "png", "webp"]
)

st.markdown('</div>', unsafe_allow_html=True)


if uploaded_file is not None:

    pil_image    = Image.open(uploaded_file).convert("RGB")
    model        = load_model()
    image_tensor, original_image = preprocess_for_gradcam(pil_image)

    with torch.no_grad():
        output     = model(image_tensor).squeeze(1)
        prob       = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0

    fake_prob = prob
    real_prob = 1 - prob

    if fake_prob >= 0.85:
        risk      = "High risk"
        badge_cls = "badge-danger"
        fill_cls  = "metric-fill-danger"
        recommend = "Reject — flag for manual review"
        verdict   = "AI generated"
        val_cls   = "danger"
    elif fake_prob >= 0.60:
        risk      = "Medium risk"
        badge_cls = "badge-warning"
        fill_cls  = "metric-fill-warning"
        recommend = "Caution — additional verification needed"
        verdict   = "Suspicious"
        val_cls   = "danger"
    else:
        risk      = "Low risk"
        badge_cls = "badge-success"
        fill_cls  = "metric-fill-success"
        recommend = "Pass — image appears authentic"
        verdict   = "Authentic"
        val_cls   = "success"

    # ── Result ────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">Verification result</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(pil_image, caption="Uploaded photo", use_column_width=True)

    with col2:
        st.markdown(f'<div class="badge {badge_cls}">{risk}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Authenticity score</div>
            <div class="metric-value">{real_prob*100:.1f}%</div>
            <div class="metric-track">
                <div class="{fill_cls}" style="width:{real_prob*100:.1f}%"></div>
            </div>
        </div>
        <div class="metric-card">
            <div class="info-row">
                <span class="info-key">Verdict</span>
                <span class="info-value {val_cls}">{verdict}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Document type</span>
                <span class="info-value">{doc_type}</span>
            </div>
            <div class="info-row">
                <span class="info-key">AI probability</span>
                <span class="info-value {val_cls}">{fake_prob*100:.1f}%</span>
            </div>
            <div class="info-row">
                <span class="info-key">Recommendation</span>
                <span class="info-value">{recommend}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Grad-CAM ──────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">Neural attention map</div>', unsafe_allow_html=True)
    st.markdown("<p>Red regions highlight areas that most influenced the verdict.</p>", unsafe_allow_html=True)

    with st.spinner("Running analysis..."):
        heatmap = get_gradcam(model, image_tensor, original_image)

    col3, col4 = st.columns(2)
    with col3:
        st.image(original_image, caption="Original", use_column_width=True)
    with col4:
        st.image(heatmap, caption="Attention heatmap", use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary ───────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">Analysis summary</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.error(f"Threat detected — unnatural textures and patterns consistent with AI-generated imagery. This {doc_type.lower()} photo is likely fraudulent.")
    else:
        st.success(f"Verified — natural grain, lighting and texture patterns consistent with authentic photography. This {doc_type.lower()} appears genuine.")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Authentik — AI Identity Verification &nbsp;|&nbsp;
    EfficientNet-B0 &nbsp;|&nbsp;
    Accuracy: 90.1% &nbsp;|&nbsp;
    v1.0.0
</div>
""", unsafe_allow_html=True)