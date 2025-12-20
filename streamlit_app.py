# ================================
# FILE: streamlit_app.py
# FINAL VERSION – LOCKED
# Project: Mango Plant Disease Detection
# ================================

import streamlit as st
from pathlib import Path
from PIL import Image

from app.inference import load_models, predict_image
from app.config import DATA_ROOT, IMAGE_EXTENSIONS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mango Plant Disease Detection",
    page_icon="🥭",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def init_models():
    load_models()
    return True

init_models()

# ---------------- DISEASE → TREATMENT MAP ----------------
DISEASE_TREATMENT = {
    "Anthracnose": {
        "cause": "Fungal infection causing dark spots on leaves and fruits.",
        "treatment": "Spray Carbendazim 0.1% or Copper Oxychloride 0.3%",
        "prevention": "Avoid overhead irrigation and prune infected parts"
    },
    "Bacterial Canker": {
        "cause": "Bacterial disease causing lesions and cracking of tissues.",
        "treatment": "Spray Streptocycline (0.01%) + Copper fungicide",
        "prevention": "Use disease-free planting material"
    },
    "Powdery Mildew": {
        "cause": "White powdery fungal growth on leaves.",
        "treatment": "Spray Sulphur 0.2% or Hexaconazole",
        "prevention": "Maintain proper air circulation"
    },
    "Die Back": {
        "cause": "Fungal disease causing drying of branches.",
        "treatment": "Prune affected branches and spray Carbendazim",
        "prevention": "Apply Bordeaux paste on cut surfaces"
    },
    "Sooty Mould": {
        "cause": "Fungal growth due to honeydew from insects.",
        "treatment": "Control insects using Imidacloprid",
        "prevention": "Manage aphids and scale insects"
    },
    "Gall Midge": {
        "cause": "Insect pest damaging flowers and young shoots.",
        "treatment": "Spray Lambda-cyhalothrin or Thiamethoxam",
        "prevention": "Timely pest monitoring"
    },
    "Cutting Weevil": {
        "cause": "Beetle cutting tender shoots and buds.",
        "treatment": "Spray Chlorpyrifos 0.05%",
        "prevention": "Remove and destroy affected shoots"
    },
    "Healthy": {
        "cause": "No disease detected.",
        "treatment": "No treatment required",
        "prevention": "Maintain good orchard hygiene"
    }
}

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 Image Browser")

disease_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
disease_names = [d.name for d in disease_dirs]

selected_disease = st.sidebar.selectbox(
    "Select Disease Category",
    disease_names
)

image_dir = DATA_ROOT / selected_disease
images = sorted([i for i in image_dir.iterdir() if i.suffix.lower() in IMAGE_EXTENSIONS])

selected_image = st.sidebar.selectbox(
    "Select Image",
    [img.name for img in images]
)

selected_image_path = image_dir / selected_image

# Load image once (robust method)
img = Image.open(selected_image_path).convert("RGB")

# Sidebar image preview (small)
st.sidebar.image(
    img,
    caption="Selected Leaf",
    width=220
)

# ---------------- MAIN UI ----------------
st.title("🥭 Mango Plant Disease Detection")
st.caption("Single Test UI — Image → Predict → Action")
st.divider()

# Layout: Image | Treatment | Model Output
col2, col3 = st.columns([2.6, 1.3])

# ---------- LEFT: IMAGE PREVIEW ----------

# ---------- MIDDLE: DIAGNOSIS & TREATMENT ----------
# ---------- MIDDLE: DIAGNOSIS & TREATMENT (TEXT ONLY) ----------
with col2:
    st.subheader("🩺 Diagnosis & Treatment")

    run = st.button("🔍 RUN DIAGNOSIS")

    if run:
        with st.spinner("Analyzing image..."):
            result = predict_image(str(selected_image_path))

        if result.get("error"):
            st.error(result["error"])
        else:
            disease = result["predicted_label"]
            info = DISEASE_TREATMENT.get(disease, {})

            st.success("Diagnosis Complete")

            st.markdown("---")

            st.markdown(
                f"""
### 🦠 **Detected Disease**
- **{disease}**

### 📌 **Cause**
- 🧫 {info.get('cause', 'N/A')}

### 💊 **Recommended Treatment**
- 🧴 {info.get('treatment', 'N/A')}

### 🌱 **Prevention Tips**
- ✅ {info.get('prevention', 'N/A')}
""",
                unsafe_allow_html=False
            )


# ---------- RIGHT: MODEL OUTPUT ----------
with col3:
    st.subheader("🤖 Model Output")
    if 'result' in locals() and not result.get("error"):
        st.metric("Disease", result["predicted_label"])
        st.metric("Confidence", f"{result['confidence']*100:.2f}%")
        st.progress(result["confidence"])
    else:
        st.info("Run diagnosis to see model output")
