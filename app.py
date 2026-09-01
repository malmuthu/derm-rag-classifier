"""
Streamlit demo: upload a lesion image, get a classification, retrieved similar cases, and a grounded explanation.
Run: streamlit run app.py
"""

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import sys
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src" / "data"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
sys.path.append(str(PROJECT_ROOT / "src" / "retrieval"))
sys.path.append(str(PROJECT_ROOT / "src" / "explain"))
sys.path.append(str(PROJECT_ROOT / "src" / "utils"))

from dataset import eval_transform, CLASSES
from model_factory import build_model
from calibration import apply_temperature
from embeddings import load_embedding_model, get_embedding
from search import load_case_db, search_similar_cases
from templates import generate_explanation
from config import load_config

@st.cache_resource
def load_everything():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Classifier
    clf_config = load_config(PROJECT_ROOT / "configs" / "vit_b_16_dropout.yaml")
    clf = build_model(clf_config["backbone"]).to(device)
    clf.load_state_dict(torch.load(PROJECT_ROOT / clf_config["checkpoint_dir"] / "best_model.pt", map_location=device))
    clf.eval()

    temp_path = PROJECT_ROOT / clf_config["checkpoint_dir"] / "temperature.json"
    with open(temp_path) as f:
        temperature = json.load(f)["temperature"]

    # Embedding model + retrieval index
    retrieval_config = load_config(PROJECT_ROOT / "configs" / "retrieval.yaml")
    embed_model = load_embedding_model(PROJECT_ROOT / retrieval_config["embedding_checkpoint"], device)
    index, metadata = load_case_db()

    return {
        "device": device,
        "clf": clf,
        "temperature": temperature,
        "embed_model": embed_model,
        "index": index,
        "metadata": metadata,
    }

def main():
    st.title("Dermatology Lesion Classifier + Retrieval")
    st.caption(
        "Portfolio project - not a validated diagnostic tool. "
        "Trained on HAM10000 dermoscopy images; results on other image types are not meaningful."
    )
    st.write(
        "Upload a dermoscopic lesion image to get a classification, "
        "similar historical cases, and a grounded explanation."
    )

    with st.spinner("Loading models..."):
        resources = load_everything()

    uploaded_file = st.file_uploader("Upload a lesion image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error(
                "Couldn't read this file as an image. Please upload a valid JPG or PNG file."
            )
            st.stop()

        if image.size[0] < 50 or image.size[1] < 50:
            st.warning(
                "This image is unusually small - results may be unreliable. "
                "This model was trained on standard dermoscopy images."
            )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            device = resources["device"]
            tensor = eval_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = resources["clf"](tensor)
                calibrated_logits = apply_temperature(logits, resources["temperature"])
                probs = F.softmax(calibrated_logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()
            predicted_class = CLASSES[pred_idx]

        with st.spinner("Retrieving similar cases..."):
            embedding = get_embedding(resources["embed_model"], image, device)
            retrieved_cases = search_similar_cases(embedding, resources["index"], resources["metadata"], k=5)

        explanation = generate_explanation(predicted_class, confidence, retrieved_cases)

        with col2:
            st.subheader(f"Prediction: {predicted_class}")
            st.markdown(explanation)

        st.subheader("Retrieved similar cases")
        case_cols = st.columns(5)
        for i, case in enumerate(retrieved_cases):
            with case_cols[i]:
                st.write(f"**{case['diagnosis']}**")
                st.write(f"sim: {case['similarity']:.2f}")
                st.write(f"{case['dx_type']}")

if __name__ == "__main__":
    main()