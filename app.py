import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os

# =============================
# 1. KONFIGURASI DASAR
# =============================

MODEL_WEIGHTS_PATH = 'ViT_model.pth'
CLASSES_FILE_PATH = 'classes.txt'
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit config harus paling atas
st.set_page_config(
    page_title="ViT Face Recognition Demo",
    layout="wide",
    initial_sidebar_state="auto"
)

# =============================
# 2. LOAD KELAS
# =============================

def load_classes(path):
    try:
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.error(f"Gagal memuat kelas: File '{path}' tidak ditemukan.")
        return ["Unknown 0", "Unknown 1"]

CLASSES = load_classes(CLASSES_FILE_PATH)
NUM_CLASSES = len(CLASSES)

# =============================
# 3. TRANSFORMASI GAMBAR
# =============================

data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =============================
# 4. LOAD MODEL
# =============================

@st.cache_resource
def load_model():
    try:
        model_name = 'vit_tiny_patch16_224'
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)

        if os.path.exists(MODEL_WEIGHTS_PATH):
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
            loaded = True
        else:
            # fallback
            model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
            loaded = False

        model.to(DEVICE)
        model.eval()
        return model, loaded

    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        return None, False

# =============================
# 5. PREDIKSI
# =============================

def predict_image(model, image):
    try:
        tensor_img = data_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor_img)

        probs = nn.functional.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)

        return CLASSES[idx.item()], confidence.item()

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        return None, 0.0

# =============================
# 6. STREAMLIT APP
# =============================

def main():

    st.title("👨‍🔬 ViT-Tiny Face Recognition Demo")
    st.markdown(
        f"Aplikasi demo pengenalan wajah menggunakan **Vision Transformer (ViT-Tiny)** "
        f"dengan total **{NUM_CLASSES} kelas**."
    )

    # Load model
    model, loaded = load_model()

    if model is None:
        st.stop()

    if loaded:
        st.toast("✅ Bobot model berhasil dimuat.", icon="🎉")
    else:
        st.warning(f"⚠️ Bobot '{MODEL_WEIGHTS_PATH}' tidak ditemukan. Menggunakan pretrained ImageNet.")

    uploaded_file = st.file_uploader(
        "Unggah gambar wajah (JPG/JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar wajah", use_column_width=True)

            if st.button("Tampilkan Prediksi Identitas"):
                with st.spinner("Memproses model..."):
                    name, conf = predict_image(model, image)

                st.session_state["prediction_result"] = {
                    "name": name,
                    "confidence": conf
                }
        else:
            st.info("Silakan unggah gambar terlebih dahulu.")

    with col2:
        st.header("Hasil Analisis")

        if "prediction_result" in st.session_state:
            result = st.session_state["prediction_result"]
            conf_pct = result["confidence"] * 100

            # Tentukan warna metric (Streamlit valid: normal/inverse/off)
            if conf_pct >= 90:
                metric_color = "normal"
                status_icon = "✅"
            elif conf_pct >= 70:
                metric_color = "inverse"
                status_icon = "🟡"
            else:
                metric_color = "off"
                status_icon = "❌"

            st.subheader(f"{status_icon} Identitas Diprediksi")

            st.markdown(f"""
            <div style="background-color:#eef4ff; padding:20px; border-radius:10px;
                        border-left:5px solid #4f46e5;">
                <h3 style="margin:0; color:#1e40af;">{result['name']}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                label="Tingkat Keyakinan (Confidence)",
                value=f"{conf_pct:.2f}%",
                delta=None,
                delta_color=metric_color
            )

        else:
            st.info("Hasil prediksi akan muncul di sini.")

if __name__ == "__main__":
    main()
