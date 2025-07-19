import streamlit as st
import torch
import json
import requests
import os
from torchvision import models, transforms
from PIL import Image
from urllib.request import urlretrieve

# --- ATUR PATH MODEL DAN LABEL (gunakan direktori yang dapat ditulis di Hugging Face Spaces) ---
BASE_DIR = "/tmp/streamlit_app"

# Pastikan STREAMLIT_HOME berada di direktori yang dapat ditulis
os.environ["STREAMLIT_HOME"] = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "models")
LABELS_DIR = os.path.join(BASE_DIR, "labels")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

MODEL_FILENAME = os.getenv("MODEL_FILENAME","mobilenetv2.pth")
LABELS_FILENAME = os.getenv("LABELS_FILENAME", "labels.json")

model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
labels_path = os.path.join(LABELS_DIR, LABELS_FILENAME)

MODEL_URL = os.getenv("MODEL_URL","https://download.pytorch.org/models/mobilenet_v2-b0353104.pth")
LABELS_URL = os.getenv("LABELS_URL", "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json")

# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Klasifikasi Gambar (PyTorch) 📸",
    page_icon="🖼️",
    layout="centered"
)

# --- FUNGSI-FUNGSI ---
@st.cache_resource
def load_model():
    """Memuat model MobileNetV2 dari file lokal atau mengunduh jika belum ada."""
    if not os.path.exists(model_path):
        st.info("Mengunduh model MobileNetV2...")
        try:
            urlretrieve(MODEL_URL, model_path)
            st.success("Model berhasil diunduh.")
        except Exception as e:
            st.error(f"Gagal mengunduh model: {str(e)}")
            return None

    try:
        # Buat model tanpa weight
        model = models.mobilenet_v2(weights=None)
        # Muat state_dict dari file lokal
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None



@st.cache_data
def load_labels():
    """Memuat label dari file lokal atau mengunduh jika belum ada."""
    if not os.path.exists(labels_path):
        st.info("Mengunduh label ImageNet...")
        try:
            response = requests.get(LABELS_URL)
            response.raise_for_status()
            with open(labels_path, 'w') as f:
                json.dump(response.json(), f)
            st.success("Label berhasil diunduh.")
        except Exception as e:
            st.error(f"Gagal mengunduh label: {str(e)}")
            return None

    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        return labels
    except Exception as e:
        st.error(f"Gagal memuat label: {str(e)}")
        return None



def preprocess_image(image):
    """Melakukan pra-pemrosesan gambar agar sesuai dengan input model PyTorch."""
    try:
        # Definisikan transformasi
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Terapkan transformasi dan tambahkan dimensi batch
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)
        return batch_t
    except Exception as e:
        st.error(f"Gagal memproses gambar: {str(e)}")
        return None

def predict(image, model, labels):
    """Melakukan prediksi klasifikasi pada gambar."""
    try:
        st.info("🧠 Model sedang menganalisis gambar...")

        # Pra-pemrosesan gambar
        batch_t = preprocess_image(image)
        if batch_t is None:
            return None

        # Lakukan prediksi tanpa menghitung gradien
        with torch.no_grad():
            output = model(batch_t)

        # Dapatkan probabilitas dengan softmax
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Dapatkan 3 kelas dengan probabilitas tertinggi
        top3_prob, top3_catid = torch.topk(probabilities, 3)

        # Siapkan hasil
        results = []
        for i in range(top3_prob.size(0)):
            class_name = labels[top3_catid[i]]
            probability = top3_prob[i].item()
            results.append((class_name, probability))

        return results
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {str(e)}")
        return None

# --- TAMPILAN UTAMA APLIKASI ---

st.title("🖼️ Aplikasi Klasifikasi Gambar (PyTorch)")
st.write(
    "Unggah sebuah gambar, dan AI akan mencoba menebak objek apa yang ada di dalamnya! "
    "Aplikasi ini menggunakan model **MobileNetV2** dari PyTorch."
)

# Muat model dan label
try:
    model = load_model()
    labels = load_labels()

    if model is None or labels is None:
        st.error("Aplikasi tidak dapat dijalankan karena gagal memuat model atau label.")
        st.stop()
except Exception as e:
    st.error(f"Kesalahan saat inisialisasi aplikasi: {str(e)}")
    st.stop()

# Komponen untuk unggah file
uploaded_file = st.file_uploader(
    "Pilih sebuah gambar...",
    type=["jpg", "jpeg", "png"],
    help="Format file yang didukung: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    try:
        # Buka dan tampilkan gambar yang diunggah
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Anda Unggah', use_column_width=True)

        # Tombol untuk memulai klasifikasi
        if st.button('✨ Klasifikasikan Gambar Ini!'):
            with st.spinner('Tunggu sebentar...'):
                # Lakukan prediksi
                predictions = predict(image, model, labels)

                if predictions is not None:
                    st.subheader("✅ Hasil Prediksi Teratas:")
                    for i, (label, score) in enumerate(predictions):
                        st.write(f"{i+1}. **{label.replace('_', ' ').title()}** - Keyakinan: {score:.2%}")
                else:
                    st.error("Prediksi gagal. Silakan coba lagi atau unggah gambar lain.")
    except Exception as e:
        st.error(f"Kesalahan saat memproses gambar yang diunggah: {str(e)}")
        # Tambahan debugging untuk membantu identifikasi
        st.write("Detail error: Periksa koneksi internet atau format gambar.")

st.divider()
st.markdown(
    "Dibuat dengan ❤️ menggunakan [Streamlit](https://streamlit.io), [PyTorch](https://pytorch.org/) & [Hugging Face Spaces](https://huggingface.co/spaces)."
)
