import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Coffee Bean Classification Dashboard",
    page_icon="☕",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]

# ==============================
# MODEL DEFINITIONS
# ==============================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*14*14,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

@st.cache_resource
def load_models():
    # CNN Scratch
    cnn = CNN().to(DEVICE)
    cnn.load_state_dict(torch.load(r"C:\Users\IKBAR\uap_ML\src\model\cnn_scratch.pkl", map_location=DEVICE))
    cnn.eval()

    # ResNet18
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    resnet.load_state_dict(torch.load(r"C:\Users\IKBAR\uap_ML\src\model\resnet18.pkl", map_location=DEVICE))
    resnet = resnet.to(DEVICE)
    resnet.eval()

    # MobileNetV2
    mobilenet = models.mobilenet_v2(pretrained=False)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 4)
    mobilenet.load_state_dict(torch.load(r"C:\Users\IKBAR\uap_ML\src\model\mobilenetv2.pkl", map_location=DEVICE))
    mobilenet = mobilenet.to(DEVICE)
    mobilenet.eval()

    return cnn, resnet, mobilenet


cnn_model, resnet_model, mobilenet_model = load_models()

# ==============================
# IMAGE PREPROCESS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(model, image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("☕ Coffee Bean Classifier")
st.sidebar.markdown("""
Aplikasi klasifikasi biji kopi menggunakan:
- CNN Scratch  
- ResNet18 (Pretrained)  
- MobileNetV2 (Pretrained)
""")

st.sidebar.markdown("---")
st.sidebar.write("📌 **Dataset**: Coffee Bean Images")
st.sidebar.write("📌 **Classes**: Dark, Green, Light, Medium")

# ==============================
# MAIN UI
# ==============================
st.title("☕ Coffee Bean Classification Dashboard")
st.markdown("Upload gambar biji kopi untuk melihat hasil prediksi dari **3 model berbeda**.")

uploaded_file = st.file_uploader(
    "📤 Upload gambar kopi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(image, caption="Input Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Model sedang memprediksi..."):
            cnn_probs = predict(cnn_model, image)
            resnet_probs = predict(resnet_model, image)
            mobilenet_probs = predict(mobilenet_model, image)

        results = {
            "CNN Scratch": cnn_probs,
            "ResNet18": resnet_probs,
            "MobileNetV2": mobilenet_probs
        }

        st.subheader("📊 Hasil Prediksi")

        for model_name, probs in results.items():
            pred_class = CLASS_NAMES[np.argmax(probs)]
            confidence = np.max(probs) * 100

            st.markdown(f"### {model_name}")
            st.write(f"**Prediksi**: `{pred_class}`")
            st.write(f"**Confidence**: `{confidence:.2f}%`")

            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, probs)
            ax.set_ylim(0,1)
            ax.set_ylabel("Probability")
            ax.set_title(f"Probabilitas Kelas - {model_name}")
            st.pyplot(fig)

    # ==============================
    # COMPARISON TABLE
    # ==============================
    st.markdown("---")
    st.subheader("📈 Perbandingan Prediksi Model")

    comparison_data = {
        "Model": [],
        "Predicted Class": [],
        "Confidence (%)": []
    }

    for model_name, probs in results.items():
        comparison_data["Model"].append(model_name)
        comparison_data["Predicted Class"].append(CLASS_NAMES[np.argmax(probs)])
        comparison_data["Confidence (%)"].append(round(np.max(probs)*100, 2))

    st.table(comparison_data)

else:
    st.info("⬆️ Silakan upload gambar biji kopi untuk memulai klasifikasi.")