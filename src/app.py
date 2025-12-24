import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Coffee Bean Classification",
    page_icon="☕",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]

# =========================
# MODEL DEFINITIONS
# =========================
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
        return self.fc(self.conv(x))


@st.cache_resource
def load_models():
    cnn = CNN().to(DEVICE)
    cnn.load_state_dict(torch.load("C:\\Users\\IKBAR\\uap_ML\\src\\model\\cnn_scratch.pkl", map_location=DEVICE))
    cnn.eval()

    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    resnet.load_state_dict(torch.load("C:\\Users\\IKBAR\\uap_ML\\src\\model\\resnet18.pkl", map_location=DEVICE))
    resnet = resnet.to(DEVICE)
    resnet.eval()

    mobilenet = models.mobilenet_v2(pretrained=False)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 4)
    mobilenet.load_state_dict(torch.load("C:\\Users\\IKBAR\\uap_ML\\src\\model\\mobilenetv2.pkl", map_location=DEVICE))
    mobilenet = mobilenet.to(DEVICE)
    mobilenet.eval()

    return cnn, resnet, mobilenet


cnn_model, resnet_model, mobilenet_model = load_models()

# =========================
# PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(model, image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(image)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("☕ Navigation")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["📊 Training Overview", "🧪 Model Testing"]
)

st.sidebar.markdown("---")
st.sidebar.write("**Dataset**: Coffee Bean Images")
st.sidebar.write("**Classes**: Dark, Green, Light, Medium")

# =========================
# PAGE 1: TRAINING OVERVIEW
# =========================
if page == "📊 Training Overview":
    st.title("📊 Training Overview")

    st.markdown("""
    Halaman ini menampilkan ringkasan hasil pelatihan model deep learning
    untuk klasifikasi biji kopi serta perbandingan performa antar model.
    """)

    # =====================
    # LOAD METRICS
    # =====================
    with open(r"C:\Users\IKBAR\uap_ML\src\assets\training_metrics.json") as f:
        metrics = json.load(f)

    # =====================
    # METRIC CARDS
    # =====================
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "CNN Scratch Accuracy",
        f"{metrics['CNN Scratch']['final_acc']*100:.2f}%"
    )
    col2.metric(
        "ResNet18 Accuracy",
        f"{metrics['ResNet18']['final_acc']*100:.2f}%"
    )
    col3.metric(
        "MobileNetV2 Accuracy",
        f"{metrics['MobileNetV2']['final_acc']*100:.2f}%"
    )

    st.markdown("---")

    # =====================
    # CNN TRAINING CURVE
    # =====================
    st.subheader("📈 CNN Scratch Training Performance")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(metrics["CNN Scratch"]["train_acc"], label="Train Accuracy")
        ax.plot(metrics["CNN Scratch"]["val_acc"], label="Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title("Accuracy Curve")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(metrics["CNN Scratch"]["train_loss"], label="Train Loss")
        ax.plot(metrics["CNN Scratch"]["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Loss Curve")
        st.pyplot(fig)

    st.markdown("---")
    
    # =====================
    # ResNet18 TRAINING CURVE
    # =====================
    st.subheader("📈 ResNet18 Training Performance")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(metrics["ResNet18"]["train_acc"], label="Train Accuracy")
        ax.plot(metrics["ResNet18"]["val_acc"], label="Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title("Accuracy Curve")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(metrics["ResNet18"]["train_loss"], label="Train Loss")
        ax.plot(metrics["ResNet18"]["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Loss Curve")
        st.pyplot(fig)

    st.markdown("---")
    
    # =====================
    # MobileNetV2 TRAINING CURVE
    # =====================
    st.subheader("📈 MobileNetV2 Training Performance")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(metrics["MobileNetV2"]["train_acc"], label="Train Accuracy")
        ax.plot(metrics["MobileNetV2"]["val_acc"], label="Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title("Accuracy Curve")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(metrics["MobileNetV2"]["train_loss"], label="Train Loss")
        ax.plot(metrics["MobileNetV2"]["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Loss Curve")
        st.pyplot(fig)

    st.markdown("---")
    
    # =====================
    # MODEL COMPARISON
    # =====================
    st.subheader("📊 Model Accuracy Comparison")

    model_names = []
    accuracies = []

    for model_name, value in metrics.items():
        model_names.append(model_name)
        accuracies.append(value["final_acc"] * 100)

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(model_names, accuracies)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final Validation Accuracy per Model")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center"
        )

    st.pyplot(fig)

    st.info(
        "Grafik di atas menunjukkan bahwa model pretrained "
        "(ResNet18 dan MobileNetV2) memiliki performa lebih baik "
        "dibandingkan CNN yang dilatih dari awal."
    )

# =========================
# PAGE 2: MODEL TESTING
# =========================
if page == "🧪 Model Testing":
    st.title("🧪 Model Testing")

    selected_model = st.selectbox(
        "Pilih Model",
        ["CNN Scratch", "ResNet18", "MobileNetV2"]
    )

    uploaded_file = st.file_uploader(
        "Upload Gambar Biji Kopi",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", width=300)

        model_map = {
            "CNN Scratch": cnn_model,
            "ResNet18": resnet_model,
            "MobileNetV2": mobilenet_model
        }

        probs = predict(model_map[selected_model], image)

        pred_class = CLASS_NAMES[np.argmax(probs)]
        confidence = np.max(probs) * 100

        st.success(f"**Prediksi:** {pred_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, probs)
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    else:
        st.warning("Silakan upload gambar terlebih dahulu.")
