# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from streamlit_option_menu import option_menu
from PIL import Image
import os

# ---------------- Load Model ----------------
MODEL_PATH = "densenet_signature.keras"
model = load_model(MODEL_PATH)

# ---------------- App Configuration ----------------
st.set_page_config(page_title="Signature Verification", layout="wide")

# ---------------- Sidebar Navigation ----------------
with st.sidebar:
    selected = option_menu(
        menu_title="DashBoard",
        options=["Home", "Prediction", "Performance", "About"],
        icons=["house", "cloud-upload", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# ---------------- Home Page ----------------
if selected == "Home":
    st.title("✒️ Signature Verification System")
    st.subheader("Verify if a signature is Real or Forged using AI")

    st.markdown("""
    **About the Project:**  
    This system uses a **DenseNet169 CNN model** to classify uploaded signature images into Real or Forged.  
    - Achieved **~91.77% accuracy** on test set  
    - Provides **confidence score** for prediction  
    - Real-time prediction from uploaded images  
    """)

    # Sample Signatures
    sample_images = [
        "ui_images/image1.jpg",
        "ui_images/image2.png",
    ]

    cols = st.columns(len(sample_images))
    for i, img_path in enumerate(sample_images):
        if os.path.exists(img_path):
            cols[i].image(img_path, width=300)

# ---------------- Prediction Page ----------------
elif selected == "Prediction":
    st.title("📤 Upload Signature for Verification")
    st.markdown("Upload an image of a signature to predict if it's **Real** or **Forged**.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Signature", width=300)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        class_labels = ['Forged', 'Real']
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Class: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

# ---------------- Performance Page ----------------
elif selected == "Performance":
    st.title("📈 Model Training Performance")
    st.markdown("Visualize the model's **training** and **validation** accuracy and loss over epochs.")

    import matplotlib.pyplot as plt
    import pickle

    try:
        # Load training history
        with open("training_history.pkl", "rb") as f:
            history = pickle.load(f)

        col1, col2 = st.columns(2)

        # Accuracy Graph
        with col1:
            st.subheader("Accuracy over Epochs")
            fig_acc, ax_acc = plt.subplots(figsize=(5,3))
            ax_acc.plot(history['accuracy'], label='Training Accuracy', marker='o')
            ax_acc.plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
            ax_acc.set_xlabel('Epochs')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.legend(fontsize=8)
            ax_acc.grid(True)
            st.pyplot(fig_acc)

        # Loss Graph
        with col2:
            st.subheader("Loss over Epochs")
            fig_loss, ax_loss = plt.subplots(figsize=(5,3))
            ax_loss.plot(history['loss'], label='Training Loss', marker='o')
            ax_loss.plot(history['val_loss'], label='Validation Loss', marker='o')
            ax_loss.set_xlabel('Epochs')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend(fontsize=8)
            ax_loss.grid(True)
            st.pyplot(fig_loss)

    except FileNotFoundError:
        st.error("⚠️ training_history.pkl not found. Please generate and save it after training your model.")

# ---------------- About Page ----------------
elif selected == "About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    **Project Name:** Signature Verification Using DenseNet169  
    **Developer:** Naga Sunil  
    **Tools:** Python, TensorFlow, Keras, Streamlit  
    **Dataset:** Signatures Dataset (Real vs Forged)  

    **Description:**  
    A web-based system to predict the authenticity of signature images.  
    Provides confidence scores and visual performance metrics.
    """)
