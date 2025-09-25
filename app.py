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
from streamlit_option_menu import option_menu

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

    # Optional carousel of sample signatures
    sample_images = [
        r"C:\Users\nagas\Downloads\Signature_Verification_DenseNet\image1.jpg",
        r"C:\Users\nagas\Downloads\Signature_Verification_DenseNet\image2.png",
        "sample_signatures/forged1.png",
        "sample_signatures/forged2.png"
    ]

    # st.markdown("### Sample Signatures")
    cols = st.columns(len(sample_images))
    for i, img_path in enumerate(sample_images):
        if os.path.exists(img_path):
            cols[i].image(img_path, use_container_width=True)

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
# elif selected == "Performance":
#     st.title("📊 Model Performance Metrics")
#     st.markdown("Evaluate the model on test data (Confusion Matrix, ROC-AUC).")

#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import confusion_matrix, roc_curve, auc
#     from tensorflow.keras.preprocessing.image import ImageDataGenerator

#     TEST_DATA_PATH = "C:/Users/nagas/Downloads/Signatures_Dataset/Test"
#     test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#     test_generator = test_datagen.flow_from_directory(
#         TEST_DATA_PATH,
#         target_size=(224, 224),
#         batch_size=1,
#         class_mode='categorical',
#         classes=['forged', 'real'],
#         shuffle=False
#     )

#     # Predictions
#     y_true = test_generator.classes
#     y_pred_prob = model.predict(test_generator, verbose=1)
#     y_pred = np.argmax(y_pred_prob, axis=1)

#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)
#     st.subheader("Confusion Matrix")
#     fig, ax = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=['Forged', 'Real'], yticklabels=['Forged', 'Real'])
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     st.pyplot(fig)

#     # ROC Curve & AUC
#     st.subheader("ROC Curve & AUC")
#     fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
#     roc_auc = auc(fpr, tpr)
#     fig2, ax2 = plt.subplots()
#     ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     ax2.set_xlim([0.0, 1.0])
#     ax2.set_ylim([0.0, 1.05])
#     ax2.set_xlabel('False Positive Rate')
#     ax2.set_ylabel('True Positive Rate')
#     ax2.legend(loc="lower right")
#     st.pyplot(fig2)

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
