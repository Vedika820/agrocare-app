import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

# Load trained model
@st.cache_resource(show_spinner=False)
def load_cnn_model():
    return load_model("crop_disease_model.h5", compile=False)

model = load_cnn_model()

# Load class label mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_label = {int(v): k for k, v in class_indices.items()}

# Streamlit UI
st.title("🌿 Smart AgroCare – Crop Doctor")
st.markdown("Upload a **plant leaf image** to detect the disease using AI")

uploaded_file = st.file_uploader("📤 Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Leaf Image", use_container_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = index_to_label[class_index]

    st.success(f"🩺 **Predicted Disease:** `{predicted_class}`")

    # ---- Advice section begins here (INSIDE the block) ----

    disease_advice = {
        "Tomato_Early_blight": "🧪 Treatment: Use fungicides like chlorothalonil. 🧼 Remove infected leaves. 💧 Avoid overhead watering.",
        "Tomato_Late_blight": "🧪 Treatment: Use copper-based fungicide. 🌬 Improve air circulation. 🔥 Destroy infected plants.",
        "Tomato_Leaf_Mold": "🧪 Treatment: Apply sulfur sprays. 🚿 Keep foliage dry. 💨 Ensure ventilation.",
        "Pepper__bell___Bacterial_spot": "🧪 Treatment: Apply copper sprays. 🌱 Use resistant seeds. ✂️ Prune infected leaves.",
        "Potato___Early_blight": "🧪 Treatment: Spray mancozeb every 7 days. ♻️ Rotate crops. 🌞 Maintain dry conditions.",
        "Potato___Late_blight": "🧪 Treatment: Use preventive fungicides. 🌬 Remove affected leaves. ❌ Avoid overcrowding.",
        "healthy": "✅ Your plant is healthy! 🚿 Keep watering regularly. 🛡️ Monitor weekly for changes."
    }

    seasonal_advice = {
        "Tomato": "📅 **Crop Tip:** Sow in June-July. Harvest in 60-70 days. Avoid over-irrigation.",
        "Pepper__bell": "📅 **Crop Tip:** Sow in warm months. Maintain high sunlight. Harvest in ~80 days.",
        "Potato": "📅 **Crop Tip:** Plant in winter. Harvest in 90-100 days. Avoid waterlogging."
    }

    advice = disease_advice.get(predicted_class, "ℹ️ No specific treatment advice available.")
    st.info(f"💡 **Advice:** {advice}")

    # Flexibly extract crop name
    if "___" in predicted_class:
     crop_name = predicted_class.split("___")[0]
    else:
     crop_name = predicted_class.split("_")[0]

    season_tip = seasonal_advice.get(crop_name, "")
    if season_tip:
     st.warning(season_tip)


# END OF SCRIPT
