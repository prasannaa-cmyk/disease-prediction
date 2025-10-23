import streamlit as st
import joblib
import numpy as np

# ==========================
# 🎨 PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="Disease Predictor 🤖",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# 🧩 LOAD MODEL AND ENCODER
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load("weighted_knn_disease_model.pkl")
    encoder = joblib.load("symptom_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# ==========================
# 💡 PAGE TITLE
# ==========================
st.markdown("""
    <h1 style='text-align: center; color: #3EB489;'>
        🧠 Intelligent Disease Prediction System
    </h1>
    <h4 style='text-align: center; color: gray;'>
        Enter your symptoms below to get the top 5 possible diseases.
    </h4>
    <hr style='border: 1px solid #3EB489;'>
""", unsafe_allow_html=True)

# ==========================
# 🔍 USER INPUT SECTION
# ==========================
all_symptoms = list(encoder.classes_)

with st.container():
    st.markdown("### 🩺 Select or type your symptoms:")
    selected_symptoms = st.multiselect(
        "Start typing to search symptoms...",
        options=all_symptoms,
        help="Choose one or more symptoms from the list."
    )

# ==========================
# 🧮 PREDICTION LOGIC
# ==========================
def predict_diseases(symptoms):
    if not symptoms:
        return []

    input_vector = encoder.transform([symptoms])
    neighbors = model.kneighbors(input_vector, n_neighbors=5, return_distance=True)
    distances, indices = neighbors

    diseases = np.array(model._y)[indices[0]]
    distances = distances[0]

    # Compute weights based on inverse distances
    weights = 1 / (distances + 1e-5)
    disease_scores = {}

    for dis, w in zip(diseases, weights):
        dis = str(dis)  # Convert to string for safety
        disease_scores[dis] = disease_scores.get(dis, 0) + w

    # Sort by score (top 5)
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return sorted_diseases

# ==========================
# 🔘 PREDICT BUTTON
# ==========================
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        with st.spinner("Analyzing symptoms..."):
            results = predict_diseases(selected_symptoms)
        
        if results:
            st.success("✅ Prediction Complete! Here are the top 5 possible diseases:")
            for i, (disease, score) in enumerate(results, start=1):
                st.markdown(f"""
                    <div style='background-color:#f0f9f9;padding:10px;border-radius:10px;margin:8px 0;'>
                        <b>{i}. {disease.title()}</b> — <span style='color:gray;'>Score: {score:.2f}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ No matching diseases found. Try adding more symptoms.")
