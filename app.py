import streamlit as st
import joblib
import numpy as np

# ==========================
# 🎨 PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="AI Disease Predictor 🤖",
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
    distances, indices = model.kneighbors(input_vector, n_neighbors=5)
    
    # Get top 5 nearest neighbors’ diseases
    neighbor_diseases = model.predict(input_vector.reshape(1, -1))
    probs = model.predict_proba(input_vector)
    
    # Estimate probabilities by summing inverse distances
    neighbors = model.kneighbors(input_vector, return_distance=True)
    distances = neighbors[0][0]
    indices = neighbors[1][0]
    
    diseases = np.array(model._y)[indices]
    weights = 1 / (distances + 1e-5)
    unique, probs = np.unique(diseases, return_counts=True)
    disease_scores = dict(zip(unique, probs * weights[:len(unique)]))
    
    # Sort top 5
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
