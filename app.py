import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Disease Predictor 🤖",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# LOAD MODEL & ENCODERS
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load("weighted_knn_disease_model.pkl")
    symptom_encoder = joblib.load("symptom_encoder.pkl")
    disease_encoder = joblib.load("disease_label_encoder.pkl")
    return model, symptom_encoder, disease_encoder

model, symptom_encoder, disease_encoder = load_model()

# ==========================
# PAGE TITLE
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
# USER INPUT
# ==========================
all_symptoms = list(symptom_encoder.classes_)
st.markdown("### 🩺 Select or type your symptoms:")
selected_symptoms = st.multiselect(
    "Start typing to search symptoms...",
    options=all_symptoms,
    help="Choose one or more symptoms from the list."
)

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_diseases(symptoms):
    if not symptoms:
        return []

    input_vector = symptom_encoder.transform([symptoms])
    distances, indices = model.kneighbors(input_vector, n_neighbors=10, return_distance=True)

    diseases = np.array(model._y)[indices[0]]
    distances = distances[0]

    # Weight by inverse distance
    weights = 1 / (distances + 1e-5)
    disease_scores = {}

    for dis, w in zip(diseases, weights):
        dis_name = disease_encoder.inverse_transform([dis])[0]
        disease_scores[dis_name] = disease_scores.get(dis_name, 0) + w

    # Top 5 unique diseases
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return sorted_diseases

# ==========================
# PREDICT BUTTON
# ==========================
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        with st.spinner("Analyzing symptoms..."):
            results = predict_diseases(selected_symptoms)

        if results:
            st.success("✅ Prediction Complete! Here are the top 5 possible diseases:")

            # Display results with visible text
            for i, (disease, score) in enumerate(results, start=1):
                st.markdown(f"""
                    <div style='background-color:#f0f9f9;padding:10px;border-radius:10px;margin:8px 0;'>
                        <b style='color:black;'>{i}. {disease}</b> — <span style='color:gray;'>Score: {score:.2f}</span>
                    </div>
                """, unsafe_allow_html=True)

            # Horizontal bar chart
            df_results = pd.DataFrame(results, columns=['Disease', 'Score'])
            df_results = df_results[::-1]  # reverse for better display
            fig, ax = plt.subplots(figsize=(7,4))
            ax.barh(df_results['Disease'], df_results['Score'], color='#3EB489')
            ax.set_xlabel("Weighted Score")
            ax.set_title("Top 5 Predicted Diseases")
            st.pyplot(fig)

        else:
            st.error("❌ No matching diseases found. Try adding more symptoms.")
