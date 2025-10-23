import streamlit as st
import numpy as np
import joblib

# ==============================
# 🎨 Page Configuration
# ==============================
st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🧬",
    layout="centered",
)

# Hide Streamlit default footer & menu
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==============================
# 💾 Load Model and Encoders
# ==============================
@st.cache_resource
def load_resources():
    model = joblib.load("weighted_knn_disease_model.pkl")
    mlb = joblib.load("symptom_encoder.pkl")
    le = joblib.load("disease_label_encoder.pkl")
    return model, mlb, le

model, mlb, le = load_resources()

# ==============================
# 🧠 Helper Function
# ==============================
def predict_disease(symptoms):
    """Return top 5 predicted diseases with confidence percentages."""
    user_input = mlb.transform([symptoms])
    distances, indices = model.kneighbors(user_input, n_neighbors=5)
    neighbor_labels = model._y[indices[0]]

    # Weighted scores based on inverse distance
    scores = {}
    for label, dist in zip(neighbor_labels, distances[0]):
        disease = le.inverse_transform([label])[0]
        scores[disease] = scores.get(disease, 0) + (1 / (dist + 1e-5))

    # Sort and convert to confidence percentages
    sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total_score = sum(score for _, score in sorted_diseases)
    percent_results = [(disease, (score / total_score) * 100) for disease, score in sorted_diseases]

    return percent_results[:5]

# ==============================
# 🧬 UI Design
# ==============================
st.markdown(
    """
    <div style='text-align:center;'>
        <h1 style='color:#0099cc;'>🧬 AI Disease Prediction</h1>
        <p style='color:gray; font-size:18px;'>
        Select your symptoms and discover the top possible diseases predicted by a Weighted KNN model.
        </p>
    </div>
    """, unsafe_allow_html=True
)

all_symptoms = sorted(list(mlb.classes_))
selected_symptoms = st.multiselect("Select your symptoms:", all_symptoms)

if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom to proceed.")
    else:
        results = predict_disease(selected_symptoms)

        st.markdown(
            """
            <div style='margin-top:30px;'>
                <h3 style='color:#006666;'>✅ Prediction Complete!</h3>
                <p style='color:gray;'>Here are the top 5 possible diseases based on your symptoms:</p>
            </div>
            """, unsafe_allow_html=True
        )

        for i, (disease, confidence) in enumerate(results, start=1):
            st.markdown(
                f"""
                <div style='background-color:#f0f9f9;
                            border-radius:10px;
                            padding:15px;
                            margin-bottom:10px;
                            box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color:#004d4d; margin:0;'>{i}. {disease}</h4>
                    <p style='color:#007777; margin:0;'>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True
            )

        st.balloons()

else:
    st.info("👆 Select symptoms and click **Predict Disease** to begin.")

# ==============================
# End of App
# ==============================
