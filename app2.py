import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# -----------------------------
# STEP 1: Load, preprocess, and train model (fully cached)
# -----------------------------
@st.cache_resource
def prepare_model():
    # Load dataset once
    data = pd.read_csv("DiseaseAndSymptoms.csv")
    data = data.fillna('none')
    for col in data.columns:
        data[col] = data[col].str.lower()

    # Collect all unique symptoms efficiently
    symptoms = sorted(set(
        val for i in range(1, 18) for val in data[f'Symptom_{i}'].unique() if val != 'none'
    ))

    # One-hot encoding (optimized)
    symptom_data = pd.DataFrame(0, index=data.index, columns=symptoms)
    for i in range(1, 18):
        valid = data[f'Symptom_{i}'] != 'none'
        symptom_data.loc[valid, data.loc[valid, f'Symptom_{i}']] = 1

    # Combine and encode
    df = pd.concat([data['Disease'], symptom_data], axis=1)
    le = LabelEncoder()
    df['Disease'] = le.fit_transform(df['Disease'])

    X = df.drop('Disease', axis=1)
    y = df['Disease']

    # Train XGBoost model once
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)

    # Mapping for quick input conversion
    symptom_index = {s: i for i, s in enumerate(symptoms)}

    return model, symptoms, symptom_index, le

# Load all once (cached)
model, symptoms, symptom_index, le = prepare_model()

# -----------------------------
# STEP 2: Streamlit UI
# -----------------------------
st.title("🩺 Disease Prediction System (Fast & Intelligent)")
st.write("Select your symptoms to get the top predicted diseases instantly using AI.")

selected_symptoms = st.multiselect("🧩 Choose Symptoms:", symptoms)

col1, col2 = st.columns(2)

# -----------------------------
# STEP 3: Prediction
# -----------------------------
with col1:
    if st.button("🔍 Predict Disease"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            # Fast vectorized input creation
            input_data = np.zeros(len(symptom_index))
            for s in selected_symptoms:
                if s in symptom_index:
                    input_data[symptom_index[s]] = 1

            # Predict
            probs = model.predict_proba([input_data])[0]
            top_idx = np.argsort(probs)[-3:][::-1]
            top_diseases = le.inverse_transform(top_idx)

            st.success("🧠 Top Disease Predictions:")
            for i, idx in enumerate(top_idx, 1):
                st.write(f"**{i}. {top_diseases[i-1].title()}** — {probs[idx]*100:.2f}%")
                st.progress(int(probs[idx]*100))

with col2:
    if st.button("🔄 Reset Selection"):
        selected_symptoms = []
        st.experimental_rerun()

