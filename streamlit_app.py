import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Crystal Hardness Predictor", page_icon="🔮")
st.title("🔮 Crystal & Mineral Hardness Predictor")
st.markdown("Enter material properties below to estimate hardness (in GPa):")

# 예시 특성들 (train.py에서 어떤 컬럼들이 쓰였는지 맞춰야 함)
feature_names = [
    "Density", "Band_gap", "Formation_energy", "Volume",
    "Atomic_mass", "Electronegativity", "Thermal_conductivity"
]

input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

if st.button("🔍 Predict Hardness"):
    try:
        model = joblib.load("model/model.pkl")
        X = pd.DataFrame([input_data], columns=feature_names)
        # 필요시 더미 처리 등 추가
        prediction = model.predict(X)[0]
        st.success(f"💎 Predicted Hardness: {prediction:.2f} GPa")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
