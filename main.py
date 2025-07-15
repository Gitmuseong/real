import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 타이틀 및 설명
st.set_page_config(page_title="Hardness Predictor", layout="centered")
st.title("🪨 Hardness Predictor")
st.markdown("""
이 앱은 인공 결정(Artificial Crystals)의 특성을 기반으로 **비커스 경도(Vickers Hardness)**를 예측합니다.  
왼쪽에 입력값을 넣으면 경도 예측 결과를 확인할 수 있어요!
""")

# 모델 불러오기
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

model = load_model()

# 입력받기
st.sidebar.header("📥 입력값을 설정하세요")
feature_names = ['Density', 'Formation_energy', 'Gap', 'Elasticity', 'Stability']

user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, value=1.0)

# 예측
if st.button("경도 예측하기"):
    input_array = np.array([list(user_input.values())])
    prediction = model.predict(input_array)
    st.success(f"💎 예측된 비커스 경도: **{prediction[0]:.2f} GPa**")
