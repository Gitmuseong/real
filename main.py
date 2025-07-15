import streamlit as st
import pandas as pd
import joblib  # 모델 로딩용
import numpy as np

# 모델 로드
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# 데이터 전처리 함수 (필요시 정의)
def preprocess_input(df):
    # 예: 불필요한 열 제거, 인코딩, 스케일링 등
    return df

# 예측 함수
def predict_hardness(model, input_df):
    processed = preprocess_input(input_df)
    predictions = model.predict(processed)
    return predictions

# Streamlit 메인 함수
def main():
    st.title("📊 경도 예측 앱 (Hardness Predictor)")
    st.write("천연 광물 및 인공 결정의 물성 정보를 바탕으로 경도를 예측합니다.")

    # 사용자 파일 업로드
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("입력 데이터 미리보기")
        st.write(input_df)

        model = load_model()
        predictions = predict_hardness(model, input_df)

        st.subheader("🔍 예측 결과")
        input_df["예측 경도"] = predictions
        st.write(input_df)

        # 다운로드 기능
        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("예측 결과 다운로드", data=csv, file_name="predicted_hardness.csv", mime="text/csv")

if __name__ == "__main__":
    main()
