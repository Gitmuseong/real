import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

def main():
    st.title("📊 경도 예측 앱 (Hardness Predictor)")
    st.write("천연 광물 및 인공 결정의 물성 정보를 바탕으로 경도를 예측합니다.")

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("입력 데이터 미리보기")
        st.write(df)

        model = load_model()
        prediction = model.predict(df)

        df["예측 경도"] = prediction
        st.subheader("예측 결과")
        st.write(df)

        st.download_button("📥 결과 다운로드", data=df.to_csv(index=False).encode("utf-8"), file_name="predicted_result.csv")

if __name__ == "__main__":
    main()
