import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# 데이터 로딩
mineral_df = pd.read_csv("../data/Mineral_Dataset_Supplementary_Info.csv")
artificial_df = pd.read_csv("../data/Artificial_Crystals_Dataset.csv")

# 공통 컬럼 맞추기 및 병합
common_cols = list(set(mineral_df.columns) & set(artificial_df.columns))
combined_df = pd.concat([mineral_df[common_cols], artificial_df[common_cols]], ignore_index=True)

# 결측값 제거
combined_df = combined_df.dropna()

# 타겟: Hardness
X = combined_df.drop("Hardness", axis=1)
y = combined_df["Hardness"]

# 범주형 처리 (있다면)
X = pd.get_dummies(X)

# 학습-테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 모델 저장
ios.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/model.pkl")

# 평가 출력
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
