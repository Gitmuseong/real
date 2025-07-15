import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 불러오기
df = pd.read_csv("Mineral_Dataset_Supplementary_Info.csv")

# 2. 사용할 특성과 타겟 설정
features = [
    "density_Average", "atomicweight_Average", "val_e_Average",
    "ionenergy_Average", "el_neg_chi_Average"
]
target = "Hardness"

# 3. 결측치 제거
df = df[features + [target]].dropna()

# 4. 입력(X), 타겟(y) 설정
X = df[features]
y = df[target]

# 5. 학습/테스트 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 성능 평가
y_pred = model.predict(X_test)
print("🔍 RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("📈 R² score:", r2_score(y_test, y_pred))

# 8. 모델 저장
joblib.dump(model, "model.pkl")
print("✅ model.pkl 파일 저장 완료!")
