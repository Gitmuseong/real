import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 데이터 불러오기
df = pd.read_csv("Mineral_Dataset.csv")

# 사용할 특성 정의
features = [
    "density_Average", "atomicweight_Average", "val_e_Average",
    "ionenergy_Average", "el_neg_chi_Average"
]
target = "Hardness"

# 결측치 제거
df = df[features + [target]].dropna()

# 입력(X), 타겟(y) 분리
X = df[features]
y = df[target]

# 학습용 / 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 모델 저장
joblib.dump(model, "model.pkl")
print("✅ model.pkl 저장 완료!")
