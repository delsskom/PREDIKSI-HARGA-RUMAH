import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor  # 🔥 TAMBAHAN

# =========================
# LOAD DATASET
# =========================
try:
    df = pd.read_csv("AmesHousing_clean.csv")
except FileNotFoundError:
    print("❌ File AmesHousing.csv tidak ditemukan!")
    exit()

# =========================
# BERSIHKAN NAMA KOLOM
# =========================
df.columns = df.columns.str.replace(" ", "")

print("✅ Kolom dataset:")
print(df.columns.tolist())

# =========================
# PILIH FITUR
# =========================
features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath']
target = 'SalePrice'

# Validasi fitur
missing = [col for col in features if col not in df.columns]
if missing:
    print(f"❌ Kolom tidak ditemukan: {missing}")
    exit()

# =========================
# AMBIL DATA
# =========================
X = df[features]
y = df[target]

# 🔥 HANDLE MISSING VALUE
X = X.fillna(X.mean())

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
print("🚀 Training Random Forest...")
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

print("🚀 Training Gradient Boosting...")
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)

print("🚀 Training XGBoost...")  # 🔥 TAMBAHAN
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================
def evaluate(model, name):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    print(f"\n📊 {name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

evaluate(rf, "Random Forest")
evaluate(gb, "Gradient Boosting")
evaluate(xgb, "XGBoost")  # 🔥 TAMBAHAN

# =========================
# SIMPAN MODEL
# =========================
joblib.dump(rf, "rf_model.pkl")
joblib.dump(gb, "gb_model.pkl")
joblib.dump(xgb, "xgb_model.pkl")  # 🔥 TAMBAHAN

print("\n✅ Model berhasil disimpan sebagai:")
print("- rf_model.pkl")
print("- gb_model.pkl")
print("- xgb_model.pkl")