import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load dataset
df = pd.read_csv("AmesHousing.csv")

# 🔥 FIX NAMA KOLOM
df.columns = df.columns.str.replace(" ", "")

# ✅ FITUR SUDAH SESUAI
features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath']

X = df[features]
y = df['SalePrice']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)

# Save
joblib.dump(rf, "rf_model.pkl")
joblib.dump(gb, "gb_model.pkl")

print("Model berhasil disimpan!")