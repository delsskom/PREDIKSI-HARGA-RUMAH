# Prediksi Harga Rumah

Aplikasi ini digunakan untuk memprediksi harga rumah menggunakan Machine Learning.

## Deskripsi
Program ini menerima beberapa input seperti luas rumah, kualitas bangunan, jumlah garasi, luas basement, dan jumlah kamar mandi, kemudian menghasilkan prediksi harga rumah.

## Input
- GrLivArea (>0) : Luas rumah
- OverallQual (1–10) : Kualitas rumah
- GarageCars (≥0) : Kapasitas garasi
- TotalBsmtSF (≥0) : Luas basement
- FullBath (≥0) : Jumlah kamar mandi

## Model
Model yang digunakan adalah Random Forest, Gradient Boost dan  XGBoost Regressor.

## Cara Menjalankan
1. Install library:
pip install -r requirements.txt

2. Jalankan program:
python app.py

3. Buka di browser:
http://127.0.0.1:7860

## Demo
Aplikasi dijalankan secara lokal pada:
http://127.0.0.1:7860

## Struktur File
- app.py
- gb_model.pkl
- rf_model.pkl
- xgb_model.pkl
- requirements.txt