# Prediksi Harga Rumah

Aplikasi ini digunakan untuk memprediksi harga rumah menggunakan Machine Learning.

## Deskripsi
Program ini menerima beberapa input seperti luas rumah, jumlah kamar, jumlah kamar mandi, dan kualitas bangunan, kemudian menghasilkan prediksi harga rumah.

## Input
- GrLivArea (Luas rumah)
- BedroomAbvGr (Jumlah kamar tidur)
- FullBath (Jumlah kamar mandi)
- OverallQual (Kualitas rumah)

## Model
Model yang digunakan adalah XGBoost Regressor.

## Cara Menjalankan
1. Install library:
pip install -r requirements.txt

2. Jalankan program:
python app.py

3. Buka di browser:
http://127.0.0.1:7860

## Demo
Aplikasi dijalankan secara lokal menggunakan browser pada alamat:
http://127.0.0.1:7860

## Struktur File
- app.py
- model_xgb.pkl
- requirements.txt