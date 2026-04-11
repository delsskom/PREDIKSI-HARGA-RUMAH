import gradio as gr
import pandas as pd
import joblib

# Load model
model = joblib.load("model_xgb.pkl")

features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath']

def predict(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):
    data = pd.DataFrame([[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath]],
                        columns=features)
    
    pred = model.predict(data)[0]
    return f"💰 Prediksi Harga: ${pred:,.0f}"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="GrLivArea"),
        gr.Number(label="OverallQual"),
        gr.Number(label="GarageCars"),
        gr.Number(label="TotalBsmtSF"),
        gr.Number(label="FullBath")
    ],
    outputs="text",
    title="🏠 Prediksi Harga Rumah"
).launch()