from flask import Flask, render_template
import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
rf = joblib.load("rf_model.pkl")
gb = joblib.load("gb_model.pkl")
xgb = joblib.load("xgb_model.pkl")

features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath']

# =========================
# FUNGSI PREDIKSI
# =========================
def predict_price(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):
    data = pd.DataFrame([[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath]],
                        columns=features)

    rf_pred = rf.predict(data)[0]
    gb_pred = gb.predict(data)[0]
    xgb_pred = xgb.predict(data)[0]

    avg_pred = (rf_pred + gb_pred + xgb_pred) / 3

    return f"RF: ${rf_pred:,.0f} | GB: ${gb_pred:,.0f} | XGB: ${xgb_pred:,.0f} | AVG: ${avg_pred:,.0f}"

# =========================
# GRADIO APP
# =========================
def create_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# 🏠 Dashboard Prediksi Harga Rumah")

        grlivarea = gr.Number(label="GrLivArea")
        overallqual = gr.Number(label="OverallQual")
        garagecars = gr.Number(label="GarageCars")
        totalbsmtsf = gr.Number(label="TotalBsmtSF")
        fullbath = gr.Number(label="FullBath")

        btn = gr.Button("Prediksi")
        output = gr.Textbox()

        btn.click(
            predict_price,
            inputs=[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath],
            outputs=output
        )

    return demo

gradio_app = create_gradio_app()

# =========================
# ROUTE FLASK
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/app")
def app_page():
    return gradio_app.launch(
        inline=True,
        share=False
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)