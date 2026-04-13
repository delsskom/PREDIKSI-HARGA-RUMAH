import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL
# =========================
rf = joblib.load("rf_model.pkl")
gb = joblib.load("gb_model.pkl")
xgb = joblib.load("xgb_model.pkl")

features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath']

# =========================
# VALIDASI INPUT
# =========================
def validate_input(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):

    if None in [grlivarea, overallqual, garagecars, totalbsmtsf, fullbath]:
        return False

    if grlivarea <= 0:
        return False

    if overallqual < 1 or overallqual > 10:
        return False

    if garagecars < 0:
        return False

    if totalbsmtsf < 0:
        return False

    if fullbath < 0:
        return False

    return True

# =========================
# PREDIKSI
# =========================
def predict_price(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):

    if not validate_input(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):
        return "❌ Input tidak valid!\n- GrLivArea harus > 0\n- OverallQual 1–10\n- Tidak boleh ada nilai negatif"

    data = pd.DataFrame([[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath]],
                        columns=features)

    rf_pred = rf.predict(data)[0]
    gb_pred = gb.predict(data)[0]
    xgb_pred = xgb.predict(data)[0]

    avg_pred = (rf_pred + gb_pred + xgb_pred) / 3

    return (
        f"🏠 Random Forest : ${rf_pred:,.0f}\n"
        f"🌲 Gradient Boost : ${gb_pred:,.0f}\n"
        f"⚡ XGBoost        : ${xgb_pred:,.0f}\n\n"
        f"💡 Rata-rata      : ${avg_pred:,.0f}"
    )

# =========================
# VISUALISASI BAR CHART
# =========================
def compare_models(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):

    if not validate_input(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):
        return None

    data = pd.DataFrame([[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath]],
                        columns=features)

    rf_pred = rf.predict(data)[0]
    gb_pred = gb.predict(data)[0]
    xgb_pred = xgb.predict(data)[0]

    models = ["Random Forest", "Gradient Boost", "XGBoost"]
    values = [rf_pred, gb_pred, xgb_pred]

    plt.figure()
    plt.bar(models, values)
    plt.title("Perbandingan Prediksi Model")
    plt.xlabel("Model")
    plt.ylabel("Harga Rumah")

    return plt

# =========================
# UI GRADIO
# =========================
with gr.Blocks() as app:

    gr.Markdown("# 🏠 Prediksi Harga Rumah")
    gr.Markdown("### Model: Random Forest | Gradient Boosting | XGBoost")

    # INPUT
    with gr.Row():
        grlivarea = gr.Number(label="GrLivArea (>0)", minimum=1)
        overallqual = gr.Number(label="OverallQual (1–10)", minimum=1, maximum=10)
        garagecars = gr.Number(label="GarageCars (≥0)", minimum=0)
        totalbsmtsf = gr.Number(label="TotalBsmtSF (≥0)", minimum=0)
        fullbath = gr.Number(label="FullBath (≥0)", minimum=0)

    # BUTTON
    with gr.Row():
        pred_btn = gr.Button("🔍 Prediksi Harga")
        comp_btn = gr.Button("📊 Bandingkan Model")

    # OUTPUT
    output_pred = gr.Textbox(label="Hasil Prediksi")
    output_chart = gr.Plot(label="Visualisasi Perbandingan Model")

    # ACTION
    pred_btn.click(
        predict_price,
        inputs=[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath],
        outputs=output_pred
    )

    comp_btn.click(
        compare_models,
        inputs=[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath],
        outputs=output_chart
    )

# =========================
# RUN
# =========================
app.launch()