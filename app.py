import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODEL
# =========================
rf = joblib.load("rf_model.pkl")
gb = joblib.load("gb_model.pkl")
xgb = joblib.load("xgb_model.pkl")

features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath']

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("AmesHousing_clean.csv")

# hanya numeric untuk korelasi
df_corr = df.select_dtypes(include=['number'])

# =========================
# PREDIKSI
# =========================
def predict_price(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):
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
# 📊 PIE CHART (REPLACE SCATTER)
# =========================
def plot_pie():
    fig, ax = plt.subplots(figsize=(7,7))

    quality_counts = df["OverallQual"].value_counts().sort_index()

    ax.pie(
        quality_counts,
        labels=quality_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )

    ax.set_title("Distribusi Kualitas Rumah (OverallQual)", fontsize=14, fontweight='bold')
    return fig

# =========================
# HISTOGRAM
# =========================
def plot_histogram():
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df["SalePrice"], bins=30, color="skyblue", edgecolor="black")
    ax.set_title("Distribusi Harga Rumah", fontsize=14, fontweight='bold')
    ax.set_xlabel("Harga")
    ax.set_ylabel("Jumlah")
    return fig

# =========================
# BOXPLOT
# =========================
def plot_boxplot():
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x="OverallQual", y="SalePrice", data=df, ax=ax)
    ax.set_title("Kualitas Rumah vs Harga", fontsize=14, fontweight='bold')
    return fig

# =========================
# HEATMAP KORELASI
# =========================
def plot_correlation():
    fig, ax = plt.subplots(figsize=(9,6))
    sns.heatmap(df_corr.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Heatmap Korelasi Fitur", fontsize=14, fontweight='bold')
    return fig

# =========================
# PERBANDINGAN MODEL
# =========================
def compare_models(grlivarea, overallqual, garagecars, totalbsmtsf, fullbath):
    data = pd.DataFrame([[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath]],
                         columns=features)

    preds = [
        rf.predict(data)[0],
        gb.predict(data)[0],
        xgb.predict(data)[0]
    ]

    names = ["Random Forest", "Gradient Boost", "XGBoost"]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(names, preds, color=["skyblue", "orange", "green"])
    ax.set_title("Perbandingan Prediksi Model", fontsize=14, fontweight='bold')
    ax.set_ylabel("Harga Rumah")
    return fig

# =========================
# GRADIO UI
# =========================
with gr.Blocks() as app:

    gr.Markdown("# 🏠 Prediksi Harga Rumah (ML Dashboard) 🔥")
    gr.Markdown("Model: Random Forest | Gradient Boost | XGBoost")

    # INPUT
    with gr.Row():
        grlivarea = gr.Number(label="GrLivArea")
        overallqual = gr.Number(label="OverallQual")
        garagecars = gr.Number(label="GarageCars")
        totalbsmtsf = gr.Number(label="TotalBsmtSF")
        fullbath = gr.Number(label="FullBath")

    btn = gr.Button("🔍 Prediksi Harga")
    output = gr.Textbox(label="Hasil Prediksi")

    btn.click(
        predict_price,
        inputs=[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath],
        outputs=output
    )

    # =========================
    # VISUALISASI
    # =========================
    gr.Markdown("## 📊 Visualisasi Dataset")

    with gr.Row():
        pie_btn = gr.Button("🥧 Pie Chart Kualitas")
        hist_btn = gr.Button("📊 Histogram Harga")

    pie_out = gr.Plot()
    hist_out = gr.Plot()

    pie_btn.click(plot_pie, outputs=pie_out)
    hist_btn.click(plot_histogram, outputs=hist_out)

    with gr.Row():
        box_btn = gr.Button("📦 Boxplot")
        corr_btn = gr.Button("🔥 Heatmap Korelasi")

    box_out = gr.Plot()
    corr_out = gr.Plot()

    box_btn.click(plot_boxplot, outputs=box_out)
    corr_btn.click(plot_correlation, outputs=corr_out)

    # =========================
    # MODEL COMPARISON
    # =========================
    gr.Markdown("## 📈 Perbandingan Model")

    compare_btn = gr.Button("Bandingkan Model")
    compare_out = gr.Plot()

    compare_btn.click(
        compare_models,
        inputs=[grlivarea, overallqual, garagecars, totalbsmtsf, fullbath],
        outputs=compare_out
    )

# RUN APP
app.launch()