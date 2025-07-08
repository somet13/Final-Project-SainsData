import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")
st.title("🫀 Prediksi Penyakit Jantung - Final Project Data Sains")

# Tahap 1: Pengumpulan Data & Pra-pemrosesan
st.subheader("1️⃣ Pengumpulan Data & Pra-pemrosesan")

try:
    df = pd.read_csv("heart.csv")  # Pastikan file ini ada di folder yang sama
    st.success("✅ Dataset berhasil dimuat secara otomatis!")
    st.write("📋 Preview Data:")
    st.dataframe(df.head())

    st.write("🔍 Cek Missing Value:")
    st.write(df.isnull().sum())

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tahap 2: Pemodelan & Pembelajaran Mesin
    st.subheader("2️⃣ Pemodelan & Pembelajaran Mesin")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    st.success("✅ Model berhasil dilatih!")

    # Tahap 3: Validasi & Evaluasi Model
    st.subheader("3️⃣ Validasi & Evaluasi Model")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"🎯 Akurasi Model: **{acc * 100:.2f}%**")

    st.text("📄 Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Tahap 4: Penerapan Model
    st.subheader("4️⃣ Penerapan Model (Simulasi Prediksi)")
    sample_input = pd.DataFrame([X.mean()], columns=X.columns)
    sample_input_scaled = scaler.transform(sample_input)
    prediction = model.predict(sample_input_scaled)

    if prediction[0] == 1:
        st.error("💔 Prediksi: Positif (berpotensi penyakit jantung)")
    else:
        st.success("❤️ Prediksi: Negatif (tidak terindikasi penyakit jantung)")

    st.write("🔎 Data Simulasi (Mean dari Dataset):")
    st.dataframe(sample_input)

except FileNotFoundError:
    st.error("⛔ File `heart.csv` tidak ditemukan. Pastikan file berada di direktori yang sama dengan script ini.")
