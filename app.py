import streamlit as st
import numpy as np
import pickle

# ------------------------------
# Load Model dan Scaler
# ------------------------------
try:
    model = pickle.load(open('model_ann.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# ------------------------------
# Judul dan Penjelasan Aplikasi
# ------------------------------
st.title("Prediksi Risiko Diabetes")
st.write("Masukkan data berikut untuk memprediksi risiko terkena diabetes:")

# ------------------------------
# Input User
# ------------------------------
try:
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-Laki"])
    age = st.number_input("Usia", min_value=1, max_value=120, step=1)
    hypertension = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    smoking_history = st.selectbox("Riwayat Merokok", ["Tidak Pernah", "Dulu Pernah", "Sekarang Merokok"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300)
except Exception as e:
    st.error(f"Input error: {e}")
    st.stop()

# ------------------------------
# Proses Input ke Format Model
# ------------------------------
input_data = np.array([
    1 if gender == "Laki-Laki" else 0,
    age,
    1 if hypertension == "Ya" else 0,
    1 if heart_disease == "Ya" else 0,
    {"Tidak Pernah": 0, "Dulu Pernah": 1, "Sekarang Merokok": 2}[smoking_history],
    bmi,
    hba1c,
    glucose
]).reshape(1, -1)

# ------------------------------
# Prediksi
# ------------------------------
if st.button("Prediksi"):
    try:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        if prediction[0] == 1:
            st.error("Hasil Prediksi: Berisiko Diabetes")
        else:
            st.success("Hasil Prediksi: Tidak Berisiko Diabetes")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
