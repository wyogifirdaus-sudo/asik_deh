import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# ==================================
# Memasukkan Model dan Pipeline
# ==================================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_ann_terbaik.keras')

@st.cache_resource
def load_pipeline():
    with open('pipeline.pkl', 'rb') as f:
        return pickle.load(f)

model = load_my_model()
pipeline = load_pipeline()

# ==========================================
# 2. TAMPILAN DASHBOARD STREAMLIT
# ==========================================
st.title("Aplikasi Prediksi Kelayakan Gaji Pekerja Data Science")
st.write("Masukkan spesifikasi pekerjaan di bawah ini untuk memprediksi tingkat kategori gaji.")

st.subheader("Karakteristik Pekerjaan & Perusahaan")

# Input pilihan sesuai mapping ordinal pada EDA kamu
experience_level = st.selectbox("Experience Level", options=["EN", "MI", "SE", "EX"])
employment_type = st.selectbox("Employment Type", options=["FT", "CT", "PT", "FL"])
remote_ratio = st.selectbox("Remote Ratio", options=[0, 50, 100])
company_size = st.selectbox("Company Size", options=["S", "M", "L"])

# Input text untuk kolom kategorikal yang di-OneHotEncoder oleh pipeline
job_title = st.text_input("Job Title", value="Data Scientist")
employee_residence = st.text_input("Employee Residence (Kode Negara, misal: US, ID)", value="US")
company_location = st.text_input("Company Location (Kode Negara, misal: US, ID)", value="US")

# ==========================================
# 3. PROSES PREDIKSI
# ==========================================
if st.button("Prediksi Kategori Gaji"):
    # Mapping manual sesuai nilai encode yang kamu buat di EDA
    exp_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
    emp_map = {'FT': 1, 'CT': 2, 'PT': 3, 'FL': 4}
    size_map = {'S': 1, 'M': 2, 'L': 3}
    
    # Buat ke format dataframe satu baris agar bisa ditransform oleh pipeline
    input_data = pd.DataFrame([{
        'experience_level': exp_map[experience_level],
        'employment_type': emp_map[employment_type],
        'remote_ratio': remote_ratio,
        'company_size': size_map[company_size],
        'job_title': job_title,
        'employee_residence': employee_residence,
        'company_location': company_location
    }])
    
    try:
        # Preprocessing input dengan pipeline transformer dari notebook
        input_processed = pipeline.transform(input_data)
        
        # Prediksi menggunakan model ANN Sequential
        prediction_prob = model.predict(input_processed)[0][0]
        prediction_class = 1 if prediction_prob >= 0.5 else 0
        
        st.subheader("Hasil Analisis Model ANN:")
        if prediction_class == 1:
            st.success(f"Gaji diprediksi **DI ATAS** Rata-rata (Probabilitas: {prediction_prob*100:.2f}%)")
        else:
            st.error(f"Gaji diprediksi **DI BAWAH** Rata-rata (Probabilitas: {(1 - prediction_prob)*100:.2f}%)")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Catatan: Pastikan teks 'Job Title' atau kode negara yang diinput merupakan data yang valid/pernah ada di dataset asli saat training.")