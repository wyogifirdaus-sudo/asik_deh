import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ==========================================
# 1. CONFIGURATION & CUSTOM DARK THEME (CSS)
# ==========================================
st.set_page_config(
    page_title="Prediksi Upah Pekerja",
    page_icon="🧑‍💻",
    layout="centered"
)

# Kustomisasi CSS untuk menyisipkan warna Hitam, Biru, Kuning, dan Hijau
st.markdown("""
    <style>
    /* Mengubah background utama menjadi Hitam/Gelap */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Style Judul Utama (Kombinasi Biru & Kuning) */
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #FFD700; /* Kuning */
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #38BDF8; /* Biru Muda */
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Mengubah warna teks label input menjadi Biru Terang agar kontras di latar hitam */
    label {
        color: #38BDF8 !important;
        font-weight: bold !important;
    }

    /* BARU: Mengubah warna tulisan indikator di dalam st.radio menjadi Kuning */
    div[data-testid="stRadio"] label p {
        color: #FFD700 !important;
        font-weight: 500;
    }
    
    /* BARU: Mengubah angka hasil tingkat keyakinan (st.metric) menjadi Putih murni agar terbaca */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    
    /* Desain tombol kustom (Latar Biru, Teks Hitam/Putih) */
    div.stButton > button:first-child {
        background-color: #1E40AF; /* Biru Tua */
        color: white;
        border-radius: 8px;
        border: 1px solid #38BDF8;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #FFD700; /* Kuning saat di-hover */
        color: #0E1117;
        border-color: #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

# ==================================
# 2. MEMASUKKAN MODEL DAN PIPELINE
# ==================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_model = os.path.join(BASE_DIR, "model_ann_terbaik.keras")
path_pipeline = os.path.join(BASE_DIR, "pipeline.pkl")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(path_model)

@st.cache_resource
def load_pipeline():
    with open(path_pipeline, 'rb') as f:
        return pickle.load(f)

model = load_my_model()
pipeline = load_pipeline()

# ==========================================
# 3. TAMPILAN DASHBOARD STREAMLIT
# ==========================================
# Header dengan Ikon Orang Bekerja & Kombinasi Warna
st.markdown('<div class="main-title">🧑‍💻 Data Science Salary Predictor 💼</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Kecerdasan Buatan (ANN) Pengukur Kelayakan Gaji Kerja</div>', unsafe_allow_html=True)

st.write("---")

with st.container():
    st.subheader("🏢 Karakteristik Pekerjaan & Perusahaan")
    
    # Baris Pertama
    col1, col2 = st.columns(2)
    with col1:
        experience_level = st.selectbox(
            "📈 Experience Level", 
            options=["EN", "MI", "SE", "EX"],
            help="EN: Entry-level, MI: Mid-level, SE: Senior, EX: Executive"
        )
    with col2:
        employment_type = st.selectbox(
            "💼 Employment Type", 
            options=["FT", "CT", "PT", "FL"],
            help="FT: Full-Time, CT: Contract, PT: Part-Time, FL: Freelance"
        )
        
    # Baris Kedua
    col3, col4 = st.columns(2)
    with col3:
        remote_ratio = st.radio(
            "🏠 Remote Ratio (%)", 
            options=[0, 50, 100], 
            horizontal=True
        )
    with col4:
        company_size = st.radio(
            "🏢 Company Size", 
            options=["S", "M", "L"], 
            horizontal=True
        )

    st.write("---")
    st.subheader("📍 Detail Posisi & Lokasi")
    
    # 1. DAFTAR POSISI KERJA (JOB TITLE)
    job_titles_list = [
        'Data Scientist', 'Machine Learning Scientist', 'Big Data Engineer',
        'Product Data Analyst', 'Machine Learning Engineer', 'Data Analyst',
        'Lead Data Scientist', 'Business Data Analyst', 'Lead Data Engineer',
        'Lead Data Analyst', 'Data Engineer', 'Data Science Consultant',
        'BI Data Analyst', 'Director of Data Science', 'Research Scientist',
        'Machine Learning Manager', 'Data Engineering Manager',
        'Machine Learning Infrastructure Engineer', 'ML Engineer', 'AI Scientist',
        'Computer Vision Engineer', 'Principal Data Scientist',
        'Data Science Manager', 'Head of Data', '3D Computer Vision Researcher',
        'Data Analytics Engineer', 'Applied Data Scientist',
        'Marketing Data Analyst', 'Cloud Data Engineer', 'Financial Data Analyst',
        'Computer Vision Software Engineer', 'Director of Data Engineering',
        'Data Science Engineer', 'Principal Data Engineer',
        'Machine Learning Developer', 'Applied Machine Learning Scientist',
        'Data Analytics Manager', 'Head of Data Science', 'Data Specialist',
        'Data Architect', 'Finance Data Analyst', 'Principal Data Analyst',
        'Big Data Architect', 'Staff Data Scientist', 'Analytics Engineer',
        'ETL Developer', 'Head of Machine Learning', 'NLP Engineer',
        'Lead Machine Learning Engineer', 'Data Analytics Lead'
    ]
    
    job_title = st.selectbox("💻 Job Title (Posisi Kerja)", options=job_titles_list)
    
    # 2. DAFTAR NEGARA ASAL KARYAWAN
    employee_residence_list = sorted([
        'DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'PL', 'PT', 
        'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'AT', 'NG', 'PH', 'ES', 'DK', 'RU', 
        'IT', 'HR', 'BG', 'SG', 'BR', 'IQ', 'VN', 'BE', 'UA', 'MT', 'CL', 'RO', 
        'IR', 'CO', 'MD', 'KE', 'SI', 'HK', 'TR', 'RS', 'PR', 'LU', 'JE', 'CZ', 
        'AR', 'DZ', 'TN', 'MY', 'EE', 'AU', 'BO', 'IE', 'CH'
    ])
    
    # 3. DAFTAR LOKASI PERUSAHAAN
    company_location_list = sorted([
        'DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 
        'AE', 'NL', 'MX', 'CA', 'AT', 'NG', 'ES', 'PT', 'DK', 'IT', 'HR', 'LU', 
        'PL', 'SG', 'RO', 'IQ', 'BR', 'BE', 'UA', 'IL', 'RU', 'MT', 'CL', 'IR', 
        'CO', 'MD', 'KE', 'SI', 'CH', 'VN', 'AS', 'TR', 'CZ', 'DZ', 'EE', 'MY', 
        'AU', 'IE'
    ])
    
    col5, col6 = st.columns(2)
    with col5:
        default_res_idx = employee_residence_list.index("US") if "US" in employee_residence_list else 0
        employee_residence = st.selectbox(
            "🌍 Employee Residence (Negara Tinggal)", 
            options=employee_residence_list, 
            index=default_res_idx
        )
    with col6:
        default_loc_idx = company_location_list.index("US") if "US" in company_location_list else 0
        company_location = st.selectbox(
            "🏢 Company Location (Lokasi Perusahaan)", 
            options=company_location_list, 
            index=default_loc_idx
        )

# ==========================================
# 4. PROSES PREDIKSI & OUTPUT JALUR HIJAU / KUNING
# ==========================================
st.write("")
predict_btn = st.button("🔮 Hitung Kelayakan Gaji", use_container_width=True)

if predict_btn:
    with st.spinner('Model ANN sedang menganalisis data...'):
        
        exp_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
        emp_map = {'FT': 1, 'CT': 2, 'PT': 3, 'FL': 4}
        size_map = {'S': 1, 'M': 2, 'L': 3}
        
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
            input_processed = pipeline.transform(input_data)
            prediction_prob = model.predict(input_processed)[0][0]
            prediction_class = 1 if prediction_prob >= 0.5 else 0
            
            st.write("---")
            st.markdown("### 📊 Hasil Analisis Model:")
            
            # OUTPUT JALUR HIJAU (Gaji di Atas Rata-rata)
            if prediction_class == 1:
                st.balloons()
                st.success("### 🎉 DI ATAS RATA-RATA!")
                
                st.metric(
                    label="Tingkat Keyakinan Model (Probabilitas Gaji Tinggi)", 
                    value=f"{prediction_prob*100:.2f}%"
                )
                st.info("💡 Posisi ini memiliki nilai pasar yang sangat menguntungkan!")
                
            # OUTPUT JALUR KUNING/ORANYE (Gaji di Bawah Rata-rata)
            else:
                st.warning("### ⚠️ DI BAWAH RATA-RATA")
                st.metric(
                    label="Tingkat Keyakinan Model (Probabilitas Gaji Rendah)", 
                    value=f"{(1 - prediction_prob)*100:.2f}%"
                )
                st.markdown("""
                    <div style="background-color: #3b2001; padding: 10px; border-left: 5px solid #FFD700; border-radius: 4px;">
                        <span style="color: #FFD700;">💡 <b>Tips Kontribusi:</b></span> Cobalah targetkan ukuran perusahaan yang lebih besar (L) atau tingkatkan level pengalaman untuk mendongkrak algoritma nilai gaji.
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
            st.info("ℹ️ Pastikan seluruh input data sudah sesuai dengan format pipeline training.")