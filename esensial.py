import streamlit as st

st.title("Judul ini cuy")
st.header("Ini buat halaman")
st.subheader("Ini sub header")
st.write("apa ajalah")
st.markdown("**bold**, *italic*")

# Menampilkan data

import pandas as pd

df = pd.DataFrame({
    "Nama": ["Budi", "Ali", "Cici"],
    "Umur": [22, 21, 24],
    "Kota": ["Tegal", "Bandung", "Ngawi"]
})

