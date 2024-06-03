import streamlit as st
import numpy as np

# Fungsi dekomposisi Crout
def crout_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.double)
    U = np.eye(n, dtype=np.double)
    steps = []

    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))
        for i in range(j + 1, n):
            U[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(j))) / L[j, j]
            if np.isnan(U[j][i]):
                U[j][i] = 0
        steps.append((L.copy(), U.copy()))
    
    return L, U, steps

# Fungsi dekomposisi Doolittle
def doolittle_decomposition(A):
    n = A.shape[0]
    L = np.eye(n, dtype=np.double)
    U = np.zeros((n, n), dtype=np.double)
    steps = []

    for j in range(n):
        for i in range(j + 1):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for i in range(j, n):
            L[i, j] = (A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))) / U[j, j]
            if np.isnan(L[i][j]):
                L[i][j] = 0
        steps.append((L.copy(), U.copy()))

    return L, U, steps

# Fungsi untuk mengubah input teks menjadi matriks numpy
def text_to_matrix(text):
    try:
        rows = text.strip().split('\n')
        matrix = [list(map(float, row.split())) for row in rows]
        return np.array(matrix)
    except Exception as e:
        st.error(f"Error converting input to matrix: {e}")
        return None

# Streamlit antarmuka
st.title("Kalkulator Dekomposisi Matriks")
st.header("Selamat Datang di Kalkulator Dekomposisi Crout & Doolittle by Kelompok 10 Kelas B")

matrix_input = st.text_area("Masukkan matriks Anda (dengan cara pisahkan elemen dengan spasi dan baris dengan baris baru/enter)")

metode = st.selectbox("Pilih metode dekomposisi", ["Crout", "Doolittle"])

if st.button("Dekomposisi"):
    A = text_to_matrix(matrix_input)
    if A is not None:
        if metode == "Crout":
            L, U, steps = crout_decomposition(A)
            st.subheader('Crout Decomposition')
            for step, (L_step, U_step) in enumerate(steps):
                st.write(f"Step {step + 1}:")
                st.write("L =", L_step)
                st.write("U =", U_step)
            st.write("Final L =", L)
            st.write("Final U =", U)
            st.write("L x U =", L @ U)

        elif metode == "Doolittle":
            L, U, steps = doolittle_decomposition(A)
            st.subheader('Doolittle Decomposition')
            for step, (L_step, U_step) in enumerate(steps):
                st.write(f"Step {step + 1}:")
                st.write("L =", L_step)
                st.write("U =", U_step)
            st.write("Final L =", L)
            st.write("Final U =", U)
            st.write("L x U =", L @ U)
