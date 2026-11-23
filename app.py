import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Invernadero Inteligente U4", layout="wide")
st.title("Optimización Inferencial del Rendimiento Agrícola (U4)")[cite: 33, 43]

# Filtro interactivo en barra lateral (Fase 5)[cite: 54]
tratamiento_sel = st.sidebar.multiselect(
    "Filtrar por tratamiento",
    options=["Tratamiento_A", "Tratamiento_B"],
    default=["Tratamiento_A", "Tratamiento_B"],
)[cite: 37, 44, 54]

@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=":memory:")
    con.execute("SELECT setseed(0.42);")[cite: 37, 43]
    con.execute("""
    CREATE TABLE telemetria_parcelas AS
    WITH base_parcelas AS (
        SELECT
            i AS parcela_id,
            ROUND(100.0 + (random() * (300.0 - 100.0)), 2) AS volumen_riego,
            ROUND(50.0 + (random() * (150.0 - 50.0)), 2) AS fertilizante_kg,
            random() AS prob_tratamiento,
            -0.8 + (random() * (0.8 - (-0.8))) AS ruido_aleatorio
        FROM generate_series(1, 100) AS t(i)
    ),
    con_tratamiento AS (
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            CASE WHEN prob_tratamiento < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            CASE WHEN prob_tratamiento < 0.50 THEN 0.0 ELSE 1.8 END AS efecto_tratamiento,
            ruido_aleatorio
        FROM base_parcelas
    )
    SELECT
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        ROUND(
            5.0 + (0.035 * volumen_riego) + (0.020 * fertilizante_kg) + efecto_tratamiento + ruido_aleatorio,
            2
        ) AS rendimiento_ton
    FROM con_tratamiento
    ORDER BY parcela_id;
    """)[cite: 43, 44, 45, 46]
    return con.execute("SELECT * FROM telemetria_parcelas").df()

df_full = cargar_datos()
df = df_full[df_full["tipo_tratamiento"].isin(tratamiento_sel)].copy()

if df.empty:
    st.warning("Selecciona al menos un tratamiento en la barra lateral.")
    st.stop()

# Modelo OLS y residuos sobre el subconjunto filtrado[cite: 36, 39, 48, 51]
X_ols = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]])
modelo_ols = sm.OLS(df["rendimiento_ton"], X_ols).fit()
df["residuo"] = df["rendimiento_ton"] - modelo_ols.predict(X_ols)

# Prueba t-Student (se calcula sobre el dataset completo si ambos grupos están presentes)[cite: 37, 49]
ga = df_full[df_full["tipo_tratamiento"] == "Tratamiento_A"]["rendimiento_ton"]
gb = df_full[df_full["tipo_tratamiento"] == "Tratamiento_B"]["rendimiento_ton"]
_, p_val = stats.ttest_ind(gb, ga, equal_var=False, alternative="greater")

# Tarjetas KPI (Fase 5)[cite: 52]
c1, c2, c3 = st.columns(3)
c1.metric("Rendimiento Promedio Global", f"{df['rendimiento_ton'].mean():.2f} ton/ha")[cite: 44, 52]
c2.metric("R² Ajustado (OLS)", f"{modelo_ols.rsquared_adj * 100:.2f}%")[cite: 36, 48, 52]
c3.metric("p-valor Prueba t (B > A)", f"{p_val:.6f}")[cite: 37, 49, 52]

# Cuadrícula 2x2 con las 4 gráficas (Fase 4 y 5)[cite: 39, 40, 50, 51, 52, 54]
matriz_corr = df[["volumen_riego", "fertilizante_kg", "rendimiento_ton"]].corr(method="pearson")

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("1. Heatmap de Correlación (Pearson)")[cite: 39, 50]
    st.pyplot(fig)

with row1_col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton", ax=ax)
    ax.set_title("2. Dispersión y Regresión OLS")[cite: 39, 50]
    st.pyplot(fig)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_full, x="tipo_tratamiento", y="rendimiento_ton", ax=ax)
    ax.set_title("3. Boxplot por Tratamiento (A vs. B)")[cite: 40, 51]
    st.pyplot(fig)

with row2_col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["residuo"], kde=True, bins=15, ax=ax)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("4. Histograma y KDE de Residuos")[cite: 40, 51]
    st.pyplot(fig)