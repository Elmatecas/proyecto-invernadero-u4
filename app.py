# ==============================================================================
# APP STREAMLIT: PROYECTO INTEGRADOR U4 - TELEMETRÍA AGRÍCOLA (INC-2505)
# ==============================================================================

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import streamlit as st
import seaborn as sns

# Configuración general de la página en Streamlit
st.set_page_config(
    page_title="Proyecto Integrador U4 - Telemetría Agrícola",
    page_icon="🌾",
    layout="wide",
)
sns.set_theme(style="whitegrid")


@st.cache_data
def simular_y_analizar(semilla: float = 0.42, n_parcelas: int = 100):
    """Ejecuta la simulación en DuckDB, el modelo OLS y la prueba t-Student."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"SELECT setseed({semilla});")

    con.execute(f"""
    CREATE TABLE telemetria_parcelas AS
    WITH base_parcelas AS (
        SELECT
            i AS parcela_id,
            ROUND(100.0 + (random() * (300.0 - 100.0)), 2) AS volumen_riego,
            ROUND(50.0 + (random() * (150.0 - 50.0)), 2) AS fertilizante_kg,
            random() AS prob_tratamiento,
            -0.8 + (random() * (0.8 - (-0.8))) AS ruido_aleatorio
        FROM generate_series(1, {n_parcelas}) AS t(i)
    ),
    con_tratamiento AS (
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            CASE
                WHEN prob_tratamiento < 0.50 THEN 'Tratamiento_A'
                ELSE 'Tratamiento_B'
            END AS tipo_tratamiento,
            CASE
                WHEN prob_tratamiento < 0.50 THEN 0.0
                ELSE 1.8
            END AS efecto_tratamiento,
            ruido_aleatorio
        FROM base_parcelas
    )
    SELECT
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        ROUND(
            5.0
            + (0.035 * volumen_riego)
            + (0.020 * fertilizante_kg)
            + efecto_tratamiento
            + ruido_aleatorio,
            2
        ) AS rendimiento_ton
    FROM con_tratamiento
    ORDER BY parcela_id;
    """)

    df = con.execute("""
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            tipo_tratamiento,
            rendimiento_ton
        FROM telemetria_parcelas
        WHERE volumen_riego IS NOT NULL
          AND fertilizante_kg IS NOT NULL
          AND rendimiento_ton IS NOT NULL
    """).df()

    # Matriz de correlación de Pearson
    columnas_numericas = ["volumen_riego", "fertilizante_kg", "rendimiento_ton"]
    matriz_corr = df[columnas_numericas].corr(method="pearson")

    # Modelo OLS
    X = df[["volumen_riego", "fertilizante_kg"]]
    y = df["rendimiento_ton"]
    X_ols = sm.add_constant(X)
    modelo_ols = sm.OLS(y, X_ols).fit()

    df["rendimiento_pred"] = modelo_ols.predict(X_ols)
    df["residuo"] = df["rendimiento_ton"] - df["rendimiento_pred"]

    rmse = float(np.sqrt(np.mean(df["residuo"] ** 2)))
    r2_adj = float(modelo_ols.rsquared_adj)

    # Tabla de coeficientes OLS en formato limpio
    tabla_coef = pd.DataFrame({
        "Variable": modelo_ols.params.index,
        "Coeficiente (coef)": modelo_ols.params.values,
        "Error Estándar (std err)": modelo_ols.bse.values,
        "Estadístico t": modelo_ols.tvalues.values,
        "p-valor (P>|t|)": modelo_ols.pvalues.values,
        "IC 95% Inf": modelo_ols.conf_int()[0].values,
        "IC 95% Sup": modelo_ols.conf_int()[1].values,
    })

    # Prueba t-Student independiente (unilateral: Tratamiento_B > Tratamiento_A)
    grupo_a = df[df["tipo_tratamiento"] == "Tratamiento_A"]["rendimiento_ton"]
    grupo_b = df[df["tipo_tratamiento"] == "Tratamiento_B"]["rendimiento_ton"]

    t_stat, p_val_unilateral = stats.ttest_ind(
        grupo_b, grupo_a, equal_var=False, alternative="greater"
    )

    diff_medias = float(grupo_b.mean() - grupo_a.mean())
    n_a, n_b = len(grupo_a), len(grupo_b)
    var_a, var_b = grupo_a.var(ddof=1), grupo_b.var(ddof=1)
    se_diff = np.sqrt((var_a / n_a) + (var_b / n_b))

    gl = ((var_a / n_a + var_b / n_b) ** 2) / (
        ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    )
    t_crit = stats.t.ppf(0.975, df=gl)
    margen_error = t_crit * se_diff
    ic_95_inf = float(diff_medias - margen_error)
    ic_95_sup = float(diff_medias + margen_error)

    resumen_t = {
        "media_a": float(grupo_a.mean()),
        "media_b": float(grupo_b.mean()),
        "n_a": n_a,
        "n_b": n_b,
        "diff_medias": diff_medias,
        "ic_95_inf": ic_95_inf,
        "ic_95_sup": ic_95_sup,
        "t_stat": float(t_stat),
        "p_val": float(p_val_unilateral),
    }

    return df, matriz_corr, modelo_ols, tabla_coef, r2_adj, rmse, resumen_t


# ==============================================================================
# INTERFAZ PRINCIPAL EN STREAMLIT
# ==============================================================================

st.title("🌾 Proyecto Integrador U4: Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
st.markdown("**Materia:** Análisis y Exploración de Datos (INC-2505)")
st.markdown(
    "En este tablero se presenta el **Algoritmo General de Simulación de Telemetría** en **DuckDB** "
    "($N=100$ parcelas), el análisis de **Correlación y Regresión Lineal (OLS)**, la **Prueba t-Student** "
    "entre `Tratamiento_A` y `Tratamiento_B`, y las **4 gráficas ejecutivas** del proyecto."
)

# Barra lateral con parámetros opcionales (por defecto usa semilla 0.42 y N=100)
with st.sidebar:
    st.header("Parámetros de Simulación")
    semilla = st.number_input("Semilla (`setseed`)", value=0.42, step=0.01, format="%.2f")
    n_parcelas = st.number_input("Número de parcelas (N)", min_value=20, max_value=1000, value=100, step=10)
    mostrar_todo_df = st.checkbox("Mostrar las 100 parcelas completas", value=False)

df, matriz_corr, modelo_ols, tabla_coef, r2_adj, rmse, resumen_t = simular_y_analizar(
    semilla=semilla, n_parcelas=int(n_parcelas)
)

# ------------------------------------------------------------------------------
# SECCIÓN 1: SIMULACIÓN DE TELEMETRÍA EN DUCKDB
# ------------------------------------------------------------------------------
st.header("1. Algoritmo General de Simulación de Telemetría (Paso a Paso)")
st.markdown("""
Siguiendo el algoritmo del proyecto:
1. Fijamos la semilla determinista `setseed(0.42)`.
2. Generamos `parcela_id` del `1` al `100`.
3. Simulamos `volumen_riego` uniforme entre `100.0` y `300.0` $\\text{m}^3/\\text{ha}$ (redondeado a 2 decimales).
4. Simulamos `fertilizante_kg` uniforme entre `50.0` y `150.0` $\\text{kg}/\\text{ha}$ (redondeado a 2 decimales).
5. Asignamos `tipo_tratamiento` (`Tratamiento_A` si probabilidad menor a `0.50`, `Tratamiento_B` si mayor o igual a `0.50`).
6. Calculamos `rendimiento_ton` con intercepto `5.0`, sensibilidad al riego `0.035`, sensibilidad al fertilizante `0.020`, bono de `+1.8` para `Tratamiento_B` y ruido aleatorio entre `-0.8` y `+0.8`.
""")

conteo_trat = df["tipo_tratamiento"].value_counts().reset_index()
conteo_trat.columns = ["Tipo de Tratamiento", "Cantidad de Parcelas"]

col_txt1, col_tbl1 = st.columns([1, 2])
with col_txt1:
    st.markdown(f"**Registros cargados:** `{len(df)}`")
    st.table(conteo_trat)

with col_tbl1:
    st.subheader("Muestra de Datos Simulados (`df.head(10)`)" if not mostrar_todo_df else "Dataset Completo de Parcelas")
    tabla_mostrar = df[["parcela_id", "volumen_riego", "fertilizante_kg", "tipo_tratamiento", "rendimiento_ton"]]
    if not mostrar_todo_df:
        tabla_mostrar = tabla_mostrar.head(10)
    st.dataframe(tabla_mostrar, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# SECCIÓN 2: CORRELACIÓN Y REGRESIÓN LINEAL MÚLTIPLE (OLS)
# ------------------------------------------------------------------------------
st.header("2. Análisis de Correlación y Modelo Predictivo OLS")
st.markdown(
    "Calculamos la **matriz de correlación de Pearson ($r$)** para identificar qué variable tiene mayor "
    "relación lineal con `rendimiento_ton`, y ajustamos un modelo de **Regresión Lineal Múltiple (OLS)** "
    "con `statsmodels` incluyendo intercepto ($\\beta_0$) para evaluar $R^2$ ajustado, p-valores y residuos."
)

st.subheader("Matriz de Correlación de Pearson (r)")
st.dataframe(matriz_corr.round(4), use_container_width=True)

st.subheader("Tabla de Coeficientes del Modelo OLS")
st.dataframe(
    tabla_coef.style.format({
        "Coeficiente (coef)": "{:.4f}",
        "Error Estándar (std err)": "{:.4f}",
        "Estadístico t": "{:.3f}",
        "p-valor (P>|t|)": "{:.4f}",
        "IC 95% Inf": "{:.4f}",
        "IC 95% Sup": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True,
)

c_m1, c_m2, c_m3 = st.columns(3)
c_m1.metric("R² Ajustado", f"{r2_adj:.4f} ({r2_adj*100:.2f}%)")
c_m2.metric("RMSE", f"{rmse:.4f} ton/ha")
c_m3.metric("R² del Modelo", f"{modelo_ols.rsquared:.4f}")

with st.expander("Ver salida textual completa de `modelo_ols.summary()`"):
    st.code(modelo_ols.summary().as_text(), language="text")

# ------------------------------------------------------------------------------
# SECCIÓN 3: PRUEBA DE HIPÓTESIS INFERENCIAL (t-STUDENT)
# ------------------------------------------------------------------------------
st.header("3. Prueba de Hipótesis Inferencial (t-Student de Dos Muestras Independientes)")
st.markdown("""
Evaluamos si `Tratamiento_B` incrementa el rendimiento promedio respecto a `Tratamiento_A`:
- **Hipótesis nula ($H_0$):** $\\mu_B - \\mu_A \\le 0$
- **Hipótesis alternativa ($H_1$):** $\\mu_B - \\mu_A > 0$ (prueba unilateral derecha, nivel $\\alpha = 0.05$)

Calculamos el estadístico $t$, el valor $p$, el **intervalo de confianza al 95%** para la diferencia de medias y el veredicto técnico.
""")

tabla_t = pd.DataFrame({
    "Métrica": [
        f"Media Tratamiento_A (n={resumen_t['n_a']})",
        f"Media Tratamiento_B (n={resumen_t['n_b']})",
        "Diferencia de medias (B - A)",
        "IC 95% diferencia de medias",
        "Estadístico t (Welch)",
        "p-valor (unilateral)",
    ],
    "Valor": [
        f"{resumen_t['media_a']:.2f} ton/ha",
        f"{resumen_t['media_b']:.2f} ton/ha",
        f"{resumen_t['diff_medias']:.2f} ton/ha",
        f"[{resumen_t['ic_95_inf']:.2f}, {resumen_t['ic_95_sup']:.2f}] ton/ha",
        f"{resumen_t['t_stat']:.4f}",
        f"{resumen_t['p_val']:.6f}",
    ],
})
st.table(tabla_t)

if resumen_t["p_val"] < 0.05:
    st.success(
        "**VEREDICTO:** Se rechaza $H_0$ ($p < 0.05$). `Tratamiento_B` incrementa significativamente el rendimiento."
    )
else:
    st.warning(
        "**VEREDICTO:** No se rechaza $H_0$ ($p \\ge 0.05$). No hay evidencia suficiente de incremento."
    )

# ------------------------------------------------------------------------------
# SECCIÓN 4: KPIs Y CUADRÍCULA DE 4 GRÁFICAS
# ------------------------------------------------------------------------------
st.header("4. Visualizaciones Clave y Métricas Resumen (KPIs)")
st.markdown("""
Generamos las **4 gráficas** solicitadas en el proyecto en una cuadrícula $2 \\times 2$:
1. **Heatmap de Correlación de Pearson**
2. **Dispersión con Línea de Regresión OLS** (`volumen_riego` vs. `rendimiento_ton`)
3. **Boxplot Comparativo de Tratamientos** (`Tratamiento_A` vs. `Tratamiento_B`)
4. **Histograma y Curva KDE de Residuos** (`Y_real - Y_predicho`)
""")

k1, k2, k3 = st.columns(3)
k1.metric("1. Rendimiento Promedio Global", f"{df['rendimiento_ton'].mean():.2f} ton/ha")
k2.metric("2. R² Ajustado del Modelo OLS", f"{r2_adj*100:.2f}%")
k3.metric("3. p-valor Prueba t (B > A)", f"{resumen_t['p_val']:.6f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Dashboard de Análisis Agronómico - Telemetría de Invernadero", fontsize=15, fontweight="bold")

# Gráfica 1: Heatmap
sns.heatmap(
    matriz_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    ax=axes[0, 0],
)
axes[0, 0].set_title("1. Heatmap de Correlación (Pearson)", fontweight="bold")

# Gráfica 2: Dispersión y Regresión OLS
sns.regplot(
    data=df,
    x="volumen_riego",
    y="rendimiento_ton",
    scatter_kws={"alpha": 0.65, "color": "#1f77b4"},
    line_kws={"color": "#d62728", "linewidth": 2},
    ax=axes[0, 1],
)
axes[0, 1].set_title("2. Dispersión y Regresión (Riego vs. Rendimiento)", fontweight="bold")
axes[0, 1].set_xlabel("Volumen de Riego (m³/ha)")
axes[0, 1].set_ylabel("Rendimiento (ton/ha)")

# Gráfica 3: Boxplot Comparativo
sns.boxplot(
    data=df,
    x="tipo_tratamiento",
    y="rendimiento_ton",
    hue="tipo_tratamiento",
    palette={"Tratamiento_A": "#ff7f0e", "Tratamiento_B": "#2ca02c"},
    legend=False,
    ax=axes[1, 0],
)
axes[1, 0].set_title("3. Rendimiento por Tratamiento (A vs. B)", fontweight="bold")
axes[1, 0].set_xlabel("Tipo de Tratamiento")
axes[1, 0].set_ylabel("Rendimiento (ton/ha)")

# Gráfica 4: Histograma y KDE de Residuos
sns.histplot(
    df["residuo"],
    kde=True,
    bins=15,
    color="#6a3d9a",
    ax=axes[1, 1],
)
axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1.2)
axes[1, 1].set_title("4. Histograma y KDE de Residuos (Y_real - Y_pred)", fontweight="bold")
axes[1, 1].set_xlabel("Residuo (ton/ha)")
axes[1, 1].set_ylabel("Frecuencia")

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
st.pyplot(fig)
