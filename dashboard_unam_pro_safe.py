#############################################
# CAPA 0: Librerías, constantes y configuración institucional UNAM
#############################################

import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import warnings

import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# =========================
# CONFIGURACIÓN INICIAL
# =========================

st.set_page_config(page_title="SAPEED", layout="wide")

# =========================
# AUTENTICACIÓN
# =========================

def check_password():

    if "APP_PASSWORD" not in st.secrets:
        st.error("APP_PASSWORD not configured in secrets.")
        st.stop()

    def password_entered():
        if st.session_state.get("password", "") == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "🔐 Contraseña institucional",
            type="password",
            key="password",
            on_change=password_entered
        )
        return False

    elif not st.session_state["password_correct"]:
        st.text_input(
            "🔐 Contraseña institucional",
            type="password",
            key="password",
            on_change=password_entered
        )
        st.error("❌ Contraseña incorrecta")
        return False

    return True


if not check_password():
    st.stop()

if st.button("Cerrar sesión"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# =============================
# Use demo data if no files uploaded
# =============================

mode = st.radio(
    "Mode de datos",
    ["Datos demostrativos", "Suba sus archivos CSV"]
)

# =============================
# Función para cargar datos de demostración (si no se suben archivos)
# =============================

def load_demo_data():
    demo_files = list(DATA_DIR.glob("*.csv"))
    return demo_files

warnings.filterwarnings("ignore")


####
if mode == "Datos demostrativos":
    demo_files = load_demo_data("data")
    uploaded_files = demo_files
else:
    uploaded_files = st.file_uploader(
        "Upload professorIDXXXXXX.csv files",
        type=["csv"],
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.warning("Suba al menos un archivo CSV.")
        st.stop()

####


# =============================
# CONFIGURACIÓN INSTITUCIONAL UNAM
# =============================

st.set_page_config(
    page_title="Sistema de Análisis Psicométrico y Estructural de Evaluación Docente - FESAc-UNAM",
    layout="wide"
)

UNAM_BLUE = "#003366"
UNAM_GOLD = "#C9A227"

st.markdown(f"""
<style>
.stApp {{
    background-color: {UNAM_BLUE};
    color: white;
}}
h1, h2, h3 {{
    color: white;
}}
.stButton>button {{
    background-color: {UNAM_GOLD};
    color: black;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

st.title("📊 SAPEED - FESAc-UNAM")
st.subheader("Plataforma exploratoria estructural y psicométrica para monitoreo institucional.")
st.markdown(
    "Carga archivos por profesor con respuestas estudiantiles a nivel ítem "
    "para calcular confiabilidad, promedios institucionales y relaciones estructurales entre dimensiones."
)



#############################################
# CAPA 1: Carga múltiple de archivos y procesamiento
# Nivel estudiante
#############################################

st.header("📂 Carga de archivos por profesor (Respuestas a nivel estudiante)")

uploaded_files = st.file_uploader(
    "Sube archivos tipo profesorIDXXXXXX.csv",
    type=["csv"],
    accept_multiple_files=True
)

# if not uploaded_files:
#     st.stop()

if not uploaded_files:
    st.info("No files uploaded. Loading synthetic demo data.")
    demo_files = load_demo_data("data")
    uploaded_files = demo_files
else:
    st.success("Using uploaded files.")


ITEM_RE = re.compile(r"^D(\d+)Q(\d+)$")

all_professor_data = []
professor_dimension_means = []

reference_dim_structure = None
reference_item_structure = None

for file in uploaded_files:

    filename = file.name
    prof_id = filename.replace(".csv", "")

    df = pd.read_csv(file)

    # -------------------------
    # Detectar columnas tipo DkQj
    # -------------------------
    item_cols = []
    dim_item_map = {}

    for col in df.columns:
        m = ITEM_RE.match(col)
        if m:
            item_cols.append(col)
            dim = f"D{m.group(1)}"
            dim_item_map.setdefault(dim, []).append(col)

    if not item_cols:
        st.warning(f"No se encontraron columnas válidas tipo DkQj en {filename}")
        continue

    # Ordenar dimensiones e ítems
    dimension_cols = sorted(dim_item_map.keys(), key=lambda x: int(x[1:]))

    for dim in dim_item_map:
        dim_item_map[dim] = sorted(
            dim_item_map[dim],
            key=lambda x: int(ITEM_RE.match(x).group(2))
        )

    # -------------------------
    # Verificar estructura consistente del instrumento
    # -------------------------
    current_item_structure = {
        dim: tuple(dim_item_map[dim]) for dim in dimension_cols
    }

    if reference_dim_structure is None:
        reference_dim_structure = dimension_cols
        reference_item_structure = current_item_structure
    else:
        if dimension_cols != reference_dim_structure:
            st.error(f"Estructura de dimensiones inconsistente en {filename}")
            st.stop()

        if current_item_structure != reference_item_structure:
            st.error(f"Estructura de ítems inconsistente en {filename}")
            st.stop()

    # -------------------------
    # Convertir ítems a numérico
    # -------------------------
    df[item_cols] = df[item_cols].apply(pd.to_numeric, errors="coerce")

    # -------------------------
    # Construcción de puntajes por dimensión (nivel estudiante)
    # -------------------------
    for dim in dimension_cols:
        df[dim] = df[dim_item_map[dim]].mean(axis=1)

    # -------------------------
    # Resumen por profesor (solo visualización)
    # -------------------------
    prof_means = df[dimension_cols].mean().to_dict()
    prof_means["ProfesorID"] = prof_id
    prof_means["N_estudiantes"] = len(df)

    professor_dimension_means.append(prof_means)

    # Agregar ID de profesor al nivel estudiante
    df["ProfesorID"] = prof_id
    all_professor_data.append(df)

# ---------------------------------
# Verificar que haya al menos un archivo válido
# ---------------------------------
if not all_professor_data:
    st.error("No se cargaron archivos válidos de profesores.")
    st.stop()

# ---------------------------------
# Consolidación institucional (nivel estudiante)
# ---------------------------------
institutional_df = pd.concat(all_professor_data, ignore_index=True)
professor_means_df = pd.DataFrame(professor_dimension_means)

dimension_cols = reference_dim_structure

st.success(f"Se cargaron {len(professor_means_df)} profesores correctamente.")
st.write("Dimensiones detectadas:", dimension_cols)



#############################################
# CAPA 2: Selección de Profesor + Análisis de Desempeño
# (Gráfica de Barras + Radar + Benchmark Institucional)
#############################################

st.header("👤 Comparación de Profesor")

selected_professor = st.selectbox(
    "Seleccionar Profesor",
    sorted(professor_means_df["ProfesorID"].unique())
)

# Datos a nivel estudiante del profesor seleccionado
df_selected = institutional_df[
    institutional_df["ProfesorID"] == selected_professor
]

# -------------------------------------------
# Promedio Institucional (ponderado a nivel estudiante)
# -------------------------------------------

st.header("🏛 Promedio Institucional (ponderado por estudiante)")

institutional_average = institutional_df[dimension_cols].mean()
st.write(institutional_average)

# -------------------------------------------
# Panel de Desempeño del Profesor
# -------------------------------------------

st.header("📊 Resumen de Desempeño del Profesor")

# -------------------------------------------
# Función auxiliar: Media + Intervalo de Confianza
# -------------------------------------------
def mean_ci(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return (float(s.mean()) if n == 1 else np.nan, np.nan)
    mean = float(s.mean())
    ci = 1.96 * float(s.std(ddof=1)) / np.sqrt(n)
    return mean, ci

# -------------------------------------------
# Cálculo de métricas
# -------------------------------------------
prof_means = []
inst_means = []
inst_upper = []
inst_lower = []
classification = []

for dim in dimension_cols:

    m_prof, _ = mean_ci(df_selected[dim])
    m_inst, ci_inst = mean_ci(institutional_df[dim])

    prof_means.append(m_prof)
    inst_means.append(m_inst)

    upper = m_inst + ci_inst if not np.isnan(ci_inst) else np.nan
    lower = m_inst - ci_inst if not np.isnan(ci_inst) else np.nan

    inst_upper.append(upper)
    inst_lower.append(lower)

    # Lógica de clasificación
    if np.isnan(m_prof) or np.isnan(lower):
        classification.append("NA")
    elif m_prof > upper:
        classification.append("Superior al institucional (Significativo)")
    elif m_prof < lower:
        classification.append("Inferior al institucional (Significativo)")
    else:
        classification.append("Dentro del rango institucional")

# -------------------------------------------
# Construcción de DataFrame comparativo
# -------------------------------------------
df_compare = pd.DataFrame({
    "Dimensión": dimension_cols,
    "Media Profesor": prof_means,
    "Media Institucional": inst_means,
    "Límite Inferior 95% Inst.": inst_lower,
    "Límite Superior 95% Inst.": inst_upper,
    "Clasificación": classification
})

# -------------------------------------------
# Layout en dos columnas
# -------------------------------------------
col1, col2 = st.columns(2)

# -------------------------------------------
# IZQUIERDA — Gráfica de Barras
# -------------------------------------------
with col1:

    st.subheader("📊 Comparación en Barras")

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=df_compare["Dimensión"],
        y=df_compare["Media Profesor"],
        name=selected_professor,
        marker_color=UNAM_BLUE
    ))

    fig_bar.add_trace(go.Bar(
        x=df_compare["Dimensión"],
        y=df_compare["Media Institucional"],
        name="Media Institucional",
        marker_color=UNAM_GOLD
    ))

    # Intervalos de confianza institucionales
    fig_bar.add_trace(go.Scatter(
        x=df_compare["Dimensión"],
        y=df_compare["Media Institucional"],
        mode="markers",
        error_y=dict(
            type="data",
            symmetric=False,
            array=df_compare["Límite Superior 95% Inst."] - df_compare["Media Institucional"],
            arrayminus=df_compare["Media Institucional"] - df_compare["Límite Inferior 95% Inst."],
            thickness=1.5,
            width=5
        ),
        marker=dict(color=UNAM_GOLD, size=4),
        showlegend=False
    ))

    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),  # Texto general negro
        xaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        ),
        legend=dict(
            font=dict(color="black")
        )
    )

    st.plotly_chart(fig_bar, width="stretch")

# -------------------------------------------
# DERECHA — Gráfica Radar
# -------------------------------------------
with col2:

    st.subheader("🕸 Comparación Radar")

    theta = dimension_cols + [dimension_cols[0]]

    prof_closed = prof_means + [prof_means[0]]
    inst_closed = inst_means + [inst_means[0]]
    upper_closed = inst_upper + [inst_upper[0]]
    lower_closed = inst_lower + [inst_lower[0]]

    fig_radar = go.Figure()

    # Banda IC institucional
    fig_radar.add_trace(go.Scatterpolar(
        r=upper_closed,
        theta=theta,
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=lower_closed,
        theta=theta,
        fill="tonext",
        fillcolor="rgba(201,162,39,0.2)",
        line=dict(width=0),
        name="IC 95% Institucional"
    ))

    # Media institucional
    fig_radar.add_trace(go.Scatterpolar(
        r=inst_closed,
        theta=theta,
        fill="toself",
        name="Media Institucional",
        line=dict(color=UNAM_GOLD)
    ))

    # Profesor seleccionado
    fig_radar.add_trace(go.Scatterpolar(
        r=prof_closed,
        theta=theta,
        fill="toself",
        name=selected_professor,
        line=dict(color=UNAM_BLUE)
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1,5])),
        showlegend=True
    )

    st.caption(
        "La gráfica radar compara las medias del profesor seleccionado "
        "contra la media institucional. El área sombreada representa "
        "el intervalo de confianza del 95% institucional."
    )

    st.plotly_chart(fig_radar, width="stretch")

# -------------------------------------------
# Tabla de interpretación
# -------------------------------------------
st.subheader("📋 Interpretación por Dimensión")

st.dataframe(
    df_compare[["Dimensión","Media Profesor","Media Institucional","Clasificación"]],
    width="stretch"
)

st.caption("""
Criterios de clasificación:
• Superior → la media del profesor supera el IC 95% institucional  
• Inferior → la media del profesor está por debajo del IC 95% institucional  
• Dentro del rango → sin diferencia estadísticamente relevante  
""")






#############################################
# CAPA 3: Análisis de Confiabilidad
# (Cronbach α, α Estratificada, por Profesor)
#############################################

st.header("🔬 Análisis de Confiabilidad")

# =====================================================
# FUNCIONES
# =====================================================

def cronbach_alpha(df_items):
    df_items = df_items.dropna()

    k = df_items.shape[1]
    if k < 2 or len(df_items) < 2:
        return np.nan

    var_items = df_items.var(axis=0, ddof=1)
    total_score = df_items.sum(axis=1)
    var_total = total_score.var(ddof=1)

    if var_total == 0 or np.isnan(var_total):
        return np.nan

    return (k / (k - 1)) * (1 - var_items.sum() / var_total)


def stratified_alpha(data, item_structure):

    sub_scores = {}
    sub_info = []

    for dim, cols in item_structure.items():

        df_items = data[list(cols)].dropna()

        if len(df_items) < 2:
            continue

        alpha_dim = cronbach_alpha(df_items)
        S = df_items.sum(axis=1)

        sub_scores[dim] = S
        sub_info.append({
            "Dimensión": dim,
            "Alpha": alpha_dim,
            "Varianza": S.var(ddof=1)
        })

    if not sub_info:
        return np.nan

    sub_df = pd.DataFrame(sub_info)
    S_mat = pd.DataFrame(sub_scores).dropna()

    if len(S_mat) < 2:
        return np.nan

    T = S_mat.sum(axis=1)
    var_T = T.var(ddof=1)

    numerator = (sub_df["Varianza"] * (1 - sub_df["Alpha"])).sum()

    if var_T == 0 or np.isnan(var_T):
        return np.nan

    return 1 - numerator / var_T


# =====================================================
# CONFIABILIDAD INSTITUCIONAL (MÉTRICAS SUPERIORES)
# =====================================================

col1, col2 = st.columns(2)

inst_alpha_strat = stratified_alpha(
    institutional_df,
    reference_item_structure
)

col1.metric(
    "α Estratificada Institucional",
    f"{inst_alpha_strat:.3f}" if not np.isnan(inst_alpha_strat) else "NA"
)

col2.metric(
    "Total de respuestas estudiantiles",
    f"{len(institutional_df)}"
)

# =====================================================
# TABLA DE CONFIABILIDAD POR DIMENSIÓN
# =====================================================

results = []

for dim in reference_dim_structure:

    item_cols = list(reference_item_structure[dim])

    # α institucional por dimensión
    inst_alpha = cronbach_alpha(institutional_df[item_cols])

    # α del profesor seleccionado
    df_prof = institutional_df[
        institutional_df["ProfesorID"] == selected_professor
    ]

    prof_alpha = cronbach_alpha(df_prof[item_cols])

    results.append([
        dim,
        inst_alpha,
        prof_alpha
    ])

rel_df = pd.DataFrame(
    results,
    columns=[
        "Dimensión",
        "α Institucional",
        f"α {selected_professor}"
    ]
)

st.subheader("Confiabilidad por Dimensión")

st.dataframe(rel_df, width="stretch")

st.caption("""
Guía general de interpretación de α de Cronbach:

• < 0.60 → Baja consistencia interna  
• 0.70 → Aceptable  
• 0.80 → Buena  
• 0.90+ → Muy alta (posible redundancia de ítems)
""")


#############################################
# CAPA 4: Análisis de Correlaciones (Limpio y robusto)
#############################################

st.header("📈 Análisis de Correlaciones")

dimension_cols = reference_dim_structure

#--------------------------------------------
# PASO 1 — Institucional (Nivel Estudiante, combinado)
#--------------------------------------------

st.subheader("🏛 Correlación Institucional (nivel estudiante, combinado)")

n_students_total = len(institutional_df)

student_corr = institutional_df[dimension_cols].corr(method="pearson")

fig_student = px.imshow(
    student_corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    aspect="auto",
)

# fig_student.update_layout(
#     coloraxis_colorbar=dict(title="r"),
#     paper_bgcolor="white",
#     plot_bgcolor="white"
# )

fig_student.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="black"),
    coloraxis_colorbar=dict(
        title=dict(text="r", font=dict(color="black")),
        tickfont=dict(color="black")
    ),
    xaxis=dict(
        title=dict(font=dict(color="black")),
        tickfont=dict(color="black")
    ),
    yaxis=dict(
        title=dict(font=dict(color="black")),
        tickfont=dict(color="black")
    )
)

# Forzar texto dentro de las celdas
fig_student.update_traces(
    textfont=dict(color="black")
)

st.plotly_chart(fig_student, width="stretch")

st.caption(f"Total de respuestas estudiantiles combinadas: N = {n_students_total}")

st.markdown("""
**Interpretación (nivel estudiante combinado):**
- Mide cómo co-varían las dimensiones considerando a **todos** los estudiantes.
- r alta → estudiantes que califican alto en una dimensión tienden a calificar alto en otra.
- r cercana a 0 → dimensiones relativamente independientes.
- r negativa → patrón inverso en la percepción estudiantil.
""")

#--------------------------------------------
# PASO 2 — Correlación a nivel Profesor (Nivel ecológico)
#--------------------------------------------

st.subheader("👨‍🏫 Correlación entre Profesores (nivel ecológico)")

if len(professor_means_df) > 1:

    professor_corr = professor_means_df[dimension_cols].corr(method="pearson")

    fig_prof = px.imshow(
        professor_corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
    )

    fig_prof.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        coloraxis_colorbar=dict(
            title=dict(text="r", font=dict(color="black")),
            tickfont=dict(color="black")            
        ),
        xaxis=dict(
            title=dict(font=dict(color="black")),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title=dict(font=dict(color="black")),
            tickfont=dict(color="black")
        )
    )

    fig_prof.update_traces(
        textfont=dict(color="black")
    )

    st.plotly_chart(fig_prof, width="stretch")

    st.caption(f"Número de profesores: {len(professor_means_df)}")

    st.markdown("""
    **Interpretación (nivel profesor / ecológico):**
    - Mide la asociación entre **promedios por profesor**.
    - Representa un patrón de nivel superior (institucional/estructural).
    - Las correlaciones ecológicas pueden diferir de las correlaciones a nivel estudiante.
    """)

else:
    st.info("Se requieren al menos 2 profesores para calcular correlación a nivel profesor.")

#--------------------------------------------
# PASO 3 — Correlación dentro del Profesor Seleccionado (Nivel estudiante)
#--------------------------------------------

st.subheader("🔍 Correlación dentro del Profesor Seleccionado")

df_selected = institutional_df[
    institutional_df["ProfesorID"] == selected_professor
]

n_students_selected = len(df_selected)

if n_students_selected > 5:

    corr_selected = df_selected[dimension_cols].corr(method="pearson")

    fig_sel = px.imshow(
        corr_selected,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto"
    )

    fig_sel.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        coloraxis_colorbar=dict(
            title=dict(text="r", font=dict(color="black")),
            tickfont=dict(color="black")            
        ),
        xaxis=dict(
            title=dict(font=dict(color="black")),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title=dict(font=dict(color="black")),
            tickfont=dict(color="black")
        )
    )
    fig_sel.update_traces(
        textfont=dict(color="black")
    )


    st.plotly_chart(fig_sel, width="stretch")

    st.caption(f"Estudiantes para {selected_professor}: N = {n_students_selected}")

else:
    st.warning(
        f"No hay suficientes estudiantes (N={n_students_selected}) "
        "para una matriz de correlación estable. Recomendado: N ≥ 10."
    )





#############################################
# CAPA 5: Regresión Ridge (Objetivo seleccionable)
# Modelado estructural entre dimensiones (nivel estudiante combinado)
#############################################

st.header("🧠 Regresión Ridge + Validación Cruzada (Red de Dimensiones)")

dimension_cols = reference_dim_structure  # ya definido en CAPA 1/2/4

# -----------------------------
# Seleccionar dimensión objetivo
# -----------------------------
target_dim = st.selectbox(
    "Selecciona la dimensión objetivo a modelar",
    dimension_cols
)

predictors = [d for d in dimension_cols if d != target_dim]

# Preparar marco de modelado
df_model = institutional_df[predictors + [target_dim]].dropna()

n_model = len(df_model)
st.caption(f"Filas usadas a nivel estudiante (después de eliminar NA): N = {n_model}")

if n_model < 20:
    st.warning("⚠️ Tamaño de muestra bajo puede producir estimaciones inestables (recomendado N ≥ 30).")

X = df_model[predictors]
y = df_model[target_dim]

# -----------------------------
# Parámetros de Validación Cruzada
# -----------------------------
k_folds = st.slider("Pliegues de validación cruzada (K)", 3, 10, 5)

if n_model <= k_folds:
    st.error(f"No hay suficientes observaciones (N={n_model}) para K={k_folds} pliegues.")
    st.stop()

lambda_grid = np.logspace(-4, 4, 50)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

mse_mean = []
mse_std = []
coef_by_lambda = {}  # para trayectorias/estabilidad

# -----------------------------
# Ciclo de Validación Cruzada
# -----------------------------
for lam in lambda_grid:

    fold_mse = []
    fold_coefs = []

    for tr, te in kf.split(X):

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(lam)))
        ])

        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])

        fold_mse.append(mean_squared_error(y.iloc[te], pred))
        fold_coefs.append(model.named_steps["ridge"].coef_)

    mse_mean.append(float(np.mean(fold_mse)))
    mse_std.append(float(np.std(fold_mse, ddof=1)) if len(fold_mse) > 1 else 0.0)
    coef_by_lambda[float(lam)] = np.array(fold_coefs)  # (pliegues, p)

best_idx = int(np.argmin(mse_mean))
best_lambda = float(lambda_grid[best_idx])
best_mse = float(mse_mean[best_idx])

# -----------------------------
# Ajuste final del modelo con λ*
# -----------------------------
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=best_lambda))
])

final_model.fit(X, y)

beta = pd.Series(
    final_model.named_steps["ridge"].coef_,
    index=predictors
)

# -----------------------------
# Mostrar resultados
# -----------------------------
st.subheader(f"Dimensión objetivo: {target_dim}")

col1, col2 = st.columns(2)
col1.metric("Mejor λ* (regularización Ridge)", f"{best_lambda:.6f}")
col2.metric("MSE_CV(λ*)", f"{best_mse:.4f}")

st.caption(
    "Interpretación: λ* controla la contracción (fuerza de regularización). "
    "MSE_CV(λ*) estima el error de predicción fuera de muestra (menor es mejor)."
)

st.subheader("Coeficientes (β) en λ*")
st.dataframe(
    beta.sort_values(ascending=False).to_frame("β"),
    width="stretch"
)

# -----------------------------
# Curva de regularización (MSE ± 1 desviación estándar)
# -----------------------------
fig_lambda = go.Figure()

fig_lambda.add_trace(go.Scatter(
    x=lambda_grid,
    y=mse_mean,
    mode="lines+markers",
    name="MSE promedio (CV)"
))

fig_lambda.add_trace(go.Scatter(
    x=lambda_grid,
    y=(np.array(mse_mean) + np.array(mse_std)),
    mode="lines",
    name="+1 desv. estándar",
    line=dict(dash="dot")
))

fig_lambda.add_trace(go.Scatter(
    x=lambda_grid,
    y=(np.array(mse_mean) - np.array(mse_std)),
    mode="lines",
    name="-1 desv. estándar",
    line=dict(dash="dot")
))

fig_lambda.update_layout(
    xaxis_type="log",
    title="Curva de Regularización (λ vs Error CV)",
    xaxis_title="λ (escala log)",
    yaxis_title="Error cuadrático medio (CV)",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color=UNAM_BLUE),
    legend=dict(font=dict(color=UNAM_BLUE), bgcolor="white")
)

st.plotly_chart(fig_lambda, width="stretch")

# -----------------------------
# Trayectorias de estabilidad de coeficientes vs λ
# -----------------------------
st.subheader("Trayectorias de estabilidad (β vs λ)")

fig_path = go.Figure()

for j, pred_name in enumerate(predictors):
    path = [coef_by_lambda[float(lam)][:, j].mean() for lam in lambda_grid]

    fig_path.add_trace(go.Scatter(
        x=lambda_grid,
        y=path,
        mode="lines",
        name=f"β({pred_name})"
    ))

fig_path.update_layout(
    xaxis_type="log",
    title="Estabilidad de coeficientes a través de λ",
    xaxis_title="λ (escala log)",
    yaxis_title="β promedio (entre pliegues)",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color=UNAM_BLUE),
    legend=dict(font=dict(color=UNAM_BLUE), bgcolor="white")
)

st.plotly_chart(fig_path, width="stretch")

# -----------------------------
# Aclaración α vs λ (evitar confusión)
# -----------------------------
with st.expander("ℹ️ Aclaración de parámetros: α psicométrica vs λ de Ridge"):
    st.markdown("""
**α psicométrica (Cronbach / α estratificada)**  
- Mide la confiabilidad del instrumento (consistencia interna).  
- Se calcula a partir de la estructura de covarianzas entre ítems.

**λ de Ridge (fuerza de regularización)**  
- Hiperparámetro que controla la contracción de coeficientes en regresión Ridge.  
- En scikit-learn se llama `alpha`, pero matemáticamente corresponde a λ.

**En este dashboard:**  
- α → confiabilidad (calidad del instrumento)  
- λ → regularización del modelo (modelado predictivo/estructural)
""")