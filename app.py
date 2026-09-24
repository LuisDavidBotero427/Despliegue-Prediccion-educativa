"""
app.py — Predicción de Aprobación de Curso
--------------------------------------------
Despliegue en Streamlit del mejor modelo predictivo (Random Forest) del
proyecto de Minería de Datos. No asume nombres de columnas fijos: los
descubre automáticamente inspeccionando el pipeline guardado en el .pkl
(ver modelo_utils.py), así que si el pipeline cambia, la app se adapta sola.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

import modelo_utils as mu

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y PALETA DE COLORES
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Aprobación",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MORADO = "#7C3AED"
MORADO_OSCURO = "#5B21B6"
MORADO_CLARO = "#EDE9FE"
VERDE_LIMA = "#A3E635"
VERDE_LIMA_OSCURO = "#65A30D"
NARANJA = "#FB923C"
NARANJA_OSCURO = "#C2410C"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, #FAF5FF 0%, #FFFFFF 35%);
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {MORADO_OSCURO} 0%, {MORADO} 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
        color: #1F2937 !important;
    }}

    .encabezado-app {{
        background: linear-gradient(120deg, {MORADO} 0%, {MORADO_OSCURO} 100%);
        padding: 1.6rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.25);
    }}
    .encabezado-app h1 {{
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
    }}
    .encabezado-app p {{
        margin: 0.3rem 0 0 0;
        opacity: 0.92;
        font-size: 1.02rem;
    }}
    .franja-acento {{
        height: 6px;
        border-radius: 6px;
        background: linear-gradient(90deg, {VERDE_LIMA} 0%, {NARANJA} 100%);
        margin: -1rem 0 1.6rem 0;
    }}

    .stButton > button, .stFormSubmitButton > button {{
        background: {MORADO} !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.4rem !important;
        transition: transform 0.08s ease-in-out;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        background: {MORADO_OSCURO} !important;
        transform: translateY(-1px);
    }}

    .stDownloadButton > button {{
        background: {VERDE_LIMA_OSCURO} !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {MORADO_CLARO};
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 700;
        color: {MORADO_OSCURO};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {MORADO} !important;
        color: white !important;
    }}

    .tarjeta-resultado {{
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }}
    .tarjeta-aprueba {{
        background: linear-gradient(120deg, {VERDE_LIMA_OSCURO} 0%, {VERDE_LIMA} 100%);
    }}
    .tarjeta-riesgo {{
        background: linear-gradient(120deg, {NARANJA_OSCURO} 0%, {NARANJA} 100%);
    }}
    .tarjeta-resultado h2 {{
        margin: 0 0 0.3rem 0;
        font-size: 1.5rem;
    }}

    .barra-proba-fondo {{
        background-color: rgba(255,255,255,0.35);
        border-radius: 999px;
        height: 14px;
        width: 100%;
        margin-top: 0.6rem;
        overflow: hidden;
    }}
    .barra-proba-relleno {{
        background-color: white;
        height: 100%;
        border-radius: 999px;
    }}

    .caja-info {{
        background-color: {MORADO_CLARO};
        border-left: 5px solid {MORADO};
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        font-size: 0.92rem;
        color: #3B0764;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="encabezado-app">
        <h1>🎓 Predicción de Aprobación de Curso</h1>
        <p>Modelo Random Forest — Proyecto de Minería de Datos</p>
    </div>
    <div class="franja-acento"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CARGA DEL MODELO
# ---------------------------------------------------------------------------
RUTA_MODELO_DEFECTO = "mejor_modelo.pkl"

with st.sidebar:
    st.markdown("## ⚙️ Modelo")
    archivo_modelo = st.file_uploader(
        "Cargar archivo .pkl (opcional si ya está junto a la app)",
        type=["pkl"],
    )
    st.markdown("---")
    st.markdown("## 🏷️ Etiquetas de la predicción")
    st.caption(
        "Se usan solo si el .pkl no trae ya un mapa_target guardado. "
        "Verifica que coincidan con la polaridad real de tu variable Aprobado."
    )
    etiqueta_0 = st.text_input("Clase 0 significa:", value="No aprueba")
    etiqueta_1 = st.text_input("Clase 1 significa:", value="Aprueba")


@st.cache_resource(show_spinner="Cargando modelo...")
def _cargar_desde_bytes_cacheado(contenido_bytes):
    return mu.cargar_desde_bytes(contenido_bytes)


@st.cache_resource(show_spinner="Cargando modelo...")
def _cargar_desde_ruta_cacheada(ruta):
    return mu.cargar_desde_ruta(ruta)


try:
    if archivo_modelo is not None:
        pipeline, metadata, info_cols = _cargar_desde_bytes_cacheado(archivo_modelo.getvalue())
    elif os.path.exists(RUTA_MODELO_DEFECTO):
        pipeline, metadata, info_cols = _cargar_desde_ruta_cacheada(RUTA_MODELO_DEFECTO)
    else:
        st.warning(
            f"No se encontró **{RUTA_MODELO_DEFECTO}** junto a la app. "
            "Sube el archivo .pkl desde la barra lateral para continuar."
        )
        st.stop()
except Exception as error:
    st.error(f"No se pudo cargar el modelo: {error}")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.success(f"Modelo cargado: **{metadata.get('nombre_modelo', 'Random Forest')}**")
    with st.expander("Columnas detectadas"):
        st.write("**Categóricas:**", info_cols["categoricas"])
        st.write("**Numéricas:**", info_cols["numericas"])

# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE VISUALIZACIÓN
# ---------------------------------------------------------------------------
def _clase_riesgo(etiqueta_texto):
    """Heurística simple para decidir el color de la tarjeta de resultado."""
    texto = etiqueta_texto.strip().lower()
    palabras_riesgo = ["no aprueba", "reprobado", "reprueba", "no", "0", "riesgo"]
    return any(p == texto or texto.startswith(p) for p in palabras_riesgo)


def mostrar_tarjeta_resultado(etiqueta, probabilidad_riesgo=None):
    es_riesgo = _clase_riesgo(str(etiqueta))
    clase_css = "tarjeta-riesgo" if es_riesgo else "tarjeta-aprueba"
    icono = "⚠️" if es_riesgo else "✅"

    barra_html = ""
    if probabilidad_riesgo is not None:
        pct = round(probabilidad_riesgo * 100, 1)
        barra_html = f"""
        <div class="barra-proba-fondo">
            <div class="barra-proba-relleno" style="width:{pct}%;"></div>
        </div>
        <p style="margin-top:0.4rem; font-weight:600;">Probabilidad de no aprobar: {pct}%</p>
        """

    st.markdown(
        f"""
        <div class="tarjeta-resultado {clase_css}">
            <h2>{icono} {etiqueta}</h2>
            {barra_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def construir_fila_desde_formulario(info_cols, valores):
    return pd.DataFrame([valores])[info_cols["todas"]]


def _probabilidad_de_clase(pipeline, X_fila, valor_clase_riesgo=0):
    if not hasattr(pipeline, "predict_proba"):
        return None
    proba = pipeline.predict_proba(X_fila)
    clases = list(getattr(pipeline, "classes_", []))
    if valor_clase_riesgo in clases:
        idx = clases.index(valor_clase_riesgo)
    else:
        idx = 0
    return proba[:, idx]


# ---------------------------------------------------------------------------
# TABS: PREDICCIÓN INDIVIDUAL / PREDICCIÓN POR LOTE
# ---------------------------------------------------------------------------
tab_individual, tab_lote = st.tabs(["🧍 Predicción individual", "📁 Predicción por lote (datos futuros)"])

# ----- TAB 1: INDIVIDUAL -----
with tab_individual:
    st.markdown("#### Ingresa los datos del registro")
    st.markdown(
        '<div class="caja-info">Los campos se generaron automáticamente a partir '
        'de las columnas que el modelo aprendió a usar. Las categóricas muestran '
        'únicamente valores que el modelo ya conoce.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("form_individual"):
        columnas_ui = st.columns(2)
        valores = {}
        campos = info_cols["categoricas"] + info_cols["numericas"]

        for i, campo in enumerate(campos):
            contenedor = columnas_ui[i % 2]
            if campo in info_cols["categoricas"]:
                opciones = info_cols["categorias"].get(campo, [])
                if opciones:
                    valor_defecto = info_cols["valores_defecto"].get(campo, opciones[0])
                    idx_defecto = opciones.index(valor_defecto) if valor_defecto in opciones else 0
                    valores[campo] = contenedor.selectbox(campo, options=opciones, index=idx_defecto)
                else:
                    valores[campo] = contenedor.text_input(campo)
            else:
                valor_defecto = float(info_cols["valores_defecto"].get(campo, 0.0))
                valores[campo] = contenedor.number_input(campo, value=valor_defecto)

        enviado = st.form_submit_button("🔮 Predecir")

    if enviado:
        fila = construir_fila_desde_formulario(info_cols, valores)
        prediccion_cruda = pipeline.predict(fila)[0]
        etiqueta = mu.traducir_prediccion(
            prediccion_cruda, metadata.get("mapa_target"), etiqueta_0, etiqueta_1
        )
        proba_riesgo = _probabilidad_de_clase(pipeline, fila, valor_clase_riesgo=0)
        mostrar_tarjeta_resultado(
            etiqueta, proba_riesgo[0] if proba_riesgo is not None else None
        )

# ----- TAB 2: LOTE -----
with tab_lote:
    st.markdown("#### Sube un archivo con datos de futuras inscripciones")
    st.markdown(
        f'<div class="caja-info">El archivo debe incluir estas columnas: '
        f'{", ".join(info_cols["todas"])}</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    archivo_datos = st.file_uploader(
        "Archivo CSV o Excel con datos futuros", type=["csv", "xlsx"], key="datos_futuros"
    )

    if archivo_datos is not None:
        try:
            if archivo_datos.name.lower().endswith(".csv"):
                df_futuro = pd.read_csv(archivo_datos)
            else:
                df_futuro = pd.read_excel(archivo_datos)
        except Exception as error:
            st.error(f"No se pudo leer el archivo: {error}")
            st.stop()

        faltantes = set(info_cols["todas"]) - set(df_futuro.columns)
        if faltantes:
            st.error(f"Faltan columnas requeridas: {sorted(faltantes)}")
        else:
            X_nuevo = df_futuro[info_cols["todas"]]
            predicciones = pipeline.predict(X_nuevo)
            proba_riesgo = _probabilidad_de_clase(pipeline, X_nuevo, valor_clase_riesgo=0)

            df_resultado = df_futuro.copy()
            df_resultado["Prediccion"] = [
                mu.traducir_prediccion(p, metadata.get("mapa_target"), etiqueta_0, etiqueta_1)
                for p in predicciones
            ]
            if proba_riesgo is not None:
                df_resultado["Probabilidad_No_Aprobar"] = np.round(proba_riesgo, 4)

            n_riesgo = sum(_clase_riesgo(str(e)) for e in df_resultado["Prediccion"])
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Registros evaluados", len(df_resultado))
            col_m2.metric("En riesgo (predicho)", n_riesgo)
            col_m3.metric(
                "% en riesgo", f"{(n_riesgo / len(df_resultado) * 100):.1f}%" if len(df_resultado) else "0%"
            )

            st.dataframe(df_resultado, use_container_width=True)

            csv_bytes = df_resultado.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar resultados (CSV)",
                data=csv_bytes,
                file_name="predicciones_aprobacion.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "¿No tienes un archivo a mano? Genera uno de prueba con "
            "`generar_datos_futuros_demo.py` (incluido junto a esta app)."
        )

st.markdown("---")
st.caption(
    "⚠️ Este modelo es una herramienta de apoyo a la decisión, no un diagnóstico "
    "definitivo. Precisión y recall de la clase de riesgo están documentados en "
    "el informe del proyecto — úsalo para priorizar seguimiento, no como filtro "
    "automático de estudiantes."
)
