"""
generar_datos_futuros_demo.py
-------------------------------
Genera un CSV de ejemplo con el formato EXACTO que espera el pipeline
guardado en mejor_modelo.pkl, para poder probar la pestaña "Predicción
por lote" de la app sin tener todavía un archivo real de inscripciones
futuras.

IMPORTANTE — esto es un set de datos SINTÉTICO para verificar que el
formato/las columnas encajen con la app. Los valores numéricos se generan
con ruido alrededor de la mediana que el propio imputer aprendió en
entrenamiento, y los categóricos se muestrean al azar entre las categorías
que el modelo ya conoce. No representan estudiantes reales ni deben usarse
para nada distinto de probar el despliegue. Reemplázalo por datos reales
de inscripciones futuras antes de usar la app en producción.

Uso (en el mismo entorno donde está mejor_modelo.pkl, por ejemplo Colab):
    python generar_datos_futuros_demo.py
"""

import numpy as np
import pandas as pd

import modelo_utils as mu

RUTA_MODELO = "mejor_modelo.pkl"
RUTA_SALIDA = "datos_futuros_demo.csv"
N_REGISTROS = 30
SEMILLA = 42


def generar(info_cols, n_registros, semilla=42):
    rng = np.random.default_rng(semilla)
    datos = {}

    for col in info_cols["categoricas"]:
        opciones = info_cols["categorias"].get(col)
        if opciones:
            datos[col] = rng.choice(opciones, size=n_registros)
        else:
            defecto = info_cols["valores_defecto"].get(col, "")
            datos[col] = [defecto] * n_registros

    for col in info_cols["numericas"]:
        centro = info_cols["valores_defecto"].get(col, 0.0)
        # Ruido proporcional al valor central; con piso para columnas centradas en 0.
        dispersion = max(abs(centro) * 0.25, 1.0)
        valores = rng.normal(loc=centro, scale=dispersion, size=n_registros)
        datos[col] = np.round(valores, 2)

    df = pd.DataFrame(datos)[info_cols["todas"]]
    return df


if __name__ == "__main__":
    _, _, info_cols = mu.cargar_desde_ruta(RUTA_MODELO)
    df_demo = generar(info_cols, N_REGISTROS, SEMILLA)
    df_demo.to_csv(RUTA_SALIDA, index=False)
    print(f"Generado {RUTA_SALIDA} con {len(df_demo)} registros sintéticos.")
    print("Columnas:", list(df_demo.columns))
    print(df_demo.head())
