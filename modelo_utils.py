"""
modelo_utils.py
----------------
Utilidades compartidas por app.py y generar_datos_futuros_demo.py.

La idea central: en vez de escribir a mano la lista de columnas que espera
el modelo (frágil: si el pipeline cambia, la app queda desactualizada),
estas funciones abren el archivo .pkl y le preguntan directamente al
ColumnTransformer ya ajustado qué columnas usó, cuáles son categóricas,
qué categorías válidas aprendió cada una, y qué valor usa el imputer por
defecto. Así la app se auto-adapta al pipeline real que se suba.
"""

import io
import joblib
import pandas as pd


def extraer_pipeline(objeto):
    """Acepta un Pipeline/ImbPipeline ya ajustado, un GridSearchCV ya ajustado,
    o un diccionario que contenga alguno de esos bajo una llave conocida
    (por ejemplo el 'metadata_despliegue' que arma el notebook del curso).
    """
    if hasattr(objeto, "named_steps"):
        return objeto
    if hasattr(objeto, "best_estimator_"):
        return objeto.best_estimator_
    if isinstance(objeto, dict):
        for llave in ("pipeline", "best_estimator_", "mejor_pipeline", "model", "modelo"):
            candidato = objeto.get(llave)
            if candidato is not None:
                try:
                    return extraer_pipeline(candidato)
                except ValueError:
                    continue
    raise ValueError(
        "No se pudo identificar un pipeline de scikit-learn/imblearn dentro del "
        "archivo .pkl. Se esperaba un Pipeline (con .named_steps), un "
        "GridSearchCV ya ajustado (con .best_estimator_), o un diccionario que "
        "contenga alguno de esos bajo una llave como 'pipeline', 'model' o "
        "'modelo'."
    )


def extraer_metadata(objeto):
    """Si el .pkl es un diccionario tipo 'metadata_despliegue', recupera el
    mapa para traducir la predicción (0/1) a la etiqueta original. Si no
    existe, la app pedirá esas etiquetas manualmente en la barra lateral.
    """
    if isinstance(objeto, dict):
        return {
            "mapa_target": objeto.get("mapa_target"),
            "nombre_modelo": objeto.get("nombre_modelo", "Modelo cargado"),
        }
    return {"mapa_target": None, "nombre_modelo": "Modelo cargado"}


def _buscar_column_transformer(pipeline):
    if "encode_smote" in pipeline.named_steps:
        return pipeline.named_steps["encode_smote"]
    for step in pipeline.named_steps.values():
        if hasattr(step, "transformers_"):
            return step
    raise ValueError(
        "No se encontró un ColumnTransformer ya ajustado dentro del pipeline; "
        "no es posible inferir automáticamente las columnas de entrada."
    )


def introspeccionar_columnas(pipeline):
    """Reconstruye, a partir del ColumnTransformer ya ajustado:
      - qué columnas son categóricas y cuáles numéricas
      - qué categorías válidas aprendió cada columna categórica
      - el valor que usa el imputer (mediana/moda) como valor por defecto
    """
    ct = _buscar_column_transformer(pipeline)

    columnas_categoricas, columnas_numericas = [], []
    categorias, valores_defecto = {}, {}

    for nombre, transformador, columnas in ct.transformers_:
        columnas = list(columnas)
        pasos = getattr(transformador, "named_steps", {})
        encoder = pasos.get("ordinal") or pasos.get("onehot") or pasos.get("encoder")
        imputer = pasos.get("imputer")

        es_categorica = encoder is not None or nombre == "cat"

        if es_categorica:
            columnas_categoricas.extend(columnas)
            if encoder is not None and hasattr(encoder, "categories_"):
                for col, cats in zip(columnas, encoder.categories_):
                    categorias[col] = [c for c in cats if pd.notna(c)]
            if imputer is not None and hasattr(imputer, "statistics_"):
                for col, val in zip(columnas, imputer.statistics_):
                    valores_defecto[col] = val
        else:
            columnas_numericas.extend(columnas)
            if imputer is not None and hasattr(imputer, "statistics_"):
                for col, val in zip(columnas, imputer.statistics_):
                    valores_defecto[col] = float(val)

    if not columnas_categoricas and not columnas_numericas:
        raise ValueError(
            "El ColumnTransformer no reportó columnas (transformers_ vacío). "
            "¿El pipeline realmente fue ajustado (fit) antes de guardarlo?"
        )

    return {
        "categoricas": columnas_categoricas,
        "numericas": columnas_numericas,
        "categorias": categorias,
        "valores_defecto": valores_defecto,
        "todas": columnas_categoricas + columnas_numericas,
    }


def cargar_desde_bytes(contenido_bytes):
    objeto = joblib.load(io.BytesIO(contenido_bytes))
    pipeline = extraer_pipeline(objeto)
    metadata = extraer_metadata(objeto)
    info_columnas = introspeccionar_columnas(pipeline)
    return pipeline, metadata, info_columnas


def cargar_desde_ruta(ruta):
    objeto = joblib.load(ruta)
    pipeline = extraer_pipeline(objeto)
    metadata = extraer_metadata(objeto)
    info_columnas = introspeccionar_columnas(pipeline)
    return pipeline, metadata, info_columnas


def traducir_prediccion(valor, mapa_target, etiqueta_0="No aprueba", etiqueta_1="Aprueba"):
    """Traduce el 0/1 crudo del modelo a una etiqueta legible.
    Usa el mapa guardado en el .pkl si existe; si no, usa las etiquetas
    manuales que se definan en la app (por defecto 0=No aprueba, 1=Aprueba,
    AJUSTAR si la polaridad real de su variable Aprobado es la contraria).
    """
    if mapa_target:
        try:
            if valor in mapa_target:
                return mapa_target[valor]
        except TypeError:
            pass
        if int(valor) in mapa_target:
            return mapa_target[int(valor)]
    if int(valor) == 0:
        return etiqueta_0
    if int(valor) == 1:
        return etiqueta_1
    return str(valor)
