# Despliegue — Predicción de Aprobación

## Archivos incluidos
- `app.py` — la app de Streamlit (interfaz morado / verde lima / naranja).
- `modelo_utils.py` — carga el `.pkl` y **descubre solo** qué columnas espera el modelo (no hay nombres de columnas escritos a mano; si el pipeline cambia, la app se adapta).
- `generar_datos_futuros_demo.py` — genera un CSV sintético con el formato exacto que pide el modelo, para probar la pestaña de predicción por lote antes de tener datos reales.
- `requirements.txt` — dependencias.
- `.streamlit/config.toml` — tema de colores para Streamlit Cloud.

## Antes de desplegar
1. Copia tu `mejor_modelo.pkl` (el que descargaste de Colab) a esta misma carpeta.
2. (Opcional, para probar la pestaña de lote) corre, en el mismo entorno donde tengas `mejor_modelo.pkl`:
   ```bash
   pip install -r requirements.txt
   python generar_datos_futuros_demo.py
   ```
   Esto genera `datos_futuros_demo.csv`. **Es sintético** — sirve para validar que el formato encaje, no para sacar conclusiones reales.
3. Prueba localmente:
   ```bash
   streamlit run app.py
   ```

## Desplegar en Streamlit Community Cloud (como en la clase de Github-Streamlit)
1. Crea un repositorio en GitHub y sube **todos** los archivos de esta carpeta, incluyendo `mejor_modelo.pkl` (revisa que no sea información confidencial antes de subirlo).
2. Ve a streamlit.io/cloud → *Deploy a public app from GitHub*.
3. Selecciona el repositorio, la rama y `app.py` como archivo principal → *Deploy*.
4. Para actualizar después: edita el archivo en GitHub → *Commit changes* → en Streamlit Cloud usa *Clear cache and restart* si cambiaste el modelo.

## Qué revisar si algo no carga
- Si la app no encuentra el modelo: usa el cargador de `.pkl` en la barra lateral (funciona igual sin tener el archivo en el repo).
- Si las etiquetas de la predicción salen como "0"/"1" en vez de texto: tu `.pkl` no trae guardado `mapa_target`. Ajusta manualmente "Clase 0 significa" / "Clase 1 significa" en la barra lateral — por defecto asume 0=No aprueba, 1=Aprueba, **verifica que coincida con tu codificación real**.
- Si sube un archivo de lote y marca "Faltan columnas": el archivo debe tener exactamente los nombres de columna que muestra el aviso azul de esa pestaña (los mismos con los que se entrenó el modelo).
