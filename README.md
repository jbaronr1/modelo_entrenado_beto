# Clasificación de contratos con BETO

Modelo de clasificación de texto que asigna una categoría a cada contrato a partir de su **objeto contractual**. Está construido sobre [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) (`dccuchile/bert-base-spanish-wwm-cased`), un BERT entrenado en español, ajustado (fine-tuning) para **55 categorías** de contratos de prestación de servicios.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `2025_octubre_optimizado_contratación_pipeline_final.py` | Pipeline completo (exportado de Google Colab): entrenamiento, evaluación y predicción sobre contratos nuevos. |
| `config.json` | Configuración del modelo (`BertForSequenceClassification`, 55 etiquetas). |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` | Tokenizador. |
| `training_args.bin` | Argumentos usados en el entrenamiento. |

> **Nota:** los pesos del modelo (`model.safetensors`, ~440 MB) **no están incluidos** porque superan el límite de 100 MB por archivo de GitHub. Para usar el modelo entrenado, copia ese archivo en la raíz del repositorio.

## Cómo funciona el pipeline

1. **Datos de entrenamiento:** `datos_clasificados.csv` (separador `;`, codificación `latin1`) con las columnas `objeto` y `clasificacion_del_contrato`.
2. **Preparación:** las etiquetas se codifican con `LabelEncoder` y los datos se dividen 90 % entrenamiento / 10 % validación (estratificado, `random_state=42`).
3. **Entrenamiento:** BETO con `max_length=128`, 2 épocas, batch de 16, `weight_decay=0.01` y `fp16`.
4. **Evaluación:** `classification_report` de scikit-learn sobre el conjunto de validación.
5. **Predicción:** se cargan textos sin clasificar (`2025.csv`, columna `Objeto`) y se clasifican con un `TextClassificationPipeline`. El resultado (texto, predicción y score) se exporta a CSV y Excel.

## Uso rápido para predecir

```python
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline

ruta = "."  # carpeta con model.safetensors y el tokenizador
model = BertForSequenceClassification.from_pretrained(ruta, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(ruta, local_files_only=True)

clasificador = TextClassificationPipeline(model=model, tokenizer=tokenizer)
print(clasificador("Prestar servicios profesionales para apoyar la gestión contractual de la entidad"))
```

El `config.json` guardado tiene etiquetas genéricas (`LABEL_0` … `LABEL_54`). El script asigna los nombres reales de las 55 categorías (lista `etiquetas_reales`) antes de predecir.

## Requisitos

```bash
pip install -r requirements.txt
```

El script fue escrito para **Google Colab** (usa `google.colab.files` y Google Drive). Para ejecutarlo localmente hay que reemplazar esas partes por lectura de archivos locales y quitar la línea `!pip install`.
