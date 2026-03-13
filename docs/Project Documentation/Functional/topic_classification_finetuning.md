# Claude Code — Prompt: Topic Classification Fine-Tuning
# Proyecto: mistral-lens | Tarea: MI-FT01 a MI-FT07
# Fecha: Marzo 2026

---

## CONTEXTO DEL PROYECTO

Estás trabajando en **mistral-lens**, una aplicación de document intelligence construida exclusivamente sobre modelos Mistral AI. La app extrae texto, topic y respuestas Q&A de documentos PDF y evalúa la calidad de la extracción.

### Estructura del proyecto (ya existente)

```
mistral-lens/
  app/
    main.py          # Gradio UI — 3 tabs
    extractor.py     # Llamadas a la API de Mistral (OCR, topic, Q&A)
    metrics.py       # WER, ROUGE-L, LLM-judge
    prompts.py       # Templates de prompts (versionados)
  scripts/
    run_evaluation.py  # Script de evaluación batch
  data/
    .gitkeep
  results/
    dev_eval.csv     # Resultados precalculados del dev set
  .env               # MISTRAL_API_KEY (ya configurado)
  requirements.txt
```

### Stack técnico
- Python 3.11+
- mistralai SDK
- datasets (HuggingFace) — carga repliqa
- jiwer — WER
- rouge-score — ROUGE-L
- gradio — UI
- python-dotenv

### Dataset
- **repliqa** de ServiceNow (HuggingFace: `ServiceNow/repliqa`)
- Split `repliqa_0` — 50 docs usados para desarrollo y tuning de prompts
- Split `repliqa_3` — 15 docs holdout, nunca vistos durante desarrollo
- Campos relevantes: `article` (texto), `topic` (categoría), `question`, `answer`

### Situación actual del topic
El pipeline actual genera topics con `mistral-large-latest` usando un prompt abierto. Produce etiquetas específicas del documento ("Urban Flood Resilience in Karachi") mientras el ground truth de repliqa usa categorías genéricas ("Neighborhood Stories"). Esto causa un mismatch de granularidad que baja el topic score a 3.1/5 en el LLM-judge.

### Objetivo de esta tarea
Implementar fine-tuning supervisado del clasificador de topic para que:
1. Los topics devueltos pertenezcan siempre al conjunto de categorías del dataset
2. La métrica cambie de LLM-judge (1-5) a exact match accuracy (%)
3. El accuracy resultante supere el 80% sobre repliqa_3

---

## TAREAS A EJECUTAR — EN ORDEN ESTRICTO

---

### MI-FT01 — Inspeccionar distribución de topics en dev y holdout

**Crea el script** `scripts/inspect_topics.py`:

```python
# Carga repliqa_0 y repliqa_3
# Para cada split, imprime:
#   - Número total de documentos
#   - Lista de categorías únicas con su frecuencia (value_counts)
#   - Categorías que están en repliqa_3 pero NO en repliqa_0 (gap crítico)
#   - Categorías que están en repliqa_0 pero NO en repliqa_3
#   - Número mínimo de ejemplos por categoría en repliqa_0
```

**Ejecuta** el script y guarda el output completo en `results/topic_distribution.txt`.

**Criterio de éxito:** El script corre sin errores y el output muestra la distribución completa. Documenta aquí si hay gaps (categorías en eval no cubiertas en dev).

---

### MI-FT02 — Ampliar el dev set si hay gaps de cobertura

**Condición:** Si en MI-FT01 encontraste categorías en repliqa_3 que tienen 0 ejemplos en repliqa_0, necesitas cubrir esos gaps.

**Acción:**
- Carga repliqa_1 y repliqa_2 (misma fuente HuggingFace)
- Para cada categoría con gap, toma hasta 10 documentos de esos splits
- Combina con repliqa_0 para crear el dataset de entrenamiento ampliado
- El dataset ampliado NO debe incluir ningún documento de repliqa_3 (es el holdout sagrado)

**Si no hay gaps:** salta a MI-FT03 directamente usando solo repliqa_0.

**Criterio de éxito:** Todas las categorías que aparecen en repliqa_3 tienen al menos 5 ejemplos en el dataset de entrenamiento.

---

### MI-FT03 — Construir el dataset de fine-tuning

**Crea el script** `scripts/build_finetune_dataset.py`:

El script genera `data/topic_finetune_train.jsonl` con este formato exacto por cada documento del dev set:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Classify the following document into exactly one of these categories:\n{CATEGORY_LIST}\n\nReturn only the exact category name, nothing else.\n\nDocument:\n{ARTICLE_TEXT_TRUNCATED_TO_2000_CHARS}"
    },
    {
      "role": "assistant",
      "content": "{GROUND_TRUTH_TOPIC}"
    }
  ]
}
```

Donde:
- `{CATEGORY_LIST}` — lista de todas las categorías únicas del dataset de entrenamiento, una por línea
- `{ARTICLE_TEXT_TRUNCATED_TO_2000_CHARS}` — los primeros 2000 caracteres del artículo (suficiente para clasificación, controla coste de tokens)
- `{GROUND_TRUTH_TOPIC}` — el campo `topic` exacto del documento en repliqa

**Además**, genera `data/topic_finetune_val.jsonl` con los mismos 15 documentos de repliqa_3 en el mismo formato (para validación durante el fine-tuning, SIN usarlos en entrenamiento).

**Guarda** también `data/category_list.txt` con las categorías únicas, una por línea — se usará después para actualizar el prompt.

**Criterio de éxito:**
- `topic_finetune_train.jsonl` existe y tiene al menos 40 líneas válidas
- `topic_finetune_val.jsonl` existe y tiene 15 líneas
- Cada línea es JSON válido con el formato correcto
- Imprime estadísticas: número de ejemplos por categoría

---

### MI-FT04 — Lanzar el job de fine-tuning en la API de Mistral

**Crea el script** `scripts/run_finetune.py`:

```python
# 1. Sube topic_finetune_train.jsonl a la API de Mistral
#    client.files.upload(file={"file_name": ..., "content": ...}, purpose="fine-tune")
#
# 2. Sube topic_finetune_val.jsonl como validation file
#
# 3. Lanza el job:
#    client.fine_tuning.jobs.create(
#      model="open-mistral-7b",   # modelo base — económico para clasificación
#      training_files=[{"file_id": train_file_id, "purpose": "fine-tune"}],
#      validation_files=[{"file_id": val_file_id, "purpose": "fine-tune"}],
#      hyperparameters={
#        "training_steps": 100,
#        "learning_rate": 0.0001
#      },
#      suffix="topic-classifier"
#    )
#
# 4. Imprime el job.id y job.status
# 5. Guarda el job_id en data/finetune_job.txt para el siguiente paso
```

**Ejecuta** el script.

**Criterio de éxito:** El script imprime un job_id válido y status "QUEUED" o "RUNNING". El job_id está guardado en `data/finetune_job.txt`.

---

### MI-FT05 — Monitorizar el job hasta completar

**Crea el script** `scripts/check_finetune_status.py`:

```python
# Lee el job_id de data/finetune_job.txt
# Llama a client.fine_tuning.jobs.retrieve(job_id)
# Imprime: status, trained_tokens, loss (train y val si disponibles)
# Si status == "SUCCESS": imprime el fine_tuned_model ID
# Guarda el fine_tuned_model ID en data/finetuned_model.txt
```

**Ejecuta** el script en loop hasta que el status sea SUCCESS o FAILED.
- Si SUCCESS: continúa a MI-FT06
- Si FAILED: revisa los logs del job e informa del error antes de continuar

**Criterio de éxito:** `data/finetuned_model.txt` contiene el model ID del modelo fine-tuned (formato: `ft:open-mistral-7b:...`).

---

### MI-FT06 — Actualizar el prompt y el extractor

**Actualiza `app/prompts.py`** — cambia el prompt de topic classification:

```python
# ANTES (prompt abierto):
TOPIC_PROMPT = """Given this document, what is the main topic?
Be specific and descriptive.

Document:
{text}"""

# DESPUÉS (prompt con taxonomía fija):
TOPIC_PROMPT = """Classify the following document into exactly one of these categories:
{categories}

Return only the exact category name, nothing else. Do not add explanations.

Document:
{text}"""

# Carga las categorías desde data/category_list.txt al inicializar
# Las categorías se inyectan en {categories} en cada llamada
```

**Actualiza `app/extractor.py`** — función `extract_topic()`:
- Lee `data/category_list.txt` al importar el módulo (una sola vez)
- Lee `data/finetuned_model.txt` para obtener el model ID
- Usa el modelo fine-tuned en lugar de `mistral-large-latest` para el topic
- Inyecta la lista de categorías en el prompt
- Si `finetuned_model.txt` no existe, usa `mistral-large-latest` como fallback con el prompt con taxonomía (para no romper el sistema si el fine-tuning no se ha corrido)

**Criterio de éxito:** `app/extractor.py` corre sin errores de importación. La función `extract_topic()` usa el modelo fine-tuned y el prompt con taxonomía.

---

### MI-FT07 — Re-evaluar sobre repliqa_3 y actualizar Business Case

**Actualiza `app/metrics.py`** — añade o verifica la función `compute_topic_accuracy()`:

```python
def compute_topic_accuracy(predicted: str, reference: str) -> float:
    """Exact match accuracy para topic classification.
    Returns 1.0 si coinciden (case-insensitive, stripped), 0.0 si no."""
    return float(predicted.strip().lower() == reference.strip().lower())
```

**Actualiza `scripts/run_evaluation.py`** — asegura que:
- La columna `topic_accuracy` usa `compute_topic_accuracy()` (exact match)
- Ya NO llama al LLM-judge para el topic (solo para Q&A)
- El CSV resultante tiene las columnas: `doc_id, wer, rouge, topic_accuracy, answer_score, latency, cost`

**Ejecuta la re-evaluación sobre el holdout:**
```bash
python scripts/run_evaluation.py --split repliqa_3 --limit 15
```

**Guarda los resultados** en `results/holdout_eval_finetuned.csv`.

**Imprime el resumen:**
```
=== RESULTADOS FINALES — repliqa_3 holdout (15 docs) ===
WER:              X.XXX  (objetivo: < 0.15)
ROUGE-L:          X.XXX  (objetivo: > 0.80)
Topic Accuracy:   XX.X%  (objetivo: > 80%)
Answer Score:     X.X/5  (objetivo: > 4.0/5)
Latencia media:   X.Xs/doc
Coste medio:      $X.XXX/doc
```

**Actualiza `app/main.py`** (Tab 3 — Business Case):
- La fila de Topic en la tabla de benchmark muestra `Topic Accuracy: XX.X%` en lugar de `Topic Score: 3.1/5`
- Si el accuracy es ≥ 80%: celda verde con checkmark
- Si el accuracy es < 80%: celda ámbar con nota metodológica
- Elimina cualquier referencia al LLM-judge score de topic (3.1/5) de la tabla principal
- Actualiza la Evaluation Methodology card con los nuevos resultados y la fecha de hoy

**Criterio de éxito:**
- `results/holdout_eval_finetuned.csv` existe con 15 filas
- Topic Accuracy ≥ 80% (si no se alcanza, reporta el número real sin modificarlo)
- La app corre sin errores en `localhost:7860`
- El Tab 3 muestra el accuracy actualizado

---

## NOTAS IMPORTANTES PARA LA EJECUCIÓN

### API key
El `.env` ya tiene `MISTRAL_API_KEY`. Usa siempre `python-dotenv` para cargarla:
```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
```

### Modelo base para fine-tuning
Usa `open-mistral-7b` — es el más económico y suficiente para clasificación. NO uses `mistral-large-latest` para el fine-tuning (caro e innecesario para esta tarea).

### El holdout es sagrado
**Nunca** incluyas documentos de `repliqa_3` en el dataset de entrenamiento. Solo en el validation file del fine-tuning (que el modelo no aprende, solo mide).

### Truncado del texto
En el dataset de fine-tuning, trunca el artículo a 2000 caracteres. Para la inferencia en extractor.py, también trunca a 2000 chars antes de enviar al modelo fine-tuned — la clasificación no necesita el texto completo y controla el coste.

### Coste estimado del fine-tuning
Con ~50 ejemplos de ~2000 chars cada uno son ~25.000 tokens de entrenamiento. A los precios actuales de Mistral el job costará menos de $1. El modelo fine-tuned tiene inferencia a ~$0.0001/llamada.

### Si el fine-tuning falla o el accuracy no mejora
Reporta el error y el número real. NO modifiques los resultados. La nota metodológica en el Business Case existe precisamente para este caso — actualízala con la explicación técnica correcta.

### Compatibilidad hacia atrás
El sistema debe seguir funcionando en modo upload (Tab 1) y evaluación manual (Tab 2) aunque el fine-tuning no haya corrido — usa el fallback a `mistral-large-latest` con prompt de taxonomía fija.

---

## ENTREGABLES ESPERADOS

Al terminar todas las tareas, el repo debe tener:

```
scripts/
  inspect_topics.py          # MI-FT01
  build_finetune_dataset.py  # MI-FT03
  run_finetune.py            # MI-FT04
  check_finetune_status.py   # MI-FT05
  run_evaluation.py          # actualizado MI-FT07

data/
  topic_finetune_train.jsonl # MI-FT03
  topic_finetune_val.jsonl   # MI-FT03
  category_list.txt          # MI-FT03
  finetune_job.txt           # MI-FT04
  finetuned_model.txt        # MI-FT05

results/
  topic_distribution.txt          # MI-FT01
  holdout_eval_finetuned.csv      # MI-FT07

app/
  prompts.py      # actualizado MI-FT06
  extractor.py    # actualizado MI-FT06
  metrics.py      # actualizado MI-FT07
  main.py         # actualizado MI-FT07
```

El comando final para verificar que todo funciona:
```bash
python scripts/run_evaluation.py --split repliqa_3 --limit 15
python app/main.py
```

---

*Proyecto: mistral-lens | Sprint: fine-tuning topic classifier | Autor: Javier Lombana | Marzo 2026*
