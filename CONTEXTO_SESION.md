# Contexto de trabajo — rag_trauma

Resumen para retomar el trabajo en otro computador. Última sesión: **25–26 de
septiembre de 2026**. Leer primero las secciones 1, 5 y 6.

---

## 1. Qué es el proyecto

Un RAG local que sirve como **asistente de apoyo al diagnóstico en trauma**.
Recupera casos similares de un registro hospitalario anonimizado (hasta 2018),
muestra qué se hizo en ellos y sugiere qué valorar. **Regla central: no
inventar nada.** Todo tiene que salir de los casos recuperados, con cita
`[caso: ID]`.

```
rag_trauma/
  .venv/                 entorno compartido (Python 3.13) — NO se copia entre PCs
  requirements.txt       instala las dos partes (-r a cada carpeta)
  README.md              visión general
  CONTEXTO_SESION.md     este archivo
  anonimizacion/         pipeline de limpieza y anonimización
    trauma.csv           DATOS CRUDOS CON PHI — cuidado al copiarlo
    outputs/             corpus_indexable.jsonl (8.198 casos), corpus.parquet, excluidos_rag.csv
    reports/             residuo_phi.csv, muestra_revision_manual.csv, reporte HTML
  rag_pipeline/          índices, evaluación y chatbot
    app.py               chatbot Streamlit
    rag/                 config, encoder, retrieve, index, build_prompt, hallazgos, generate
    indices/             BM25 + Qdrant local (8.198 puntos)
    MODELOS.md           modelos recomendados para 4 GB y 8 GB de VRAM
    SETUP.md             instalación y variables de entorno
```

Único vínculo entre las dos partes: `rag_pipeline/rag/config.py` lee
`../anonimizacion/outputs/corpus_indexable.jsonl` (se puede cambiar con
`CORPUS_JSONL`).

---

## 2. Lo que se hizo, en orden

### 2.1 Separación en dos carpetas
Antes todo estaba en la raíz. Se movió a `anonimizacion/` y `rag_pipeline/`
sin romper nada: tests, pipeline, evaluación (`eval_rutas.csv` idéntico) y
chatbot verificados.

### 2.2 Corpus solo con casos completos (`anonimizacion/`)
Objetivo del usuario: que al generador no lleguen casos truncados ni con
campos vacíos, porque son los que llevan al modelo a inventar.

- **Detector de truncamiento corregido** (`src/quality.py`, `_es_truncado`):
  - El export cortó *examen físico* y *hospitalización* a 500 caracteres. Un
    campo de esos dos con ≥ 480 caracteres (`UMBRAL_TOPE`) cuenta como
    truncado: hay un pico anómalo en 495–499.
  - *Motivo de consulta* NO tiene tope (llega a 5.270): ya no se marca por
    longitud.
  - Un campo que termina en palabra funcional (`...SE`, `...CON`) cuenta como
    truncado.
  - Terminar sin punto ya NO cuenta: la heurística anterior marcaba ~2.200
    notas completas (`...SE DA SALIDA`).
- **Flags nuevos:** `flag_incompleto` (algún campo vacío), `campos_vacios`,
  `apto_rag = indexable & ~truncado & ~incompleto`.
- **Export:** `corpus_indexable.jsonl` solo lleva `apto_rag` (se conservó el
  nombre del archivo). Nuevo `outputs/excluidos_rag.csv` con el motivo de
  cada exclusión.
- **Ablación:** `python run_pipeline.py --incluir-incompletos` reproduce el
  criterio viejo (22.176 casos).
- **Resultado: 8.198 casos aptos** de 22.176 indexables. Fuera: 2.865
  truncados y 11.113 con algún campo vacío (casi siempre hospitalización).
- **Tests:** nuevo `tests/test_calidad.py`. Pasan los 83.
- **Documentación:** actualizados `README.md` y `DICCIONARIO_DATOS.md`.
- El usuario ya regeneró `outputs/` y reindexó (`python -m rag.index`).

### 2.3 Error HTTP 500 del chatbot (resuelto)
- **Causa:** Ollama sin RAM para cargar el modelo. Log:
  `%LOCALAPPDATA%\Ollama\server.log` → `failed to allocate CPU_REPACK buffer`.
- **Equipo de la sesión:** sin GPU NVIDIA (solo Intel UHD), 7,7 GB de RAM, unos
  2,3 GB libres.
- **Cambios en `rag/config.py` y `rag/encoder.py`:**
  - `MODELO_LLM` por defecto `qwen2.5:3b` (antes el 7B, que no cabía);
  - bge-m3 en `bfloat16` para las consultas (`PRECISION_CONSULTA`, ~1,1 GB en
    vez de 2,3 GB; coseno 0,9999 contra fp32, mismo top-5);
  - `TIMEOUT_LLM=400`;
  - el verificador de citas acepta `[12345]` además de `[caso: 12345]`.

### 2.4 Prompt de asistente de diagnóstico (lo más importante)
El usuario pidió que las respuestas muestren lo que se hizo en casos similares
y además sugieran, sin inventar. **Probado con el 3B: solo con el prompt, el
modelo inventaba.** Sugería "estudios de imagen (4 de 5)" sin que ningún caso
los mencionara, atribuía procedimientos a casos equivocados y contaba mal.
Solución: **el código pone los datos; el modelo solo elige.**

Cómo se arma la respuesta (`rag/generate.py: ensamblar`):

| Sección | Quién la hace | Dónde |
|---|---|---|
| Casos similares | Código: conteos | `build_prompt.sintesis_calculada` |
| Qué se hizo | Código: hallazgos + manejo citado textual | `build_prompt.seccion_que_se_hizo` |
| Sugerencias | El modelo elige; el código redacta con conteo y citas exactos | `generate.elegir_sugerencias` |
| Lo que el registro no permite afirmar | Código | `build_prompt.limites_calculados` |

- **`rag/hallazgos.py` (nuevo):** léxico de 32 términos (Hallazgos, Estudios,
  Manejo) con manejo de negaciones ("SIN sangrado activo" no cuenta). Para
  agregar un término basta una línea en `LEXICO`.
- **Reglas de las sugerencias:**
  - si la sugerencia del modelo no corresponde a ningún elemento extraído de
    los casos, se descarta;
  - no se sugiere un hallazgo que la consulta ya menciona;
  - al menos una sugerencia es de manejo;
  - van entre 3 y 4.
- **`SINTESIS_LLM=1`:** el modelo vuelve a escribir la síntesis y los
  límites. Apagado por defecto; solo para modelos grandes y revisando las
  respuestas a mano.
- **Prompt de resumen de un caso:** tiene uno propio, `SISTEMA_RESUMEN`.
- **Textos actualizados** en `app.py`: "sugiere solo lo que esos casos
  respaldan; no reemplaza el juicio clínico".
- **Resultado en CPU con el 3B:** 40–60 s por respuesta, 0 citas inválidas.

### 2.5 `rag_pipeline/MODELOS.md`
Dos tablas (4 GB y 8 GB de VRAM) para cumplir **≤ 15 s por respuesta**.
Recomendados: **`qwen3:4b`** para 4 GB y **`qwen3:8b`** para 8 GB. Los tiempos
son ESTIMADOS; solo `qwen2.5:3b` se probó.

---

## 3. Equipo nuevo (destino)

Windows, **16 GB de RAM, GPU de 4 GB de VRAM**. Modelo sugerido: `qwen3:4b`,
pendiente de medir.

---

## 4. Cómo retomar en el PC nuevo

1. **Copiar el proyecto sin `.venv/`**, porque tiene rutas absolutas y no
   funciona en otro PC. `anonimizacion/trauma.csv` tiene datos personales:
   transferirlo por un medio seguro, o dejarlo y copiar solo `outputs/`.
2. Crear el entorno:
   ```powershell
   cd <ruta>\rag_trauma
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
3. Instalar Ollama y descargar el modelo:
   ```powershell
   ollama pull qwen3:4b
   $env:MODELO_LLM = "qwen3:4b"
   $env:PRECISION_CONSULTA = "fp32"   # con 16 GB sobra RAM y la consulta va más rápida
   ```
4. Comprobar el entorno: `cd rag_pipeline; python scripts\revisar_entorno.py`
5. Construir los índices. No están en git (ver `.gitignore`); el corpus sí:
   ```powershell
   python -m rag.index     # desde rag_pipeline\; con GPU tarda pocos minutos
   ```
   Solo si falta `anonimizacion/outputs/`, antes: `cd ..\anonimizacion; python run_pipeline.py`
6. Probar: `python scripts\probar_respuestas.py "herida por arma de fuego en tórax con hipoventilación"`
7. Chatbot: `streamlit run app.py`

---

## 5. Pendientes, por prioridad

1. **`"think": false` en `_ollama()`** (`rag/generate.py`). Sin esto, `qwen3`
   y `qwen3.5` razonan antes de responder y pasan de 15 s. Es obligatorio
   para usar los modelos recomendados en `MODELOS.md`.
2. **Bajar `MAX_TOKENS` a ~250** en `rag/config.py` (hoy 900). El modelo solo
   escribe las sugerencias.
3. **Medir modelos en el PC nuevo.** Se ofreció crear
   `scripts/comparar_modelos.py`: correr las 28 consultas de
   `eval/consultas.yaml` con cada modelo y reportar tiempo típico y peor caso
   (contra 15 s), sugerencias descartadas y pertinencia. Verificar con
   `ollama ps` que el modelo diga `100% GPU`.
4. **Resumen por identificador** (`resumir_caso`): todavía lo escribe el
   modelo, y en una prueba cambió "neumoconiótico" por "neumógeno". Hacerlo
   determinista, como "Qué se hizo".
5. **Recalibrar `UMBRAL_ABSTENCION`** (0,45). Con el corpus nuevo, "manejo de
   cetoacidosis diabética" (fuera de dominio) obtiene 0,499 y ya no se
   abstiene. Correr `python -m eval.evaluate`.
6. **Desenlace sin clasificar:** 35 % de los casos aptos quedan como
   `no_determinado` aunque tienen la hospitalización llena. Mejorar
   `clasificar_desenlace` en `anonimizacion/src/enrich.py`.
7. **Revisar 33 hallazgos de datos personales** en
   `anonimizacion/reports/residuo_phi.csv` (0,40 % de los casos aptos).
8. **Restos del manejo de truncados:**
   - `app.py`: el filtro "Excluir registros truncados" (línea ~106) y la
     etiqueta "truncado" ya no aplican;
   - `eval/consultas.yaml`: la línea ~197 menciona truncados;
   - `SETUP.md`: el paso 6 del guion de demo ("Caso truncado") ya no aplica.
9. **Tests para `rag_pipeline`**: no hay. Candidatos: `hallazgos._afirmado`
   (negaciones), `elegir_sugerencias` y `ensamblar`.
10. **Control de versiones:** ya existe `.gitignore`, pensado para un
    repositorio **privado**. Excluye solo `.venv/`, `rag_pipeline/indices/`
    (Qdrant pesa ~83 MB) y las cachés de Python. `trauma.csv` y `outputs/` SÍ
    se versionan. Falta `git init` y el primer commit. Después de clonar hay
    que correr `python -m rag.index`.

---

## 6. Trampas conocidas

- **Qdrant local admite un solo proceso.** Si el chatbot está abierto,
  `rag.index`, `eval.evaluate` o `probar_respuestas.py` fallan con
  `already accessed by another instance`. Cerrar streamlit, o usar una copia
  de `indices/` con `INDICE_DIR=<copia>`.
- **HTTP 500 del chatbot = casi siempre falta de memoria en Ollama.** El
  detalle está en `%LOCALAPPDATA%\Ollama\server.log`.
- **Ollama descarga el modelo a los 5 minutos sin uso**, y la consulta
  siguiente es lenta. Se evita con `OLLAMA_KEEP_ALIVE=-1`.
- **Prompt:** mide entre 1.362 y 2.437 tokens, más la salida, dentro de la
  ventana de 4.096 que Ollama usa por defecto. Si se sube `TOP_K`, revisar
  `num_ctx`.
- **Al editar prompts con scripts de Python:** las líneas con `\` de
  continuación dentro de strings fallan con `str.replace`. Usar edición
  directa del archivo.
- **`probar_respuestas.py` → "palabras sin raíz en los casos":** la mayoría
  es lenguaje común. Solo preocupa si aparece un término clínico.

---

## 7. Variables de entorno relevantes (`rag_pipeline/rag/config.py`)

| Variable | Valor por defecto | Nota |
|---|---|---|
| `MODELO_LLM` | `qwen2.5:3b` | en el PC nuevo: `qwen3:4b` (ver `MODELOS.md`) |
| `PRECISION_CONSULTA` | `bf16` | en el PC nuevo: `fp32` |
| `DISPOSITIVO_CONSULTA` | `cpu` | dejar en CPU: la VRAM es para el generador |
| `MAX_TOKENS` | `900` | pendiente: bajar a ~250 |
| `TIMEOUT_LLM` | `400` | |
| `SINTESIS_LLM` | apagado | no encender sin revisar respuestas |
| `UMBRAL_ABSTENCION` | `0.45` | pendiente: recalibrar |
| `TOP_K` | `5` | |
