# Contexto de trabajo — rag_trauma

Resumen para retomar el trabajo. Última sesión: **26 de septiembre de 2026**, en
el PC nuevo (GTX 1650). **Empezar por la sección 0**, después 1, 5 y 6.

---

## 0. Para retomar mañana

**Principio que manda sobre todo lo demás:** es un **asistente al
diagnóstico**. Lo más importante de la respuesta es la **sugerencia**: va
primero, corta y fácil de usar. El respaldo (conteos, citas, tabla) va después.
Y la regla de siempre: nada inventado, todo sale de los casos con `[caso: ID]`.

**Estado del código:** todo lo de la sesión del 26/09 está **sin commit**. El
único commit es `5c3a035` (el prototipo anterior). Cambios pendientes de
commit:

```
M  .gitignore                     (+ rag_pipeline/eval/revision/)
M  CONTEXTO_SESION.md
M  rag_pipeline/MODELOS.md
M  rag_pipeline/app.py            (tiempo de respuesta, aviso fijo, tarjetas resaltadas, sin truncados)
M  rag_pipeline/rag/build_prompt.py   (formato nuevo de la respuesta)
M  rag_pipeline/rag/config.py     (qwen3:4b-instruct, NUM_GPU, NUM_CTX, MAX_TOKENS, SIMILITUD_BAJA)
M  rag_pipeline/rag/generate.py   (think:false, ventana dinámica, ensamblar, elegir_sugerencias)
M  rag_pipeline/rag/hallazgos.py  (extractor corregido, obvios, resaltar_html)
M  rag_pipeline/scripts/probar_respuestas.py
?? rag_pipeline/.streamlit/config.toml
?? rag_pipeline/scripts/exportar_revision.py
?? rag_pipeline/tests/            (test_hallazgos.py, test_ensamblar.py: 46 tests)
```

**Primeros pasos mañana:**
1. Comprobar que todo sigue verde: `cd rag_pipeline; ..\.venv\Scripts\python.exe -m pytest tests/ -q`
   (deben pasar los 46) y en `anonimizacion\` los 83.
2. Decidir si se hace commit de lo del 26/09 (el usuario no lo ha pedido aún).
   `exportar_revision.py` compara contra `5c3a035`, así que sigue sirviendo
   después del commit.
3. Abrir el chatbot desde `rag_pipeline\`: `streamlit run app.py`. Si en esa
   terminal quedó `$env:MODELO_LLM = "qwen3:4b"`, borrarla
   (`Remove-Item Env:MODELO_LLM`): el valor por defecto ya es el correcto.
4. Seguir con los pendientes de la sección 5. Lo más importante: la
   **revisión médica** (pendiente 1) y seguir mejorando **las sugerencias**.

---

## 1. Qué es el proyecto

Un RAG local que sirve como **asistente de apoyo al diagnóstico en trauma**.
Recupera casos similares de un registro hospitalario anonimizado (hasta 2018),
sugiere qué valorar y muestra qué se hizo en ellos. **Regla central: no
inventar nada.** Todo tiene que salir de los casos recuperados, con cita
`[caso: ID]`. **El código pone los datos; el modelo solo elige** qué
sugerencias mostrar.

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
    .streamlit/          config.toml: no recargar rag/ en caliente (ver sección 6)
    rag/                 config, encoder, retrieve, index, build_prompt, hallazgos, generate
    indices/             BM25 + Qdrant local (8.198 puntos) — no está en git
    tests/               46 tests del extractor y de la respuesta (sin Ollama ni índices)
    scripts/             revisar_entorno, probar_respuestas, exportar_revision
    eval/                consultas.yaml (28 consultas), evaluate.py, revision/ (planilla médica, no en git)
    MODELOS.md           modelos recomendados para 4 GB y 8 GB de VRAM (con mediciones)
    SETUP.md             instalación y variables de entorno
```

Único vínculo entre las dos partes: `rag_pipeline/rag/config.py` lee
`../anonimizacion/outputs/corpus_indexable.jsonl` (se puede cambiar con
`CORPUS_JSONL`).

---

## 2. Lo que se hizo, en orden

### 2.1 Separación en dos carpetas (25/09)
Antes todo estaba en la raíz. Se movió a `anonimizacion/` y `rag_pipeline/`
sin romper nada: tests, pipeline, evaluación (`eval_rutas.csv` idéntico) y
chatbot verificados.

### 2.2 Corpus solo con casos completos (`anonimizacion/`, 25/09)
Objetivo: que al generador no lleguen casos truncados ni con campos vacíos,
porque son los que llevan al modelo a inventar.

- **Detector de truncamiento corregido** (`src/quality.py`, `_es_truncado`):
  - el export cortó *examen físico* y *hospitalización* a 500 caracteres; un
    campo con ≥ 480 caracteres (`UMBRAL_TOPE`) cuenta como truncado;
  - *motivo de consulta* no tiene tope y ya no se marca por longitud;
  - un campo que termina en palabra funcional (`...SE`, `...CON`) cuenta
    como truncado; terminar sin punto ya no.
- **Flags:** `flag_incompleto`, `campos_vacios`,
  `apto_rag = indexable & ~truncado & ~incompleto`.
- **Export:** `corpus_indexable.jsonl` solo lleva `apto_rag`;
  `outputs/excluidos_rag.csv` dice por qué se excluyó cada caso.
- **Ablación:** `python run_pipeline.py --incluir-incompletos` (22.176 casos).
- **Resultado: 8.198 casos aptos** de 22.176 indexables.
- **Tests:** `tests/test_calidad.py`; pasan los 83.

### 2.3 HTTP 500 en el PC anterior (25/09, resuelto)
Ollama sin RAM (PC sin GPU NVIDIA, 7,7 GB). Se pasó a `qwen2.5:3b` y bge-m3
en bf16 para las consultas. Log: `%LOCALAPPDATA%\Ollama\server.log`.

### 2.4 Prompt de asistente de diagnóstico (25/09)
Probado con el 3B: solo con el prompt, el modelo inventaba (sugería
"estudios de imagen 4 de 5" sin que ningún caso los mencionara, contaba mal).
Solución, que sigue vigente: **el código pone los datos; el modelo solo
elige.** El modelo recibe los casos y una lista de "elementos documentados"
(`hallazgos.py`) y responde solo con viñetas de sugerencias; el código toma
de ahí la ELECCIÓN y escribe todo el texto. `SINTESIS_LLM=1` deja que el
modelo escriba la síntesis (apagado; solo con modelos grandes y revisando).
El formato de la respuesta de esa sesión fue reemplazado el 26/09 (2.7).

### 2.5 Velocidad en el PC nuevo: de ~100 s a ~16 s (26/09)
Medido en la GTX 1650:
- **`qwen3:4b` es la variante Thinking** (Qwen3-4B-Thinking-2507): razona
  siempre, aunque se envíe `"think": false`. Gastaba los 900 tokens
  razonando en inglés y **nunca escribía las sugerencias**. Se cambió a
  **`qwen3:4b-instruct`** (valor por defecto en `config.py`).
- `_ollama()` envía `"think": false`; `MAX_TOKENS` bajó a 250 (el modelo
  escribe ~100); el resumen de un caso usa `MAX_TOKENS_RESUMEN` (700).
- **Ollama dejaba 10 de 37 capas en CPU** (`33%/67% CPU/GPU`) aunque cabían.
  Se envía `num_gpu=99` y `num_ctx=3072` → `100% GPU`, 42 tok/s, ~3,0 GB de
  VRAM. Si un prompt no cabe (p. ej. 10 casos), la ventana se agranda sola
  para esa llamada y no se fuerzan capas (~55 s).
- De los ~16 s, ~11 s son leer el prompt (~2.000 tokens): es el límite de
  cómputo de la tarjeta.
- Detalle en `rag_pipeline/MODELOS.md`.

### 2.6 Interfaz del chatbot (26/09)
- Bajo cada respuesta, en pequeño: tiempo y modelo (`16.2 s · qwen3:4b-instruct`).
- Aviso fijo bajo cada respuesta ("casos históricos hasta 2018, no son
  frecuencias generales, no reemplaza el juicio clínico").
- Expander "Casos consultados · texto completo": las tarjetas **resaltan lo
  que coincide con la consulta** (términos del léxico y palabras de la
  consulta por raíz) y el texto va escapado como HTML.
- Se quitó el filtro y la etiqueta de "truncado".
- `rag_pipeline/.streamlit/config.toml` evita el error de Qdrant al guardar
  archivos (sección 6).

### 2.7 Respuesta pensada para médicos (26/09) — lo más importante
Revisada contra el corpus y contra otros asistentes clínicos:
- **OpenEvidence:** respuesta breve con citas verificables.
- **Evaluaciones en el punto de atención:** los médicos piden la conclusión
  primero, algo fácil de escanear y accionable, y la incertidumbre explícita.
- **Guía CDS de la FDA, criterio 4:** el médico debe poder revisar por su
  cuenta la base de cada recomendación.
- **Rúbrica de Med-PaLM:** factualidad, omisión, posible daño y utilidad.

**Extractor (`hallazgos.py`) corregido.** Los conteos que ve el médico eran
falsos en parte. Todo se verificó sobre los 8.198 casos, con revisión a mano de
muestras:
- **Remisión:** solo la de salida ("se remite", "remitir"). Bajó de 1.098 a
  208: "remitido por HPAF" es cómo llegó el paciente, no un manejo.
- **Cirugía:** solo si se hizo o se indicó (llevado a cx, procedimiento o
  hallazgos quirúrgicos, requiere cx, un procedimiento concreto). Bajó de
  1.365 a 897. "Valorado por cx plástica" o "por neurocirugía" son
  interconsultas y no cuentan.
- **Por sección:** manejo y estudios no se leen del motivo de consulta.
- **Negación a la derecha:** "Blumberg negativo", "(-)". No se usa "no" a la
  derecha, porque "fractura no desplazada" sí es una fractura.
- **Términos nuevos:**
  - laparatomía (mal escrita);
  - Blumberg / abdomen agudo → irritación peritoneal;
  - hemoperitoneo / líquido libre;
  - TCE: pérdida de conciencia, amnesia, emesis, cefalea, otorragia,
    convulsión, anisocoria;
  - quemaduras: curación, escarectomía / injerto, líquidos endovenosos.
- **Descartado a propósito:** "signos de hipoperfusión". Es raro en el
  corpus y ambiguo: "frialdad distal" es un signo vascular.

**Formato de la respuesta** (`build_prompt.py` + `generate.ensamblar`), en este orden:
1. **Sugerencias según casos similares.** Van primero y en orden
   diagnóstico: **Valorar** (hallazgos), luego **Considerar** estudios, luego
   **Considerar** manejo. Cada una tiene dos líneas:
   ```
   - **Valorar irritación peritoneal** — presente en 2 de 5 casos similares
     En esos casos se hizo: laparotomía / laparoscopia 1 · desenlace: alta 1, sin clasificar 1 [caso: 3377] [caso: 27722]
   ```
   Reglas (`generate.elegir_sugerencias`):
   - de 3 a 4 sugerencias;
   - la elección es del modelo, pero se descarta la que no esté en la lista;
   - **al menos una de manejo;**
   - **al menos un "Valorar…"** si hay hallazgos, con preferencia por los de
     los casos que escalaron (cirugía, UCI, fallecimiento). Sin esto, el
     modelo elegía solo manejo y dejaba fuera la irritación peritoneal de los
     operados;
   - no se sugiere lo que la consulta ya dice o implica (`hallazgos.obvios`:
     "herida penetrante" si la consulta es por una herida por arma);
   - "cirugía" no se repite si ya está el procedimiento concreto.
2. **En síntesis**, con:
   - mecanismo y región;
   - lo más documentado;
   - la escalada más grave con sus hallazgos y su cita;
   - una línea de respaldo: n y rango de similitud, y cuántos casos se
     hallaron solo por palabras clave;
   - un aviso de **similitud baja** si el mejor caso está por debajo de 0,55.
3. **Tabla**, un caso por fila: caso, similitud, lesión, hallazgos (en negrita
   los de la consulta), manejo y desenlace. "Cirugía" como desenlace se
   muestra como "operado".
4. **Para afinar la búsqueda:** hallazgos que diferencian los casos y que no
   van ya sugeridos.
5. **Límites de estos casos**, en una línea: sin desenlace clasificado,
   signos vitales anotados y Glasgow.

**El prompt del modelo no cambió.** Sus títulos (`TITULO_SUGERENCIAS`, etc.) son
de protocolo; los que ve el médico son `ROTULO_*`.

**Verificación:**
- 46 tests nuevos en `rag_pipeline/tests/`;
- 0 citas inválidas y 0 "palabras sin raíz" en las consultas de prueba;
- la app probada sin navegador con `streamlit.testing.v1.AppTest`.

**Planilla para el médico:** `python scripts\exportar_revision.py` compara a
ciegas el formato anterior (`git show 5c3a035`) con el actual, usando la misma
recuperación y la misma elección del modelo. Ya está generada en
`rag_pipeline/eval/revision/`: `revision.html` y `planilla.csv` son para el
médico; **`clave.csv` no se le entrega**.

---

## 3. Equipo actual

Windows 11, **16 GB de RAM, GPU GTX 1650 de 4 GB**. Ollama 0.34.4 con
`qwen3:4b-instruct` (también está `qwen3:4b`, la variante Thinking, que no
sirve). Respuesta típica ~16 s; la primera tras cargar el modelo ~25–35 s.

---

## 4. Cómo instalar en un PC nuevo

1. **Copiar el proyecto sin `.venv/`**, porque tiene rutas absolutas.
   `anonimizacion/trauma.csv` tiene datos personales: transferirlo por un
   medio seguro, o copiar solo `outputs/`.
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
   ollama pull qwen3:4b-instruct      # valor por defecto de config.py; NO qwen3:4b
   setx OLLAMA_KEEP_ALIVE -1          # opcional: no descargar el modelo a los 5 min
   ```
4. Comprobar el entorno: `cd rag_pipeline; python scripts\revisar_entorno.py`
5. Índices (no están en git): `python -m rag.index` desde `rag_pipeline\`.
   Si falta `anonimizacion/outputs/`, antes: `cd ..\anonimizacion; python run_pipeline.py`.
6. Tests: `python -m pytest tests/ -q` (46).
7. Probar: `python scripts\probar_respuestas.py "herida por arma de fuego en tórax con hipoventilación"`
8. Chatbot: `streamlit run app.py` (**desde `rag_pipeline\`**).

---

## 5. Pendientes, por prioridad

1. **Revisión médica.** Entregar `eval/revision/revision.html` y
   `planilla.csv`, sin `clave.csv`. La rúbrica va de 1 a 5 en conteos y citas,
   omisiones, riesgo y utilidad, más si se lee en 30 s. Con la clave, comparar
   el formato anterior con el actual. **Cada error que marque se vuelve un
   test** en `tests/test_hallazgos.py`. Si se cambia el formato, regenerar la
   planilla con `INDICE_DIR` apuntando a una copia de los índices si el
   chatbot está abierto.
2. **Seguir mejorando las sugerencias** (el principio de la sección 0). Ideas
   abiertas:
   - el léxico solo reconoce lo que tiene. En quemaduras no hay hallazgos que
     valorar (profundidad, % de superficie corporal), y en TCE falta el
     Glasgow como valor. Ampliar `LEXICO` con verificación sobre el corpus,
     como se hizo el 26/09;
   - que "Valorar…" priorice los hallazgos asociados a escalada con más
     señal: ver si el médico lo encuentra útil;
   - medir cuántas veces el modelo elige solo manejo y cuántas completa el
     código.
3. **Commit** de todo lo del 26/09 (sección 0), cuando el usuario lo pida.
4. **Resumen por identificador** (`resumir_caso`): todavía lo escribe el
   modelo y cambia términos: "neumoconiótico" por "neumógeno", y el 26/09
   "empema" por "Empiema". Hacerlo determinista, como el resto.
5. **Recalibrar `UMBRAL_ABSTENCION`** (0,45). "Manejo de cetoacidosis
   diabética" (fuera de dominio) obtiene 0,47 y no se abstiene; por ahora
   sale el aviso de similitud baja (< 0,55). Correr `python -m eval.evaluate`.
6. **Errores del pipeline de anonimización** encontrados al revisar
   respuestas:
   - el clasificador de mecanismo pone el caso 24012 ("me pegaron un
     puntazo", HPACP) como contundente;
   - la expansión de abreviaturas escribe "Tx (tratamiento) renal" en el caso
     6561, donde probablemente es trauma;
   - también escribe "Qx (quirúrgico) de II y III grado" en el caso 2589,
     donde es quemadura.

   Van en `anonimizacion/src/`, y después hay que regenerar el corpus y
   reindexar.
7. **Desenlace sin clasificar:** 35 % de los casos aptos quedan como
   `no_determinado`. Mejorar `clasificar_desenlace` en
   `anonimizacion/src/enrich.py`.
8. **Bajar de 16 s a menos de 15 s:** la única palanca es acortar el prompt,
   con menos texto por caso en `construir_prompt`, porque el modelo solo elige
   de la lista. Antes, medir si cambian las sugerencias elegidas.
9. **Medir otros modelos:** crear `scripts/comparar_modelos.py` con las 28
   consultas: tiempo típico y peor caso, sugerencias descartadas. Verificar
   `100% GPU` con `ollama ps`.
10. **Revisar 33 hallazgos de datos personales** en
    `anonimizacion/reports/residuo_phi.csv`.
11. **Restos de truncados:** `eval/consultas.yaml` (línea ~197) y el paso 6
    del guion de demo en `SETUP.md`.

---

## 6. Trampas conocidas

- **Qdrant local admite un solo proceso.** Con el chatbot abierto, fallan
  `rag.index`, `eval.evaluate`, `probar_respuestas.py` y
  `exportar_revision.py` con `already accessed by another instance`. Cerrar
  Streamlit o usar una copia:
  `$env:INDICE_DIR = "<copia de rag_pipeline\indices>"`. La copia sin el
  archivo `.lock` funciona.
- **El mismo error dentro del chatbot tras guardar un archivo:** Streamlit
  recarga todos los módulos del proyecto y `rag.retrieve` abre un segundo
  cliente. Lo evita `rag_pipeline/.streamlit/config.toml`
  (`folderWatchBlacklist = ["**/rag"]`), que **solo se lee si Streamlit se
  lanza desde `rag_pipeline\`**. Al editar `rag/*.py`, reiniciar Streamlit.
- **`qwen3:4b` ≠ `qwen3:4b-instruct`.** La primera es Thinking e ignora
  `think: false`. Con `qwen:4b` (sin el 3) se baja Qwen 1.5, que es otro
  modelo.
- **HTTP 500 del chatbot = casi siempre falta de memoria en Ollama.** Si pasa
  con `NUM_GPU=99` (el navegador también usa VRAM), `$env:NUM_GPU = "-1"`.
  Log en `%LOCALAPPDATA%\Ollama\server.log`.
- **Ollama descarga el modelo a los 5 minutos sin uso.** Se evita con
  `OLLAMA_KEEP_ALIVE=-1`.
- **Al editar archivos desde Bash con heredoc**, las tildes, el "·" y los `\n`
  dentro de strings se dañaron varias veces. Usar la herramienta de edición
  directa, o un script en un archivo aparte.
- **Probar la app sin navegador:** `streamlit.testing.v1.AppTest.from_file`
  con ruta absoluta y `INDICE_DIR` apuntando a una copia. Así apareció un
  error de sintaxis en `app.py` que habría tumbado el chatbot.
- **`probar_respuestas.py` → "palabras sin raíz en los casos":** el
  vocabulario del código ya no cuenta. Si aparece un término clínico, hay que
  revisarlo.

---

## 7. Variables de entorno relevantes (`rag_pipeline/rag/config.py`)

| Variable | Valor por defecto | Nota |
|---|---|---|
| `MODELO_LLM` | `qwen3:4b-instruct` | NO `qwen3:4b` (Thinking). Sin GPU: `qwen2.5:3b` |
| `NUM_GPU` | `99` | todas las capas en GPU; `-1` = que decida Ollama |
| `NUM_CTX` | `3072` | se agranda sola para la llamada si el prompt no cabe |
| `MAX_TOKENS` | `250` | `900` con `SINTESIS_LLM=1` |
| `MAX_TOKENS_RESUMEN` | `700` | resumen de un caso por identificador |
| `TIMEOUT_LLM` | `400` | |
| `SINTESIS_LLM` | apagado | no encender sin revisar respuestas |
| `UMBRAL_ABSTENCION` | `0.45` | pendiente: recalibrar |
| `SIMILITUD_BAJA` | `0.55` | por debajo, la respuesta avisa que los casos pueden no corresponder |
| `TOP_K` | `5` | con 10 el prompt no cabe en 3.072 y se vuelve lento (~55 s) |
| `PRECISION_CONSULTA` | `bf16` | con 16 GB de RAM se puede usar `fp32` |
| `DISPOSITIVO_CONSULTA` | `cpu` | dejar en CPU: la VRAM es para el generador |
| `INDICE_DIR` | `rag_pipeline/indices` | apuntar a una copia para correr scripts con el chatbot abierto |
