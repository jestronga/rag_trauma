# Contexto de trabajo — rag_trauma

Resumen para retomar el trabajo. Última sesión: **26 de septiembre de 2026**, en
el PC nuevo (GTX 1650). **Empezar por la sección 0**, después 1, 5 y 6.

---

## 0. Para retomar mañana

**Principio que manda sobre todo lo demás:** es un **asistente al
diagnóstico**. Desde el 28/09 la respuesta es una **sugerencia en prosa**, como
la daría un asistente médico: **el modelo redacta, el código verifica**
(`rag/verificar.py`). Si la prosa nombra algo que no consta en los casos, cita
mal, cuenta mal u ordena en vez de orientar, se muestra la versión armada por el
código (sección 2.10). Regla de siempre: nada inventado, todo sale de los casos
con `[caso: ID]`; el origen se revisa en "Casos consultados" de la app.

**Estado del código:** lo del 26/09 está en el commit `a9f9f1c`. Todo lo del
27–28/09 (secciones 2.8–2.10) está **sin commit**; el usuario no lo ha pedido.

**Primeros pasos, en el PC de 8 GB:**
1. Tests: `cd rag_pipeline; ..\.venv\Scripts\python.exe -m pytest tests/ -q`
   (93) y en `anonimizacion\` los 83.
2. Modelos: `ollama pull qwen3:4b-instruct` y bajar el reranker (`SETUP.md`,
   sección 2).
3. `python scripts\calibrar_umbral.py` → debe decir `cuda` y < 1 s por consulta.
4. `python scripts\comparar_modelos.py --modos prosa` → tiempos reales,
   `gpu_pct` = 100 y cuántas prosas pasan la verificación.
5. Revisión médica (pendiente 1), ahora con la prosa: regenerar la planilla.

---

## 1. Qué es el proyecto

Un RAG local que sirve como **asistente de apoyo al diagnóstico en trauma**.
Recupera casos similares de un registro hospitalario anonimizado (hasta 2018),
sugiere qué valorar y muestra qué se hizo en ellos. **Regla central: no
inventar nada.** Todo tiene que salir de los casos recuperados, con cita
`[caso: ID]`. **El modelo redacta la sugerencia; el código la verifica** y, si
no pasa, pone él los datos (antes del 28/09 el modelo solo elegía qué
sugerencias mostrar; sigue así con `SALIDA=sugerencias`).

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

### 2.8 Evaluación corregida y reranker (27/09)
Inspirado en `medical_RAG_system` (Stuhlmann et al., SDS 2025): reranker como
filtro y evaluación por etapas. Análisis completo en la página "Análisis RAG
trauma vs medical_RAG_system" (claude.ai).

- **Error en `eval/evaluate.py`:** el ideal del nDCG salía de la propia ruta,
  así que las rutas no eran comparables. Ahora el ideal es la unión de lo que
  traen todas las rutas evaluadas (pooling). Nuevas columnas `R@10` y
  `relevantes_pool`.
- **Doble sigmoide en `buscar_rerank`:** sentence-transformers >= 4 ya aplica
  la sigmoide; aplicarla otra vez apiñaba todo en [0,5, 0,73]. Ahora se piden
  logits crudos (`_probabilidades`).
- **`Resultado.relevancia`** guarda la probabilidad del reranker;
  `score_norm` sigue siendo el coseno (lo leen `SIMILITUD_BAJA` y la tabla).
- **Segunda guarda de abstención** (`generate.filtrar_relevantes`): solo
  pasan los casos con relevancia >= `UMBRAL_RERANK`; con menos de
  `MIN_CASOS`, se abstiene.
- **Calibración** (`scripts/calibrar_umbral.py`, 10 consultas fuera de dominio
  nuevas en `consultas.yaml`, `neg-04`…`neg-10`): con 0,2, 0 de 25 consultas
  de trauma se abstienen y 1 de 10 fuera de dominio responde (apendicitis,
  0,42). Solo con coseno respondían 8 de 10.
- **Latencia:** en CPU (i5-12450H) el reranker tarda ~31 s por consulta; no es
  viable. `RUTA=auto` lo usa solo si hay CUDA.
- `RERANKER=1` estaba documentado pero ningún código lo leía: eliminado.

### 2.9 Salida estructurada (27/09)
El modelo ya no escribe viñetas: responde `{"sugerencias": [...]}` y el
campo `format` de Ollama (JSON Schema con `enum`) le impide elegir algo que no
esté en la lista (`generate.esquema_sugerencias`, `generar_eleccion`).
- **La lista que ve el modelo = el `enum`** = `hallazgos.elegibles`: sin lo
  que la consulta ya dice ni "cirugía" repetida. Antes el modelo veía todo y el
  código descartaba después.
- **Sin elementos elegibles no se llama al modelo** (no hay qué elegir).
- `SALIDA_ESTRUCTURADA=0` vuelve a las viñetas; con `SINTESIS_LLM` también.
  `exportar_revision.py` sigue en modo texto, a propósito: el código de
  `5c3a035` no sabe leer JSON.
- **Medido** (25 consultas, mismos casos, `qwen3:4b-instruct`,
  `reports/comparar_salida.csv`; hoy se repite con
  `scripts/comparar_modelos.py --modos texto,json`):

  | | texto | JSON |
  |---|---|---|
  | propuestas descartadas | 13 de 83 | 4 de 99 (todas repeticiones) |
  | sugerencias completadas por el código | 11 | 1 |
  | tokens de salida (mediana) | 109 | 31 |
  | JSON inválido | — | 0 de 25 |
  | el modelo no eligió ningún hallazgo | 5 | 4 |

  En texto el modelo inventaba elementos ("abrasión en región frontal",
  "fijación maxilar") y sugería lo obvio ("hipoventilación"). Los tiempos de
  esa corrida no sirven: PC anterior sin GPU y con 0,1 GB de RAM libre.
- **Cambio visible:** el modelo propone siempre 4 (el máximo) y las
  sugerencias finales coinciden con las del modo texto solo en 7 de 25
  consultas. Tiende a elegir más hallazgos "Valorar…" de 1 caso. **Que lo
  juzgue el médico** en la revisión (pendiente 1).
- `maxItems` = mínimo entre 4 y los elegibles: con 1 elegible el modelo lo
  repetía 4 veces.

### 2.10 Sugerencia en prosa, verificada (28/09)
Pedido del usuario: que responda como medical_RAG_system, en prosa de
asistente médico; el origen ya se ve en "Casos consultados". Reemplazó el
paso de "prompt corto". Decisiones del usuario: prosa + línea de respaldo; si
no pasa la verificación, versión determinista; `qwen3:4b-instruct` + reranker.
- **Flujo** (`SALIDA=prosa`, por defecto): `generate.generar_prosa` (JSON
  `{"sugerencia": …}`, `SISTEMA_PROSA` + `INSTRUCCION_PROSA`, sin datos
  calculados) → `verificar.suavizar` ("Se debe considerar" → "Conviene
  considerar") → `verificar.verificar_prosa` → `ensamblar_prosa` (prosa +
  `linea_respaldo`) o, si falla, `ensamblar("")` con la elección de las reglas
  (sin segunda llamada). `Respuesta.prosa_verificada` / `motivos`; la app dice
  por qué se muestra la versión determinista.
- **Reglas del verificador** (`rag/verificar.py`): citas existentes, términos
  del léxico presentes en los casos, atribución por oración, conteos ("dos de
  ellos", "en dos casos", "todos", "la mayoría"), tono (sin "se debe", "hay
  que") y palabras sin raíz en los casos (el chequeo que vivía en
  `probar_respuestas.py`).
- **Medido** (25 consultas, `scripts/comparar_modelos.py --modos prosa`):
  11/25 con el primer prompt → 19/25 con el prompt ajustado → **18/25 (72 %)**
  tras corregir el léxico (abajo). Los 7 rechazos finales son justos
  (radiografías que ningún caso tiene, "sangrado activo" sin respaldo,
  condiciones agregadas: "signos de infección", "profundidad").
- **Límite conocido:** el verificador comprueba términos, citas y conteos, no
  el razonamiento. Entre las verificadas hubo deslices menores: una condición
  quizá agregada ("sangrado abundante"), un caso poco pertinente citado y
  redacción torpe ("Glasgow con [caso: …]"). **La revisión médica tiene que
  ver la prosa** (regenerar la planilla con `exportar_revision.py`; hoy compara
  solo formatos deterministas).
- `PALABRAS_COMUNES` (en `verificar.py`) se amplía solo con falsos rechazos
  medidos; nunca con términos clínicos. Ojo: toda palabra que aparezca en el
  código de `build_prompt.py` o `hallazgos.py` queda permitida, así que no
  poner ejemplos clínicos en los prompts.
- **Error del léxico corregido:** `sutura` era `sutur|rafia\b` y encontraba
  "-grafía": **1.209 de 2.426 casos contaban sutura sin tenerla** (solo tenían
  radiografía, tomografía o ecografía). Ahora `(?<!g)rafia\b`; tests en
  `test_hallazgos.py`. Afectaba los conteos de sutura de toda la respuesta
  determinista y la lista de elementos del prompt.

### 2.11 Preparado para la demo en la GTX 1650 (28/09)
- `RUTA=auto` mira la VRAM (`retrieve.vram_gb`): rerank solo con ≥ 7 GB. Antes
  bastaba con que hubiera CUDA, y en la 1650 el reranker (~1,3 GB) junto a
  `qwen3:4b-instruct` (~3,0 GB) habría dejado a Ollama sin memoria.
- `UMBRAL_ABSTENCION` 0,45 → 0,52: sin reranker, el botón "5 · Fuera de
  alcance" (cetoacidosis, coseno 0,500) no se abstenía. Medido con preguntas
  cortas: trauma más bajo "TCE" 0,541; bloquea diabetes, apendicitis, asma e
  infarto; **no** separa "dolor torácico" (0,59) ni "dolor de cabeza" (0,62).
- `KEEP_ALIVE=2h` en cada llamada y **precarga al abrir la app**
  (`preparar_modelos`, `st.cache_resource`): la primera pregunta no espera la
  carga de bge-m3 ni del modelo.
- `scripts/preparar_demo.py`: GPU, % en VRAM y los 5 botones por el pipeline
  completo. Corrido en el PC sin GPU: los 5 correctos (4 responden, el 5 se
  abstiene), 0 citas inválidas. **Falta correrlo en la 1650.**
- Texto del botón 4 sin "registros truncados"; guion de `SETUP.md` §6 al día.

---

## 3. Equipos

- **GTX 1650 (demo):** Windows 11, 16 GB de RAM, 4 GB de VRAM. Ollama 0.34.4
  con `qwen3:4b-instruct` (también está `qwen3:4b`, la variante Thinking, que
  no sirve). Con salida estructurada ~16 s; con la prosa, ~20 s estimados.
  Ruta `rrf` (el reranker no cabe).
- **PC sin GPU** (Intel UHD, 7,7 GB de RAM): donde se trabajó el 27–28/09. Todo
  corre en CPU (~2 min por respuesta); sus tiempos no representan a los otros.
- **PC de 8 GB de VRAM:** destino final, con reranker en GPU.

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
   - ~~medir cuántas veces el modelo elige solo manejo y cuántas completa el
     código~~ medido el 27/09 (sección 2.9): 4 y 1 de 25 con salida
     estructurada.
3. **Commit** de todo lo del 26/09 (sección 0), cuando el usuario lo pida.
4. **Resumen por identificador** (`resumir_caso`): todavía lo escribe el
   modelo y cambia términos: "neumoconiótico" por "neumógeno", y el 26/09
   "empema" por "Empiema". Hacerlo determinista, como el resto.
5. ~~Recalibrar `UMBRAL_ABSTENCION`~~ **Hecho el 27/09 con el reranker**
   (sección 2.8). Falta **medirlo en el PC de 8 GB**: `python
   scripts\calibrar_umbral.py` debe decir `cuda` y < 1 s por consulta, y
   `ollama ps` seguir en `100% GPU` con el reranker cargado.
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
8. **Latencia con la prosa:** ~150 tokens de salida (mediana 145) además de
   leer el prompt (~1.700 tokens). En la GTX 1650 serían ~16 s + ~3–4 s. El
   "prompt corto" ya no sirve: la prosa necesita el texto de los casos. Medir
   en el PC de 8 GB con `comparar_modelos.py --modos prosa`.
9. **Medir otros modelos en el PC de 8 GB:** el script ya existe (sección
   2.10): `python scripts\comparar_modelos.py --modelos
   qwen3:4b-instruct,qwen3:8b --ruta rrf`. Con `--ruta rrf` los dos modelos
   ven los mismos casos y el reranker no ocupa VRAM (el 8B no cabe con él).
   Aparte, con la configuración real (`RUTA=auto`, reranker en GPU), correrlo
   sin `--modelos` para confirmar `gpu_pct` = 100 con el reranker cargado.
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
| `UMBRAL_ABSTENCION` | `0.52` | primera guarda: coseno del mejor caso (0,45 hasta el 28/09; sin reranker no bloqueaba cetoacidosis) |
| `KEEP_ALIVE` | `2h` | se envía en cada llamada a Ollama: el modelo no se descarga entre preguntas |
| `UMBRAL_RERANK` | `0.2` | segunda guarda (ruta rerank): prob. mínima por caso; < 2 casos → abstención |
| `RUTA` | `auto` | `rerank` con ≥ 7 GB de VRAM (`VRAM_MIN_RERANK`); `rrf` en la GTX 1650 y sin GPU |
| `DISPOSITIVO_RERANKER` | `auto` | `cuda` con ≥ 7 GB; con `qwen3:8b` en 8 GB poner `cpu` y `RUTA=rrf` |
| `PROFUNDIDAD_RERANK` | `30` | candidatos de RRF que se reordenan |
| `SIMILITUD_BAJA` | `0.55` | por debajo, la respuesta avisa que los casos pueden no corresponder |
| `TOP_K` | `5` | con 10 el prompt no cabe en 3.072 y se vuelve lento (~55 s) |
| `PRECISION_CONSULTA` | `bf16` | con 16 GB de RAM se puede usar `fp32` |
| `DISPOSITIVO_CONSULTA` | `cpu` | dejar en CPU: la VRAM es para el generador |
| `INDICE_DIR` | `rag_pipeline/indices` | apuntar a una copia para correr scripts con el chatbot abierto |
