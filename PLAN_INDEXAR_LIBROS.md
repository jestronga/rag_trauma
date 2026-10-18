# Plan — indexar los libros de trauma en el RAG

Pedido de la reunión (semana del 30/09/2026): con solo casos el asistente tiene
sesgo (lo que se hizo en el HUV hasta 2018 no es necesariamente lo que se
debe hacer). Se agregan dos libros como segunda fuente.

## Las fuentes (`anonimizacion/data/`, fuera de git)

| | ATLS 11.ª ed. (Course Manual) | Paciente Traumatizado — Manual de Manejo Inicial |
|---|---|---|
| Archivo | `ATLS_11thEdition_CourseManual_ENG_17.pdf` (80 MB) | `LIBRO PACIENTE TRAUMATIZADO MANUAL DE MANEJO INICIAL.pdf` (185 MB) |
| Idioma | inglés | español (HUV / Univalle, Cali) |
| Páginas | 377 (16 casi sin texto) | 300 (26 casi sin texto) |
| Texto | capa de texto, 1,42 M caracteres (~350 k tokens) | capa de texto, 0,44 M caracteres (~110 k tokens) |
| Índice incrustado | no | no |
| Estructura | 3 secciones, ~25 capítulos; título de capítulo en fuente 34, encabezado "COURSE MANUAL \| Chapter N \| …" | 23 capítulos con página de inicio en el índice (pp. 6–7) |
| Ruido a quitar | encabezados y pies, ATLS-OE, índice alfabético (p. 356+) | bibliografía al final de cada capítulo, pies de foto ("Cortesía…"), palabras partidas en líneas sueltas |

No hace falta OCR. Estimado: ~900 fragmentos del ATLS y ~300 del manual del
HUV, frente a los 8.198 casos.

## Principio

Los libros **no reemplazan** la regla del proyecto, la amplían. **Casos y
libros pesan lo mismo** (decisión del 07/10): la respuesta se arma con las
dos fuentes, cada una con su espacio, y cualquiera de las dos basta para
responder. Hay dos tipos de afirmación y nunca se mezclan:

- **Casos:** "qué se hizo en casos parecidos". Lleva conteos y `[caso: ID]`.
- **Guía:** "qué recomienda el manual". Sin conteos, con `[guía: ID]` que
  apunta a libro, capítulo y página, y el fragmento visible en la app.

Por eso los libros van en **una colección aparte**, no mezclados con los
casos: los puntajes no son comparables, y `hallazgos.py`, la tabla y
`verificar.py` cuentan sobre casos ("2 de 5"). Si un fragmento de guía entrara
en la lista de casos, rompería todos los conteos.

---

## Decisiones (tomadas el 07/10/2026)

1. **Permiso de uso.** Los dos libros tienen derechos de autor (el ATLS es del
   American College of Surgeons). Uso local y académico, sin publicar el
   texto: el texto extraído y los índices quedan fuera de git.
2. **El ATLS queda en inglés.** bge-m3 es multilingüe: una consulta en español
   encuentra el pasaje en inglés por la vía densa (BM25 no). El modelo redacta
   en español y la app muestra el fragmento original. En las oraciones
   citadas al ATLS el verificador solo comprueba la cita y los números. No se
   traduce: con un modelo de 4B, una mala traducción se volvería "la guía
   dice".
3. **Mismo peso para casos y libros.** Consecuencias:
   - **Abstención por fuente:** se responde si pasa cualquiera de las dos
     guardas (casos o libros); se abstiene solo si no pasa ninguna. Con una
     sola fuente, la respuesta lo dice ("no hay casos similares en el
     registro" / "los manuales no tratan esta situación").
   - **Mismo espacio en el prompt:** presupuesto de tokens parecido para cada
     fuente (p. ej. 4 casos + 3 fragmentos, ~1.400 tokens cada lado), no el
     mismo número, porque un fragmento mide más que un caso.
   - **La prosa da las dos miradas:** lo que se hizo en casos similares y lo
     que recomiendan los manuales, sin que una quede como nota al pie de la
     otra. Si no coinciden, la respuesta lo dice; no elige un ganador.
   - **El respaldo determinista también tiene las dos secciones.**
   - **Las dos guardas se calibran con las mismas consultas fuera de
     dominio:** ninguna fuente puede responder sola a "cetoacidosis".

---

## Fases

### Fase 1 — Extracción y limpieza (hecha el 07/10/2026; falta la revisión a mano)

**Resultado:** `python scripts\extraer_guias.py --muestra 30` →
`guias/secciones.jsonl` con 926 secciones: ATLS 676 en 24 capítulos (1,03 M
caracteres) y HUV 250 en 23 capítulos (0,34 M). Descartado: bibliografía
(132 k + 65 k), flujogramas y rótulos sueltos (1,6 k + 7,5 k). Tests en
`tests/test_guias.py` (42).

Cómo quedó (`rag/guias.py`, reglas por libro en `guias/perfiles.yaml`):
- cada bloque recibe un rol por su fuente dominante: títulos (4 niveles por
  tamaño y color), cuerpo, pie ("Table 7-4:", "Tabla 1."), bibliografía (por
  el texto: en el ATLS viene en dos fuentes distintas) o material;
- orden de lectura: los bloques que cruzan el centro parten la página en
  franjas; en cada franja, columna izquierda y después derecha;
- el material va con su pie (tabla o figura); el que no tiene pie se agrupa
  en "Recuadro:" (NEXUS Chest, indicaciones de dren pleural, grados de
  trauma renal). **Los flujogramas se descartan** (se reconocen por "SI",
  "NO", "(+)", "¿…?"): sin las flechas no se sabe qué rama lleva a qué;
- tablas y recuadros van al final de su sección (si no, cortan oraciones);
- páginas impresas para citar (ATLS: PDF − 19), portadas con objetivos del
  ATLS fuera; capítulos no clínicos marcados (`clinico: false`): ATLS 2,
  16, 18, 19, 20 y HUV 1.

**Pendiente de la fase 1:** revisar a mano `guias/muestra_revision.md` (30
secciones) contra el PDF. Revisado por Claude: sin encabezados, sin guiones
blandos ni caracteres de Wingdings; las 11 "et al." que quedan son citas en
el texto, no bibliografía; los "hypo- and" son guiones suspendidos reales.

Plan original:
- Script `rag_pipeline/scripts/extraer_guias.py` con PyMuPDF (open source,
  local). Va a `requirements.txt`.
- Un perfil por libro (`rag_pipeline/guias/perfiles.yaml`): páginas a saltar
  (portada, colaboradores, índice alfabético, ATLS-OE) y patrones de
  encabezado y pie. Capítulos: el del HUV, desde su índice; el del ATLS, por
  el tamaño de fuente del título y el encabezado de página.
- Limpieza: unir palabras partidas y líneas sueltas, quitar la bibliografía y
  los pies de foto, conservar las tablas como texto (las del ATLS tienen
  umbrales: ABC score, PAS ≤ 90, FC ≥ 120).
- Salida: `rag_pipeline/guias/secciones.jsonl` (libro, capítulo, sección,
  páginas, texto). Fuera de git.
- **Verificación:** tests de la limpieza (sin encabezados, sin referencias,
  sin palabras cortadas) y revisión a mano de 30 secciones contra el PDF.

### Fase 2 — Fragmentación (en curso desde el 07/10/2026)

El usuario pidió comparar varias configuraciones y elegir con mediciones. Se
evalúa por SECCIÓN (cada fragmento hereda su `seccion_id`), así todas las
configuraciones se miden contra la misma anotación. Decide R@ppto: secciones
de grado 2 que caben en los 1.400 tokens del prompt. Verdad base: el conjunto
anotado decide; las ~100 preguntas sintéticas confirman la tendencia.
Pasos para la 1650: `CONTEXTO_SESION.md`, sección 00.

**Hecho:**
- `rag/fragmentar.py`: 7 configuraciones (`seccion`, `fijo-256`, `fijo-512`,
  `parrafos-350`, `parrafos-350-sin-ruta`, `oraciones-128` small-to-big,
  `semantico`). Con el tokenizador real, ninguna pasa su tope: entre 1.552
  (`fijo-512`) y 4.747 (`oraciones-128`) fragmentos.
- `eval/consultas_guias.yaml`: 51 consultas (25 de trauma, 16 de
  procedimiento, 10 fuera de dominio), con 156 secciones de grado 2 y 223 de
  grado 1, propuestas por Claude. **Falta que el usuario o el médico las
  valide.**
- `scripts/comparar_fragmentacion.py` (probado de punta a punta con
  `ENCODER=falso`) y `scripts/generar_preguntas_guias.py`.
- Tests: `test_fragmentar.py` y `test_comparar_fragmentacion.py`.
- **Ojo:** los IDs de sección (`ATLS-07-019`) salen del orden de extracción.
  Si se cambian las reglas de la fase 1 y se re-extrae, los IDs pueden
  correrse y la anotación queda desalineada. El script avisa si falta un ID
  anotado, pero no si apunta a otra sección.

**Corrida 1 (07–08/10).** 89 preguntas sintéticas generadas. densa, bm25
y rrf en la GTX 1650 → `reports/comparar_fragmentacion.csv`. rerank en
**Kaggle** (T4) → `reports/comparar_fragmentacion_rerank.csv`. En la 1650
el reranker llevaba 3 h sin terminar la primera configuración (más de 20 h
en total) y se cortó. Notebook: `scripts/kaggle_chunking_rerank.ipynb`;
dataset privado `kaggle-rerank` con el código, `secciones.jsonl` y
`guias/cache/`, nunca índices ni casos. El zip se arma con `tar.exe -a`:
`Compress-Archive` escribe `\` en las rutas y Kaggle lo rechaza.

R@ppto, sin las consultas fuera de dominio (anotado: 41 consultas;
sintético: 89):

| Configuración | densa anot. | rerank anot. | rrf anot. | bm25 anot. | densa sint. | rerank sint. |
|---|---|---|---|---|---|---|
| fijo-256 | **0,401** | 0,348 | 0,297 | 0,266 | 0,787 | 0,809 |
| semantico | 0,380 | **0,358** | 0,304 | 0,261 | **0,809** | 0,764 |
| parrafos-350-sin-ruta | 0,380 | 0,306 | 0,281 | 0,232 | 0,787 | 0,787 |
| fijo-512 | 0,349 | 0,299 | 0,272 | 0,261 | 0,730 | 0,730 |
| seccion | 0,347 | 0,299 | 0,279 | 0,244 | 0,753 | 0,764 |
| parrafos-350 | 0,337 | 0,335 | 0,277 | 0,249 | 0,764 | 0,775 |
| oraciones-128 | 0,302 | 0,322 | 0,293 | 0,217 | 0,787 | **0,831** |

Conclusiones:
- **La densa es la mejor ruta en el conjunto anotado.** BM25 casi no
  encuentra nada en el ATLS (R@ppto_ATLS ≈ 0,04: consultas en español,
  texto en inglés), y RRF sobre los dos libros baja el ATLS a ~0,07.
- **El rerank queda limitado por sus candidatos.** Reordena los 30
  primeros de RRF, que casi no traen secciones del ATLS: en el ATLS cae a
  ~0,17 (densa ~0,31), pero en el HUV es lo mejor (0,69 en `semantico`,
  frente a 0,57 de la densa). Confirma la decisión de la fase 4: BM25 solo
  sobre el HUV.
- **Para la abstención sirve el rerank, no el coseno.** Con la densa, las
  consultas fuera de dominio llegan a 0,589 y la de trauma más baja está en
  0,48–0,51: se pisan. Con el rerank, ninguna fuera de dominio pasa de 0,31
  y la mediana de trauma es ~0,70. Pero 12 de 41 consultas de trauma
  (coloquiales, abreviaturas, ortografía) también quedan bajo 0,31, y además
  se recuperan mal. Con la decisión 3 los libros se abstienen y responden
  los casos.
- **Configuración candidata: `fijo-256`**, primera o segunda en casi todas
  las combinaciones y la más simple. Las diferencias con `semantico` están
  dentro del ruido: con 41 consultas, una sola mueve unos 2,4 puntos.
  `oraciones-128` queda última con la densa en el anotado. Sacar la ruta de
  títulos no empeora (parrafos-350-sin-ruta 0,380 contra 0,337).
- Categorías débiles en todas las configuraciones: "fina" (≈0,05–0,18),
  "ortográfica" (≈0,1) y "vocabulario". Procedimiento va bien (0,54–0,71).

**Corrida 2 (08/10, Kaggle T4)** → `reports/comparar_fragmentacion_rerank2.csv`.
Rutas nuevas en `comparar_fragmentacion.py`: `rrf-huv` (densa + BM25 solo
sobre el HUV), `rerank-densa` (el reranker reordena los 30 primeros de la
densa) y `rerank-huv` (reordena `rrf-huv`), con `fijo-256` y `semantico`. La
`densa` repetida da exactamente lo mismo que en la corrida 1, así que las dos
corridas se pueden comparar. El notebook trae su propia copia del script
(`%%writefile`) y comprueba la salida antes de descargarla.

| Ruta | fijo-256 anot. | semantico anot. | fijo-256 sint. | semantico sint. |
|---|---|---|---|---|
| **densa** | **0,401** | **0,380** | 0,787 | 0,809 |
| **rerank-densa** | 0,374 | 0,371 | **0,876** | **0,888** |
| rerank (sobre rrf) | 0,348 | 0,358 | 0,809 | 0,764 |
| rerank-huv | 0,334 | 0,333 | 0,809 | 0,764 |
| rrf | 0,297 | 0,304 | 0,494 | 0,506 |
| rrf-huv | 0,254 | 0,255 | 0,461 | 0,472 |

- **rerank-densa gana claramente en las sintéticas** (en el ATLS sube de 0,74
  a 0,84). Darle al reranker los candidatos de la densa resolvió el
  problema del ATLS.
- **En las anotadas no supera a la densa.** Con fijo-256, consulta por
  consulta, mejora 6 y empeora 10; con semantico, 7 y 7. Está dentro del
  ruido. Explicación probable: las sintéticas las escribió el modelo leyendo
  la sección y comparten su vocabulario, lo que favorece al reranker. Las
  anotadas tienen consultas coloquiales, abreviaturas y errores de
  ortografía, más parecidas al uso real (en "ortográfica", rerank-densa da
  0,00).
- **BM25 solo sobre el HUV no sirve en una lista fusionada:** con `rrf-huv`,
  R@ppto_ATLS cae a 0,014. Los fragmentos del HUV suben porque los traen las
  dos listas, y llenan el presupuesto. Si se quiere BM25 para el HUV, cada
  libro necesita su propio espacio en el prompt. **Descartadas `rrf-huv` y
  `rerank-huv`.**
- **Abstención:** rerank-densa con fijo-256 es la que mejor separa: ninguna
  consulta fuera de dominio pasa de **0,173** (0,31 con el rerank sobre rrf;
  0,589 con el coseno). Aun así, 10 de 41 consultas de trauma quedan por
  debajo: las coloquiales y con errores.

**Decisión de la fase 2 (08/10, provisional hasta validar las anotaciones):**
- **Configuración: `fijo-256`**, la más estable en las dos corridas.
- **Ruta:** `densa` en la GTX 1650, porque el reranker no cabe junto al
  generador. `rerank-densa` en el PC de 8 GB, sobre todo como **guarda de
  abstención de los libros** (umbral de partida ~0,2, a calibrar). Si
  también ordena, se decide con las anotaciones validadas.
- Sin BM25 en los libros por ahora.

**Falta de la fase 2:** validar las 51 consultas anotadas (usuario o
médico). Si cambian, repetir la densa tarda segundos con la caché;
rerank-densa se repite en Kaggle (`scripts/kaggle_chunking_rerank.ipynb`).

Diseño original:
- Por sección, sin cruzar capítulos. De 250 a 450 tokens de bge-m3, con un
  párrafo de solapamiento. **Límite duro: 512**, el `max_length` del encoder;
  todo lo que pase de ahí se pierde sin aviso.
- Cada fragmento lleva su ruta de títulos al inicio ("ATLS › Chapter 7
  Disability › Cerebral contusion"): mejora la recuperación y le dice al
  modelo de qué trata.
- ID estable: `ATLS-07-012`, `HUV-11-004`. Metadatos: `fuente`, `idioma`,
  `capitulo`, `seccion`, `pagina_ini`, `pagina_fin`, `region_anatomica`
  (opcional, con el mismo léxico de los casos).
- **Verificación:** distribución de tamaños, 0 fragmentos > 512 tokens, cada
  página con texto cubierta.

### Fase 3 — Indexación (hecha el 17/10/2026, en el PC de 8 GB)

**Resultado:** `python -m rag.index --fuente guias` → colección `guias` en
el mismo Qdrant local, con **2.272 fragmentos `fijo-256`** (ATLS 1.632, HUV
640; 562 de tablas), mediana de 214 y 168 tokens, máximo 256. Todas las
secciones tienen al menos un fragmento. Embeber tarda 17 s en la RTX 4060.

Cómo quedó (`rag/index.py`):
- **Cada fuente reemplaza solo su colección** (`delete_collection`). Antes
  `construir_denso` borraba la carpeta entera de Qdrant: reindexar los casos
  habría borrado los libros.
- `--fuente casos | guias | todas`. Por defecto `casos`, como antes. Sin
  BM25 para los libros (fase 2), así que `--solo` es solo para los casos.
- Payload de cada fragmento: `record_id` (ID del fragmento,
  `ATLS-07-019-02`) y `text` (lo que se embebe, con la ruta de títulos),
  como en los casos, más `seccion_id`, `fuente`, `libro`, `capitulo`,
  `titulos`, `pagina_ini`/`pagina_fin` (páginas impresas, para citar),
  `tipo` (texto | tabla), `texto_contexto`, `n_tokens` e `idioma`.
  **`clinico` se guarda para que la fase 4 decida con mediciones** si se
  filtran los capítulos no clínicos (111 secciones; están indexadas).
- La indexación **falla** si algún fragmento pasa de 510 tokens: el
  encoder lo truncaría sin avisar.
- `meta.json` tiene una entrada por fuente (`fuentes.casos`,
  `fuentes.guias`) y migra el formato anterior.
- Config: `COLECCION_GUIAS` (`guias`) y `FRAGMENTACION_GUIAS` (`fijo-256`).
  La ruta de las secciones es el `GUIAS_SECCIONES` que ya existía (el
  `GUIAS_JSONL` del plan). **`TOP_K_GUIAS` y `GUIAS` entran en la fase 4**,
  con el código que los lee (no repetir lo de `RERANKER=1`, documentado sin
  que nada lo leyera).
- Tests: `tests/test_index.py` (6): reindexar una fuente no borra la otra,
  payload con la cita, el límite de tokens y `meta.json`.

**Verificado:**
- Extracción idéntica a la de los otros PC (`926 1937563090e0`).
- `comparar_fragmentacion.py --configs fijo-256 --rutas densa,rerank-densa`
  reproduce **exactamente** la fase 2: 0,401 y 0,374 (anotadas), 0,787 y
  0,876 (sintéticas), máximo fuera de dominio 0,173 con el reranker.
- **Leyendo de Qdrant**, con la consulta embebida como en el chatbot (CPU,
  bf16): R@ppto densa 0,406 en las anotadas y 0,787 en las sintéticas.
  `trauma` sigue con 8.198 puntos.

### Fase 4 — Recuperación y evaluación (hecha el 17/10/2026, en el PC de 8 GB)

**Resultado:** `retrieve.buscar_guias(consulta)` devuelve los fragmentos de
los libros para el prompt, o una lista vacía si los libros no tratan la
consulta. Medido con `scripts/calibrar_guias.py`, leyendo de Qdrant:

| Ruta (equipo) | Guarda | R@ppto anot. | R@ppto sint. | Trauma sin libros | Proc. sin libros | Fuera de dominio que pasan |
|---|---|---|---|---|---|---|
| `rerank` (8 GB) | prob. ≥ 0,2 | 0,331 | 0,854 | 11/25 | 0/16 | 0/10 |
| `densa` (1650) | coseno ≥ 0,60 | 0,335 | 0,787 | 16/25 | 0/16 | 0/10 |
| (sin guarda, rerank) | — | 0,374 | 0,876 | 0/25 | 0/16 | 10/10 |

Las dos guardas juntas (`calibrar_umbral.py`, decisión 3, ruta rerank): las
25 consultas de trauma reciben respuesta (14 con las dos fuentes, 11 solo con
casos). De las fuera de dominio responde 1 de 10, `neg-05` (apendicitis), y
**solo por los casos**: es el problema conocido desde el 27/09 (0,423 en el
reranker de casos). Los libros no agregan ningún fallo nuevo.

Cómo quedó (`rag/retrieve.py`):
- `FragmentoGuia`: tipo aparte de `Resultado`, para que un fragmento no
  pueda entrar en la lista de casos. Trae libro, capítulo, títulos, páginas
  impresas, tipo, idioma, `texto_contexto` (va al prompt), coseno y
  relevancia.
- `candidatos_guias(consulta, ruta)`: los 30 primeros de la densa
  (`PROFUNDIDAD_RERANK`); con `rerank`, reordenados por el cross-encoder.
  `seleccionar_guias(candidatos, umbral, presupuesto)`: guarda y presupuesto,
  sin modelos (testeable, y la calibración prueba umbrales sin repetir la
  búsqueda). `buscar_guias` = las dos con la configuración.
- **Guarda por el mejor fragmento:** si no llega al umbral, los libros se
  abstienen; si llega, el presupuesto se llena en orden. Filtrar fragmento
  por fragmento recuperaba menos (0,297 contra 0,331) y la precisión por
  tramo de probabilidad no muestra un corte limpio.
- `RUTA_GUIAS=auto`: `rerank` si el reranker cabe en la GPU (≥ 7 GB), si no
  `densa`. Latencia en la RTX 4060: 0,88 s por consulta con rerank.
- **`PRESUPUESTO_GUIAS` (1.400 tokens) en lugar de `TOP_K_GUIAS`:** lo que
  mide R@ppto y lo que pide la decisión 3. Con el umbral: 1.000 → 0,286,
  1.400 → 0,331, 1.800 → 0,354.
- **Capítulos no clínicos: no se filtran.** Medido: ±0,005 de R@ppto y una
  consulta fuera de dominio. No vale una variable.
- `index.py` guarda `n_tokens_contexto`: lo que cuenta para el presupuesto
  (difiere de `n_tokens` solo en small-to-big).
- `calibrar_umbral.py` agrega la columna "libros" y la cuenta de las dos
  guardas juntas. `scripts/calibrar_guias.py` calibra la de los libros.
- Tests: `tests/test_buscar_guias.py` (14) y uno más en `test_index.py`.
- `GUIAS=1` (encender o apagar los libros) entra en la fase 5, con
  `generate.responder`, que es quien lo lee.

**Decisión abierta (usuario o médico): ¿cuota por libro?** Con el
reranker, el ATLS se lleva el 34 % de los fragmentos aunque es el 72 % del
índice: una consulta en español prefiere el texto en español. Medido con la
mitad del presupuesto para cada libro (lo que uno no usa pasa al otro):

| rerank | R@ppto anot. | sint. | anot. ATLS | anot. HUV | % fragmentos ATLS |
|---|---|---|---|---|---|
| presupuesto común (actual) | 0,331 | 0,854 | 0,203 | 0,572 | 34 % |
| cuota por libro | 0,345 | 0,843 | 0,318 | 0,494 | 50 % |

El total no cambia más allá del ruido: es un balance entre el protocolo
local (HUV) y el estándar internacional (ATLS), no una mejora. Se queda el
presupuesto común hasta que se decida.

**Ideas para medir después:**
- 11 de 25 consultas de trauma quedan sin libros: abreviaturas ("HPAF",
  "FX", "TCE"), ortografía ("qemadura", "luxacion") y coloquiales ("sujeto
  baleado"). Expandir las abreviaturas de la consulta solo para los libros
  (hay un diccionario en `anonimizacion/config/abreviaturas.yaml`).
- La fuera de dominio más alta en el reranker de los libros es "cuál es la
  capital de Colombia" (0,173): el margen con el umbral de 0,2 es corto.
  Agregar más consultas fuera de dominio antes de bajar el umbral.

Plan original:
- `retrieve.buscar_guias(consulta, k)`: **densa sobre los dos libros**
  (fase 2: BM25 en una lista fusionada saca al ATLS del presupuesto). Con ≥ 7
  GB, el reranker sobre los 30 primeros de la densa (`rerank-densa`). Umbral
  propio de relevancia: un fragmento flojo no entra.
- Abstención: `UMBRAL_GUIAS` para los libros, al lado de `UMBRAL_ABSTENCION`
  y `UMBRAL_RERANK` para los casos (decisión 3).
- Evaluación: agregar a `eval/consultas.yaml` el capítulo esperado de cada
  libro para las 25 consultas de trauma, más algunas consultas de
  procedimiento ("¿cuándo poner un tubo de tórax?"), que se responden solo con
  los libros. Medir Recall@3 por capítulo y comparar densa contra RRF.
  `calibrar_umbral.py` calibra las dos guardas: las 10 consultas fuera de
  dominio no deben pasar ninguna.

### Fase 5 — Generación y verificación
- El prompt tiene dos bloques del mismo tamaño, "Casos similares" y
  "Fragmentos de los manuales". La prosa separa "En casos similares…" de "El
  ATLS / el manual del HUV recomienda…" y dice si coinciden o no.
- Si solo pasó una fuente, el prompt lleva solo esa y la respuesta lo dice.
- `NUM_CTX`: 4 casos + 3 fragmentos (~2.800 tokens con las reglas) no caben
  en 3.072. Pasa a 4.096; medir VRAM y tiempo en la GTX 1650 y en el PC de 8
  GB. `TOP_K` de casos baja de 5 a 4: medir antes y después.
- `verificar.py`:
  - citas `[guía: ID]` existentes;
  - en una oración citada a una guía: los números deben aparecer en el
    fragmento y, si está en español, los términos del léxico también;
  - no se atribuye a un caso lo que solo dice la guía, ni al revés;
  - sin cambios en las reglas de casos.
- Respaldo determinista: si la prosa no pasa, se muestra lo de siempre más
  "Según las guías", con título y página de cada fragmento (sin parafrasear).

### Fase 6 — Aplicación
- Expander "Fragmentos de guía consultados", con libro, capítulo, página y
  texto.
- El aviso fijo nombra las fuentes: ATLS 11.ª ed. y el manual del HUV.
- `preparar_demo.py` y los botones de ejemplo al día.

### Fase 7 — Revisión médica
- Planilla con la prosa nueva (pendiente 1 de `CONTEXTO_SESION.md`) y una
  pregunta más en la rúbrica: ¿la guía citada corresponde a la situación?
- Cada error marcado se vuelve un test.

---

## Orden de trabajo propuesto

1. ~~Decisiones 1–3~~ (07/10).
2. Fases 1–2: el resultado se revisa a ojo antes de indexar nada.
3. Fases 3–4: se mide si la recuperación funciona **antes** de tocar el
   prompt.
4. Fases 5–6, y después 7 con el médico.
