# Contexto de trabajo — rag_trauma

Resumen para retomar el trabajo. Última sesión: **19 de octubre de 2026**, en
el **PC de 8 GB** (RTX 4060), preparado desde un clon limpio. **Desde el 17/10
solo se mide en la RTX 4060** (decisión del usuario): la GTX 1650 ya no es
objetivo. **Empezar por la sección 00** (libros de trauma: fases 1 a 6 hechas,
cobertura de los libros corregida el 19/10 con `UMBRAL_GUIAS` recalibrado, y
`PROSA_SIEMPRE` decidido para la demo; sigue la revisión médica, fase 7),
después 0, 1, 5 y 6.

**Antes, desde el 23/10: indexar el ATLS 10.ª ed. en español** (pedido).
Plan en **`PLAN_INDEXAR_ATLS10.md`** (raíz). El PDF
(`anonimizacion/data/ATLS_10_ed_espanol.pdf`, fuera de git) parece traducción
automática de la edición anterior. **Fase 1 (extracción) hecha el 23/10**:
689 secciones (`ATLS10-03-005`), 1,12 M caracteres; el ATLS 11 y el HUV
salen idénticos byte a byte. **Fase 2 hecha el 23/10**: anotaciones del
ATLS 10 en las 41 consultas (falta validarlas), 38 preguntas sintéticas,
tabla de capítulos 10.ª → 11.ª y 10 consultas fuera de dominio nuevas (neg-11
a neg-20, en los dos archivos). Esas consultas destaparon que, con los
umbrales de hoy, responderían 3 de 20 fuera de dominio (neg-05, neg-15 ACV,
neg-19 insuficiencia cardiaca); no se movió ningún umbral (decisión 5 del
plan). **Fase 3 hecha el 23/10**: la colección `guias` tiene los tres libros
(4.013 fragmentos), pero **`ATLS10=0` por defecto** lo deja fuera de la
búsqueda: la app responde como antes (línea base reproducida). No encenderlo
hasta la fase 5: el prompt y el verificador no lo conocen. Con `ATLS10=1`, el
ATLS 10 deja sin libros a 9 de 25 consultas de trauma (antes 12), pero
pasarían 3 de 20 fuera de dominio por los libros. Sigue la fase 4 (cuota por
libro y guarda). Cinco decisiones para el usuario; la
primera es si quien lo pidió sabe lo de la traducción.

**Próxima reunión (pedido del 19/10): "un chatbot mucho más sofisticado".**
Plan en **`PLAN_CHATBOT_CLINICO.md`** (raíz), sin empezar: que pregunte lo que
falta, que ordene la respuesta como el ATLS (xABCDE), una ficha del paciente
para seguir la conversación y abreviaturas expandidas para buscar en los
manuales. Orden: fase 0 (traer las notas de la demo y línea base) → 4 → 3 → 1
→ 2. Tiene cuatro decisiones abiertas para el usuario.

---

## 00. Libros de trauma (desde el 07/10)

**Cobertura de los libros corregida el 19/10** (`PLAN_INDEXAR_LIBROS.md`,
fase 1, "Cambios del 19/10"): entraba el 74 % del texto del ATLS y el 77 % del
HUV; lo que faltaba era casi todo a propósito, salvo tres huecos:
- el final del capítulo 24 del ATLS (`fin` 350 → 355: ATLS-OE, Hartford,
  resumen);
- **los 23 flujogramas y 3 tablas del HUV, transcritos a mano** en
  `anonimizacion/data/transcripciones_guias.yaml` (fuera de git, viaja con
  los PDF). Antes sus cajas entraban sueltas, sin las flechas;
- los pies de foto y dibujo del HUV, que entraban como 99 recuadros, ahora se
  descartan.

Ningún ID se corrió. 935 secciones, 2.201 fragmentos, 236 tests. **Las
transcripciones las tiene que revisar el médico** (fase 7): las dudas de
lectura están en sus `notas`.

**`UMBRAL_GUIAS` 0,2 → 0,34 (19/10, decisión del usuario).** Con la tabla del
quemado transcrita, `neg-09` ("hemorragia digestiva alta por úlcera péptica")
pasaba la guarda de los libros con 0,326 (antes el máximo fuera de dominio era
0,173). Con 0,34 vuelve a pasar 0 de 10 por los libros, y `voc-04` queda sin
libros (12 de 25 de trauma sin libros, antes 11; igual la responden los
casos). Las dos guardas juntas (`calibrar_umbral.py`): las 25 de trauma
responden (13 con las dos fuentes, 12 solo casos); fuera de dominio solo
`neg-05` (apendicitis, por los casos, como desde el 27/09). Margen corto por
los dos lados (`fin-04` 0,351): agregar consultas fuera de dominio antes de
moverlo.

**Fase 3 hecha el 17/10** (`PLAN_INDEXAR_LIBROS.md`, fase 3): colección
`guias` en el mismo Qdrant, 2.272 fragmentos `fijo-256` (2.201 desde el 19/10), con
`python -m rag.index --fuente guias`. Reindexar una fuente ya no borra la
otra. Leyendo de Qdrant reproduce la fase 2 (R@ppto densa 0,406 en las
anotadas, 0,787 en las sintéticas).

**Fase 4 hecha el 17/10** (`PLAN_INDEXAR_LIBROS.md`, fase 4):
`retrieve.buscar_guias(consulta)` devuelve `FragmentoGuia` (tipo aparte de
los casos) hasta `PRESUPUESTO_GUIAS` (1.400 tokens), o nada si el mejor
fragmento no pasa la guarda. En el PC de 8 GB, `rerank` con `UMBRAL_GUIAS`
0,2: ninguna fuera de dominio pasa, las 16 de procedimiento responden y 11 de
25 de trauma se quedan sin libros (les responden los casos); desde el 19/10,
0,34 y 12 de 25 (arriba). Sin el reranker
en GPU, `densa` con `UMBRAL_GUIAS_COSENO` 0,60 (respaldo; deja sin libros a
16 de 25). Calibración: `scripts/calibrar_guias.py`; las dos guardas juntas:
`calibrar_umbral.py`.

**Decisión abierta para el usuario o el médico: ¿cuota por libro?** Con el
reranker, el ATLS se lleva el 34 % de los fragmentos (la consulta en español
prefiere el HUV). La mitad del presupuesto para cada libro sube el ATLS
(0,203 → 0,318) y baja el HUV (0,572 → 0,494); el total no cambia más allá
del ruido. Tabla en el plan, fase 4.

**Fase 5 hecha el 17/10** (`PLAN_INDEXAR_LIBROS.md`, fase 5, con la tabla
y las iteraciones): `generate.responder` usa las dos guardas (se abstiene solo
si no pasa ninguna), prompt con los dos bloques (solo casos: el de siempre,
sin cambios), cita por sección `[guía: HUV-11-003]`, verificador con las
reglas 7 a 11 (las de los casos no cambian) y **respaldo por fuente**: cada
parte en prosa si sus oraciones pasan, si no en determinista. Medido en 41
consultas: con las dos fuentes, 7 de 21 en prosa completa y 18 de 21 con
alguna parte en prosa. Con libros, 7,6 s de mediana (sin libros 5,3 s).
`GUIAS=1` por defecto, `NUM_CTX` 6.144 con libros, `MAX_TOKENS_PROSA` 600.
**Desvío del plan:** se quitó la oración en que el modelo comparaba casos y
manuales (juzgaba la atención y nunca pasaba); lo dice el código, en general:
`AVISO_DOS_FUENTES`.

**`PROSA_SIEMPRE`: decidido para la demo (18/10, usuario): `True`, todas las
respuestas en prosa.** La verificación corre igual (`Respuesta.motivos`,
`prosa_verificada`) pero no reemplaza nada ni se muestra la nota. Riesgo
conocido y aceptado para la demo: con los libros, la prosa sin verificar puede
citar la sección equivocada o agregar detalles clínicos. `preparar_demo.py`
marca "prosa SIN verificar" para saberlo antes. Después de la demo, volver a
pensarlo (con `False`, respaldo por fuente y nota arriba). Los tests fijan
`PROSA_SIEMPRE` en cada caso y pasan con cualquier valor.

**Fase 6 hecha el 17/10** (`PLAN_INDEXAR_LIBROS.md`, fase 6): la app tiene
el apartado "Fragmentos de los manuales" (cada pasaje con libro, capítulo,
sección, página y texto; el ATLS en inglés), el interruptor "Manuales de
trauma" en la barra lateral, los textos con las dos fuentes y **la nota de
verificación arriba de la respuesta** (pendiente 13), que dice qué parte va
armada desde las fuentes y por qué. `preparar_demo.py` revisa los manuales y
pasa 6 consultas (la nueva, `6 · Solo manuales`, es `proc-09` de
`consultas_guias.yaml`); el 17/10 terminó en "Todo listo". Guion de la demo
en `SETUP.md` §6, al día (sin botones, PC de 8 GB, dos fuentes).

**Respuesta por apartados (18/10, pedido del usuario para la demo):**
`generate.armar_secciones` arma la respuesta en `Respuesta.secciones`:
"Casos similares del registro" (prosa sobre los casos + línea de respaldo;
sin tabla, pedido del usuario) y "Manuales de trauma (ATLS y HUV)" (prosa sobre los
manuales; sin lista de pasajes, pedido del usuario: están en el desplegable;
sin prosa, `aviso_pasajes` remite a él), más el aviso de que pueden no
coincidir. La app pinta cada apartado con su desplegable debajo
(`app.cuerpo_respuesta`). La prosa se parte por fuente con
`verificar_por_fuente` (oraciones con `[guía]` → manuales; el resto → casos):
con `PROSA_SIEMPRE` se muestran las dos partes; sin él, solo las que pasaron
(`prosa_por_fuente`). Si una fuente no responde, su apartado lo dice.
`Respuesta.texto` es la unión de los apartados (historial, scripts).

**Pantalla para la demo (19/10, pedido del usuario):** la app no muestra la
línea "*Respaldo: n casos · similitud…*" (`build_prompt.RX_LINEA_RESPALDO`, con
test que la ata al formato de `linea_respaldo`) ni el apartado "pie" con
`AVISO_DOS_FUENTES` (`app.APARTADOS_OCULTOS`); el aviso de similitud baja sí
se muestra. Las citas `[caso]` y `[guía]` van en rojo (`app.citas_en_rojo`,
`:red[…]` de Streamlit). Solo cambia la pantalla: `Respuesta.texto` y los
scripts siguen igual. Que las fuentes pueden no coincidir queda en "Alcance"
(barra lateral) y en el guion (`SETUP.md` §6).

**Lo que sigue con los libros:**
1. ~~Decidir `PROSA_SIEMPRE`~~ decidido para la demo: `True` (arriba).
2. **Fase 7, revisión médica:** planilla con la prosa nueva y la pregunta "¿la
   guía citada corresponde a la situación?"; mirar las oraciones del ATLS
   (solo se verifican cita y cifras), la decisión abierta de la cuota por
   libro y **las 25 transcripciones del HUV** (19/10).
3. Validar las 51 consultas anotadas (usuario o médico). Si cambian, repetir
   `calibrar_guias.py` (segundos) y, si cambia la fragmentación,
   `FRAGMENTACION_GUIAS` y reindexar (~20 s).
4. Revisar a mano `guias/muestra_revision.md` (fase 1). ~~Visto el 17/10: un
   "Recuadro: Foto. Herida por arma de fuego…" del HUV (cap. 23) entra como
   pasaje~~ corregido el 19/10: los pies de foto y dibujo se descartan. La
   muestra es del 07/10; regenerarla con `extraer_guias.py --muestra 30`.
5. Ideas medibles: expandir abreviaturas de la consulta solo para los libros
   (11 de 25 de trauma quedan sin libros, casi todas por abreviaturas u
   ortografía); "ocluar"/"ocluye" y "quinto"/"5to" rechazan oraciones del
   HUV por la forma de la palabra.

En la reunión de la semana del 30/09 pidieron no indexar solo casos (sesgo):
se agregan dos libros como segunda fuente, **con el mismo peso que los casos**
(decisión del usuario). Todo el plan y su estado están en
**`PLAN_INDEXAR_LIBROS.md`** (raíz). Resumen:
- **Fase 1, extracción (hecha):** `rag/guias.py` + `guias/perfiles.yaml` →
  `guias/secciones.jsonl`, con 926 secciones (ATLS 676, HUV 250); 935 desde
  el 19/10 (ATLS 682, HUV 253). Falta revisar a mano
  `guias/muestra_revision.md` contra el PDF.
- **Fase 2, fragmentación (hecha el 08/10, provisional):** 7
  configuraciones en `rag/fragmentar.py`, comparadas con
  `scripts/comparar_fragmentacion.py` sobre 51 consultas anotadas
  (`eval/consultas_guias.yaml`, **falta que el usuario o el médico las
  valide**) y 89 sintéticas (`eval/preguntas_sinteticas_guias.jsonl`). Dos
  corridas: densa, bm25 y rrf en la 1650; el reranker en **Kaggle** (en la
  1650 tardaba más de 20 h). Resultados en `reports/comparar_fragmentacion*.csv`
  y en `PLAN_INDEXAR_LIBROS.md`, fase 2. Resumen:
  - **`fijo-256` + densa** es lo mejor en las anotadas (R@ppto 0,401);
  - **`rerank-densa`** gana en las sintéticas (0,876) y empata dentro del
    ruido en las anotadas. Es la mejor guarda: ninguna consulta fuera de
    dominio pasa de 0,173;
  - BM25 sobre los libros (en una lista fusionada) saca al ATLS del
    presupuesto: **descartado**.

**Kaggle** (dataset privado `kaggle-rerank`): sube solo `rag/`, `eval/*.py`,
`consultas_guias.yaml`, `preguntas_sinteticas_guias.jsonl`,
`scripts/comparar_fragmentacion.py`, `guias/secciones.jsonl`,
`perfiles.yaml` y `guias/cache/*.npy`. **Nunca `indices/`, `eval/revision/`
ni nada de `anonimizacion/`**, porque ahí hay casos. El zip se arma con
`tar.exe -a -c -f <zip> -C <carpeta> rag_pipeline`: `Compress-Archive` (PS
5.1) escribe las rutas con `\` y Kaggle lo rechaza.

**Qué NO viaja por git** (está en `.gitignore`):
- `anonimizacion/data/`: `trauma.csv` (PHI), los dos PDF (80 y 185 MB, con
  derechos de autor) y, desde el 19/10, `transcripciones_guias.yaml` (los
  flujogramas y tablas del HUV transcritos a mano: **no se regenera**, se
  copia con los PDF). Desde el 07/10 `trauma.csv` vive ahí, no en
  `anonimizacion/`.
- `rag_pipeline/guias/*` salvo `perfiles.yaml`: el texto extraído se
  regenera en cada PC.

**Pasos para preparar los libros en un PC nuevo, en orden** (hechos en la 1650
el 08/10; sirven para el PC de 8 GB):
1. **ANTES del `git pull`**, copiar `trauma.csv` a su carpeta nueva. El
   commit del 07/10 lo saca de git (lo mueve a `data/`, fuera de git) y, para
   git, eso es un borrado: el pull lo eliminaría del disco.
   ```powershell
   mkdir anonimizacion\data
   copy anonimizacion\trauma.csv anonimizacion\data\
   ```
2. Copiar los PDF (USB o red) a `anonimizacion\data\`:
   `ATLS_11thEdition_CourseManual_ENG_17.pdf`,
   `LIBRO PACIENTE TRAUMATIZADO MANUAL DE MANEJO INICIAL.pdf` y, desde el
   23/10, `ATLS_10_ed_espanol.pdf` (sin él, `extraer_guias.py` se detiene:
   `--libro atls` o `--libro huv` extraen uno solo, pero sobrescriben
   `secciones.jsonl`). Con ellos, `transcripciones_guias.yaml` (19/10; sin
   él, `extraer_guias.py` avisa y la huella del paso 5 no coincide).
3. `git pull`.
4. Instalar y extraer:
   ```powershell
   cd rag_pipeline
   ..\.venv\Scripts\python.exe -m pip install "pymupdf==1.28.2"
   ..\.venv\Scripts\python.exe scripts\extraer_guias.py
   ..\.venv\Scripts\python.exe -m pytest tests/ -q        # 271 desde el 23/10
   ```
   Después de extraer: `..\.venv\Scripts\python.exe -m rag.index --fuente guias`.
   **Versión exacta de PyMuPDF a propósito:** otra versión podría extraer
   distinto, los IDs de sección (`ATLS-07-019`) se correrían y
   `eval/consultas_guias.yaml` quedaría apuntando a otras secciones.
5. Comprobar que la extracción es idéntica a la del PC de 8 GB.
   `extraer_guias.py` imprime una huella por libro (23/10): ATLS
   `682 0fb5ef517866`, HUV `253 6c83224cbb24`, ATLS10 `689 3bbb01c493d0`.
   Las tres juntas, con este comando, **`1624 a15bc4fb7152`** (ATLS y HUV
   solos, `935 9885b7f88286`, como desde el 19/10; hasta entonces
   `926 1937563090e0`):
   ```powershell
   ..\.venv\Scripts\python.exe -c "import json,hashlib; ids=[json.loads(l)['id']+'|'+' > '.join(json.loads(l)['titulos']) for l in open('guias/secciones.jsonl',encoding='utf-8')]; print(len(ids), hashlib.sha1('\n'.join(ids).encode()).hexdigest()[:12])"
   ```
6. Las corridas:
   ```powershell
   ..\.venv\Scripts\python.exe scripts\generar_preguntas_guias.py      # Ollama abierto
   # cerrar Ollama: libera la VRAM para el reranker
   ..\.venv\Scripts\python.exe scripts\comparar_fragmentacion.py --rutas densa,bm25,rrf,rerank
   ```
   Salida: `reports/comparar_fragmentacion.csv` y la tabla en consola. Decide
   la columna **R@ppto**: secciones correctas que caben en los 1.400 tokens
   del prompt.
7. La validación de las anotaciones puede ir antes o después: si cambian,
   se repite el paso 6. Los embeddings quedan en `guias/cache/` y la
   repetición tarda segundos.

**PHI:** el commit del 07/10 saca `trauma.csv` del repo de aquí en adelante,
pero sigue en el historial de GitHub. Limpiar el historial sigue pendiente
(pendiente 0).

---

## 0. Para retomar (estado al 28/09)

**Principio que manda sobre todo lo demás:** es un **asistente al
diagnóstico**. Desde el 28/09 la respuesta es una **sugerencia en prosa**, como
la daría un asistente médico: **el modelo redacta, el código verifica**
(`rag/verificar.py`). Si la prosa nombra algo que no consta en los casos, cita
mal, cuenta mal u ordena en vez de orientar, se muestra la versión armada por el
código (sección 2.10). Regla de siempre: nada inventado, todo sale de los casos
con `[caso: ID]`; el origen se revisa en "Casos consultados" de la app.

**Estado del código:** lo del 26/09 está en `a9f9f1c` y lo del 27–28/09
(secciones 2.8–2.11) en `dfbc688`; los dos están en `origin/main`. `71214d9`
(contexto) también está en `origin/main`. Los cambios del día de la demo
(sección 2.12) entraron en `45662e9`, **salvo `CONSULTAS_DEMO.md` y
`PREGUNTAS_DEMO.md`**, que siguen solo en la GTX 1650. No agregar a git la
captura de pantalla ni `desktop.ini`.

**Lo primero después de la demo:**
1. **Pasar a limpio las notas de `PREGUNTAS_DEMO.md`** (respuestas, consultas que
   escribieron los asistentes, acuerdos) y convertirlas en pendientes aquí.
2. ~~**Decidir `PROSA_SIEMPRE`**~~ decidido para la demo (18/10, usuario):
   `True`; repensarlo después de la demo (sección 00).
3. ~~Commit de los cambios de la demo~~ hecho en `45662e9` (salvo los dos `.md`
   de arriba).

**Urgente, después de la demo — PHI en GitHub:** `anonimizacion/trauma.csv`
(datos crudos), `outputs/log_reemplazos.csv` (texto original de cada
reemplazo), `reports/residuo_phi.csv` y `reports/muestra_revision_manual.csv`
están versionados desde `5c3a035` y ya están en `origin/main`
(github.com/jestronga/rag_trauma). No se tocó antes de la demo: sacarlos de git
haría que el `git pull` en el PC de la demo los borre del disco. Ver
pendiente 0.

**Primeros pasos, en el PC de 8 GB** (1 a 4 hechos el 17/10, sección 3):
1. ~~Tests~~ pasan los 271 de `rag_pipeline\` (23/10) y los 83 de `anonimizacion\`
   (19/10).
2. ~~Modelos~~ `qwen3:4b-instruct` y el reranker descargados.
3. ~~`calibrar_umbral.py`~~ `cuda`, 0,56 s de mediana por consulta.
4. ~~`comparar_modelos.py --modos prosa`~~ fase 5 de los libros: 5,3 s de
   mediana sin libros y 7,6 s con libros, generador 100 % en GPU.
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
  .venv/                 entorno compartido (Python 3.12 en el PC de 8 GB) — NO se copia entre PCs
  requirements.txt       instala las dos partes (-r a cada carpeta)
  README.md              visión general
  CONTEXTO_SESION.md     este archivo
  CONSULTAS_DEMO.md      6 consultas con prosa verificada para la demo, reservas y cuáles evitar
  PREGUNTAS_DEMO.md      preguntas para los médicos en la demo, con espacio para notas y acuerdos
  anonimizacion/         pipeline de limpieza y anonimización
    data/                fuera de git (07/10): trauma.csv (DATOS CRUDOS CON PHI) y los
                         dos libros (ATLS 11.ª ed., manual del HUV) — ver PLAN_INDEXAR_LIBROS.md
    outputs/             corpus_indexable.jsonl (8.198 casos), corpus.parquet, excluidos_rag.csv
    reports/             residuo_phi.csv, muestra_revision_manual.csv, reporte HTML
  rag_pipeline/          índices, evaluación y chatbot
    app.py               chatbot Streamlit
    .streamlit/          config.toml: no recargar rag/ en caliente (ver sección 6)
    rag/                 config, encoder, retrieve, index, build_prompt, hallazgos, generate,
                         verificar, guias (extracción de los libros), fragmentar
    guias/               perfiles.yaml (en git) y el texto extraído de los libros (no está en git)
    indices/             BM25 + Qdrant local (casos: 8.198 puntos; guias: 2.201) — no está en git
    tests/               271 tests del extractor, los libros, la respuesta y el verificador (sin Ollama ni índices)
    scripts/             revisar_entorno, probar_respuestas, exportar_revision, preparar_demo,
                         calibrar_umbral/_guias, comparar_modelos/_fragmentacion, extraer_guias
    eval/                consultas.yaml (35 consultas), consultas_guias.yaml (51, libros), evaluate.py,
                         revision/ (planilla médica, no en git)
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

### 2.12 Día de la demo: prosa siempre e interfaz nueva (28/09, sin commit)
El usuario corrió el chatbot en la 1650 y una consulta ("arma de fuego en
abdomen") salió con la versión determinista: la prosa no pasó la verificación y
el motivo solo aparecía en un caption gris al pie, que no se veía. Pidió, para
la demo, **solo prosa y debajo los casos consultados**.
- **`PROSA_SIEMPRE`** (`rag/config.py`, por defecto `True`): `generate.responder`
  muestra la prosa aunque `verificar_prosa` la rechace. La verificación se sigue
  corriendo y queda en `Respuesta.prosa_verificada` / `motivos`. Si no hay prosa
  (sin elementos elegibles o JSON cortado), sale la determinista.
  `app.py` oculta el aviso de "no pasó la verificación" con `PROSA_SIEMPRE`.
  **Temporal: va contra la regla de nada inventado** (sección 0).
- **Interfaz (`app.py`):**
  - título "Asistente Clínico de Trauma" (también el de la pestaña), en una
    cabecera con franja rosada, borde e ícono rojos y el lema "Apoyo al
    diagnóstico basado en casos similares del registro de trauma";
  - **se quitaron los botones de demostración** (y `consultas_demo`, el import
    de `yaml` y su CSS) y el subtítulo. Los botones siguen en
    `eval/consultas.yaml` (campo `boton`) por si se vuelven a poner;
    `preparar_demo.py` los sigue usando. Toda consulta escrita se reescribe con
    el historial (antes los botones lo evitaban);
  - tres tarjetas de bienvenida (describa / revise / verifique) solo mientras no
    hay conversación;
  - la respuesta nueva aparece letra por letra (`escribir_de_a_poco`,
    `st.write_stream`, como máximo ~2 s; el texto ya está completo) y el
    spinner dice "Pensando…";
  - letra más grande: `baseFontSize = 18` en el tema y los textos pequeños
    (casos, etiquetas, avisos, tiempo) subidos.
- **Tema (`.streamlit/config.toml`, `[theme]`):** claro, blanco con rojo
  `#C8102E`, fondo secundario `#FCEEEF`. Cambios del tema requieren reiniciar
  Streamlit.
- **`SETUP.md` §6 quedó desactualizado:** el guion habla de "botones 1-5". Las
  consultas de la demo están ahora en `CONSULTAS_DEMO.md`: elegidas de
  `reports/comparar_modelos.csv` (prosa verificada, similitud ≥ 0,57, 3-5 casos
  citados), con reservas y las que conviene evitar (se abstienen con 0,52 o su
  prosa no pasa).
- El usuario pidió **no correr pruebas extra** en estos cambios: no se corrieron
  tests ni la app. Si falla algo al abrirla, empezar por `app.py`.

---

## 3. Equipos

- **GTX 1650 (demo del 28/09; desde el 17/10 ya no se mide en ella):** Windows 11, 16 GB de RAM, 4 GB de VRAM. Ollama 0.34.4
  con `qwen3:4b-instruct` (también está `qwen3:4b`, la variante Thinking, que
  no sirve). Con salida estructurada ~16 s; con la prosa, ~20 s estimados.
  Ruta `rrf` (el reranker no cabe).
- **PC sin GPU** (Intel UHD, 7,7 GB de RAM): donde se trabajó el 27–28/09. Todo
  corre en CPU (~2 min por respuesta); sus tiempos no representan a los otros.
- **PC de 8 GB de VRAM (RTX 4060, 16 GB de RAM):** destino final, con
  reranker en GPU. Preparado el 17/10 desde un clon limpio: `.venv` con
  **Python 3.12** (no el 3.13 de la Store), torch 2.14.1+cu126, Ollama 0.40.6
  con `qwen3:4b-instruct`. Dependencias más nuevas que en los otros PC
  (transformers 5.20, pandas 3.0, qdrant-client 1.19) y pasan todos los tests.
  Medido: casos indexados en 53 s y libros en 17 s; reranker 0,56 s de mediana
  (`calibrar_umbral.py`, `cuda`, mismos números que el 27/09); **una respuesta
  con prosa, ~5 s** con todo cargado (3 de 3 verificadas); generador 100 % en
  VRAM (3,0 GB) con el reranker cargado. La primera consulta de un proceso
  tarda ~45 s por la carga de bge-m3 y del reranker (la app los precarga).

---

## 4. Cómo instalar en un PC nuevo

1. **Copiar el proyecto sin `.venv/`**, porque tiene rutas absolutas.
   `anonimizacion/data/trauma.csv` tiene datos personales: transferirlo por un
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
   setx LLAMA_ARG_CACHE_RAM 0         # Ollama >= 0.40: sin caché de prompts en RAM (sección 6)
   ```
4. Comprobar el entorno: `cd rag_pipeline; python scripts\revisar_entorno.py`
5. Índices (no están en git): `python -m rag.index` desde `rag_pipeline\`, y
   para los libros `python -m rag.index --fuente guias` (antes, la sección 00).
   Si falta `anonimizacion/outputs/`, antes: `cd ..\anonimizacion; python run_pipeline.py`.
6. Tests: `python -m pytest tests/ -q` (271).
7. Probar: `python scripts\probar_respuestas.py "herida por arma de fuego en tórax con hipoventilación"`
8. Antes de una demo: `python scripts\preparar_demo.py` (debe decir "Todo
   listo" y 100 % en GPU).
9. Chatbot: `streamlit run app.py` (**desde `rag_pipeline\`**).

---

## 5. Pendientes, por prioridad

**Nuevo desde el 07/10, antes que todo lo demás:** libros de trauma, sección 00
y `PLAN_INDEXAR_LIBROS.md` (fases 1 a 6 hechas; sigue la 7, revisión médica).

**`CONSULTAS_DEMO.md` y `PREGUNTAS_DEMO.md` nunca entraron a git** (el commit
`45662e9` llevó `app.py`, `config.py` y `generate.py`, no esos dos): solo
existen en el PC de la GTX 1650. Traerlos para el pendiente 12.

0. **Sacar la PHI de git** (sección 0), cuando el usuario lo decida:
   revisar si el repositorio es o fue público; `git rm --cached` de los
   cuatro archivos y agregarlos a `.gitignore` (antes, copiarlos a salvo en
   cada PC: el `pull` los borraría); reescribir el historial (`git
   filter-repo`) y forzar el push. Si el repositorio fue público, tratar los
   datos como expuestos. Mientras tanto, no hacer `git add` de esas rutas.
1. **Revisión médica**, ahora con la prosa (sección 2.10: el verificador no ve
   el razonamiento clínico). Entregar `eval/revision/revision.html` y
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
3. **Push** del commit local con esta actualización del contexto, cuando el
   usuario lo pida (antes, ver pendiente 0).
4. **Resumen por identificador** (`resumir_caso`): todavía lo escribe el
   modelo y cambia términos: "neumoconiótico" por "neumógeno", y el 26/09
   "empema" por "Empiema". Hacerlo determinista, como el resto.
5. ~~Recalibrar `UMBRAL_ABSTENCION`~~ **Hecho el 27/09 con el reranker**
   (sección 2.8). **Medido en el PC de 8 GB el 17/10:** `cuda`, 0,56 s de
   mediana, y el generador sigue 100 % en VRAM con el reranker cargado
   (sección 3).
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
   "prompt corto" ya no sirve: la prosa necesita el texto de los casos.
   **En el PC de 8 GB, ~5 s por respuesta** con todo cargado (17/10, 3
   consultas). ~~Falta la corrida completa~~ hecha el 17/10 (fase 5 de los
   libros, 25 consultas de trauma): 5,3 s de mediana sin libros (peor 6,4) y
   7,6 s con libros (peor 11,5).
9. **Medir otros modelos en el PC de 8 GB:** el script ya existe
   (`scripts/comparar_modelos.py`, secciones 2.9 y 2.10): `python scripts\comparar_modelos.py --modelos
   qwen3:4b-instruct,qwen3:8b --ruta rrf`. Con `--ruta rrf` los dos modelos
   ven los mismos casos y el reranker no ocupa VRAM (el 8B no cabe con él).
   Aparte, con la configuración real (`RUTA=auto`, reranker en GPU), correrlo
   sin `--modelos` para confirmar `gpu_pct` = 100 con el reranker cargado.
10. **Revisar 33 hallazgos de datos personales** en
    `anonimizacion/reports/residuo_phi.csv`.
11. ~~Restos de truncados en `consultas.yaml` y en el guion de `SETUP.md`~~
    Hecho el 28/09 (sección 2.11).
12. **Después de la demo (sección 2.12):** notas de `PREGUNTAS_DEMO.md` →
    pendientes; ~~decidir `PROSA_SIEMPRE`~~ decidido para la demo (18/10):
    `True` (sección 00). ~~Actualizar el guion de `SETUP.md`
    §6; correr los tests y abrir la app~~ hecho el 17/10 (fase 6 de los libros:
    220 tests, app probada con `AppTest`, `preparar_demo.py` en "Todo listo").
13. ~~**Motivo visible si vuelve la versión determinista**~~ hecho el 17/10:
    con `PROSA_SIEMPRE=0`, una nota arriba de la respuesta dice qué parte va
    armada desde las fuentes y por qué (`app.nota_verificacion`).

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
- **Ollama ≥ 0.40 guarda en RAM una caché de prompts que crece hasta 8 GB**
  (motor llama.cpp, `--cache-ram`, 8.192 MiB por defecto). El 17/10, tras las
  mediciones con prompts de ~4.000 tokens, llegó a 8,8 GB; con la app (7,2 GB)
  la memoria comprometida tocó el límite (29,3 de 30,6 GB) y Ollama se colgó
  guardando un prompt: la app decía "Cargando modelos…" minutos y terminaba en
  `TimeoutError: timed out`. Síntoma en `%LOCALAPPDATA%\Ollama\server.log`:
  `prompt_save` seguido de `500 | 3m0s`. **Arreglo: `setx LLAMA_ARG_CACHE_RAM 0`**
  y reiniciar Ollama (el log dice "prompt cache is disabled"). Aquí no ayuda:
  cada consulta trae casos y pasajes distintos, y leer un prompt de 3.000
  tokens tarda < 1 s en la 4060. Para revertir: `setx LLAMA_ARG_CACHE_RAM ""`.
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
| `NUM_CTX` | `6144` con `GUIAS`, `3072` sin | se agranda sola para la llamada si el prompt no cabe (y Ollama recarga el modelo) |
| `MAX_TOKENS` | `250` | `900` con `SINTESIS_LLM=1` |
| `MAX_TOKENS_RESUMEN` | `700` | resumen de un caso por identificador |
| `TIMEOUT_LLM` | `400` | |
| `SINTESIS_LLM` | apagado | no encender sin revisar respuestas |
| `SALIDA` | `prosa` | `sugerencias` = siempre la respuesta determinista |
| `PROSA_SIEMPRE` | `1` | prosa aunque no pase la verificación; decidido para la demo (18/10), repensar después (sección 00) |
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
| `GUIAS` | `1` | los libros como segunda fuente; `0` o sin la colección = solo casos, como antes |
| `ATLS10` | `0` | el ATLS 10.ª ed. en español: indexado desde el 23/10 (fase 3 de `PLAN_INDEXAR_ATLS10.md`), pero apagado no se busca (`retrieve.libros_apagados`); encenderlo después de la fase 5 |
| `MAX_TOKENS_PROSA` | `600` | 400 hasta el 17/10: con los libros se cortaba el JSON |
| `COLECCION_GUIAS` | `guias` | colección de Qdrant con los libros (17/10) |
| `FRAGMENTACION_GUIAS` | `fijo-256` | configuración de `rag/fragmentar.py` al indexar los libros |
| `RUTA_GUIAS` | `auto` | libros: `rerank` con ≥ 7 GB de VRAM, si no `densa` |
| `PRESUPUESTO_GUIAS` | `1400` | tokens de fragmentos de los libros para el prompt |
| `UMBRAL_GUIAS` | `0.34` | ruta `rerank`: prob. mínima del mejor fragmento. 0,2 hasta el 19/10; con las transcripciones, fuera de dominio máx. 0,326 (`neg-09`), sección 00 |
| `GUIAS_TRANSCRIPCIONES` | `anonimizacion/data/transcripciones_guias.yaml` | flujogramas y tablas del HUV transcritos a mano (19/10); fuera de git, viaja con los PDF |
| `UMBRAL_GUIAS_COSENO` | `0.60` | ruta `densa` (sin reranker en GPU): coseno mínimo del mejor fragmento; separa mal |
