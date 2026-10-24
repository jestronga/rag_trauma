# Puesta en marcha — prototipo RAG de trauma

Objetivo: chatbot que recupera casos análogos del registro de trauma y los
describe con citas. Todo corre **en local**: ningún texto sale de la máquina.

---

## 1. Instalar

```powershell
cd C:\Users\JOSE\Documents\dev\rag_trauma
python -m venv .venv
.\.venv\Scripts\activate
```

El `.venv` vive en la raíz y lo comparten las dos partes del proyecto
(`anonimizacion\` y `rag_pipeline\`). Salvo donde se indique, los comandos
de esta guía se corren **desde `rag_pipeline\`**.

### torch con CUDA — primero, y con el índice correcto

El índice a usar depende de tu versión de Python. **Este entorno corre Python
3.13.1**, y `cu121` no publica ruedas para 3.13: sus versiones de torch se
detuvieron en Python 3.12. De ahí el error

```
ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
```

que no significa "torch no existe" sino "no hay ninguna rueda para tu Python en
este índice". Python 3.13 necesita torch 2.6 o posterior:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Si ese índice tampoco tuviera rueda, prueba en este orden:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`nvidia-smi` (esquina superior derecha) dice qué versión de CUDA soporta tu
driver: elige un índice igual o menor.

### Si ninguno funciona, no te quedas bloqueado

```powershell
pip install torch        # versión CPU, desde PyPI
```

**Ollama trae su propio runtime de CUDA y no depende de torch**, así que la
generación sigue corriendo en la GPU. Lo único que se paga con torch en CPU es
la indexación: una sola vez, 40-60 minutos en lugar de unos pocos. Déjala
corriendo mientras preparas el resto y no vuelve a hacer falta.

### El resto

```powershell
pip install -r requirements.txt      # desde la raíz: instala ambas partes
```

**Ollama** — descárgalo de [ollama.com](https://ollama.com) y luego:

```powershell
ollama pull qwen3:4b-instruct       # NO qwen3:4b (variante Thinking)
```

Comprueba que todo está en su sitio:

```powershell
cd rag_pipeline
python scripts\revisar_entorno.py
```

---

## 2. Presupuesto de VRAM en 8 GB

Esta es la restricción que manda, y la configuración por defecto ya la respeta:

| Pieza | VRAM | Dónde corre por defecto |
|---|---|---|
| bge-m3 (indexación) | ~2,3 GB | **GPU**, una sola vez |
| bge-m3 (consulta) | — | **CPU**: embeber una consulta corta cuesta ~150 ms |
| `qwen3:4b-instruct` | ~3,0 GB con `NUM_CTX=3072` (sin libros); 3,57 GB con 6.144 (con libros) | **GPU**, vía Ollama |
| bge-reranker-v2-m3 (fp16) | ~1,3 GB | **GPU** si torch ve CUDA (`RUTA=auto` → `rerank`) |

Generador y reranker suman ~4,3 GB (~4,9 GB con los libros y `NUM_CTX=6144`) y caben
en 8 GB con margen para Windows. Medido en la RTX 4060 el 17/10: 100 % en GPU.
bge-m3 sale de la GPU en cuanto termina de indexar.

**El reranker necesita GPU.** En CPU tarda ~31 s por consulta (medido en un
i5-12450H, 30 candidatos), así que sin CUDA `RUTA=auto` usa `rrf`. Descarga
previa (~2,3 GB, una vez):

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')"
```

**Con un generador de 8B (`qwen3:8b`) el reranker no cabe:** Ollama se quedaría
sin memoria (HTTP 500). En ese caso:

```powershell
$env:DISPOSITIVO_RERANKER="cpu"; $env:RUTA="rrf"
```

Ver que el reranker realmente corre en la GPU: `python scripts\calibrar_umbral.py`
imprime el dispositivo y el tiempo de rerank por consulta (debería quedar por
debajo de 1 s).

Si el generador no arranca por falta de memoria, baja a un modelo de 3B:

```powershell
ollama pull qwen2.5:3b-instruct-q4_K_M
$env:MODELO_LLM="qwen2.5:3b-instruct-q4_K_M"
```

---

## 3. Construir corpus e índices

```powershell
cd ..\anonimizacion
python run_pipeline.py          # ~35 s  → anonimizacion\outputs\corpus_indexable.jsonl
cd ..\rag_pipeline
python -m rag.index             # GPU: pocos minutos · CPU: 40-60 min
```

`rag.index` lee el corpus de `..\anonimizacion\outputs\` por defecto; para
usar otro archivo, define `CORPUS_JSONL`.

`rag.index` construye dos índices: BM25 (segundos) y el denso con bge-m3 en
Qdrant modo local. **No hace falta Docker**: Qdrant escribe en `indices/qdrant/`.

La primera corrida descarga bge-m3 (~2,3 GB) desde Hugging Face.

Para probar rápido sin esperar a que indexe todo:

```powershell
python -m rag.index --limite 2000
```

### Libros de trauma (colección `guias`)

Los PDF van en `..\anonimizacion\data\` (fuera de git), junto con
`transcripciones_guias.yaml`: la transcripción a mano de los flujogramas y las
tablas en imagen del manual del HUV. Ese archivo no se regenera: se copia con
los PDF. Sin él, `extraer_guias.py` avisa y los flujogramas entran como cajas
sueltas. Se extraen y se indexan aparte; los casos no se reindexan:

```powershell
python scripts\extraer_guias.py         # → guias\secciones.jsonl (1.624 secciones)
python -m rag.index --fuente guias      # fijo-256: 4.013 fragmentos, ~30 s en GPU
```

Desde el 23/10 son tres libros: el ATLS 11.ª ed. (inglés), el manual del HUV y
el **ATLS 10.ª ed. en español** (`ATLS_10_ed_espanol.pdf`: la edición anterior,
en una traducción automática no oficial; `PLAN_INDEXAR_ATLS10.md`). **Se busca
por defecto desde el 24/10** (`ATLS10=1`), con 1.800 tokens de manuales
repartidos por libro (`CUOTA_GUIAS`). Con `$env:ATLS10 = "0"` el asistente
vuelve a dos manuales y 1.400 tokens, como antes.

Cada fuente vive en su colección del mismo Qdrant (`trauma` y `guias`) y
reindexar una no borra la otra. `--fuente todas` hace las dos.
`indices\meta.json` registra cuándo y con qué se indexó cada fuente.

---

## 4. Comparar las rutas de recuperación

```powershell
python -m eval.evaluate                          # bm25, densa, rrf
python -m eval.evaluate --rutas bm25,densa,rrf,rerank
```

Imprime nDCG@10, P@5 y MRR **por categoría** (léxica, vocabulario, abreviatura,
ortográfica, fina), más el comportamiento de abstención en las consultas fuera
de dominio. Deja tres archivos en `reports/`:

- `eval_rutas.csv` — métricas por consulta y ruta
- `eval_detalle.csv` — los top-5 de cada ruta, con el texto
- `validacion_manual.csv` — **para llenar a mano** la columna `relevante_humano`

Sobre esto último: la relevancia está pre-anotada por metadatos (mismo
mecanismo + misma región). Es un proxy razonable para elegir ruta, pero no es
relevancia clínica. Validar 30-40 filas a mano y reportar la concordancia es lo
que convierte el número en un argumento.

Para fijar la ruta ganadora:

```powershell
$env:RUTA="rrf"
```

---

## 5. Levantar el chatbot

```powershell
streamlit run app.py
```

Abre `http://localhost:8501`. Desde el 24/10 el diseño es minimalista:
- **Al abrir:** un saludo ("✚ Buenas tardes") y el cuadro de la consulta en el
  centro.
- **En la conversación:** la consulta va en una burbuja a la derecha y la
  respuesta como texto, sin avatares.
- **Siempre:** la paleta de siempre (blanco, rojo y rosa).

La barra lateral abre cerrada (la flecha `»` arriba a la izquierda). Ahí están:
- la ruta, el umbral de abstención y el interruptor "Manuales de trauma";
- los filtros por mecanismo, región y desenlace, en vivo, útiles para responder
  preguntas durante la demostración sin salir de la interfaz;
- "Paciente nuevo" y "Limpiar conversación".

Cada respuesta va en dos apartados, uno debajo del otro:

- **Casos similares del registro**: la prosa sobre los casos (con el aviso de
  similitud baja, si corresponde) y debajo el desplegable "Casos consultados"
  con el texto completo de cada caso y la confianza de la búsqueda.
- **Manuales de trauma (ATLS y HUV)**: la prosa sobre los manuales y debajo el
  desplegable "Fragmentos de los manuales", con libro, capítulo, sección,
  página y texto de cada pasaje (el ATLS en inglés, tal cual). Con `ATLS10=1`,
  la prosa nombra la edición ("La 10.ª edición del ATLS recomienda…"), el
  apartado termina con un aviso del código (es la edición anterior, en
  traducción no oficial; si también hay pasajes de la 11.ª, que pueden no
  coincidir) y la tarjeta de cada pasaje dice dónde está lo mismo en la 11.ª
  ("en la 11.ª ed.: cap. 7").

Las citas `[caso: ID]` y `[guía: ID]` van en rojo. Para la demo (19/10) no se
muestran la línea de respaldo ("Respaldo: 5 casos del registro · similitud…")
ni el aviso de que casos y manuales pueden no coincidir: lo dice "Alcance", en
la barra lateral (`app.APARTADOS_OCULTOS` y `RX_LINEA_RESPALDO`; siguen en
`Respuesta.texto`).

Si una fuente no responde, su apartado lo dice. Con `PROSA_SIEMPRE=0`, la parte
de la redacción del modelo que no pasó la verificación se muestra armada desde
su fuente y una nota arriba de la respuesta dice cuál y por qué.

Escribir solo un número (`40887`) recupera y resume ese caso concreto.

Desde el 24/10 (`PLAN_CHATBOT_CLINICO.md`), con un paciente presentado:

- **La ficha del paciente** (`CONVERSACION=ficha`). Arriba de cada respuesta,
  lo que entendió el código: "Paciente: arma de fuego · abdomen · hipotensión o
  choque · TA 80/50 · inestable". En un seguimiento, una línea dice qué cambió
  ("Con hipotensión o choque: entran 3 casos nuevos; aparece transfusión (2 de
  5)"). Si el mensaje empezó un paciente nuevo, lo dice.
  - En la barra lateral, "Paciente nuevo" borra la ficha.
  - Bajo la última respuesta, "Es otro paciente" o "Era el mismo paciente"
    corrige al código y responde de nuevo.
  - No se llama al modelo para entender el seguimiento.
- **Preguntas** (`PREGUNTAS=1`). Bajo la última respuesta, hasta dos preguntas
  con botones, en el orden de la revisión primaria: "C · Circulación — ¿cómo
  está hemodinámicamente?", "D · Neurológico — ¿qué Glasgow tiene?"… Un clic
  suma el dato a la ficha y responde de nuevo; "No se sabe" deja de
  preguntarlo.
- **Por letras** (`ORDEN_ATLS=1`). Dentro de cada apartado, la prosa va por
  letras ("**B · Respiración.**", "**C · Circulación.**", "**Después de la
  revisión primaria.**"). La letra la pone el código, no el modelo. Las
  preguntas de procedimiento ("¿cuándo está indicado colocar un tubo de
  tórax?") siguen con la prosa de siempre y sin preguntas.
- **La búsqueda de los manuales** expande las abreviaturas y nombra el
  mecanismo. "Fragmentos de los manuales" muestra la búsqueda usada: «HPAF
  (herida por arma de fuego) en abdomen».

---

## 6. Guion sugerido para la demostración

**Antes (10 minutos antes, con el chatbot cerrado):**

```powershell
cd rag_pipeline
python scripts\preparar_demo.py
```

Revisa la GPU, la configuración de los manuales (con el ATLS 10 y la guarda por
señal de trauma), la de la conversación (ficha, preguntas, letras), que Ollama
tenga el modelo entero en la VRAM (100 %) y pasa por el sistema completo:

- las consultas de la demo (campo `boton` de `eval/consultas.yaml` y
  `eval/consultas_guias.yaml`; la 7, la del ATLS 10, se salta con `ATLS10=0`);
- la conversación del paso 6 (sección 4), con la ficha y la respuesta anterior
  como las pasa la app.

Debe terminar en "Todo listo". Después: `streamlit run app.py`. Al abrir, la
app precarga los modelos (~30 s, una sola vez) y el modelo queda cargado 2 horas
(`KEEP_ALIVE`).

En el PC de 8 GB (RTX 4060) corre el reranker para casos y manuales. Cada
respuesta con las dos fuentes tarda **~9-10 s** (por letras; solo casos, ~5 s).
Un seguimiento tarda lo mismo: ya no hay una llamada para reescribirlo.

**La frase de la reunión: más sofisticado no es más libre.** El asistente
pregunta como un clínico, ordena como el ATLS y recuerda al paciente, y todo
sigue citado. Las preguntas, la ficha y el orden los pone el código; el modelo
solo redacta, y lo que redacta se verifica.

Las consultas ya no son botones de la app: se escriben. Diez minutos, en este orden:

**1 · Encuadre, antes de escribir nada.** El registro cubre atención de trauma
**hasta 2018**. El sistema sugiere a partir de lo que se hizo en casos
históricos; no indica conductas ni reemplaza el juicio clínico. Dicho por ti al
principio es una limitación reconocida; dicho por alguien del público a mitad
de la demo, es un agujero.

**2 · Caso directo** (1) — *"herida por arma de fuego en tórax con
hipoventilación"*. Dos miradas: lo que se hizo en casos similares, con
`[caso: ID]`, y lo que recomiendan los manuales, con `[guía: ID]`.
- **Las dos van ordenadas como la revisión primaria:** "B · Respiración"
  (hipoventilación, tubo de tórax), "C · Circulación", "Después de la revisión
  primaria". La letra de cada cosa la decide el código con una tabla que tiene
  que validar el médico; el modelo solo redacta dentro de cada letra.
- **Arriba, la ficha** ("Paciente: arma de fuego · tórax · hipoventilación").
  **Abajo, las preguntas** de lo que falta: "C · ¿cómo está
  hemodinámicamente?" y el grupo de edad.
- **Abre "Casos consultados" y "Fragmentos de los manuales":** el texto
  completo de cada caso y de cada pasaje (libro, capítulo, página). **Todo lo
  que dice la sugerencia se puede comprobar ahí.**
- Los casos son de hasta 2018 y pueden no coincidir con los manuales: el
  sistema no elige. Dilo tú; en pantalla solo lo dice "Alcance", en la barra
  lateral.

**3 · Cómo se evita que invente.**
- El modelo redacta, pero el código revisa cada oración contra su fuente antes
  de mostrarla: que lo que nombra esté en los casos o en el pasaje citado, que
  el caso o el pasaje citado lo tenga, que las cifras estén en el manual, que no
  cuente mal y que oriente en vez de ordenar.
- Por letras, además, que cada cosa esté en su letra: la hipoventilación no va
  en C.
- **Para la demo todo va en prosa** (`PROSA_SIEMPRE=1`, decisión del 18/10; por
  letras, `PROSA_SIEMPRE_ATLS` toma ese valor): el verificador corre igual y
  deja sus motivos, pero no reemplaza nada.
- Fuera de la demo, con `PROSA_SIEMPRE=0`, la parte (o la letra) que no pasa se
  muestra armada desde las fuentes y una nota arriba dice cuál.

**4 · Brecha de vocabulario** (2) — *"paciente apuñalado en el pecho,
herida penetrante"*. El corpus dice `HPACP en tórax` o `ME APUÑALARON`. Por eso
hay búsqueda densa y no solo por palabras.

**5 · Abreviatura** (3) — *"HPAF en cráneo con orificio de entrada
parietal"*.
- Los casos la entienden porque el corpus trae las abreviaturas expandidas.
- Desde el 24/10, los manuales también: su búsqueda se hace con «HPAF (herida
  por arma de fuego) en cráneo…». Ábrelo en "Fragmentos de los manuales": dice
  la búsqueda usada.
- **Ojo:** los pasajes que llegan son del HUV sobre arma de fuego en general y
  trauma penetrante de tórax y abdomen, no de cráneo. "Herida por arma de
  fuego" pesa más que "cráneo", porque la búsqueda está en español y lo del
  cráneo está en el ATLS 11, en inglés. Si alguien lo pregunta, esa es la
  respuesta.

**6 · La conversación sobre el mismo paciente** — tres turnos.
1. Escribe *"HPAF en abdomen"*. La ficha dice "Paciente: arma de fuego ·
   abdomen" y debajo salen dos preguntas: el estado hemodinámico (C) y el
   grupo.
2. Haz clic en **"Hipotenso o en choque"**. No hay que reescribir nada: el dato
   se suma a la ficha ("· hipotensión o choque · inestable") y el asistente
   responde de nuevo. Una línea dice qué cambió respecto de la respuesta
   anterior ("entran 3 casos nuevos; aparece…"); la escribe el código, no el
   modelo.
3. Escribe *"¿qué dice el ATLS del FAST?"*. Es un seguimiento: busca con el
   paciente ("HPAF en abdomen, hipotenso o en choque, qué dice el ATLS del
   FAST"), visible en "Casos consultados". El FAST no entra en la ficha, porque
   preguntar por él no es un hecho. Responde la 10.ª edición del ATLS, en
   español.

Para mostrar que se corrige con un clic: escribe *"mujer de 75 años con fractura
de cadera"*. El código lo toma como **paciente nuevo** y lo dice; si fuera el
mismo, "Era el mismo paciente" lo rehace. "Paciente nuevo", en la barra lateral,
empieza de cero.

**7 · Caso grave** (4) — *"herida por arma de fuego con desenlace fatal"*.
Con la barra lateral se puede filtrar por mecanismo, región o desenlace en vivo.
Es otro paciente y el código lo toma así: una presentación (un mecanismo o una
lesión, sin "le hicieron…", "tiene…", "además…") empieza una ficha nueva.

**7a · Solo los manuales** (6) — **antes, "Paciente nuevo"** en la barra
lateral. Luego *"cuándo activar el protocolo de transfusión masiva"*.
- Es una pregunta de procedimiento: no hay casos que se parezcan y responden
  solo el ATLS y el manual del HUV. Es lo que pidió la reunión del 30/09: que el
  asistente no dependa solo de lo que se hizo en el hospital.
- No presenta a un paciente: va con la prosa de siempre, sin letras ni
  preguntas.
- Sin "Paciente nuevo", una pregunta se toma como del paciente anterior: buscaría
  con él, y responderían también los casos.

**7b · El ATLS en español** (7) — *"TCE leve con pérdida
de conciencia transitoria"*. Sin el ATLS 10, los manuales no responden esta
consulta; con él, responde la 10.ª edición, y la prosa lo dice. Abre
"Fragmentos de los manuales": es la **edición anterior (2018), en una
traducción no oficial al español** (el aviso del apartado y la cabecera de cada
pasaje lo dicen), y la tarjeta indica dónde está lo mismo en la 11.ª. El
sistema no elige entre ediciones. Con tres manuales, el modelo a veces atribuye
un pasaje al manual equivocado ("La 11.ª edición…" con un pasaje de la 10.ª).
El verificador lo detecta, pero en la demo la prosa se muestra igual: si
alguien lo nota, la cita en rojo lleva al pasaje real en "Fragmentos de los
manuales".

**8 · Abstención** (5) — *"manejo de cetoacidosis diabética en
urgencias"*. Ni los casos ni los manuales responden: el sistema dice que no
encontró nada suficientemente parecido. **Esta es la diapositiva que convence a
un público clínico**: responde qué hace el sistema cuando no sabe.

**Preguntas del público:** de las 20 consultas fuera de dominio medidas
(apendicitis, ACV, infarto, embarazo, hipoglucemia…), desde el 23/10 no responde
ninguna: si la consulta no nombra un mecanismo, una lesión ni un procedimiento
de trauma, las dos guardas piden más (guarda por señal de trauma,
`rag/senal.py`). Una pregunta que no está en esas 20 no está medida.

**Evitar en la demo:**
- **Escribir un número de caso para resumirlo.** Ese resumen todavía lo escribe
  el modelo sin verificación y a veces cambia términos (pendiente 4 de
  `CONTEXTO_SESION.md`). Para mostrar trazabilidad, usar "Casos consultados".
- **Hacer una pregunta general después de un paciente sin "Paciente nuevo".**
  Una presentación ("HPAF en abdomen", "mujer de 75 años con…", "TCE leve…") y
  una consulta general sin trauma ("manejo de cetoacidosis…") empiezan de cero
  solas. Una pregunta ("¿cuándo…?", "¿qué dice…?") se suma al paciente
  anterior. `preparar_demo.py` revisa este guion escrito en un solo chat
  (sección 4). La ficha lo muestra y se corrige con un clic, pero en la demo
  distrae.

---

## Variables de entorno

| Variable | Por defecto | Para qué |
|---|---|---|
| `RUTA` | `auto` | `rerank` con GPU NVIDIA, `rrf` sin ella; o fijar `bm25`, `densa`, `rrf`, `rerank` |
| `TOP_K` | `5` | casos que se le pasan al generador |
| `UMBRAL_ABSTENCION` | `0.52` | coseno mínimo para responder. Sin reranker es la única guarda: bloquea cetoacidosis (0,50) y deja pasar "TCE" (0,541) |
| `UMBRAL_RERANK` | `0.2` | con `rerank`: probabilidad mínima por caso; con menos de 2 casos sobre ella, se abstiene. Calibrar con `scripts\calibrar_umbral.py` |
| `PROFUNDIDAD_RERANK` | `30` | candidatos de RRF que se reordenan |
| `SALIDA` | `prosa` | `prosa`: el modelo redacta la sugerencia con citas y `rag/verificar.py` la revisa; si no pasa, se muestra la versión armada desde los casos. `sugerencias`: siempre la versión armada desde los casos |
| `MAX_TOKENS_PROSA` | `600` | tope de la sugerencia en prosa (solo casos usa ~150–250; con los manuales, ~260–470) |
| `NUM_CTX` | `6144` con `GUIAS`, `3072` sin | ventana de Ollama. Con casos y manuales el prompt llega a ~4.400 tokens. Fija a propósito: si cambia entre llamadas, Ollama recarga el modelo (6-8 s). El 4B ocupa 3,57 GB con 6.144 |
| `GUIAS` | encendido | los libros como segunda fuente de la respuesta. `0` = solo casos, como antes (ablación). Sin la colección `guias` indexada se comporta como `0` |
| `SALIDA_ESTRUCTURADA` | encendido | el modelo responde JSON restringido a la lista de elementos (campo `format` de Ollama, requiere Ollama ≥ 0.5). `0` = viñetas de texto como antes |
| `DISPOSITIVO_RERANKER` | `auto` | `cuda` si hay GPU NVIDIA; `cpu` si el generador es de 8B |
| `MODELO_LLM` | `qwen3:4b-instruct` | generador en Ollama; según la VRAM, ver `MODELOS.md`. Sin GPU NVIDIA y poca RAM: `qwen2.5:3b` |
| `KEEP_ALIVE` | `2h` | cuánto deja Ollama el modelo cargado tras cada llamada (`-1` = siempre) |
| `VRAM_MIN_RERANK` | `7` | GB de VRAM desde los que `RUTA=auto` usa el reranker en GPU |
| `SINTESIS_LLM` | apagado | `1` para que el modelo escriba la síntesis y los límites. Apagado los arma el código con conteos: el 3B inventaba en esas secciones |
| `PRECISION_CONSULTA` | `bf16` | precisión de bge-m3 en CPU: `bf16` ~1,1 GB de RAM, `fp32` ~2,3 GB |
| `DISPOSITIVO_CONSULTA` | `cpu` | dónde embeber la consulta |
| `ENCODER` | `hf` | `hf` = transformers directo · `st` = sentence-transformers · `falso` = sustituto hash |
| `COLECCION_GUIAS` | `guias` | colección de Qdrant con los libros de trauma |
| `FRAGMENTACION_GUIAS` | `fijo-256` | configuración de `rag/fragmentar.py` al indexar los libros (reindexar si cambia) |
| `RUTA_GUIAS` | `auto` | libros: `rerank` (densa + reranker sobre 30) si el reranker cabe en la GPU, si no `densa` |
| `PRESUPUESTO_GUIAS` | `1800` (`1400` con `ATLS10=0`) | tokens de fragmentos de los libros que van al prompt, con cuota por libro si está el ATLS 10 |
| `ABREV_LIBROS` | encendido | los manuales se buscan con las abreviaturas expandidas («HPAF (herida por arma de fuego)»), salvo las que los manuales ya usan (FAST, GCS, Rx…) |
| `MECANISMO_LIBROS` | encendido | y con el nombre del mecanismo cuando la consulta lo dice con otras palabras ("baleado" → herida por arma de fuego) |
| `CONVERSACION` | `ficha` | la ficha del paciente la arma el código (`rag/ficha.py`); `llm` = el modelo reescribe cada seguimiento, como antes del 24/10 |
| `PREGUNTAS` | encendido | hasta 2 preguntas con botones bajo la respuesta (`rag/preguntas.py`); necesita `CONVERSACION=ficha` |
| `ORDEN_ATLS` | encendido | la prosa por letras de la revisión primaria (xABCDE) dentro de cada apartado, solo con un paciente (`rag/prioridades.py`) |
| `PROSA_SIEMPRE_ATLS` | el de `PROSA_SIEMPRE` | por letras: `1` = la letra que no pasa la verificación se muestra igual; `0` = va armada desde su fuente (decisión 3 del plan, pendiente) |
| `UMBRAL_GUIAS` | `0.34` | ruta `rerank`: probabilidad mínima del mejor fragmento para que los libros hablen (0,2 hasta el 19/10). Calibrar con `scripts\calibrar_guias.py` |
| `UMBRAL_GUIAS_COSENO` | `0.60` | ruta `densa` (sin el reranker en GPU): coseno mínimo del mejor fragmento. Separa mal: con 0,60 frena las 10 fuera de dominio pero deja sin libros a 16 de 25 consultas de trauma |

`ENCODER=falso` sirve para verificar la tubería completa (índices, rutas,
métricas, interfaz) sin descargar bge-m3. **No mide calidad**: es un hash de
bolsa de palabras. Úsalo solo para comprobar que todo corre.

`ENCODER=hf` (el valor por defecto) carga bge-m3 con `transformers` directo:
tokenizador, modelo, pooling CLS y normalización L2, que es exactamente lo que
hace bge-m3 para su vector denso. No pasa por sentence-transformers, así que no
se rompe cuando esa librería cambia de versión mayor.

---

## Si algo falla

**La app se queda en "Cargando modelos…" y termina en `TimeoutError: timed out`**
→ Ollama ≥ 0.40 guarda en RAM una caché de prompts que crece hasta 8 GB y puede
agotar la memoria (pasó el 17/10 en el PC de 16 GB de RAM). Desactivarla y
reiniciar Ollama (salir desde el ícono de la bandeja y volver a abrirlo):

```powershell
setx LLAMA_ARG_CACHE_RAM 0
```

En `%LOCALAPPDATA%\Ollama\server.log` debe aparecer "prompt cache is disabled".

**`Falta el índice denso`** → `python -m rag.index`

**`No hay conexión con Ollama`** → abre otra terminal y corre `ollama serve`

**La indexación va lentísima** → torch no ve la GPU. Comprueba con
`python scripts\revisar_entorno.py` y reinstala torch con el índice de CUDA.
No es bloqueante: es un costo de una sola vez y Ollama usa la GPU igual.

**`Could not find a version that satisfies the requirement torch`** → el índice
de CUDA que elegiste no tiene rueda para tu versión de Python. Con Python 3.13
hacen falta `cu126` o posterior; `cu121` solo llega hasta Python 3.12.

**`No se pudo generar la respuesta: HTTP Error 500`** → casi siempre es Ollama sin memoria para cargar el modelo. El detalle está en `%LOCALAPPDATA%\Ollama\server.log` (`failed to allocate buffer`). Sin GPU NVIDIA el modelo va entero a RAM: cierra aplicaciones o usa un modelo más pequeño.

**El generador se queda sin memoria** → si el reranker está en la GPU,
`$env:DISPOSITIVO_RERANKER="cpu"; $env:RUTA="rrf"`; si no, baja a
`qwen2.5:3b-instruct-q4_K_M`.

**`Unrecognized processing class in BAAI/bge-m3`** → sentence-transformers 6.x
intenta instanciar un `AutoProcessor` para todo modelo, y bge-m3 no tiene uno.
Ya no afecta a la ruta densa (usa `transformers` directo), pero sí al reranker:
`pip install "sentence-transformers<6"`.

**Qdrant avisa de «local mode not recommended»** → es un aviso conservador a
partir de 20.000 puntos. Medido sobre este corpus: ~85 ms por consulta densa,
~100 ms con RRF. Para pasar a Qdrant en Docker solo cambia el constructor en
`_qdrant()`, dentro de `rag/retrieve.py`.
