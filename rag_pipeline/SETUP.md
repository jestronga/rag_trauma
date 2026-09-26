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
ollama pull qwen2.5:7b-instruct-q4_K_M
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
| bge-m3 (consulta) | ~2,3 GB | **CPU** — embeber una consulta corta cuesta ~150 ms |
| Qwen2.5 7B Q4 | ~4,7 GB | **GPU**, vía Ollama |
| bge-reranker-v2-m3 | ~2,3 GB | **apagado** |

Las tres piezas juntas no caben en 8 GB. Por eso bge-m3 sale de la GPU en
cuanto termina de indexar y Ollama se queda con ella. El reranker se enciende
así, y conviene dejarlo en CPU:

```powershell
$env:RERANKER="1"       # añade ~1 s por consulta en CPU
```

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

Abre `http://localhost:8501`. La barra lateral permite cambiar de ruta, ajustar
el umbral de abstención y filtrar por mecanismo, región y desenlace en vivo —
útil para responder preguntas durante la demostración sin salir de la interfaz.

Escribir solo un número (`40887`) recupera y resume ese caso concreto.

---

## 6. Guion sugerido para la demostración

Ocho minutos, en este orden:

**1 · Encuadre, antes de escribir nada.** El registro cubre atención de trauma
**hasta 2018**. El sistema describe lo que se hizo en casos históricos; no
indica conductas. Dicho por ti al principio es una limitación reconocida;
dicho por alguien del público a mitad de la demo, es un agujero.

**2 · Caso fácil** — *"herida por arma de fuego en tórax con hipoventilación"*.
Muestra el funcionamiento normal. Abre el desplegable de casos: ahí se ven las
citas y los metadatos.

**3 · Brecha de vocabulario** — *"paciente apuñalado en el pecho, herida penetrante"*.
Ninguna de esas palabras está en el corpus, que dice `HPACP en tórax` o
`ME APUÑALARON`. Es el momento de explicar por qué hay una ruta densa y no solo
búsqueda por palabras.

**4 · Abreviatura** — *"HPAF en cráneo con orificio de entrada parietal"*.
Funciona en los dos sentidos gracias a la expansión aumentativa del corpus.

**5 · Filtro por metadatos.** Con la misma consulta, activa mecanismo = arma de
fuego y región = tórax en la barra lateral. Se ve cómo se acota el espacio de
búsqueda.

**6 · Caso truncado.** Busca uno marcado `truncado` y muestra que el sistema
declara que el registro se corta en vez de inventar el desenlace. El 22 % del
corpus está cortado a 500 caracteres por el export original.

**7 · Abstención** — *"manejo de cetoacidosis diabética en urgencias"*.
El sistema dice que no encontró casos suficientemente similares. **Esta es la
diapositiva que convence a un público clínico**, más que cualquier acierto:
responde la pregunta que todos tienen en la cabeza, que es qué hace el sistema
cuando no sabe.

**8 · Trazabilidad.** Escribe un identificador de caso y muestra el registro
crudo detrás de la respuesta, con los marcadores de anonimización a la vista.

---

## Variables de entorno

| Variable | Por defecto | Para qué |
|---|---|---|
| `RUTA` | `rrf` | ruta de recuperación: `bm25`, `densa`, `rrf`, `rerank` |
| `TOP_K` | `5` | casos que se le pasan al generador |
| `UMBRAL_ABSTENCION` | `0.45` | coseno mínimo para responder |
| `MODELO_LLM` | `qwen2.5:3b` | generador en Ollama; según la VRAM, ver `MODELOS.md`. El 7B (`qwen2.5:7b-instruct-q4_K_M`) necesita GPU NVIDIA o ~5 GB de RAM libre |
| `SINTESIS_LLM` | apagado | `1` para que el modelo escriba la síntesis y los límites. Apagado los arma el código con conteos: el 3B inventaba en esas secciones |
| `PRECISION_CONSULTA` | `bf16` | precisión de bge-m3 en CPU: `bf16` ~1,1 GB de RAM, `fp32` ~2,3 GB |
| `RERANKER` | apagado | `1` para encender el cross-encoder |
| `DISPOSITIVO_CONSULTA` | `cpu` | dónde embeber la consulta |
| `ENCODER` | `hf` | `hf` = transformers directo · `st` = sentence-transformers · `falso` = sustituto hash |

`ENCODER=falso` sirve para verificar la tubería completa (índices, rutas,
métricas, interfaz) sin descargar bge-m3. **No mide calidad**: es un hash de
bolsa de palabras. Úsalo solo para comprobar que todo corre.

`ENCODER=hf` (el valor por defecto) carga bge-m3 con `transformers` directo:
tokenizador, modelo, pooling CLS y normalización L2, que es exactamente lo que
hace bge-m3 para su vector denso. No pasa por sentence-transformers, así que no
se rompe cuando esa librería cambia de versión mayor.

---

## Si algo falla

**`Falta el índice denso`** → `python -m rag.index`

**`No hay conexión con Ollama`** → abre otra terminal y corre `ollama serve`

**La indexación va lentísima** → torch no ve la GPU. Comprueba con
`python scripts\revisar_entorno.py` y reinstala torch con el índice de CUDA.
No es bloqueante: es un costo de una sola vez y Ollama usa la GPU igual.

**`Could not find a version that satisfies the requirement torch`** → el índice
de CUDA que elegiste no tiene rueda para tu versión de Python. Con Python 3.13
hacen falta `cu126` o posterior; `cu121` solo llega hasta Python 3.12.

**`No se pudo generar la respuesta: HTTP Error 500`** → casi siempre es Ollama sin memoria para cargar el modelo. El detalle está en `%LOCALAPPDATA%\Ollama\server.log` (`failed to allocate buffer`). Sin GPU NVIDIA el modelo va entero a RAM: cierra aplicaciones o usa un modelo más pequeño.

**El generador se queda sin memoria** → baja a `qwen2.5:3b-instruct-q4_K_M`, o
apaga el reranker si lo encendiste.

**`Unrecognized processing class in BAAI/bge-m3`** → sentence-transformers 6.x
intenta instanciar un `AutoProcessor` para todo modelo, y bge-m3 no tiene uno.
Ya no afecta a la ruta densa (usa `transformers` directo), pero sí al reranker:
`pip install "sentence-transformers<6"`.

**Qdrant avisa de «local mode not recommended»** → es un aviso conservador a
partir de 20.000 puntos. Medido sobre este corpus: ~85 ms por consulta densa,
~100 ms con RRF. Para pasar a Qdrant en Docker solo cambia el constructor en
`_qdrant()`, dentro de `rag/retrieve.py`.
