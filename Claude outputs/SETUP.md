# Puesta en marcha — prototipo RAG de trauma

Objetivo: chatbot que recupera casos análogos del registro de trauma y los
describe con citas. Todo corre **en local**: ningún texto sale de la máquina.

---

## 1. Instalar

```powershell
cd C:\Users\JOSE\Documents\dev\rag_trauma
python -m venv .venv
.\.venv\Scripts\activate

# torch con CUDA PRIMERO (si no, pip instala la versión CPU y la GPU queda sin usar)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Ollama** — descárgalo de [ollama.com](https://ollama.com) y luego:

```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
```

Comprueba que todo está en su sitio:

```powershell
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
python run_pipeline.py          # ~35 s  → outputs/corpus_indexable.jsonl
python -m rag.index             # GPU: pocos minutos · CPU: 40-60 min
```

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
| `MODELO_LLM` | `qwen2.5:7b-instruct-q4_K_M` | generador en Ollama |
| `RERANKER` | apagado | `1` para encender el cross-encoder |
| `DISPOSITIVO_CONSULTA` | `cpu` | dónde embeber la consulta |
| `ENCODER` | `bge` | `falso` usa un sustituto hash sin descargar modelos |

`ENCODER=falso` sirve para verificar la tubería completa (índices, rutas,
métricas, interfaz) sin descargar bge-m3. **No mide calidad**: es un hash de
bolsa de palabras. Úsalo solo para comprobar que todo corre.

---

## Si algo falla

**`Falta el índice denso`** → `python -m rag.index`

**`No hay conexión con Ollama`** → abre otra terminal y corre `ollama serve`

**La indexación va lentísima** → torch no ve la GPU. Comprueba con
`python scripts\revisar_entorno.py` y reinstala torch con el índice de CUDA.

**El generador se queda sin memoria** → baja a `qwen2.5:3b-instruct-q4_K_M`, o
apaga el reranker si lo encendiste.

**Qdrant avisa de «local mode not recommended»** → es un aviso conservador a
partir de 20.000 puntos. Medido sobre este corpus: ~85 ms por consulta densa,
~100 ms con RRF. Para pasar a Qdrant en Docker solo cambia el constructor en
`_qdrant()`, dentro de `rag/retrieve.py`.
