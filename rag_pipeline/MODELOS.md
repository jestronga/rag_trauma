# Modelos generativos recomendados

Qué modelo usar en Ollama según la VRAM de la tarjeta gráfica, con un objetivo:
**cada respuesta en 15 segundos o menos, con la mejor calidad posible.**

Se asume Windows, tarjeta NVIDIA y 16 GB de RAM. El modelo se elige con la
variable `MODELO_LLM` (ver `SETUP.md`).

> **Los tiempos son estimaciones**, calculadas a partir del tamaño de cada
> modelo y del ancho de banda típico de cada tipo de tarjeta. Solo
> `qwen2.5:3b` se probó con este proyecto (en CPU). Antes de elegir, hay que
> medir en el equipo real (ver [Cómo verificar](#cómo-verificar)).

---

## De qué depende el tiempo de respuesta

| Etapa | Costo | Nota |
|---|---|---|
| Recuperar los casos | < 0,5 s | bge-m3 en CPU |
| Leer el prompt | 1.362 – 2.437 tokens | medido con las 28 consultas de `eval/consultas.yaml` |
| Generar la salida | ~100 – 150 tokens | el modelo solo escribe la sección de sugerencias |

El resto de la respuesta (síntesis, tabla de casos, contexto de las sugerencias, límites) lo arma el código,
así que el modelo escribe poco. Por eso la regla que decide es simple: **el
modelo tiene que caber entero en la VRAM.** Si no cabe, Ollama pasa capas a la
CPU y el tiempo se multiplica.

Hay que dejar margen: Windows ocupa entre 0,3 y 0,5 GB de VRAM, y el contexto
de la conversación suma unos cientos de MB más.

---

## Tarjeta de 4 GB de VRAM

Modelos de 3 a 4B parámetros. **Medido en una GTX 1650 (26/09/2026) con
`qwen3:4b-instruct`: ~16 s por respuesta** (5 casos, 100 % GPU). De esos, ~11 s
son la lectura del prompt (~1.800–2.000 tokens), que en esta tarjeta ya es el
límite de cómputo; la generación de ~100 tokens tarda ~2,5 s (42 tok/s).

| # | Modelo | Descarga | Por qué | Cuidado con |
|---|---|---|---|---|
| 1 | **`qwen3:4b-instruct`** | 2,5 GB | Qwen3-4B-Instruct-2507. Buen español, elige bien las sugerencias, sin razonamiento | Ollama no lo carga entero en GPU por sí solo: ver `NUM_GPU` abajo |

> **No usar `qwen3:4b` a secas.** Hoy esa etiqueta es Qwen3-4B-**Thinking**-2507:
> razona siempre, aunque se envíe `"think": false` (el razonamiento sale en
> inglés dentro de la respuesta). Medido: agotaba los 900 tokens sin llegar a
> escribir las sugerencias, ~80 s por respuesta.
| 2 | `qwen3.5:4b` | 3,4 GB | Generación más nueva; 201 idiomas | Muy justo: incluye módulo de imágenes y puede pasar capas a CPU. Apagar razonamiento |
| 3 | `qwen2.5:3b` | 1,9 GB | Probado en este proyecto; sin modo de razonamiento | Menor calidad al elegir sugerencias |
| 4 | `medgemma:4b` | 3,3 GB | Gemma 3 adaptado a medicina | Español no documentado. Puede aportar conocimiento médico general que no está en los casos. Solo para comparar |

**Descartados en 4 GB:** los de 7B o más (`qwen2.5:7b`, `qwen3:8b`,
`qwen3.5:9b`) no caben y pasan de 15 s. `gemma4` tampoco: su variante más
pequeña pesa 7,2 GB.

---

## Tarjeta de 8 GB de VRAM

Modelos de 7 a 9B parámetros. Tiempo estimado por respuesta: **5 – 12 s**.

> **Decisión desde el 27/09/2026: en 8 GB, `qwen3:4b-instruct` + reranker en
> GPU.** El reranker (bge-reranker-v2-m3, ~1,3 GB en fp16) es la guarda de
> abstención: con él, solo 1 de las 10 consultas fuera de dominio de prueba
> (apendicitis aguda) recibe respuesta, contra 8 de 10 solo con coseno. No cabe junto a un 8B.
> Como el modelo solo elige sugerencias de una lista, el 8B aporta menos que
> el reranker. Los 8B de la tabla quedan para usarse con
> `DISPOSITIVO_RERANKER=cpu` y `RUTA=rrf`, sin esa guarda.

| # | Modelo | Descarga | Por qué | Cuidado con |
|---|---|---|---|---|
| 1 | **`qwen3:8b`** | 5,2 GB | Mejor equilibrio: calidad clara sobre los 4B y margen para el contexto | Razonamiento activo: hay que apagarlo |
| 2 | `qwen3.5:9b` | 6,6 GB | El de mayor calidad que entra en 8 GB | Muy justo: con Windows y el módulo de imágenes puede pasar capas a CPU. Apagar razonamiento |
| 3 | `qwen2.5:7b-instruct-q4_K_M` | 4,7 GB | Modelo para el que se diseñó el proyecto; sin razonamiento | Un poco por debajo de `qwen3:8b` |
| 4 | `qwen3:4b-instruct` | 2,5 GB | Opción rápida (~3–5 s); deja VRAM libre para bge-m3 en GPU | Menor calidad que los de 8B |

**Descartados en 8 GB:** `qwen3:14b` (9,3 GB), `qwen2.5:14b` (9,0 GB) y
`gemma4:12b` (7,6 GB, sin espacio para el contexto).

Con 8 GB **no conviene poner bge-m3 en la GPU** si el generador es de 8–9B: se
llevaría ~1,1 GB y el modelo dejaría de caber. Solo con `qwen3:4b-instruct` hay
espacio para los dos (`DISPOSITIVO_CONSULTA=cuda`).

---

## Ajustes necesarios con cualquier modelo

**1. Apagar el razonamiento en Qwen3 y Qwen3.5.** Estos modelos "piensan"
antes de responder y eso puede sumar decenas de segundos. Hay que enviar
`"think": false` en la llamada a Ollama. `_ollama()` en `rag/generate.py` ya
lo envía siempre. Ojo: las variantes *Thinking* (como `qwen3:4b`) lo ignoran;
usar las *instruct*.

**2. Mantener el modelo cargado.** La primera consulta después de 5 minutos
sin uso recarga el modelo y puede pasar de 15 s:

```powershell
setx OLLAMA_KEEP_ALIVE -1      # luego reiniciar Ollama
```

**3. Limitar la salida.** `MAX_TOKENS` ya está en 250 (900 con
`SINTESIS_LLM=1`): el modelo solo escribe las sugerencias, ~100 tokens.

**4. Meter el modelo entero en la GPU.** Ollama calcula la memoria con mucho
margen: en la GTX 1650 dejaba 10 de 37 capas en CPU (`33%/67% CPU/GPU`, 15
tok/s) aunque cabían. `config.py` envía `num_gpu=99` (todas las capas) y
`num_ctx=3072` (el prompt con 5 casos más la salida). Resultado: `100% GPU`,
42 tok/s, ~3,0 GB de VRAM. Si Ollama da HTTP 500 por memoria (p. ej. con el
navegador usando la GPU): `$env:NUM_GPU = "-1"`. Con más casos de los que
caben en 3.072 tokens, la ventana se agranda sola para esa consulta y Ollama
vuelve a repartir capas: con 10 casos, ~55 s.

**5. No encender `SINTESIS_LLM=1` sin revisar respuestas a mano.** Con ese
ajuste el modelo escribe también la síntesis y los límites. Probado con el 3B,
inventaba en esas secciones. Los modelos de 8–9B inventan menos, pero no están
exentos.

---

## Cómo verificar

1. Descargar el modelo y apuntar el proyecto a él:

   ```powershell
   ollama pull qwen3:4b-instruct
   # es el valor por defecto de config.py; para otro modelo:
   # $env:MODELO_LLM = "<modelo>"
   ```

2. Con el chatbot cerrado, desde `rag_pipeline\`:

   ```powershell
   python scripts\comparar_modelos.py                                   # modelo de config
   python scripts\comparar_modelos.py --modelos qwen3:4b-instruct,qwen3:8b
   ```

   Corre las 25 consultas de `eval/consultas.yaml` con los mismos casos para
   todos los modelos y deja `reports/comparar_modelos.csv`. Imprime, por
   modelo:
   - **tiempo total** (mediana y peor) y por etapa: recuperación, lectura del
     prompt, salida. Cada respuesta se mide sin la caché de Ollama (el script
     descarga el modelo antes) y sin contar la recarga;
   - **`gpu_pct`**: debe ser **100**. Menos significa que el modelo no cabe y
     Ollama pasa capas a la CPU: bajar al siguiente de la tabla;
   - sugerencias descartadas y completadas por el código, y **casos citados
     frente a recuperados**.
3. Si la mediana pasa de 15 s, bajar al siguiente de la tabla.

---

## Fuentes

Tamaños de descarga tomados de la biblioteca de Ollama (septiembre de 2026):

- [qwen3](https://ollama.com/library/qwen3)
- [qwen3.5](https://ollama.com/library/qwen3.5)
- [qwen2.5](https://ollama.com/library/qwen2.5)
- [gemma4](https://ollama.com/library/gemma4)
- [medgemma](https://ollama.com/library/medgemma)
