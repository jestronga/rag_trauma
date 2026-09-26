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

El resto de la respuesta (síntesis, "Qué se hizo", límites) lo arma el código,
así que el modelo escribe poco. Por eso la regla que decide es simple: **el
modelo tiene que caber entero en la VRAM.** Si no cabe, Ollama pasa capas a la
CPU y el tiempo se multiplica.

Hay que dejar margen: Windows ocupa entre 0,3 y 0,5 GB de VRAM, y el contexto
de la conversación suma unos cientos de MB más.

---

## Tarjeta de 4 GB de VRAM

Modelos de 3 a 4B parámetros. Tiempo estimado por respuesta: **5 – 10 s**.

| # | Modelo | Descarga | Por qué | Cuidado con |
|---|---|---|---|---|
| 1 | **`qwen3:4b`** | 2,5 GB | Mejor equilibrio: buen español, buen seguimiento de instrucciones y margen de memoria | Razonamiento activo: hay que apagarlo (ver abajo) |
| 2 | `qwen3.5:4b` | 3,4 GB | Generación más nueva; 201 idiomas | Muy justo: incluye módulo de imágenes y puede pasar capas a CPU. Apagar razonamiento |
| 3 | `qwen2.5:3b` | 1,9 GB | Probado en este proyecto; sin modo de razonamiento | Menor calidad al elegir sugerencias |
| 4 | `medgemma:4b` | 3,3 GB | Gemma 3 adaptado a medicina | Español no documentado. Puede aportar conocimiento médico general que no está en los casos. Solo para comparar |

**Descartados en 4 GB:** los de 7B o más (`qwen2.5:7b`, `qwen3:8b`,
`qwen3.5:9b`) no caben y pasan de 15 s. `gemma4` tampoco: su variante más
pequeña pesa 7,2 GB.

---

## Tarjeta de 8 GB de VRAM

Modelos de 7 a 9B parámetros. Tiempo estimado por respuesta: **5 – 12 s**.

| # | Modelo | Descarga | Por qué | Cuidado con |
|---|---|---|---|---|
| 1 | **`qwen3:8b`** | 5,2 GB | Mejor equilibrio: calidad clara sobre los 4B y margen para el contexto | Razonamiento activo: hay que apagarlo |
| 2 | `qwen3.5:9b` | 6,6 GB | El de mayor calidad que entra en 8 GB | Muy justo: con Windows y el módulo de imágenes puede pasar capas a CPU. Apagar razonamiento |
| 3 | `qwen2.5:7b-instruct-q4_K_M` | 4,7 GB | Modelo para el que se diseñó el proyecto; sin razonamiento | Un poco por debajo de `qwen3:8b` |
| 4 | `qwen3:4b` | 2,5 GB | Opción rápida (~3–5 s); deja VRAM libre para bge-m3 en GPU | Menor calidad que los de 8B |

**Descartados en 8 GB:** `qwen3:14b` (9,3 GB), `qwen2.5:14b` (9,0 GB) y
`gemma4:12b` (7,6 GB, sin espacio para el contexto).

Con 8 GB **no conviene poner bge-m3 en la GPU** si el generador es de 8–9B: se
llevaría ~1,1 GB y el modelo dejaría de caber. Solo con `qwen3:4b` hay
espacio para los dos (`DISPOSITIVO_CONSULTA=cuda`).

---

## Ajustes necesarios con cualquier modelo

**1. Apagar el razonamiento en Qwen3 y Qwen3.5.** Estos modelos "piensan"
antes de responder y eso puede sumar decenas de segundos. Hay que enviar
`"think": false` en la llamada a Ollama, dentro de `_ollama()` en
`rag/generate.py`. **El código actual todavía no lo hace.** Con `qwen2.5` y
`medgemma` no hace falta.

**2. Mantener el modelo cargado.** La primera consulta después de 5 minutos
sin uso recarga el modelo y puede pasar de 15 s:

```powershell
setx OLLAMA_KEEP_ALIVE -1      # luego reiniciar Ollama
```

**3. Limitar la salida.** `MAX_TOKENS` está en 900. Como el modelo solo
escribe las sugerencias, con 250 basta, y así una respuesta que se alargue no
pasa del límite de tiempo:

```powershell
$env:MAX_TOKENS = "250"
```

**4. No encender `SINTESIS_LLM=1` sin revisar respuestas a mano.** Con ese
ajuste el modelo escribe también la síntesis y los límites. Probado con el 3B,
inventaba en esas secciones. Los modelos de 8–9B inventan menos, pero no están
exentos.

---

## Cómo verificar

1. Descargar el modelo y apuntar el proyecto a él:

   ```powershell
   ollama pull qwen3:4b
   $env:MODELO_LLM = "qwen3:4b"
   ```

2. Hacer una consulta en el chatbot y correr `ollama ps`. La columna
   `PROCESSOR` debe decir **`100% GPU`**. Si muestra un reparto como
   `20%/80% CPU/GPU`, el modelo no cabe: bajar al siguiente de la tabla.
3. Medir el tiempo de varias consultas, sin contar la primera (carga del
   modelo). Si alguna pasa de 15 s, bajar al siguiente de la tabla.

---

## Fuentes

Tamaños de descarga tomados de la biblioteca de Ollama (septiembre de 2026):

- [qwen3](https://ollama.com/library/qwen3)
- [qwen3.5](https://ollama.com/library/qwen3.5)
- [qwen2.5](https://ollama.com/library/qwen2.5)
- [gemma4](https://ollama.com/library/gemma4)
- [medgemma](https://ollama.com/library/medgemma)
