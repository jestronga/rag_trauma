# Plan — indexar el ATLS 10.ª edición en español

Pedido (23/10/2026): indexar en el RAG el ATLS 10.ª ed. en español
(`anonimizacion/data/ATLS_10_ed_espanol.pdf`), que se suma al ATLS 11.ª ed.
en inglés y al manual del HUV (`PLAN_INDEXAR_LIBROS.md`). Va **antes** del
plan del chatbot clínico (`PLAN_CHATBOT_CLINICO.md`): cambia la línea base que
ese plan mide en su fase 0.

## Lo que se encontró en el PDF (medido el 23/10)

| | ATLS 10.ª ed. (este) | ATLS 11.ª ed. (indexado) |
|---|---|---|
| Archivo | `ATLS_10_ed_espanol.pdf` (11 MB) | `ATLS_11thEdition_CourseManual_ENG_17.pdf` (80 MB) |
| Año, idioma | 2018, español | inglés |
| Páginas | 464 (37 casi sin texto) | 377 |
| Texto | capa de texto, 1,86 M caracteres (1,53 M sin las líneas repetidas) | 1,42 M caracteres |
| Índice incrustado | no; "Contenidos breves" en la p. 46 del PDF | no |
| Estructura | 13 capítulos, apéndices A–G (G = estaciones de habilidades) y 3 secciones al final, después del índice alfabético (Resultados, Prevención de lesiones, Biomecánica) | 24 capítulos, xABCDE |
| Maquetación | dos columnas (x ≈ 80 y ≈ 320); **una sola familia tipográfica** (ArialUnicodeMS) con tamaños continuos de 6,0 a 8,0 en el cuerpo | dos columnas; fuentes distintas por rol |

**1. Es una traducción automática, no la oficial (fuerte indicio).** El texto
tiene errores típicos de traducir el PDF palabra por palabra:
- la letra capitular separada del resto: "When" → "W gallina" ("hen" →
  "gallina", p. 4);
- la viñeta "▸" (la letra "n" de una fuente de símbolos) traducida como
  "norte": "( norte FIGURA 1-1 )", "norte VOLVER AL ÍNDICE";
- números de página traducidos ("dieciséis", "SG" por XL);
- términos mal traducidos: "encuesta secundaria" (*secondary survey*),
  "gerente de la vía aérea", "trampa" (*pitfall*), "instituto de vida
  conservando la terapia".

Las cifras de las tablas sí se conservan. Por ejemplo, la tabla 3-1 (clases de
hemorragia): < 15 %, 15–30 %, 31–40 %, > 40 %, déficit de base. Es justo el
riesgo que la **decisión 2 de `PLAN_INDEXAR_LIBROS.md`** quería evitar al no
traducir el ATLS: "una mala traducción se volvería 'la guía dice'".

**2. Es la edición anterior.** La 11.ª reorganiza la revisión primaria
(xABCDE, la hemorragia exanguinante primero) y actualiza conductas. Donde
las dos ediciones difieran, el asistente mostraría dos versiones del mismo
manual.

**3. Va a desplazar al ATLS 11 si no se hace nada.** Ya está medido (fase 4
de los libros): una consulta en español prefiere el texto en español. El ATLS
11 se lleva el 34 % de los fragmentos siendo el 72 % del índice. Un ATLS en
español, más largo que el inglés, competiría por los mismos temas y en el
idioma de la consulta. Puede sacar al 11 primero de los 30 candidatos de la
densa (`PROFUNDIDAD_RERANK`) y después del presupuesto.

**4. Problemas para el extractor actual (`rag/guias.py`):**
- **El 17,9 % de los caracteres está repetido en la misma caja**: cada línea
  viene 2 o 3 veces superpuesta ("…los segundo pico se produce en" ×3, p.
  32). Hoy entraría repetido.
- **4 páginas duplicadas** (PDF 75/76, 126/127, 219/220, 419/420).
- **Las reglas de rol no sirven.** Los perfiles del ATLS 11 y del HUV separan
  títulos y cuerpo por la fuente y su tamaño. Aquí todo es ArialUnicodeMS y el
  tamaño varía de forma continua, así que hay que separar por **color**:
  | Color | Rol |
  |---|---|
  | `383837` | cuerpo |
  | `ffffff` ~11,2, sobre una franja | título 1 |
  | `ac1a25` ~10,4 | título 2 |
  | `ad1d23` ~7 | cuerpo resaltado |
  | `ad1d23` pequeño + `616060` | pie de figura |
  | `545453` | tablas "trampa \| prevención" (*Pitfall \| Prevention*) |
  | `171615` | encabezado y rótulos dentro de las figuras |
- **El desfase de página no es fijo**: va de 53 (cap. 1) a 58 (al final)
  por las páginas insertadas o repetidas. `desfase_pagina` es un solo número
  y no alcanza.
- **Rótulos y créditos de las figuras como texto**: "Libélula Media Group
  11/23/2011", "Figura # 06,01".
- **Las citas no aceptan este ID.** `verificar.RX_CITA_GUIA` exige un prefijo
  de 2 a 5 letras (`[a-z]{2,5}-\d{2}-\d{3}`), así que "ATLS10-03-005" no se
  reconoce como cita. Hay que cambiar la regex o el prefijo.

## Principio

Lo de siempre: el modelo redacta, el código pone los datos y verifica; casos y
manuales en apartados separados. Con un tercer manual se agrega una regla:
**el ATLS 10 no reemplaza al ATLS 11 ni se mezcla con él.** Lleva su propio ID,
dice su edición y su origen en cada cita, y tiene su propio espacio en el
prompt. Si no se presenta así, quien lee no distingue la edición vigente de
una traducción automática de la anterior.

Todo detrás de una variable, como `GUIAS`: `ATLS10=0` (o sin el libro
indexado) deja el asistente como hoy, byte a byte.

---

## Decisiones para el usuario (antes de la fase 4)

1. **¿Quién lo pidió sabe que es una traducción automática de la edición
   anterior?** Esto decide el papel del libro:
   - **a) Tercer manual, citable** (lo que se pidió), con la etiqueta "10.ª
     ed. (2018), traducción no oficial" en cada cita y tarjeta.
     **Recomendado si quien lo pidió lo confirma sabiendo lo anterior.**
   - b) Solo para buscar: encuentra en español y lleva al pasaje equivalente
     del ATLS 11, que es lo que se cita. No se muestra su texto. Resuelve
     el problema del idioma sin el riesgo de la traducción, pero no es lo que
     se pidió.
   - c) Conseguir la traducción oficial del ACS y usar este plan con ese PDF
     (el perfil de extracción cambiaría).

   Las fases 1 a 3 sirven para las tres opciones: se puede empezar sin
   decidir.
2. **Espacio en el prompt.** Recomendado: **cuota por libro**, para que el ATLS
   10 no desplace al 11 ni al HUV. Se decide con la tabla de la fase 4.
   Retoma la decisión abierta de la cuota por libro (fase 4 de los libros).
3. **Cómo se nombra en la prosa.** Recomendado: la oración dice la edición ("La
   10.ª edición del ATLS recomienda…") y el verificador rechaza "el ATLS
   recomienda" con una cita del ATLS 10.
4. **Si aparecen las dos ediciones en una respuesta.** Recomendado: un aviso
   del código, no del modelo: "La 10.ª edición del ATLS (2018) es anterior a
   la 11.ª". El asistente no elige entre ediciones, igual que no elige entre
   casos y manuales. Si se quiere que diga "donde difieran, rige la 11.ª", lo
   decide el médico.

---

## Fase 1 — Extracción (hecha el 23/10/2026; falta la revisión a mano)

**Resultado:** `python scripts\extraer_guias.py` → `guias/secciones.jsonl` con
los tres libros, 1.617 secciones.

| | ATLS 10 |
|---|---|
| Secciones | 682 en 23 capítulos (IDs `ATLS10-03-005`) |
| Texto | 1,12 M caracteres, mediana de 1.190 por sección |
| Descartado | bibliografía 116 k; rótulos de figura y objetivos 25 k; material sin pie 4 k |
| Cobertura | el 90 % del texto de las páginas extraídas (lo demás es casi todo bibliografía) y el 75 % del libro (el ATLS 11, 74 %) |
| Páginas impresas | cuadran en las 362 páginas con número en el encabezado |
| Huella | ATLS10 `682 32bd704cd2af`; las tres juntas `1617 1a058871c67d` |

**El ATLS 11 y el HUV salen idénticos byte a byte** a la extracción del 19/10
(`935 9885b7f88286`): todo lo nuevo va detrás de opciones del perfil. Tests:
262 (25 nuevos en `test_guias.py` y uno en `test_responder_guias.py`).

**No se reindexó:** `secciones.jsonl` ya trae el ATLS 10, pero el índice de la
app sigue con dos libros hasta la fase 3. No correr `rag.index --fuente
guias` antes de la fase 5: el prompt y el verificador todavía no conocen el
ATLS 10.

Lo que hubo que agregar a `rag/guias.py` además de lo planeado, todo por
opciones del perfil:
- **Tablas armadas por columnas** (`tablas_por_columnas`). En este PDF cada
  renglón es un bloque, y las celdas de dos líneas se intercalaban entre
  columnas. Ahora las tablas "trampa | prevención" salen una fila por trampa
  y la tabla 3-1 (clases de hemorragia) completa. Para eso se guarda la caja
  de cada renglón (`Bloque.cajas`). Además, lo que cae dentro del alto y el
  ancho de una tabla va con su pie, porque el pie es más angosto que la tabla.
- **Títulos de dos renglones** (`unir_titulos`): venían en dos bloques y el
  segundo reemplazaba al primero ("Clase II" en vez de "15 % a 30 %… Hemorragia
  Clase II").
- **Leyendas de figura** (rol `leyenda`): se guardan como anexo de la
  sección, con su continuación en gris. Los rótulos de dentro de la figura y
  los créditos se descartan.
- **Objetivos de la primera página de los apéndices** (`recortar_arriba`).
- **Frases repetidas por la traducción en otra caja** (`quitar_repetidas`):
  la oración que repite una de las 3 anteriores del mismo párrafo.
- **Líneas repetidas:** se comparan con tolerancia de 1 punto, no con
  `round()` (las copias difieren en milésimas, y redondear las separaba).
- **Reemplazos:** las letras de las partes de una figura ("SEGUNDO." → "B.") y
  los rótulos "UN segundo do" sueltos.

**Lo que queda, conocido:**
- 0,5 % de texto repetido, en párrafos distintos o en tablas. Algunas
  repeticiones son legítimas: dos pacientes del escenario de triage.
- **Se descartan dos recuadros que el médico podría echar de menos:**
  - la hoja de la escala ASIA (p. 188 del PDF, texto de 2 a 4 pt);
  - el recuadro NEXUS (p. 195), que la traducción dejó ilegible ("mi - mi
    Toh (alcohol)"). Los criterios NEXUS sí están en el cuerpo del capítulo 7.
- Orden de lectura cruzado en algunas listas dentro de recuadros (cap. 2,
  intubación asistida por drogas).
- La primera sección de cada capítulo no tiene título (23); en la cabecera
  sale solo el capítulo.
- **Para la fase 5:** la cabecera escribe "cap. 14 Apéndice A: Trauma ocular",
  porque los apéndices van del 14 al 20. Mostrar solo el título cuando
  empieza por "Apéndice".
- `guias/muestra_revision.md` se regeneró el 23/10 con los tres libros (30
  secciones): **falta revisarla a mano** contra el PDF.

Plan original:

Perfil nuevo `atls10` en `guias/perfiles.yaml` (fuente `ATLS10`, idioma `es`,
`guion_corta_palabra: true`) y cambios en `rag/guias.py`, **detrás de
opciones del perfil**. Así el ATLS 11 y el HUV extraen igual que hoy.

1. **Líneas repetidas** (`deduplicar_lineas: true`): en `bloques_de_pagina`,
   una línea con la misma caja (redondeada) y el mismo texto que otra de la
   página se descarta. Medir cuánto queda: tiene que bajar del 17,9 % a ~0.
2. **Páginas duplicadas**: 76, 127, 220 y 420 a `saltar`.
3. **Página impresa por tramos** (`desfases: [[pdf_desde, desfase], …]` en
   vez de `desfase_pagina`). La tabla se arma con los números de los
   encabezados (detectados en 323 de 380 páginas) y se comprueba con un test:
   coincide con todo número detectado. No se lee el número de cada página
   porque a veces viene traducido.
4. **Roles por color** (la tabla de arriba). `_calza` ya admite `color` y
   `tam`; falta probar el orden de las reglas con bloques sintéticos, como los
   tests de hoy.
5. **Tablas y figuras:**
   - "Tabla 3-1 …" (pie arriba) con su material;
   - las tablas "trampa | prevención" como tabla, con esos rótulos y sin
     tomarlos por títulos. Son conducta concreta ("Identificar a los
     pacientes con anatomía de la vía aérea difícil");
   - las figuras ("norte FIGURA 1-4 …") se descartan con sus rótulos y
     créditos, como los pies de foto del HUV;
   - los algoritmos en imagen no se transcriben por ahora (tampoco los del
     ATLS 11).
6. **Arreglos mecánicos de la traducción**, solo los que son del PDF y no
   del libro, en una lista `reemplazos` del perfil, con test:
   - quitar "norte" delante de FIGURA, TABLA y CAPÍTULO;
   - quitar "VOLVER AL ÍNDICE" y el encabezado (`margen_sup`, `margen_inf`);
   - la letra capitular de cada capítulo ("W gallina que tratan…" → "Los
     médicos que tratan…"). Son 13 más los apéndices, uno por uno.

   **Los errores de traducción del contenido no se corrigen** ("encuesta
   secundaria"): el texto es el del libro y la tarjeta lo muestra tal cual.
   Se anotan para el médico (fase 7).
7. **Capítulos**, con su página de inicio en el PDF (de la p. 46):
   | Capítulos | Inicio en el PDF | `capitulo_num` |
   |---|---|---|
   | 1 a 13 | 55, 75, 96, 116, 137, 157, 183, 203, 224, 242, 270, 282, 296 | 1 a 13 |
   | Apéndices A a G | 313, 321, 331, 345, 360, 374, 392 | 14 a 20, con el título "Apéndice A: …" |
   | Resultados, Prevención y Biomecánica | 450, 453, 458 | 21 a 23 |
   - Fuera: preliminares (PDF 1–54), índice alfabético (435–449),
     bibliografía y las dos portadas de cada capítulo (título y "Esquema del
     capítulo / OBJETIVOS"; `saltar_portada` hoy salta una).
   - `no_clinicos`: D (desastres), E (equipo), F (escenarios de triage),
     Resultados y Prevención. Se indexan igual, como hoy.
8. **ID**: `ATLS10-03-005`. Cambiar `RX_CITA_GUIA` a un prefijo con letras y
   dígitos (`[a-z][a-z0-9]{1,6}`), con test. Los `startswith("ATLS-")` de los
   scripts no lo confunden con el 11.
9. `scripts/extraer_guias.py --libro atls10` (hoy solo acepta `atls` y `huv`).

**Hecho cuando:**
- la huella del ATLS 11 y del HUV no cambia (`935 9885b7f88286`, calculada
  solo sobre esas dos fuentes);
- ningún párrafo tiene una línea repetida;
- 0 "VOLVER AL ÍNDICE" y 0 "norte FIGURA";
- todas las páginas impresas cuadran con la tabla de desfases;
- cobertura medida como el 19/10 (qué % del texto entra y qué se descarta a
  propósito);
- muestra de 30 secciones revisada contra el PDF.

Estimado: ~1,1 M caracteres útiles, unas 600 secciones.

## Fase 2 — Conjunto de evaluación (1 día)

1. **Anotar el ATLS 10 en las 51 consultas** de `eval/consultas_guias.yaml`
   (grado 2 y 1, como las demás; propuestas por Claude). Sin esto, R@ppto
   cuenta como no relevante todo pasaje del ATLS 10 que entre al
   presupuesto, como pasó con los flujogramas del HUV el 19/10.
2. **Tabla de equivalencias de capítulos, 10.ª → 11.ª**, en el perfil
   (`equivalente_11`). Sirve para anotar más rápido y para la tarjeta de la
   app ("en la 11.ª ed.: cap. 6"). Propuesta, para que la valide el médico:
   | 10.ª | 11.ª | 10.ª | 11.ª |
   |---|---|---|---|
   | 1 Evaluación inicial | 1, 14 | 8 Musculoesquelético | 10 |
   | 2 Vía aérea y ventilación | 4, 5 | 9 Lesiones térmicas | 9 |
   | 3 Choque | 3, 6 | 10 Pediátrico | 11 |
   | 4 Tórax | 5, 21 | 11 Geriátrico | 12 |
   | 5 Abdomen y pelvis | 21 | 12 Embarazo | 13 |
   | 6 Cabeza | 7 | 13 Traslado | 15 |
   | 7 Columna y médula | 7 | A Ocular | 23 |
   | B Hipotermia y calor | 8, 9 | C Víctimas en masa, entornos austeros | 24 |
   | D Desastres | 17 | E Equipo de trauma | 2 |
   | F Escenarios de triage | 17 | G Habilidades | — |
   | Prevención | 18 | Biomecánica | 22 (en parte) |
3. **Preguntas sintéticas del ATLS 10**: unas 40, con
   `generar_preguntas_guias.py` sobre sus secciones (hoy son 89 del 11 y del
   HUV).
4. **Más consultas fuera de dominio** (5 a 10). El margen de `UMBRAL_GUIAS`
   ya es corto (`neg-09` en 0,326, el umbral en 0,34) y el ATLS 10 trae temas
   cercanos: medicamentos del adulto mayor, embarazo, violencia de pareja.
   Estaba pendiente de antes.

Las anotaciones las valida el médico junto con las 51 que faltan (fase 7 de
los libros).

## Fase 3 — Fragmentación e indexación (medio día)

- **`fijo-256`, como los otros dos.** No se repite la comparación de 7
  configuraciones. Solo una confirmación con la densa sobre las preguntas
  sintéticas del ATLS 10: `fijo-256` contra `semantico`, que empataron en la
  fase 2.
- `python -m rag.index --fuente guias`: reindexa la colección `guias` con los
  tres libros. Estimado: unos 1.600 fragmentos más (de 2.201 a ~3.800) y
  ~30 s. Ningún fragmento por encima de 510 tokens (la indexación falla si
  pasa).
- `meta.json` y el payload no cambian de formato (`fuente: ATLS10`,
  `idioma: es`).

## Fase 4 — Recuperación: que no desplace a los otros (2 días)

Es la fase que decide (decisión 2). `scripts/calibrar_guias.py` y
`comparar_fragmentacion.py` pasan a medir por libro: R@ppto y % de fragmentos
del ATLS 11, del ATLS 10 y del HUV. Configuraciones, una a la vez, con la
ruta `rerank`:

| Configuración | Qué cambia |
|---|---|
| a) común | como hoy, con los tres libros en la misma lista |
| b) cuota por libro | cada libro, una parte del presupuesto; lo que uno no usa pasa a los otros (medido el 17/10 con dos libros) |
| c) candidatos por libro | los 30 primeros de la densa **de cada libro**, y el reranker sobre los 90. Medir la latencia: hoy son 0,88 s |
| d) b + c | |

Más:
- **`PRESUPUESTO_GUIAS`**: con tres libros, 1.400 tokens son ~2 fragmentos
  por libro. Medir 1.800 (el 17/10 dio 0,354 contra 0,331). El prompt más
  largo medido es de 4.410 tokens: con 400 más y 600 de salida sigue cabiendo
  en `NUM_CTX` 6.144.
- **La guarda (`UMBRAL_GUIAS`) se recalibra** con `calibrar_guias.py` y las
  dos juntas con `calibrar_umbral.py`. Si una fuera de dominio pasa, primero
  se agregan consultas fuera de dominio y no se mueve el umbral (regla del
  19/10).

**Hecho cuando** (línea base del 19/10):
- el ATLS 11 no pierde: R@ppto anotado del ATLS ≥ 0,203 y su % de fragmentos
  no baja del 34 %;
- el HUV no baja de 0,494 (su valor con cuota);
- 0 de 10 fuera de dominio por los manuales;
- latencia de la búsqueda ≤ 1,5 s.

**Lo que se espera ganar:** de las 12 de 25 consultas de trauma sin manuales,
varias deberían encontrar pasajes del ATLS 10 (hoy fallan por el idioma, las
abreviaturas o la ortografía). Medirlo: es el argumento para la reunión.

## Fase 5 — Prompt y verificación (1 a 2 días)

- `config.NOMBRES_LIBROS`: `"ATLS10": "ATLS 10.ª ed. (2018)"`.
- `build_prompt.py`:
  - "tres manuales" en `_REGLAS_DOS_FUENTES`;
  - la cabecera del fragmento dice "(edición anterior, 2018; traducción no
    oficial)";
  - las instrucciones piden "El ATLS 11.ª ed. recomienda", "La 10.ª edición
    del ATLS recomienda" o "El manual del HUV recomienda".

  **Ojo con `_VOCABULARIO_CODIGO`:** todo lo escrito en `build_prompt.py`
  cuenta como vocabulario permitido. No poner ejemplos clínicos.
- **`verificar.py`, regla 8 por edición:** si la oración nombra la 10.ª
  ("10.ª", "décima"), tiene que citar `ATLS10-`. "El ATLS" sin edición que
  cita `ATLS10-` se rechaza, con el motivo "no dice la edición". "El ATLS"
  que cita `ATLS-` sigue como hoy.
- **Regla 10 (términos y vocabulario) aplica al ATLS 10**, porque está en
  español. Queda más verificado que el ATLS 11, del que solo se revisan cita
  y cifras. Medir si las palabras raras de la traducción producen falsos
  rechazos: el modelo tiende a "corregir" "encuesta secundaria". No se
  agregan a `PALABRAS_COMUNES` sin medirlo.
- **Aviso de las dos ediciones** (decisión 4), escrito por el código en el
  apartado de manuales, solo cuando hay pasajes de las dos.
- **Medir** (`comparar_modelos.py --modos prosa --consultas guias`, 41
  consultas, `PROSA_SIEMPRE=0` para ver la verificación) contra la línea base
  del 19/10: 7 de 21 completas y 18 de 21 con alguna parte en prosa. Además:
  0 JSON inválidos, mediana ≤ 8,5 s (hoy 7,6) y cuántas oraciones citan el
  ATLS 10 y pasan la regla de la edición. Nueva columna
  `pasajes_citados_atls10`.

## Fase 6 — App y demo (medio día)

- Tarjeta del pasaje con la etiqueta "10.ª ed. · 2018 · traducción no
  oficial" y, si hay equivalencia, "en la 11.ª ed.: cap. N".
- Textos de "Alcance", del interruptor y del aviso: los tres manuales.
- `preparar_demo.py`: comprobar que el ATLS 10 esté indexado y agregar una
  consulta que responda con él (de las que hoy quedan sin manuales). Tiene
  que terminar en "Todo listo".
- Guion en `SETUP.md` §6 y pasos de instalación en `CONTEXTO_SESION.md`:
  copiar el PDF a `anonimizacion/data/` (fuera de git, como los otros) y la
  huella nueva.
- Tests (hoy 237) y la app con `AppTest`.

## Fase 7 — Revisión médica

Se suma a la fase 7 de los libros:
- una muestra de pasajes del ATLS 10 citados: ¿la traducción dice lo que
  dice el libro? Los errores de traducción que cambien una conducta se
  anotan y deciden si la sección se excluye;
- la tabla de equivalencias 10.ª → 11.ª;
- las anotaciones de la fase 2;
- el aviso de las dos ediciones (decisión 4).

---

## Orden y tiempo

**1 → 2 → 3 → 4 (decisiones 1 y 2) → 5 → 6**, unos **6 a 8 días**. Las fases
1 a 3 no tocan lo que ve el médico: se pueden hacer mientras se responde la
decisión 1.

**Si hay poco tiempo:**
- fases 1 y 3 completas;
- en la 2, solo las consultas fuera de dominio;
- en la 4, la cuota por libro sin las otras configuraciones;
- en la 5, nombres, cabecera y regla 8;
- sin la tabla de equivalencias.

**Después**, el plan del chatbot clínico rehace su línea base (fase 0) con los
tres libros. En su fase 4 (orden xABCDE), la letra por capítulo del ATLS 10
sale de la tabla de equivalencias. Ojo: el ATLS 10 es ABCDE, sin la "x".
