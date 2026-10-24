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

1. ~~**¿Quién lo pidió sabe que es una traducción automática de la edición
   anterior?**~~ **Sí (usuario, 23/10): opción a), tercer manual citable con la
   etiqueta de edición y origen.** Las opciones eran:
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
2. ~~**Espacio en el prompt.**~~ **Decidido (usuario, 23/10): cuota por libro y
   1.800 tokens cuando el ATLS 10 esté encendido** (fase 4). Retoma la decisión
   abierta de la cuota por libro (fase 4 de los libros).
3. **Cómo se nombra en la prosa.** **Decidido (usuario, 23/10): lo recomendado.**
   La oración dice la edición ("La
   10.ª edición del ATLS recomienda…") y el verificador rechaza "el ATLS
   recomienda" con una cita del ATLS 10.
4. **Si aparecen las dos ediciones en una respuesta.** **Decidido (usuario,
   23/10): lo recomendado**, más un aviso cuando solo hay pasajes del ATLS 10
   (por la decisión 1). Un aviso
   del código, no del modelo: "La 10.ª edición del ATLS (2018) es anterior a
   la 11.ª". El asistente no elige entre ediciones, igual que no elige entre
   casos y manuales. Si se quiere que diga "donde difieran, rige la 11.ª", lo
   decide el médico.
5. ~~**Las guardas con 20 consultas fuera de dominio**~~ **Decidido (usuario,
   23/10): guarda por señal de trauma.** Sin señal, libros 0,65 y casos 0,7
   (fase 4). Lo que sigue es el planteo original.
   Con los umbrales de hoy responderían 3 de 20 (neg-05, neg-15, neg-19). Ni
   subir `UMBRAL_GUIAS` ni `UMBRAL_RERANK` sale gratis:
   - `UMBRAL_GUIAS` de 0,34 a 0,40 frena a neg-15 en los libros, pero deja
     sin libros a fin-04 (0,351): 13 de 25 de trauma sin manuales;
   - en los casos no hay hueco: neg-19 está en 0,208 y voc-06, de trauma, en
     0,215.

   Recomendado: decidirlo en la fase 4, con el ATLS 10 indexado (cambia los
   puntajes de los libros) y con la expansión de abreviaturas del plan del
   chatbot en mente, que sube las de trauma sin tocar las fuera de dominio.

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
| Huella | ATLS10 `689 3bbb01c493d0`; las tres juntas `1624 a15bc4fb7152` (fase 2) |

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

## Fase 2 — Conjunto de evaluación (hecha el 23/10/2026; falta validarla)

**Resultado:**
- **Anotaciones** (`eval/consultas_guias.yaml`): las 41 consultas de trauma y
  de procedimiento llevan al final su bloque `ATLS10-*`, con **137 secciones de
  grado 2 y 257 de grado 1** (el ATLS 11 y el HUV juntos: 156 y 223). Propuesta
  de Claude a partir del índice de secciones y de los candidatos de la
  búsqueda (densa sobre fragmentos `fijo-256` + reranker). La búsqueda sola no
  alcanzaba: en lex-03, "Manejo de la lesión cerebral grave (GCS 3-8)" salía
  con probabilidad 0,01.
- **`scripts/calibrar_guias.py` solo cuenta los libros del índice**
  (`libros_indexados`, `cargar_consultas(..., libros)`). Hasta la fase 3 las
  secciones del ATLS 10 no se pueden recuperar: si contaran, R@ppto bajaría
  sin que la búsqueda haya cambiado. Así la línea base del plan del chatbot
  sigue siendo comparable. Test en `test_comparar_fragmentacion.py`.
- **Tabla de equivalencias 10.ª → 11.ª** (`equivalente_11` en el perfil, con
  test). Comprobada contra las anotaciones: en 35 de las 37 consultas con
  grado 2 en los dos ATLS, los capítulos coinciden por la tabla. Las dos que
  no se explican: el cuello penetrante está en los capítulos 1 y 2 de la 10.ª
  y en el 22 de la 11.ª, y el E-FAST está en el apéndice de habilidades, que no
  tiene equivalente. Se agregaron a la tabla los capítulos que la 11.ª separa:
  el 3 (hemorragia exanguinante, con el torniquete), el 22 (penetrante) y el 6
  (parada traumática).
- **38 preguntas sintéticas del ATLS 10** (`sin-090` a `sin-127`), con
  `generar_preguntas_guias.py --libro ATLS10 --agregar`: las 89 de antes
  siguen idénticas. Se descartaron 14 de 52 por copiar 5 o más palabras
  seguidas del texto, lo esperable con un libro en el idioma de la pregunta.
- **10 consultas fuera de dominio nuevas** (`neg-11` a `neg-20`, en
  `consultas.yaml` y `consultas_guias.yaml`): embarazo, adulto mayor,
  anticoagulación, neurología, intoxicación, anafilaxia, insuficiencia
  cardiaca e hipoglucemia. Se dejaron fuera los agentes nerviosos, porque el
  apéndice D los trata.
- **Extracción corregida al anotar:** la traducción achicó algunas franjas de
  título a 8,4 pt ("Trauma de la cabeza", p. 258 del PDF) y el trauma craneal
  pediátrico quedaba bajo "traumatismo abdominal". Además, la traducción
  cambia la palabra de los pies de tabla ("mesa 9-1" por *Table*, "caja de
  correos 4" por *Box*). Ahora es pie el texto blanco con la palabra y un
  número, y su segundo renglón se suma al pie (`unir_pies`). El ATLS 10 pasa a
  689 secciones; el ATLS 11 y el HUV siguen idénticos. Tests: 270.

**Medido el 23/10 con el índice de hoy** (ATLS 11 y HUV, ruta `rerank`,
`calibrar_guias.py` y `calibrar_umbral.py`):
- **La línea base no se movió:** R@ppto 0,327 en las anotadas y 0,854 en las
  sintéticas, y 12 de 25 consultas de trauma sin libros, como el 19/10.
- **Las consultas fuera de dominio nuevas destapan debilidades que ya
  existían, no del ATLS 10:**
  | Consulta | Casos (p2) | Libros | Responde por |
  |---|---|---|---|
  | neg-05 apendicitis (conocida desde el 27/09) | 0,423 | 0,019 | casos |
  | neg-15 ACV isquémico con hemiparesia | 0,580 | **0,361** | casos y libros |
  | neg-19 insuficiencia cardiaca descompensada | 0,208 | 0,029 | casos (umbral 0,2) |

  Con las dos guardas juntas responderían 3 de 20 (antes, 1 de 10). Por los
  libros pasa solo neg-15 (0,361), atraída por un pasaje del ATLS 11 sobre
  el equipo de trauma y el shock en la lesión medular (`ATLS-02-018`). Por los
  casos la atraen caídas con "no le responden las piernas". neg-19 pasa la
  guarda de los casos por 0,008, y la consulta de trauma más baja (voc-06)
  está en 0,215.
- **neg-17 se cambió** (acetaminofén → migraña con aura). El registro de
  trauma tiene casos de intoxicación (mecanismo `intoxicacion`): el
  acetaminofén encontraba casos reales y no era fuera de dominio para los
  casos.
- **No se movió ningún umbral.** Es la regla del 19/10 (primero más consultas
  fuera de dominio), y ahora hay 20. Se decide en la fase 4 con el ATLS 10
  indexado, porque la guarda de los libros se recalibra igual (decisión 5,
  abajo).

Plan original:

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

## Fase 3 — Fragmentación e indexación (hecha el 23/10/2026)

**Resultado:**
- **`fijo-256` confirmado para el ATLS 10.**
  `comparar_fragmentacion.py --libros ATLS10` (opción nueva: un libro solo,
  con sus secciones y sus relevantes) da `reports/comparar_fragmentacion_atls10.csv`:
  | R@ppto, solo el ATLS 10 | anotadas (41) | sintéticas (38) |
  |---|---|---|
  | fijo-256, densa | **0,411** | **0,842** |
  | semantico, densa | 0,351 | 0,789 |
  | fijo-256, rerank-densa | **0,458** | 0,868 |
  | semantico, rerank-densa | 0,416 | **0,895** |

  `fijo-256` gana en las anotadas con las dos rutas. `semantico` gana en las
  sintéticas con el reranker por una sola pregunta (2,6 puntos), que es ruido.
- **Indexado** (`python -m rag.index --fuente guias`, 28 s): **4.013
  fragmentos** (ATLS 1.652, ATLS 10 1.812 con 408 de tablas, HUV 549), todos
  de 256 tokens o menos, sin secciones sin fragmentos. La colección `trauma`
  sigue con 8.198 puntos. El índice del 19/10 quedó respaldado en el
  scratchpad de la sesión.
- **Interruptor `ATLS10`** (`config.py`, apagado por defecto): con el ATLS 10
  en el índice, `retrieve.candidatos_guias` lo deja fuera con un filtro de
  Qdrant (`libros_apagados`). Todo pasa por ahí: la app, `generate.responder`
  y los scripts. `calibrar_guias.py` tampoco cuenta sus relevantes si está
  apagado. Test en `test_buscar_guias.py`; 271 tests.
- **Con `ATLS10=0` se reproduce la línea base** en el punto de operación
  (`UMBRAL_GUIAS` 0,34, 1.400 tokens): R@ppto 0,327 y 0,854, 12 de 25 de trauma
  sin libros, 1 de 20 fuera de dominio (neg-15, 0,361). Con otros presupuestos
  hay diferencias de 0,005 a 0,011 (a 1.000 tokens, 0,276 contra 0,281), que
  probablemente vienen de volver a calcular los embeddings en fp16 con lotes
  de otra composición.

**Vista previa para la fase 4** (`ATLS10=1`, presupuesto común, la
configuración a):
| | ATLS10=0 | ATLS10=1 |
|---|---|---|
| Trauma sin libros (de 25) | 12 | **9** |
| Fuera de dominio que pasan (de 20) | 1 (neg-15) | **3** (neg-15 0,362, neg-20 0,475, neg-12 0,541) |
| R@ppto anotadas · sintéticas | 0,327 · 0,854 | 0,221 · 0,780 |

- El ATLS 10 rescata 3 de las 12 consultas sin libros, pero el capítulo de
  embarazo atrae "sangrado vaginal en el primer trimestre" y algo del libro
  atrae "hipoglucemia en diabético". Con 0,40 de umbral seguirían pasando 2.
- **R@ppto no es comparable entre las dos columnas:** con `ATLS10=1` el
  denominador suma las 137 secciones de grado 2 del ATLS 10. Si el ATLS 11
  pierde lugar se mide en la fase 4, con R@ppto por libro.

Plan original:

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

## Fase 4 — Recuperación: que no desplace a los otros (hecha el 23/10/2026)

**Decisiones tomadas (usuario, 23/10), ya en el código:**
- **Decisión 2: cuota por libro con 1.800 tokens**, solo con el ATLS 10. En
  `config.py`, `CUOTA_GUIAS` y `PRESUPUESTO_GUIAS` toman su valor de `ATLS10`:
  apagado, 1.400 sin cuota, como antes; encendido, 1.800 con cuota.
  `CANDIDATOS_POR_LIBRO` queda apagado.
- **Decisión 5: guarda por señal de trauma** (`rag/senal.py`, `SENAL_TRAUMA`).
  Si la consulta no nombra un mecanismo, una lesión ni un procedimiento de
  trauma, los libros piden que el mejor fragmento llegue a 0,65
  (`UMBRAL_GUIAS_SIN_SENAL`, ruta rerank) y los casos piden 2 con 0,7 o más
  (`UMBRAL_RERANK_SIN_SENAL`; al prompt siguen entrando los de 0,2 o más). Sin
  señal no se abstiene: 63 de las 127 sintéticas no la tienen.
  - Las listas se escribieron antes de medir: los mecanismos de la
    anonimización y listas generales de lesiones y procedimientos, sin regiones
    ni léxico de enfermedades. Lo único que se corrigió después fue un defecto:
    patrones sin límite de palabra ("riña" en "warfarina" y "urinaria", "ácido"
    en "cetoacidosis").
  - Medido con `scripts/comparar_guardas.py`:
    | Sin señal | Libros: fuera de dominio que pasan | Sintéticas sin libros | Casos: fuera de dominio que responden | Trauma que se abstiene |
    |---|---|---|---|---|
    | como hoy (0,34 / 0,2) | 3 | 2/127 | 3 | 0/25 |
    | **0,65 / 0,7** | **0** | 5/127 | **0** | **0/25** |

    Ninguna fuera de dominio tiene señal; de las de dominio no la tienen 1 de
    25 de trauma ("qemadura", mal escrita; sus casos puntúan 0,95) y 2 de 16 de
    procedimiento (0,99 en los libros). El margen sobre la fuera de dominio
    más alta es de 0,11 en los libros (sangrado vaginal, 0,541) y de 0,12 en
    los casos (ACV, 0,580).
  - **De punta a punta** (las guardas reales de `responder`, vía
    `comparar_modelos.recuperar`, sin el modelo): con `ATLS10=1` responden las
    25 de trauma (16 con las dos fuentes, 9 solo casos) y las 16 de
    procedimiento, y se abstienen las 20 fuera de dominio. Con `ATLS10=0` (la app
    de hoy, que ya lleva esta guarda) las de trauma responden como antes (13 y
    12) y las fuera de dominio que responden bajan de 3 a 0.
  - Tests: `tests/test_senal.py` (16); en total, 290.
- **Estado del "Hecho cuando":** el ATLS 11 (0,209 ≥ 0,203) y el HUV (0,569 ≥
  0,494) se mantienen, 0 fuera de dominio (mejor que la meta) y la búsqueda
  tarda 0,80 s. No se cumple que el ATLS 11 conserve su parte del prompt (18 %,
  antes 24 %): con tres libros no puede.

**Mediciones de la fase 4** (lo que llevó a las decisiones):

**Qué se programó** (todo apagado por defecto: la app sigue como hoy):
- `retrieve.seleccionar_guias(..., cuota=True)` (`CUOTA_GUIAS`): cada libro
  cuyo mejor fragmento pasa la guarda recibe una parte igual del presupuesto;
  lo que sobra vuelve al orden de siempre.
- `retrieve.candidatos_guias(..., por_libro=True)` (`CANDIDATOS_POR_LIBRO`):
  los 30 primeros de cada libro, no de los tres juntos.
- `retrieve.libros_activos()`: los libros del índice menos los apagados (lo
  usan la búsqueda por libro y `calibrar_guias.py`).
- `scripts/comparar_libros.py`: las cuatro configuraciones con 1.400 y 1.800
  tokens, R@ppto por libro, % del prompt por libro, consultas sin libros,
  fuera de dominio y una grilla de guardas. Escribe
  `reports/comparar_libros.csv`. Tests: 274.

**Resultados** (ruta `rerank`, umbral 0,34; R@ppto en las 41 anotadas, salvo
"sint."; "% prompt" = fragmentos del prompt en las de trauma y procedimiento):
| Configuración | Tokens | R@ppto | ATLS 11 | ATLS 10 | HUV | sint. | % prompt ATLS 11 / 10 / HUV | Trauma sin libros | Búsqueda |
|---|---|---|---|---|---|---|---|---|---|
| **Hoy, sin ATLS 10** (común) | 1.400 | 0,327 | **0,203** | — | **0,563** | 0,854 | 24 / — / 76 | 12 | 0,82 s |
| común | 1.400 | 0,221 | 0,149 | 0,202 | 0,437 | 0,780 | **8** / 39 / 53 | 9 | 0,80 s |
| común | 1.800 | 0,256 | 0,171 | 0,236 | 0,534 | 0,803 | 12 / 36 / 52 | 9 | 0,80 s |
| cuota | 1.400 | 0,223 | 0,142 | 0,214 | 0,425 | 0,811 | 15 / 35 / 49 | 9 | 0,80 s |
| **cuota** | **1.800** | 0,272 | **0,209** | 0,239 | **0,569** | 0,811 | 18 / 36 / 47 | 9 | 0,80 s |
| por libro | 1.400 | 0,240 | 0,162 | 0,220 | 0,448 | 0,780 | 10 / 37 / 52 | 8 | 1,37 s |
| por libro + cuota | 1.800 | 0,279 | 0,160 | 0,258 | 0,552 | 0,819 | 18 / 35 / 47 | 8 | 1,37 s |

(R@ppto total no es comparable entre la primera fila y las demás: con el
ATLS 10 el denominador suma sus secciones de grado 2. Las columnas por libro
sí lo son.)

- **El ATLS 10 desplaza al ATLS 11, como se temía:** con el presupuesto común,
  el ATLS 11 pasa del 24 % al 8 % del prompt y su R@ppto de 0,203 a 0,149. El
  HUV también pierde (0,563 a 0,437).
- **Cuota con 1.800 tokens es la única configuración que deja al ATLS 11 y
  al HUV donde están hoy** (0,209 y 0,569) y agrega el ATLS 10 (0,239). Baja
  las consultas de trauma sin libros de 12 a 9 y no cambia la latencia. El
  ATLS 11 queda en el 18 % del prompt, no en el 24 %: no se cumple esa parte
  del "Hecho cuando". Con 1.400 tokens la cuota no alcanza, porque un tercio
  son ~2 fragmentos por libro.
- **Los candidatos por libro no compensan:** +0,57 s de búsqueda (el reranker
  ve 90), una consulta menos sin libros, y el ATLS 11 sale peor (0,160).
- Con 41 consultas, una sola mueve ~2,5 puntos por libro.

**Guardas, con el ATLS 10** (candidatos juntos; consultas de trauma sin libros
/ fuera de dominio que pasan, de 25 y de 20):
| Guarda | Trauma sin libros | Fuera de dominio | R@ppto (cuota 1.800) |
|---|---|---|---|
| mejor ≥ 0,34 (hoy) | 9 | 3 (neg-12 0,541, neg-20 0,475, neg-15 0,362) | 0,272 |
| mejor ≥ 0,50 | 11 | 1 (neg-12) | 0,272 |
| mejor ≥ 0,55 | 11 | **0** | 0,272 |
| mejor ≥ 0,34 y 2.º de otra sección ≥ 0,10 | 12 | 1 (neg-12) | 0,261 |

- **Subir el umbral no baja R@ppto:** las consultas de trauma que pierden los
  libros entre 0,34 y 0,55 (fin-04, 0,351, y fin-01, 0,463) no recibían
  ninguna de sus secciones relevantes.
- **Pero el margen es mínimo:** neg-12 ("sangrado vaginal en el primer
  trimestre") está en 0,541 y la consulta de trauma siguiente, ort-03
  ("luxación de hombro por caída"), en 0,553. Un umbral de 0,55 está calibrado
  a este conjunto, no separa de verdad.
- **La guarda de dos pasajes sirve menos de lo esperado:** frena neg-15 y
  neg-20, pero deja sin libros a 3 de trauma más y no frena neg-12 (su segundo
  pasaje, 0,214, es una tabla real de embarazo).
- Las tres fuera de dominio que pasan no tienen mecanismo ni lesión: sangrado
  en el embarazo, hipoglucemia, ACV. Otra guarda posible, no medida: exigir
  señal de trauma en la consulta (mecanismo, región o lesión del léxico) o un
  umbral más alto cuando falta. Serviría también para la guarda de los casos,
  que deja pasar neg-05, neg-15 y neg-19 por razones parecidas.

Plan original:

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
- fuera de dominio por los manuales: no más de las que pasan hoy, 1 de 20
  (neg-15, medido el 23/10), o las que deje la decisión 5;
- latencia de la búsqueda ≤ 1,5 s.

**Lo que se espera ganar:** de las 12 de 25 consultas de trauma sin manuales,
varias deberían encontrar pasajes del ATLS 10 (hoy fallan por el idioma, las
abreviaturas o la ortografía). Medirlo: es el argumento para la reunión.

## Fase 5 — Prompt y verificación (hecha el 23/10/2026)

**Qué se programó** (sin pasajes del ATLS 10, el prompt y la respuesta quedan
como antes, byte a byte):
- **Nombre del libro** (`config.NOMBRES_LIBROS`): "ATLS 10.ª ed. (2018, edición
  anterior, traducción no oficial)", en la cabecera de cada pasaje del prompt
  y en la de la tarjeta (decisión 1). Los apéndices se muestran como
  "Apéndice A: Trauma ocular", sin el número interno (14).
- **Prompt con tres manuales** (`build_prompt.SISTEMA_PROSA_GUIAS_ATLS10` y
  las dos instrucciones `_ATLS10`). Se usan solo si entre los pasajes hay
  alguno del ATLS 10 (`sistema_prosa_guias`, `construir_prompt_guias`):
  - presentan los tres manuales;
  - piden empezar con "La 11.ª edición del ATLS recomienda", "La 10.ª edición
    del ATLS recomienda" o "El manual del HUV recomienda";
  - prohíben comparar una edición con la otra.
- **Verificador** (`verificar._manual_nombrado`, regla 8 por edición): la
  edición nombrada es la citada. "El ATLS recomienda" con una cita del ATLS 10
  se rechaza ("sin decir la edición"); sin el ATLS 10, la regla es la de antes.
  La edición ("10.ª edición", "11.ª ed.") se quita antes de revisar cifras y
  vocabulario, y el separador de oraciones ya no corta en "ed.". Las reglas
  de términos y vocabulario (10) se aplican al ATLS 10 porque está en
  español: queda más vigilado que el ATLS 11.
- **Avisos del código** en el apartado de los manuales (decisiones 1 y 4),
  que la app sí muestra:
  - `AVISO_EDICIONES`, con pasajes de las dos ediciones: son dos ediciones,
    pueden no coincidir y el asistente no elige;
  - `AVISO_ATLS10`, con pasajes solo de la 10.ª: es la edición anterior, en
    traducción no oficial.
- `comparar_modelos.py` cuenta los pasajes del ATLS 10 citados y el rechazo
  "sin la edición". Tests: `tests/test_atls10_prosa.py` (9); 299 en total.

**Medido** (`comparar_modelos.py --modos prosa --consultas guias`, 41 consultas;
la verificación no depende de `PROSA_SIEMPRE`; `reports/comparar_modelos_atls10.csv`):
| | ATLS10=0 (hoy) | ATLS10=1 |
|---|---|---|
| Prosa verificada completa | **18 de 41** | 14 de 41 |
| Con las dos fuentes: completa / alguna parte en prosa | 7 / 18 de 20 | 6 / 15 de 23 |
| Solo manuales: completa | 3 de 9 | 2 de 9 |
| Cita algún pasaje del ATLS 10 (si lo tiene) | — | **22 de 24** |
| Cita algún pasaje del ATLS 11 | 13 | 15 |
| Rechazos "sin decir la edición" | — | **0** |
| Mediana / peor | 7,4 s / 10,6 s | 8,6 s / 11,8 s |
| Tokens del prompt / de salida (mediana) | 2.368 / 239 | 3.787 / 306 |
| Fuera de dominio que responden; de dominio que se abstienen | 0 de 20; 0 de 41 | 0 de 20; 0 de 41 |

Generador 100 % en GPU, sin JSON cortados.

- **El modelo usa bien el ATLS 10:** lo cita en 22 de las 24 consultas que
  traen pasajes suyos, siempre como "La 10.ª edición del ATLS recomienda…", y
  el ATLS 11 se sigue citando igual que antes.
- **La prosa verificada baja de 18 a 14.** Leídas, las oraciones del ATLS 10
  que no pasan se rechazan por:
  - **órdenes copiadas del libro**: la traducción es imperativa ("debe
    aumentar el índice de sospecha"); 6 rechazos por "orden" contra 2;
  - **términos clínicos que no están en el pasaje citado** ("hipercapnia"
    donde el pasaje dice que el CO₂ se acumula; "equimosis", "hemoneumotórax",
    "paraesternal": probablemente traducidos de un pasaje del ATLS 11 y
    atribuidos al ATLS 10). Rechazos justos;
  - **palabras de redacción** ("garantiza", "indique", "señala") y
    "correcciones" de la traducción ("motriz" donde el pasaje dice "respuesta
    del motor"): falsos rechazos, pero solo 2 consultas se rechazan
    únicamente por eso. No se agregaron a `PALABRAS_COMUNES`: aflojarían el
    verificador en los tres libros.
- **No se cumplen dos metas del "Hecho cuando":** el % verificado baja (14 contra
  18) y la mediana pasa de 8,5 s (8,6 s; son 1.400 tokens más de prompt).
  Con `PROSA_SIEMPRE=True` (la demo) la prosa se muestra igual: hay más
  oraciones sin verificar en pantalla, el riesgo aceptado el 18/10.
- **Segunda iteración (24/10):** la instrucción con el ATLS 10 pide escribir
  como recomendación lo que el pasaje dice como orden (`_TONO_TRES`). Resultado:
  15 de 41 verificadas (antes 14), mediana 8,9 s; no se nota más allá del
  ruido. Quedó porque no estorba. `reports/comparar_modelos_atls10.csv` es esta
  corrida. Leídos los rechazos:
  - las 6 órdenes que quedan son sobre todo del HUV ("se debe…"), como sin el
    ATLS 10;
  - **con tres manuales el modelo se equivoca más de manual** (6 rechazos
    "manual equivocado"): "El manual del HUV recomienda… [guía: ATLS10-14-011]",
    "La 11.ª edición del ATLS recomienda…" con un pasaje de la 10.ª. El
    verificador los atrapa (regla 8), pero **con `PROSA_SIEMPRE=True` se
    mostrarían igual**. Hay que tenerlo en cuenta al decidir si se enciende el
    ATLS 10 para la demo.

Plan original:

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

## Fase 6 — App y demo (hecha el 24/10/2026)

**Qué se hizo** (con `ATLS10=0` la app queda como estaba):
- **Tarjeta del pasaje:** la cabecera ya dice "ATLS 10.ª ed. (2018, edición
  anterior, traducción no oficial)". Se agrega la etiqueta "en la 11.ª ed.:
  cap. 7", con el título al pasar el cursor
  (`guias.equivalentes_atls11`, desde la tabla `equivalente_11`). Una etiqueta
  aparte con la edición, como pedía el plan, repetía la cabecera y se quitó.
- **Textos de la app** según `ATLS10`: ayuda del interruptor, "Alcance" (los
  tres manuales y que no elige entre ediciones), aviso bajo la respuesta (los
  libros que trajo esa respuesta) y leyenda del desplegable (el ATLS 10 tal
  como viene en la traducción, sin corregir). La precarga usa el sistema con
  tres manuales.
- **`preparar_demo.py`:** muestra la cuota, la guarda por señal de trauma y si
  el ATLS 10 está encendido e indexado. Botón nuevo, **"7 · ATLS en español"**
  (abr-03, "TCE leve con pérdida de conciencia transitoria", `atls10: true` en
  `consultas.yaml`): sin el ATLS 10 los manuales no la responden; con él, sí.
  Solo corre con `ATLS10=1`, y falla si no trae pasajes del ATLS 10.
- **Guion** (`SETUP.md` §6): el paso 7b, la variable para lanzar la demo con
  el ATLS 10 y las 20 consultas fuera de dominio de la guarda por señal.

**Verificado:**
- `preparar_demo.py` con `ATLS10=1`: **"Todo listo"**; los 7 botones bien, el 7
  con casos y manuales y prosa verificada, 100 % en GPU.
- La app con `AppTest`:
  - con `ATLS10=1`, la consulta del botón 7 sale sin errores, con "La 10.ª
    edición del ATLS recomienda…" y las citas en rojo, el aviso de la edición
    anterior y la tarjeta con "en la 11.ª ed.: cap. 7";
  - con `ATLS10=0`, nada del ATLS 10 en pantalla.
- Tests: 300.

**Pendiente para encender el ATLS 10 en la demo (decisión del usuario):** con
`PROSA_SIEMPRE=True`, las oraciones que atribuyen un pasaje al manual
equivocado se muestran igual (fase 5). Hoy `ATLS10=0`.

Plan original:

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
