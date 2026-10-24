# Plan — un asistente más clínico para la próxima reunión

Pedido (19/10/2026): para la próxima reunión, "un chatbot mucho más
sofisticado". Este plan lo traduce en cuatro mejoras que hacen que el
asistente trabaje como un clínico **sin darle más libertad al modelo**:

1. Que pregunte lo que falta.
2. Que ordene la respuesta como el ATLS (xABCDE).
3. Que siga la conversación sobre el mismo paciente.
4. Que encuentre más en los manuales.

## Principio

El de siempre (`CONTEXTO_SESION.md`, sección 0): el modelo redacta, el código
pone los datos y verifica. Tres de las cuatro mejoras (1, 3 y 4) no agregan
nada escrito por el modelo: son reglas del código. La 2 cambia la forma de la
prosa, y por eso lleva su propia verificación y su respaldo determinista.

Para la reunión: **más sofisticado no es más libre.** Pregunta como un
clínico, ordena como el ATLS, recuerda al paciente, y todo sigue citado.

Cada mejora va detrás de una variable de `config.py`, como `GUIAS`: apagada,
el asistente responde como hoy. La demo tiene marcha atrás.

| Mejora | Variable | Llamadas al modelo |
|---|---|---|
| 4 · Abreviaturas para los manuales | `ABREV_LIBROS=1` | ninguna extra |
| 3 · Ficha del paciente | `CONVERSACION=ficha` (`llm` = como hoy) | una menos en cada seguimiento |
| 1 · Preguntas | `PREGUNTAS=1` | ninguna extra |
| 2 · Orden xABCDE | `ORDEN_ATLS=1` | la misma de hoy, con otro formato |

Los tests fijan cada variable en cada caso y pasan con cualquier valor, como
con `PROSA_SIEMPRE`.

## Orden de trabajo: 0 → 4 → 3 → 1 → 2

No es el orden de la lista, a propósito:
- **La 4 primero:** es la más chica y sube cuántas consultas tienen manuales,
  que es lo que la 2 necesita para lucirse.
- **La 3 antes que la 1:** cuando el médico contesta una pregunta, ese turno
  es un seguimiento. Sin ficha, la respuesta a la pregunta no tiene dónde
  sumarse.
- **La 2 al final:** es la de más riesgo (más prosa que verificar) y se mide
  contra lo que dejen las otras.

Estimado, en días de trabajo: fase 0 medio día, 4 un día, 3 tres días, 1 dos
días, 2 de tres a cuatro días, demo un día. **Unas dos semanas.**

**Si hay menos de una semana:** fase 0, la 4 completa, la 1 con botones que
suman el dato al texto de la consulta (sin ficha) y la versión liviana de la 2
(abajo). Queda fuera la 3.

---

## Fase 0 — Antes de programar (línea base hecha el 24/10/2026; faltan las notas de la demo)

**Línea base** (RTX 4060, configuración del 24/10: tres manuales con el ATLS 10,
cuota y 1.800 tokens, guarda por señal de trauma, `PROSA_SIEMPRE=True`):
| Qué | Valor | Cómo |
|---|---|---|
| Las dos guardas juntas | 25 de 25 de trauma responden (16 con las dos fuentes, 9 solo casos); 0 de 20 fuera de dominio | `calibrar_umbral.py` |
| Manuales | R@ppto 0,272 (anotadas) y 0,819 (sintéticas); **9 de 25 de trauma sin manuales**; 0 de 16 de procedimiento; 0 de 20 fuera de dominio; 0,72 s por búsqueda | `calibrar_guias.py`, fila "app" |
| Prosa, 25 de trauma | 11 de 25 verificadas completas; mediana 8,9 s, peor 12,0 s; prompt ~4.000 tokens | `reports/linea_base_chatbot.csv` |
| Prosa, 41 con manuales | 15 de 41 verificadas; mediana 8,9 s | `reports/comparar_modelos_atls10.csv` |
| Demo | "Todo listo", 7 botones | `preparar_demo.py` |

Antes de medir, `calibrar_guias.py` y `calibrar_umbral.py` se pusieron al día:
ahora aplican la cuota y la guarda por señal de trauma, como la app. Sin eso
medían otra cosa.

**Falta** (lo hace el usuario):
- traer `PREGUNTAS_DEMO.md` y `CONSULTAS_DEMO.md` del PC de la GTX 1650;
- preguntarle a quien hizo el pedido qué le faltó.

Plan original:

1. Traer `PREGUNTAS_DEMO.md` y `CONSULTAS_DEMO.md` del PC de la GTX 1650 y
   pasar las notas a limpio (pendiente 12). Lo que dijeron los médicos en la
   demo del 28/09 puede cambiar las prioridades de este plan.
2. Preguntarle a quien hizo el pedido qué le faltó, con este plan a la vista.
3. Línea base en la RTX 4060, con el chatbot cerrado (Qdrant admite un solo
   proceso):
   ```powershell
   ..\.venv\Scripts\python.exe scripts\calibrar_umbral.py
   ..\.venv\Scripts\python.exe scripts\comparar_modelos.py --modos prosa
   ..\.venv\Scripts\python.exe scripts\comparar_modelos.py --modos prosa --consultas guias
   ..\.venv\Scripts\python.exe scripts\preparar_demo.py
   ```
   Guardar la consola de `calibrar_umbral.py` y copiar
   `reports/comparar_modelos.csv` como línea base. Valores del 19/10: 12 de
   25 de trauma sin manuales, 0 de 10 fuera de dominio por los manuales,
   R@ppto 0,327 (anotadas) y 0,854 (sintéticas), 7,6 s de mediana con
   manuales; con las dos fuentes, 7 de 21 en prosa completa verificada y 18 de
   21 con alguna parte.

---

## Fase 1 — Mejora 4: que encuentre más en los manuales (hecha el 24/10/2026)

**Problema.** 12 de las 25 consultas de trauma se quedan sin manuales
(`UMBRAL_GUIAS` 0,34). Casi todas por abreviaturas ("HPAF", "FX", "TCE"),
ortografía ("qemadura", "luxacion") o lenguaje coloquial ("sujeto baleado").
Los casos no tienen el problema: el corpus ya trae las abreviaturas
expandidas.

**Qué se hace.**
- `rag/consulta.py` (nuevo): `para_libros(consulta)` expande las abreviaturas
  con `anonimizacion/config/abreviaturas.yaml`, de forma aumentativa como el
  corpus: "HPAF en abdomen" → "HPAF (herida por arma de fuego) en abdomen".
  El YAML se lee como dato, con su ruta en `config.py` (`ABREVIATURAS_YAML`),
  como `CORPUS_JSONL`: no se importa código de `anonimizacion/`.
- Fuera las ambiguas que ya dieron errores (pendiente 6): `tx` (tratamiento o
  trauma) y `qx` (quirúrgico o quemadura). Lista `NO_EXPANDIR`, con test.
- **Solo en `retrieve.buscar_guias`.** La búsqueda de casos no cambia: así
  `eval_rutas.csv` y las calibraciones de los casos siguen valiendo.
- La app muestra en "Fragmentos de los manuales" la búsqueda usada, como
  "Casos consultados" muestra la consulta reescrita.
- **Opcional, medido aparte:**
  - ortografía: cambiar las palabras de 6 o más letras que no están en el
    vocabulario de los libros por la más parecida (`difflib`, corte alto, p.
    ej. 0,85). Solo para buscar;
  - glosario español → inglés para el ATLS: la consulta en español prefiere
    el HUV (el ATLS se lleva el 34 % de los fragmentos). Solo para buscar: el
    texto del ATLS sigue en inglés (decisión 2 de `PLAN_INDEXAR_LIBROS.md`).

**Cómo se mide.** `calibrar_umbral.py` (las dos guardas juntas) y
`calibrar_guias.py` (R@ppto en las 51 anotadas y las 89 sintéticas), un
cambio a la vez.

**Hecho cuando** (con la línea base del 24/10; la del 19/10 era 12 de 25 y
0,327 / 0,854, con dos manuales):
- las de trauma sin manuales bajan de 9 de 25 a 6 o menos;
- las fuera de dominio por los manuales siguen en 0 de 20. Ojo: las 20 se
  frenan gracias a la guarda por señal de trauma, y la expansión agrega
  palabras a la consulta;
- R@ppto no baja (0,272 y 0,819).

Si sube una fuera de dominio, no se mueve el umbral: primero se agregan
consultas fuera de dominio (`CONTEXTO_SESION.md`, sección 00).

**Resultado (24/10, RTX 4060, ruta rerank, ATLS 10 encendido).** Dos cambios,
medidos uno a la vez, los dos en `rag/consulta.py` y encendidos por defecto:

| | trauma sin manuales | R@ppto anot. | R@ppto sint. | fuera que pasan |
|---|---|---|---|---|
| línea base | 9 de 25 | 0,272 | 0,819 | 0 de 20 |
| abreviaturas, todas las del YAML menos `tx`/`qx`/`ha` | 7 de 25 | 0,271 | 0,811 | 0 de 20 |
| abreviaturas sin las que los manuales ya usan (`ABREV_LIBROS`) | 7 de 25 | 0,271 | 0,819 | 0 de 20 |
| + nombre del mecanismo (`MECANISMO_LIBROS`) | **6 de 25** | **0,280** | **0,819** | **0 de 20** |

Con `ATLS10=0`: de 12 a 10 de 25 sin manuales, 0 de 20 fuera de dominio.

- **Las que los manuales ya usan no se expanden** (`LAS_USAN_LOS_MANUALES`:
  10 veces o más en `guias/secciones.jsonl`; un test las recuenta). Con
  "FAST (ecografia FAST abdominal)", sin-114 perdía su sección. Dos significan
  otra cosa en los manuales: "Fr" es el calibre French (tubo de 28 Fr) y "AP",
  anteroposterior. "ha" quedó fuera porque casi siempre es el verbo.
- **El nombre del mecanismo** sale de `senal.MECANISMOS`, con una etiqueta por
  clase. Una consulta sin señal de trauma no cambia, así que no pesa en las
  fuera de dominio. Saca a voc-04 ("precipitación desde altura (caída)":
  0,236 → 0,390) y sube voc-02 (0,042 → 0,181) y voc-06 (0,045 → 0,293), sin
  que pasen. Ojo: las etiquetas se escribieron viendo esas consultas.
- **Cambia la búsqueda de 15 de las 188 consultas.** Ganan voc-03, voc-04, abr-04 y ort-01.
  Pierde abr-02 ("HPACP (herida por arma cortopunzante) en zona II de cuello":
  de 0,29 a 0,14, porque entra la cinemática del arma cortopunzante del HUV y
  sale el cuello del ATLS 11).
- **abr-01 ya no está "sin manuales", pero no recibe lo de cráneo.** Llegan
  9 fragmentos del HUV: de lo anotado, solo la cinemática del arma de fuego
  (grado 1). El resto es atención inicial del politraumatizado y trauma
  penetrante de tórax y abdomen: "herida por arma de fuego" pesa más que
  "cráneo". Por eso la cuenta de "sin manuales" sola no basta: hay que mirar
  R@ppto.
- La app muestra en "Fragmentos de los manuales" la búsqueda usada y resalta
  con ella. `preparar_demo.py` dice si están encendidas: "Todo listo".
- **No se hizo:**
  - la ortografía con `difflib`: ort-04 ("qemadura") está en 0,207, y lex-04,
    bien escrita ("quemadura de segundo grado…"), en 0,169; corregirla no
    alcanzaría el umbral;
  - el glosario español → inglés: es lo que le falta a abr-01 y a abr-02, que
    tienen las secciones de grado 2 en el ATLS 11.
- **No se midió la prosa.** Los pasajes cambian en 15 consultas; se medirá con
  la fase siguiente.

---

## Fase 2 — Mejora 3: que siga la conversación sobre el mismo paciente (hecha el 24/10/2026)

**Hoy.** Cada seguimiento ("¿y si está hipotenso?") pasa por
`generate.reescribir`: el modelo reescribe la pregunta con los últimos 4
turnos (`VENTANA_HISTORIAL`). Es una llamada más, el modelo puede cambiar
términos (como en `resumir_caso`, pendiente 4) y nadie ve qué entendió: solo
la etiqueta "consulta reescrita" en el desplegable.

**Qué se hace: una ficha del paciente armada por el código.**
- `rag/ficha.py` (nuevo): `Ficha` con mecanismo, regiones, hallazgos
  afirmados y negados, signos con su valor (TA, FC, FR, SatO2, Glasgow),
  grupo (niño, adulto mayor, embarazada) y estado (estable o inestable).
  `actualizar(ficha, mensaje)` la llena con reglas:
  - hallazgos: `hallazgos.terminos`, con su negación (lo mismo que ya lee los
    casos);
  - mecanismo y región: patrones para registro clínico, a partir de
    `MECANISMOS` y `REGIONES` de `anonimizacion/src/enrich.py`, copiados y
    adaptados (los de allá están hechos para la voz del paciente, "me
    dispararon"), con las etiquetas de `config.ETIQUETAS_*`;
  - signos: regex nuevas para los valores ("TA 80/50", "Glasgow 9", "GCS
    13"). `build_prompt._RX_VITALES` y `_RX_GLASGOW` hoy solo detectan si
    están.
- Cada mensaje es de uno de tres tipos, por reglas:
  - **caso por ID** ("caso 3377"): como hoy, `resumir_caso`;
  - **paciente nuevo**: el primer mensaje, uno que cambia el mecanismo o uno
    que dice "otro paciente" o "nuevo caso" → ficha nueva;
  - **seguimiento**: todo lo demás → se suma a la ficha.
- La búsqueda usa `ficha.consulta()`, un texto armado por el código ("herida
  por arma de fuego en abdomen, hipotensión o choque, Glasgow 13"), más lo
  que pregunta el mensaje. Los hallazgos negados no entran en la búsqueda: el
  embedding no entiende el "sin".
- `generate.responder` recibe la ficha y devuelve la nueva en
  `Respuesta.ficha`: los scripts la usan igual que la app.
- **La ficha se ve.** Arriba de cada respuesta, una línea: "Paciente: herida
  por arma de fuego · abdomen · hipotenso · Glasgow 13". En la barra lateral,
  un botón "Paciente nuevo" que la borra. Si el código tomó el mensaje como un
  paciente nuevo, lo dice. Así se ve qué entendió y se corrige con un clic.
- **Qué cambió.** En un seguimiento, una línea armada por el código compara
  con la respuesta anterior: "Con hipotensión: entran 3 casos nuevos; aparece
  transfusión (2 de 5)". Sale de los IDs y de `hallazgos.extraer` de las dos
  respuestas: no la escribe el modelo.
- `CONVERSACION=llm` deja la reescritura de hoy.

**Cómo se mide.**
- `eval/conversaciones.yaml` (nuevo): unas 15 conversaciones de 2 a 4 turnos,
  en registro clínico como `consultas.yaml`, con la ficha y el tipo esperados
  en cada turno. Con las trampas: negaciones ("no está hipotenso"), un
  paciente nuevo sin decirlo y un seguimiento que solo dice "¿y en niños?".
- Tests de `ficha.py` sin Ollama ni índices, como los de hoy.
- `scripts/probar_conversaciones.py` (nuevo): corre las conversaciones por el
  pipeline completo y cuenta ficha correcta, tipo correcto y si los casos del
  último turno coinciden en mecanismo y región (el proxy de `consultas.yaml`).

**Hecho cuando:** ficha correcta en el 90 % de los turnos o más, ninguna
conversación mezcla dos pacientes, y un seguimiento tarda lo mismo o menos que
hoy (se ahorra la llamada de reescritura).

**Resultado (24/10, RTX 4060).** `rag/ficha.py`, `eval/conversaciones.yaml`,
`scripts/probar_conversaciones.py`; **`CONVERSACION=ficha` por defecto**
(`llm` vuelve a la reescritura).

| | ficha | reescritura (`llm`) |
|---|---|---|
| ficha correcta, desarrollo (16 conversaciones, 47 turnos) | 47 de 47 | — |
| ficha correcta, control (10 conversaciones, 27 turnos) | 23 de 27 antes de corregir; 26 de 27 después | — |
| tipo de mensaje correcto | 74 de 74 | — |
| conversaciones que mezclan dos pacientes | 0 | — |
| seguimiento, mediana (desarrollo / control) | 9,9 s / 9,3 s | 13,8 s / 12,5 s (2,7 s de reescritura) |
| último turno, casos con mecanismo y región (grado 2) | 0,74 / 0,64 | 0,64 / 0,52 |
| prosa verificada | 19 de 46 / 10 de 25 | 12 de 45 / 10 de 26 |

- **El 100 % de desarrollo no mide nada nuevo.** Las conversaciones se
  escribieron antes que las reglas, pero las reglas se armaron mirándolas. Por
  eso se escribió el conjunto de control después, sin ajustar las reglas a él.
  Dio 23 de 27 (85 %, debajo del 90 %):
  - "mordida por perro" no era mordedura. Se agregó a `senal.py` y no solo a la
    ficha, porque sin señal de trauma la consulta se abstenía entera. Las
    guardas quedan igual: 25 de 25, 0 de 20, R@ppto 0,280 y 0,819;
  - "sin sangrado" no es "sin sangrado activo" para el léxico, que es el de los
    casos. Queda así.
- **Desvío del plan: la búsqueda son las palabras del médico,** no un texto
  armado con las etiquetas. Son las cláusulas del paciente, sin lo negado y sin
  lo que un mensaje posterior negó, con el mecanismo de la ficha si una
  hipótesis lo cambió, más lo que pregunta el mensaje de ahora. Las etiquetas
  perdían "zona II", "orificio de entrada parietal" o "evisceración". El primer
  mensaje de cada paciente se busca tal cual, como antes.
- **Lo que hacía mal la reescritura,** visto en las búsquedas:
  - perdía al paciente: "Glasgow 15, estable" quedó como "pacientes con Glasgow
    15, estable", sin la fractura de cadera, y ese turno dio 0 casos de grado
    2;
  - agregaba datos que nadie dijo, sacados de la respuesta anterior:
    "lesión punzante en mejilla derecha", "hipoventilación basal derecha";
  - "¿cuál es el manejo del dolor?" (conv-15) quedó sin señal de trauma y se
    abstuvo.
- **Reglas que se agregaron mirando los datos:**
  - "choque de moto" es una colisión, no shock;
  - las pupilas y el Glasgow no son una región;
  - "el paciente…" es el mismo, "paciente con…" es otro;
  - "TCE severo" es un paciente, aunque no nombre mecanismo ni hallazgo del
    léxico. Se vio en la fase 3; sin eso, un "¿y en niños?" después habría
    empezado un paciente nuevo.
- **La app:**
  - "Paciente: arma de fuego · abdomen · hipotensión o choque · TA 80/50 ·
    inestable" arriba de cada respuesta;
  - "Paciente nuevo: no se suma al anterior" cuando cambia de paciente;
  - la línea de qué cambió;
  - el botón "Paciente nuevo" en la barra lateral, con la ficha actual;
  - bajo la última respuesta, "Es otro paciente" o "Era el mismo paciente", que
    corrige y responde de nuevo.

---

## Fase 3 — Mejora 1: que pregunte lo que falta (hecha el 24/10/2026; falta el juicio del médico)

**Idea.** Un clínico no responde igual ante un paciente estable que ante uno
en choque, y los manuales tampoco: 9 de las 25 transcripciones de flujogramas
y tablas del HUV mencionan el shock y 4 la estabilidad. Si la consulta no lo
dice, el asistente lo pregunta.

**Responder primero, preguntar después** (recomendado, decisión 1). En el
punto de atención se pide la conclusión primero (`CONTEXTO_SESION.md`,
sección 2.7). La respuesta sale con lo que hay y, debajo, hasta dos preguntas
con botones. Un clic suma el dato a la ficha (mejora 3) y responde de nuevo.

Solo se pregunta antes de responder cuando no hay con qué buscar: si la
consulta no trae mecanismo ni región y las dos guardas se abstienen, en lugar
del mensaje de abstención se preguntan el mecanismo y la región, con botones.

**Qué se pregunta** (`rag/preguntas.py`, nuevo: reglas sobre la ficha, en el
orden de la revisión primaria):

| Falta | Cuándo se pregunta | Opciones |
|---|---|---|
| Ventilación (B) | región tórax | Normal · Hipoventilación · Dificultad respiratoria |
| Estado hemodinámico (C) | herida penetrante, o tórax, abdomen, pelvis o cuello, o sangrado | Estable · Hipotenso o en choque · No se sabe |
| Glasgow (D) | cráneo, TCE, caída o tránsito | 15 · 13–14 · 9–12 · 8 o menos |
| Superficie y profundidad (E) | quemadura | % de superficie · grado |
| Grupo | no se dijo la edad | Niño · Adulto · Adulto mayor · Embarazada |

- Como máximo dos por respuesta: las primeras según xABCDE.
- Si queda lugar, los hallazgos que diferencian los casos recuperados. Ya los
  calcula `build_prompt.para_afinar`, que hoy solo usa la respuesta
  determinista (`ensamblar`).
- Nunca se pregunta lo que la ficha ya tiene ni lo que la consulta implica
  (`hallazgos.obvios`).
- Superficie y profundidad de la quemadura ayudan a los manuales (Lund y
  Browder y la tabla del quemado están transcritos), no a los casos: el
  léxico no los reconoce (pendiente 2).
- **Los textos de las preguntas van en `preguntas.py`, no en
  `build_prompt.py` ni en `hallazgos.py`:** todo lo escrito en esos dos
  archivos cuenta como vocabulario permitido para el verificador
  (`verificar._VOCABULARIO_CODIGO`).
- En la app, botones (`st.pills`, Streamlit 1.66) solo bajo la última
  respuesta. Un clic agrega el mensaje del usuario ("Hipotenso o en choque")
  y sigue como un seguimiento.

**Cómo se mide.** En las 25 consultas de trauma: cuántas preguntas salen y
cuáles. Para cada pregunta, responderla con cada opción y ver si cambian los
casos o los pasajes: si no cambia nada, la pregunta no aporta. La pertinencia
la juzga el médico, con una columna más en la planilla: "¿es la pregunta que
usted haría?".

**Hecho cuando:** ninguna pregunta por algo que la consulta ya dice, dos como
máximo por respuesta y, como objetivo, que al menos la mitad de las respuestas
a las preguntas cambien lo que se recupera.

**Resultado (24/10, RTX 4060).** `rag/preguntas.py`,
`scripts/probar_preguntas.py` (solo recupera, con `generate.recuperar`, sin
generar) y `reports/probar_preguntas.csv`, con la columna vacía
`medico_la_haria`. **`PREGUNTAS=1` por defecto**, con `CONVERSACION=ficha`.

| En las 25 de trauma | |
|---|---|
| preguntas | 46 en 24 consultas (2 como máximo) |
| por lo que la consulta ya dice | 0 |
| qué se pregunta | grupo 19 · C 11 · D 10 · B 2 · E-superficie 1 · lo que separa los casos 3 |
| respuestas que cambian lo recuperado | 148 de 153 (97 %); las 46 preguntas tienen alguna opción que cambia |
| cuánto cambia | 2,2 casos nuevos de 5 en promedio (D 3,6 · B 3,0 · C 3,0 · grupo 1,3) y 2,4 secciones nuevas de los manuales |

- **El 97 % es una medida floja:** casi cualquier texto que se agregue a la
  búsqueda cambia algún caso del top 5. Dice que las preguntas mueven la
  recuperación, no que la mejoren. Si es la pregunta que haría, lo dice el
  médico con la columna `medico_la_haria`.
- **Sin pregunta:** ort-04 ("qemadura por gasolina…"). Con la falta de
  ortografía no se reconoce la quemadura.
- **Antes de responder** se pregunta el mecanismo y la región solo si la
  consulta nombra una lesión de trauma ("paciente con trauma"). En la primera
  medición, sin esa condición, 17 de las 20 fuera de dominio preguntaban el
  mecanismo ("cuál es la capital de Colombia") y la región no cambiaba nada.
  Ahora, 0 de 20. **Desvío del plan:** las preguntas van debajo del mensaje de
  abstención, no en su lugar, para que se vea por qué no respondió.
- **Las respuestas se suman a la ficha:**
  - el Glasgow acepta rangos ("Glasgow 13-14", "8 o menos");
  - "No se sabe" no responde de nuevo, solo deja de preguntarlo
    (`Ficha.sin_dato`);
  - un clic es siempre un seguimiento (`tipo="seguimiento"`).
- **La C no se pregunta:**
  - por una lesión ocular aislada;
  - si la consulta ya dice el estado, la TA o "choque" (pero "choque de moto"
    no cuenta).
- **La app:** `st.pills` bajo la última respuesta. Probado con `AppTest` de
  Streamlit:
  1. "HPAF en abdomen" pregunta la C y el grupo;
  2. el clic en "Hipotenso o en choque" responde de nuevo con "Con hipotensión o
     choque e inestable: entran 3 casos nuevos…";
  3. "Es otro paciente" lo rehace como paciente nuevo.
- `preparar_demo.py`: "Todo listo".

---

## Fase 4 — Mejora 2: que ordene la respuesta como el ATLS (hecha el 24/10/2026; decisión 2: dentro de cada apartado; faltan la tabla del médico y la decisión 3)

**Idea.** El ATLS 11 organiza la revisión primaria como xABCDE: hemorragia
exanguinante, vía aérea, respiración, circulación, déficit neurológico y
exposición. Sus capítulos 3 a 8 siguen ese orden, uno por letra. El manual
del HUV tiene el capítulo 2, "ABCDE y triage". Una respuesta ordenada así se
lee como piensa el médico en urgencias.

**Se respeta la decisión 3 de los libros:** casos y manuales siguen en
apartados separados y una oración nunca cita las dos fuentes. El orden va
**dentro de cada apartado** (decisión 2 de abajo):

```
### Casos similares del registro
**B · Respiración.** En casos similares se documentó hipoventilación y se colocó tubo de tórax [caso: …].
**C · Circulación.** …
**Después de la revisión primaria.** …

### Manuales de trauma (ATLS y HUV)
**B · Respiración.** El ATLS recomienda … [guía: ATLS-05-…].
**C · Circulación.** El manual del HUV recomienda … [guía: HUV-06-…].
```

**Qué decide el código** (`rag/prioridades.py`, nuevo):
- **La letra de cada término del léxico.** P. ej.: intubación → A;
  hipoventilación, neumotórax, hemotórax y tubo de tórax → B; hipotensión o
  choque, hemoperitoneo, FAST y transfusión → C; Glasgow, pérdida de
  conciencia, anisocoria y déficit neurológico → D. Lo que no es de la
  revisión primaria (fractura, sutura, antibiótico…) va en "Después de la
  revisión primaria". Casos dudosos para el médico: sangrado activo y
  compromiso vascular (¿x o C?).
- **La letra de cada pasaje.** Por capítulo: ATLS 3 → x, 4 → A, 5 → B, 6 → C,
  7 → D, 8 → E; HUV 5 (vía aérea) → A, 6 (shock) → C, 3 (hipotermia) → E. En
  los demás capítulos del HUV, por los términos del léxico del pasaje (está
  en español). Los demás del ATLS van en "Después". Se calcula al buscar,
  con `capitulo_num`: no hay que reindexar. Medir cuántos pasajes quedan con
  letra; si son pocos del ATLS, una lista de términos en inglés.
- **Qué letras aparecen:** las que tienen algún elemento o pasaje, más las
  que implica la ficha (hipotenso → C).

**Qué escribe el modelo.** La misma llamada de hoy, con otro formato. El
prompt le pasa los elementos y los pasajes agrupados por letra, y la salida
estructurada pasa de `{"sugerencia": "…"}` a `{"casos": [{"letra", "texto"}],
"manuales": [{"letra", "texto"}]}`, con `letra` restringida (`enum`) a las que
eligió el código. Los rótulos ("**B · Respiración.**") los pone el código.

**Cómo se verifica.**
- Cada texto, con `verificar_por_fuente`, como hoy.
- Regla nueva: lo que nombra un texto es de su letra (la hipoventilación no
  va en C) y la guía citada es de esa letra o de "Después".
- Respaldo por letra: si una no pasa, esa letra se arma desde su fuente, como
  hoy se hace por fuente. Con `PROSA_SIEMPRE=True` se muestra igual
  (decisión 3).
- Solo para la presentación de un paciente (ficha con mecanismo o
  hallazgos). Las preguntas de procedimiento ("¿cuándo poner un tubo de
  tórax?", las `proc-*`) siguen con la prosa de hoy.

**Cómo se mide.** `comparar_modelos.py` con un modo nuevo, `--modos
prosa,atls`, en las 25 de trauma y las 16 de procedimiento: % verificado, JSON
inválido, letras coherentes, tokens de salida y mediana de tiempo. Al médico:
"¿el orden de prioridades es el correcto?".

**Hecho cuando:** 0 JSON inválidos, el % verificado no baja respecto de la
línea base, mediana de 10 s o menos y la tabla de letras revisada por el
médico.

**Resultado (24/10, RTX 4060).** Decisión 2 del usuario: dentro de cada
apartado. **`ORDEN_ATLS=1` por defecto** desde la 3.ª medición (abajo). La
decisión 3 queda para el usuario: `PROSA_SIEMPRE_ATLS` toma el valor de
`PROSA_SIEMPRE` (encendido para la demo, 18/10).

- **`rag/prioridades.py`:**
  - una letra para cada término del léxico, con las del plan; un test cuida
    que ninguno quede sin letra;
  - 7 dudosos para el médico (`DUDOSOS`): sangrado activo (x o C), compromiso
    vascular, toracotomía, convulsión, irritación peritoneal, laparotomía y
    angiografía;
  - la letra de cada pasaje sale del capítulo (ATLS 11, del 3 al 8; HUV 3, 5,
    6 y, al mirar los títulos, 20 vascular → x, 21 ultrasonido → C, 22
    toracostomía → B; ATLS 10, 2, 3 y 15) y, si no, en español, de sus
    términos;
  - en un pasaje, "hemorragia" es C, no x: con x, 74 secciones del ATLS 10
    quedaban en "hemorragia exanguinante".
- **Pasajes con letra**, entre los recuperados en las 41 consultas: HUV 59 %,
  ATLS 10 60 %, ATLS 11 44 %. El resto va en "Después". No se agregó la lista
  de términos en inglés.
- **Formato:**
  - una clave por letra pedida, obligatoria, con un tope de 400 caracteres
    (`maxLength`, que Ollama cumple). Con la lista de `{"letra", "texto"}` y
    `enum` del plan, el 4B repetía la misma letra hasta cortarse (2 de 3
    consultas de prueba);
  - sin el tope, abr-04, con 9 pasajes en "Después", escribía una oración por
    pasaje hasta agotar los tokens;
  - el código quita la oración que el tope dejó a medias y pone los rótulos;
  - `MAX_TOKENS_ATLS` es 1000.
- **Verificación por letra:** `verificar_por_fuente` de siempre, más la regla
  de la letra.
  - **Casos:** un texto nombra su letra y lo de "Después" (lo que se hizo:
    "con hemoperitoneo se realizó laparotomía"), no otra letra primaria. La
    regla estricta del plan rechazaba esa oración; además, "laparotomía" es
    "cirugía" para el léxico.
  - **Manuales:** la oración va en la letra primaria de los pasajes que cita
    (la letra la decide el código); si cita pasajes de dos letras primarias,
    no pasa. Antes, 9 de 16 rechazos por letra eran una oración en "Después"
    que citaba un pasaje de C.
- **Solo con un paciente** (`Ficha.es_paciente`). Las preguntas de
  procedimiento ("¿cuándo está indicado colocar un tubo de tórax?", "técnica…",
  "manejo del…") no son un paciente aunque nombren una lesión: siguen con la
  prosa de siempre y no llevan preguntas. "Ácido tranexámico" ya no se lee
  como quemadura química.
- **La app:** los rótulos en negrita dentro de cada apartado y, arriba, qué
  letras van armadas desde su fuente.

| 25 presentaciones (`comparar_modelos.py --modos prosa,atls --consultas guias`) | por letras, 1.ª | 2.ª (letra por lo citado) | 3.ª (tope de 400 caracteres) | prosa de siempre |
|---|---|---|---|---|
| JSON inválido | 0 | 1 (abr-04, cortado) | **0** | — |
| verificadas enteras | 5 | 7 | 5 | 7 |
| fuentes en prosa (de 45) | 18 | 19 | 18 | 20 |
| · casos (de 25) | | | 14 | 16 |
| · manuales (de 20) | | | 4 | 4 |
| letras verificadas | 55 de 97 | 53 de 89 | 51 de 92 | — |
| de la letra equivocada | 10 | 6 | 2 | — |
| mediana | 9,4 s | 9,7 s | **9,1 s** (peor 12,8 s) | 8,7 s |
| tokens de salida | 359 | 363 | 327 (máx. 580) | 305 |

`reports/comparar_modelos_atls_orden.csv`, `…orden2.csv`, `…orden3.csv`.

**Por qué se encendió:**
- **Se cumplen** el JSON válido (0 de 25) y la mediana de 10 s o menos.
- **Lo verificado queda un poco por debajo:** 18 fuentes de 45 contra 20; las
  tres corridas dieron 18, 19 y 18, y la prosa de siempre se midió una vez.
  Estrictamente, "no baja" no se cumple; la diferencia es del orden de lo que
  varía entre corridas.
- **Falta la tabla del médico** (letras de los términos y los capítulos, con
  los 7 dudosos), como en la fase 3.

**La decisión 3, con datos (para el usuario):**
- **Con `PROSA_SIEMPRE_ATLS=0`,** la letra que no pasa va armada desde su
  fuente:
  - en los casos, "En casos similares consta hemotórax [caso: …] y tubo de
    tórax [caso: …]", que se lee mecánico pero tiene sus citas;
  - en los manuales, "Pasajes de esta letra: [guía: …]".
- **Los manuales pasan enteros solo en 4 de 20**, igual que la prosa de
  siempre. Con 0, en 16 de 20 alguna letra de los manuales queda como lista y
  la recomendación no se ve. Con `PROSA_SIEMPRE`, se ve aunque no pase, como
  hoy.
- **Por eso no se siguió lo recomendado** (0) sin preguntar: cambiaría lo que el
  usuario eligió para la demo. `PROSA_SIEMPRE_ATLS` toma el valor de
  `PROSA_SIEMPRE`.
- **Lo que hace fallar a los manuales** es lo de siempre: el manual
  equivocado, palabras fuera de la guía citada y el tono. El orden por letras
  no lo empeora ni lo arregla.
- **Lo que queda en los rechazos** es lo de siempre: palabras fuera de la guía
  citada, el manual equivocado (la "11.ª edición" con una cita del HUV) y el
  tono, más algunos de letra que son errores de verdad (hemotórax y tubo de
  tórax en "Después").
- **La prosa de siempre también varía entre corridas:** 9 de 41 en esta,
  contra 15 de 41 en la línea base.

**Riesgo principal.** Un formato más rico es más difícil para el 4B:
`MAX_TOKENS_PROSA` ya se subió a 600 porque se cortaba el JSON. **Versión
liviana** (medio día), si no da o si no hay tiempo: la prosa de hoy, con los
elementos y los pasajes ordenados por letra en el prompt y la instrucción de
seguir ese orden. Sin rótulos ni regla nueva.

---

## Fase 5 — Demo (hecha el 24/10/2026)

- `preparar_demo.py`: un escenario de tres turnos, "HPAF en abdomen" → botón
  "Hipotenso o en choque" → "¿qué dice el ATLS del FAST?". Debe terminar en
  "Todo listo".
- Guion en `SETUP.md` §6: qué mostrar de cada mejora y la frase del
  principio.
- `CONTEXTO_SESION.md` al día y los tests.

**Resultado.**
- **`preparar_demo.py`** tiene una sección 4 nueva:
  - **el guion de `SETUP.md` §6 escrito en un solo chat:** el tipo que da la
    ficha a cada mensaje, solo con las reglas y sin el modelo;
  - **los tres turnos por `generate.responder`**, como los pasa la app, con
    estas comprobaciones:
    1. la ficha (arma de fuego, abdomen) y la pregunta C con "Hipotenso o en
       choque";
    2. la hipotensión sumada, el estado inestable, la línea de qué cambió y que
       la C no se vuelva a preguntar;
    3. que la búsqueda lleve al paciente y la pregunta, que respondan los
       manuales y que el FAST no entre en la ficha.

  Termina en "Todo listo": los turnos tardan 10,6, 8,5 y 11,4 s. En el tercero
  responde la 10.ª edición del ATLS.
- **`SETUP.md`:**
  - §5: lo nuevo en pantalla;
  - §6: la frase "más sofisticado no es más libre", el paso 6 (la
    conversación), las letras y las preguntas en el paso 2, la abreviatura que
    ahora llega a los manuales (con la advertencia de abr-01: los pasajes no son
    de cráneo) y "Paciente nuevo" antes de la pregunta de procedimiento;
  - las variables nuevas y `PRESUPUESTO_GUIAS` 1800.
- **Lo que destapó el guion: en un solo chat, cada consulta se sumaba al
  paciente anterior,** también la de la abstención ("manejo de cetoacidosis
  diabética" quedaba con el paciente de un TCE). Las reglas de la fase 2 solo
  reconocían un paciente nuevo si cambiaba el mecanismo o si el mensaje decía
  quién era ("mujer de 75 años…"). Se agregó a `ficha.tipo_de_mensaje`:
  - un hecho que nombra un mecanismo o una lesión es otro paciente, salvo que
    continúe ("le hicieron…", "tiene…", "el paciente…", "además", "también");
  - una consulta general sin señal de trauma empieza de cero.

  En las 74 conversaciones de evaluación nada cambia: 47 de 47 y 26 de 27.
  Una pregunta ("¿cuándo activar el protocolo de transfusión masiva?") se sigue
  sumando al paciente anterior, a propósito: en la guardia es útil. Por eso el
  guion pide "Paciente nuevo" antes del paso 7a.

---

## Decisiones para el usuario

1. **¿Preguntar antes o después de responder?** Recomendado: después, salvo
   cuando no hay con qué buscar.
2. **¿El orden xABCDE dentro de cada apartado, o un solo bloque con las dos
   fuentes?** Recomendado: dentro de cada apartado. Respeta la decisión 3 de
   los libros y el verificador ya trabaja por fuente. Un solo bloque se lee
   mejor, pero mezcla las fuentes en la pantalla.
3. **`PROSA_SIEMPRE` con el formato nuevo.** Más prosa es más oraciones sin
   verificar en pantalla. Recomendado: medir primero y, si el % verificado
   baja, mostrar armada desde su fuente la letra que no pasa.
4. **Quién valida las tablas** (letra por término y por capítulo, preguntas y
   cuándo se hacen): el médico, en la fase 7 de los libros.

Fuera de este plan: cambiar de modelo (pendiente 9), un chat libre de
medicina y quitar el verificador.
