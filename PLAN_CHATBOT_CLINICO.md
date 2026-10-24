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

## Fase 2 — Mejora 3: que siga la conversación sobre el mismo paciente (3 días)

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

---

## Fase 3 — Mejora 1: que pregunte lo que falta (2 días)

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

---

## Fase 4 — Mejora 2: que ordene la respuesta como el ATLS (3 a 4 días)

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

**Riesgo principal.** Un formato más rico es más difícil para el 4B:
`MAX_TOKENS_PROSA` ya se subió a 600 porque se cortaba el JSON. **Versión
liviana** (medio día), si no da o si no hay tiempo: la prosa de hoy, con los
elementos y los pasajes ordenados por letra en el prompt y la instrucción de
seguir ese orden. Sin rótulos ni regla nueva.

---

## Fase 5 — Demo (1 día)

- `preparar_demo.py`: un escenario de tres turnos, "HPAF en abdomen" → botón
  "Hipotenso o en choque" → "¿qué dice el ATLS del FAST?". Debe terminar en
  "Todo listo".
- Guion en `SETUP.md` §6: qué mostrar de cada mejora y la frase del
  principio.
- `CONTEXTO_SESION.md` al día y los tests.

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
