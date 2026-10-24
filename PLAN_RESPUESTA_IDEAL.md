# Plan — respuestas como la ideal, con el modelo de 4B

**Estado (24/10/2026): hecho y encendido** (`TARJETAS=1`). Los números están
en "Resultado", antes de "Riesgos".

Pedido (24/10/2026): que el chatbot responda parecido a
`respuesta_ideal_arrollado.md`, escrita por Claude Opus 5.5 con esfuerzo alto
para la consulta "hombre arrollado por vehiculo automotor". Hay dos
condiciones: que el tiempo no suba demasiado y que la recuperación no empeore.

## Qué hace buena a la respuesta ideal, y qué responde hoy el chatbot

Para comparar se usa voc-03, "peatón arrollado por vehículo automotor"
(`reports/comparar_modelos_atls_orden3.csv`, 24/10, modo por letras).

| Rasgo de la ideal | Hoy |
|---|---|
| **Lee el mecanismo:** lo trata como de alta energía y separa el atropello del arrollamiento (aplastamiento, pelvis, desguantamiento). | La ficha dice "accidente de tránsito" y pierde que es un peatón. |
| **Dice qué hacer ahora, aunque la consulta no traiga datos:** una acción por letra, con su umbral (Glasgow ≤ 8, menos de 3 horas). | La búsqueda trae pasajes de cinemática, que es lo que se parece a la consulta. El 4B los reparte en letras que no les tocan: el patrón de lesiones del peatón queda en "B · Respiración" y los criterios de triaje en "D". Ninguna letra trae una acción. |
| **Abre ramas:** estable → TC; inestable con eFAST positivo → cirugía; fuente pélvica → empaquetamiento o angioembolización. | No hay ramas. |
| **Da el patrón de lesiones esperado.** | Sí lo da, pero en la letra equivocada y atribuido al manual equivocado (motivos de la verificación). |
| **Usa el registro como epidemiología:** "los hallazgos más frecuentes fueron…". | Da anécdotas: "En uno de los casos se observó pérdida de conciencia". |
| **Pide lo que falta y cambiaría el manejo:** incluye una pregunta propia del mecanismo (¿impacto o las ruedas por encima?) y el tiempo desde el evento. | Dos botones: D · Glasgow y grupo de edad. |
| **Se lee de un vistazo:** títulos y viñetas con la palabra clave en negrita. | Párrafos por letra. |

**Lo que no hay que copiar:**
- **Las citas vagas:** "Manual de procedimientos, sección…", "TR-XXXXX". La cita
  por oración que usa hoy el chatbot es más estricta.
- **La memoria del modelo como fuente.** En este caso todo lo que dice la ideal
  está en los manuales (abajo), pero no se puede contar con eso: con un 4B, la
  memoria no sirve de fuente, y cada viñeta va con el pasaje que la respalda.
- **El imperativo** ("Dar ácido tranexámico…"): el verificador pide orientar,
  no ordenar.

**Lo que sí está en los manuales.** Busqué 20 elementos de la ideal en
`guias/secciones.jsonl` y aparecen los 20, casi todos también en español (HUV
o ATLS 10). La primera búsqueda dio 19, porque buscaba "TC" y el HUV escribe
"TAC"; con eso corregido, también está la TAC corporal total. A veces los
manuales son más precisos que la ideal:
- **Ácido tranexámico:** "Si el paciente está en las primeras 3 horas del
  trauma y tiene un shock hemorrágico… 1 g inicial durante 10 minutos y un 1 g
  en infusión por 8 horas" (HUV-06-004).
- **Vía aérea definitiva:** "Glasgow de 8 o menos" (HUV-02-003).
- **Sospecha de politrauma:** "Accidente auto vs peatón" está entre los
  hallazgos en la escena (HUV-04-011).
- **Patrón del peatón:** lesión craneal, lesiones viscerales abdominales y
  fracturas de las extremidades inferiores o de la pelvis (ATLS10-01-027, tabla
  1-1).
- **Compresión pélvica:** con sábanas o con un dispositivo a nivel de los
  trocánteres (HUV-06-003).
- **Fractura abierta:** antibiótico "within 1 hour of arrival" (ATLS-10-014).
  La 10.ª edición solo dice que pasadas tres horas sube el riesgo
  (ATLS10-08-027).
- **Aplastamiento:** rabdomiólisis (HUV-04-009) y lesiones de desguantamiento
  que hay que sospechar por el mecanismo (ATLS10-08-038).
- **TAC corporal total:** en el politraumatizado estable o respondedor
  transitorio (HUV-06-007), con una lista de indicaciones (HUV-04-013).
- **Las tres fases del atropello:** el parachoques lesiona la extremidad
  inferior; el capó y el parabrisas, la cabeza y el torso; el suelo, la cabeza,
  la columna y las extremidades (ATLS10-23-010 y 011).

---

## El diagnóstico

**Casi toda la respuesta ideal sería igual para cualquier peatón:** la
revisión primaria, los estudios, el patrón esperado y las consideraciones. Lo
propio del paciente es poco: lo que dice la ficha, los casos parecidos y lo
que falta.

Hoy el 4B redacta todo en cada consulta, a partir de lo que trae la búsqueda.
Lo fijo es justo lo que peor le sale: los manuales pasan enteros la
verificación en 4 de 20 respuestas.

La propuesta es **repartir el trabajo**:

| Parte | Quién la escribe | Cuándo | Costo en la consulta |
|---|---|---|---|
| Lectura del mecanismo, xABCDE, estudios, patrón y consideraciones | una **tarjeta por mecanismo**, redactada una vez desde el texto de los pasajes; la verifica el código y la valida el médico | fuera de línea | 0 s |
| Lo que dice la ficha (hipotenso, Glasgow…) | reglas: la tarjeta se ajusta a la ficha | en la consulta | ~0 s |
| El registro: cuántos casos hay y qué tuvieron | código: conteos sobre los 8.198 casos | fuera de línea | 0 s |
| Los casos parecidos | el 4B, como hoy, pero solo los casos | en la consulta | menos que hoy |
| Lo que falta | la ficha y las preguntas de la fase 3 | en la consulta | 0 s |

La frase de la reunión sigue valiendo: **más sofisticado no es más libre.** El
modelo grande escribe una vez y alguien lo revisa; el chico escribe solo sobre
lo que recuperó.

**El tiempo.** En la medición del 24/10 (41 consultas, por letras), la mediana
es 9,9 s:
- leer el prompt (~4.000 tokens) toma 1,1 s;
- escribir (~320 tokens) toma 4,6 s.

Lo caro es escribir: cada 100 tokens de salida cuestan ~1,4 s, y leer 1.000
tokens más cuesta ~0,3 s. Con la tarjeta, el 4B deja de escribir las letras
de los manuales (hasta 3, de 400 caracteres cada una) y deja de leer sus
pasajes. Se espera bajar unos 2 s, pero eso se mide; no se promete.

**La recuperación.**
- La tarjeta se elige por la ficha, con reglas, no por la búsqueda.
- La búsqueda, las guardas y el presupuesto no se tocan:
  `calibrar_guias.py` y `calibrar_umbral.py` tienen que dar lo mismo que en la
  línea base.
- Los pasajes recuperados siguen en el desplegable.
- El único cambio a la búsqueda es opcional y se mide aparte (fase 3).

---

## Orden de trabajo: 0 → 1 → 2 → 3 → 4 → 5

### Fase 0 — Línea base de la brecha (medio día)

- **Lista de comprobación de la ideal**, para cada una de las 25
  presentaciones de trauma:
  - lectura del mecanismo;
  - acción por letra (cuántas de 6);
  - ramas;
  - patrón;
  - registro agregado;
  - lo que falta.

  El código cuenta lo que pueda (por ejemplo, si hay una acción por letra con
  cita) y lo demás se cuenta a mano.
- **El tiempo y la recuperación ya tienen línea base** (24/10): 9,9 s de
  mediana; R@ppto 0,280 en las anotadas y 0,819 en las sintéticas; 6 de 25
  de trauma sin manuales; 0 de 20 fuera de dominio que pasen.

### Fase 1 — Tarjetas por mecanismo (dos a tres días; el núcleo)

`tarjetas/tarjetas.yaml` tiene una tarjeta por mecanismo (en git; `guias/`
no lo está, por los derechos de autor de los libros), con una parte común (la
revisión primaria) y lo propio de cada mecanismo:

```yaml
peaton:
  nombre: peatón atropellado
  lectura:
    - texto: "Accidente auto vs peatón: el HUV lo pone entre los hallazgos en la escena que hacen sospechar politrauma"
      cita: [HUV-04-011]
  letras:
    C:
      - texto: "Compresión pélvica externa con sábanas o un dispositivo a nivel de los trocánteres si se sospecha sangrado pélvico"
        cita: [HUV-06-003]
      - texto: "Ácido tranexámico en las primeras 3 horas si hay choque hemorrágico: 1 g en 10 minutos y 1 g en 8 horas"
        cita: [HUV-06-004]
        si: no_estable        # se omite si la ficha dice estable
  despues: [...]              # estudios y ramas (estable / inestable)
  patron: [...]
  aplastamiento: [...]        # si las ruedas pasaron por encima
  preguntas: [impacto_o_arrollamiento]
```

- **Quién las redacta:** Claude, desde el texto de cada pasaje y no desde la
  memoria, en español. Se prefieren el HUV y el ATLS 10; el ATLS 11 se usa solo
  para lo que no está en los otros.
- **`tests/test_tarjetas.py`:**
  - cada viñeta pasa el verificador actual contra su pasaje: palabras, cifras,
    edición y tono;
  - cada cita existe en `secciones.jsonl`. Si se reindexa y cambia un
    identificador, el test avisa.
- **`rag/tarjetas.py`:** `elegir(ficha)` y `armar(tarjeta, ficha)`. La tarjeta
  se ajusta a la ficha con reglas:
  - las viñetas con `si` se filtran con la ficha: si el paciente está
    inestable, queda la rama del inestable y la C se marca;
  - la letra que tiene datos en la ficha los muestra: "C · Circulación — PA
    85/55".
- **En pantalla:**
  - con tarjeta, el apartado de los manuales es la tarjeta y el 4B escribe solo
    las letras de los casos (se respeta la decisión 2 del plan anterior: dentro
    de cada apartado);
  - sin tarjeta (un mecanismo que no la tiene o una consulta de procedimiento),
    todo queda como hoy.
- **Orden de las tarjetas:**
  1. peatón (el caso de la ideal);
  2. caída (2.103 casos);
  3. ocupante de carro o moto (el resto de los 1.629 de tránsito);
  4. arma de fuego (1.180);
  5. cortopunzante (770);
  6. después: contundente, quemadura y mordedura.

  Las cinco primeras cubren el 87 % de los casos con mecanismo conocido.

**Hecho cuando:**
- todas las viñetas pasan el verificador;
- "hombre arrollado por vehículo automotor" muestra la lectura, una acción por
  letra, las ramas y el patrón, todo citado;
- la mediana de tiempo no sube.

### Fase 2 — El registro como epidemiología (medio día)

- **`scripts/patrones_registro.py`** genera `tarjetas/patrones_registro.json`,
  que tiene solo conteos y ningún texto de casos. Para cada mecanismo cuenta:
  - las regiones;
  - los hallazgos y el manejo, con el léxico de `hallazgos.py`;
  - el desenlace, entre los casos que lo tienen.

  El peatón se reconoce por el texto: atropell, arrollad o peatón.
- **Medido el 24/10, peatón: 324 casos.**
  - Regiones: miembro inferior 48 %, cráneo 36 %, miembro superior 32 %,
    cara 27 %, tórax 14 %, abdomen 13 % y pelvis 12 %.
  - Fractura: 25 %.
  - Desenlace, en los 219 que lo tienen: 46 hospitalizados y 14 a cirugía.

  Es lo que la ideal dice de memoria, pero con números del registro.
- **En la consulta:**
  - una línea arriba de los casos, escrita por el código;
  - debajo, el conteo de los casos recuperados que ya existe ("fractura en 3
    de 5 [caso: …]").
- **Aviso:** las regiones y el desenlace los pone con reglas el pipeline de
  anonimización. El 35 % de los casos aptos no tiene desenlace (pendiente 7).

### Fase 3 — La ficha lee el mecanismo (un día)

- **El peatón como mecanismo propio** en `ficha.MECANISMOS`: atropellado,
  arrollado, peatón y "auto vs peatón". Hoy cae en accidente de tránsito.
- **La variante:**
  - "le pasó por encima" o "aplastado" se leen como aplastamiento;
  - "arrollado" es ambiguo, así que la tarjeta pregunta: "¿Fue impacto o le
    pasaron las ruedas por encima?" (Impacto / Por encima / No se sabe).

  La pregunta entra en la cola de la fase 3 del plan anterior (máximo 2).
- **"Datos que faltan" en una línea del código:** grupo de edad, signos
  vitales, Glasgow y tiempo desde el evento (lo necesita el ácido
  tranexámico). Se listan solo los que la tarjeta usa.
- **Opcional, medido aparte:** agregar "arrollado" → "atropello" en
  `corpus.SINONIMOS_CONSULTA`, para el BM25 de los casos (el corpus escribe
  "atropellado", nota de voc-03). Se queda solo si la evaluación de los casos
  (`eval/consultas.yaml`) no baja.

### Fase 4 — Formato (medio día)

- **El orden, con tarjeta:**
  1. lectura del mecanismo;
  2. revisión primaria;
  3. estudios y ramas ("Después de la revisión primaria");
  4. patrón esperado;
  5. el registro y los casos;
  6. lo que falta.
- **Viñetas cortas** con la palabra clave en negrita.
- **Letras de los casos más cortas:** `MAX_CARACTERES_LETRA` pasa de 400 a
  300, solo si la medición muestra que no se pierden citas.

### Fase 5 — Medición y demo (medio día)

- **Las 25 presentaciones** con `comparar_modelos.py --modos atls`: tiempo,
  JSON, lo verificado en los casos y la lista de la fase 0.
- **`calibrar_guias.py` y `calibrar_umbral.py`:** iguales a la línea base.
- **`preparar_demo.py`:** "hombre arrollado por vehículo automotor" → botón
  "Por encima" → la variante de aplastamiento.
- **`SETUP.md` §6:** el modelo grande escribió una vez y el médico lo revisa;
  el chico escribe sobre los casos.

Estimado: **unos cinco días.** Si hay menos tiempo: la fase 1 solo con el
peatón y la parte común, más la fase 2.

---

## Resultado (24/10/2026, RTX 4060)

Las fases 0 a 5 están hechas. `TARJETAS=1` por defecto; `TARJETAS=0` deja todo
como antes.

**Lo construido:**
- **Fase 1, tarjetas** (`tarjetas/tarjetas.yaml`, `rag/tarjetas.py`):
  - las 5 de la decisión 4, con 67 viñetas que citan 32 secciones, 16 del HUV y
    16 del ATLS 10, todas en español;
  - pasan las 67 el verificador contra su pasaje: 65 al primer intento, y las
    dos restantes usaban una palabra que no estaba en el pasaje ("exponer",
    "suelen");
  - `python -m rag.tarjetas` las revisa; `tests/test_tarjetas.py` suma 29
    tests y da 437 en total, que pasan con cada variable en 0 y en 1.
- **Fase 2, registro** (`scripts/patrones_registro.py`): peatón 324 casos,
  accidente de tránsito sin peatón 1.338, caída 2.102, arma de fuego 1.180 y
  cortopunzante 770.
- **Fase 3, la ficha lee el mecanismo:**
  - el peatón, y si fue un impacto o las ruedas le pasaron por encima, quedan
    como `Ficha.variantes`, sin cambiar el mecanismo, así que la búsqueda no
    cambia;
  - la pregunta del peatón y, con tarjeta, la de la C;
  - una pregunta concreta ("¿qué dice el ATLS del FAST?") no lleva tarjeta;
  - el sinónimo opcional "arrollado" no se hizo, para no tocar la búsqueda;
  - la ficha da lo mismo en las 74 conversaciones de evaluación: 47 de 47 y 26
    de 27;
  - preguntas en las 25 de trauma: siguen siendo 46, ninguna ya dicha; la C sale
    16 veces y la del peatón 1.
- **Fase 4, formato:** primero el protocolo (decisión 3), con viñetas, ramas con
  rótulo y los datos de la ficha en su letra. Debajo, "Pasajes que cita el
  protocolo" y "Otros fragmentos que encontró la búsqueda". No se tocó
  `MAX_CARACTERES_LETRA`.

**Medición** (41 consultas con `comparar_modelos.py --modos atls,tarjeta
--consultas guias`, en `reports/comparar_modelos_tarjetas.csv`; la parte de los
casos se repitió 3 veces más, en `…_repeticion.csv`):

| | Por letras | Con tarjeta |
|---|---|---|
| Presentaciones con tarjeta | — | 14 de 25 |
| Mediana, en esas 14 | 9,8 s | **5,7 s** |
| Peor, en esas 14 | 11,1 s | **7,4 s** |
| Mediana, las 25 presentaciones | 9,7 s | 6,7 s |
| Tokens de prompt y de salida (25) | 4.129 y 336 | 2.044 y 192 |
| Manuales verificados, en las 14 | 3 de 12 con pasajes | 14 de 14 (el protocolo) |
| Casos verificados, en las 14 (4 corridas) | 32 de 56 (57 %) | 25 de 56 (45 %) |
| Fuentes verificadas, en las 25 | 21 de 45 | 27 de 47 |
| JSON inválidos | 0 | 0 |

- **La recuperación no cambió.** Con `calibrar_guias.py`, en la fila de la app:
  R@ppto 0,280 y 0,819; 6 de 25 de trauma sin manuales; 0 de 16 de
  procedimiento; 0 de 20 fuera de dominio. Con `calibrar_umbral.py`: 25 de 25 de
  trauma responden (6 solo casos, 19 con las dos fuentes) y 0 de 20 fuera de
  dominio. Es la línea base.
- **`preparar_demo.py` termina en "Todo listo"**, con la sección 5 nueva: el
  peatón tarda 10,0 s y el botón "Le pasaron por encima", 6,9 s.
- **En la app**, "hombre arrollado por vehiculo automotor" tarda 7,4 s. Esa
  consulta no trae pasajes de los manuales (su guarda no pasa): justo ahí el
  protocolo llena el hueco.

**Lo que salió peor, y lo pendiente:**
- **La parte de los casos pasa algo menos con tarjeta:** 45 % contra 57 %. Sin
  el bloque de los manuales ordenado por letra, el 4B pone más seguido
  términos de la B en "Después" (8 rechazos de ese tipo contra 3). Con
  `PROSA_SIEMPRE_ATLS=1` se muestra igual. Queda por probar un ajuste del
  prompt, midiendo.
- **11 de las 25 presentaciones no tienen tarjeta:** TCE sin mecanismo (3),
  quemadura, mordedura, contundente, cuerpo extraño y lesiones sin mecanismo.
  Una tarjeta de TCE, elegida por la lesión y no por el mecanismo, cubriría 3
  (decisión para el usuario).
- **El largo:** sin datos, el peatón lleva 31 viñetas; con datos baja, porque
  se van las ramas que no corresponden. Si al médico le parece largo, se recorta
  la parte común.
- **El médico** revisa las 67 viñetas en la fase 7 (decisión 2: se muestran
  antes).

---

## Riesgos

- **Que parezca una plantilla:** la parte de los manuales sería igual para
  todos los peatones. Lo compensan:
  - las ramas que se ajustan a la ficha;
  - la letra con los datos del paciente;
  - el registro y los casos.

  Además, en la guardia, que el protocolo sea el mismo es una virtud.
- **Contenido clínico redactado por un modelo.** Por eso cada viñeta va con su
  pasaje, pasa el verificador y la valida el médico (decisión 2).
- **Las tarjetas envejecen con los manuales:** si se reindexa, el test avisa.
- **Lo que no tiene tarjeta** sigue como hoy, así que la mejora no empeora
  nada.

## Decisiones del usuario (24/10: las cuatro, como se recomendó)

1. **¿Tarjetas en lugar de la prosa del 4B sobre los manuales, cuando hay un
   paciente con un mecanismo que tiene tarjeta?** Recomendado: sí. **Decidido:
   tarjetas.**
   - Se gana una acción por letra, verificada, sin tiempo extra.
   - Se pierde que el texto de los manuales cambie con lo recuperado; eso
     queda en el desplegable.
2. **¿Se muestran antes de que el médico las revise?** Recomendado: sí, para
   la demo.
   - Cada viñeta lleva su cita y pasa el verificador, que es más de lo que pasa
     hoy la prosa del 4B con `PROSA_SIEMPRE`.
   - El médico las revisa en la fase 7, como las tablas de letras.

   **Decidido: sí, se muestran antes.**
3. **¿Qué va primero en una presentación, el protocolo o los casos?**
   Recomendado: el protocolo (la tarjeta), como en la ideal, porque es lo que se
   hace primero. Hoy van primero los casos. **Decidido: primero el
   protocolo.**
4. **¿Cuántas tarjetas para la reunión?** Recomendado: las cinco primeras
   (peatón, caída, ocupante, arma de fuego y cortopunzante), que cubren el 87 %
   de los casos con mecanismo conocido. **Decidido: las cinco.**

Fuera de este plan: cambiar de modelo (pendiente 9) y que el 4B escriba desde
su memoria.
