# Plan — indexar los libros de trauma en el RAG

Pedido de la reunión (semana del 30/09/2026): con solo casos el asistente tiene
sesgo (lo que se hizo en el HUV hasta 2018 no es necesariamente lo que se
debe hacer). Se agregan dos libros como segunda fuente.

## Las fuentes (`anonimizacion/data/`, fuera de git)

| | ATLS 11.ª ed. (Course Manual) | Paciente Traumatizado — Manual de Manejo Inicial |
|---|---|---|
| Archivo | `ATLS_11thEdition_CourseManual_ENG_17.pdf` (80 MB) | `LIBRO PACIENTE TRAUMATIZADO MANUAL DE MANEJO INICIAL.pdf` (185 MB) |
| Idioma | inglés | español (HUV / Univalle, Cali) |
| Páginas | 377 (16 casi sin texto) | 300 (26 casi sin texto) |
| Texto | capa de texto, 1,42 M caracteres (~350 k tokens) | capa de texto, 0,44 M caracteres (~110 k tokens) |
| Índice incrustado | no | no |
| Estructura | 3 secciones, ~25 capítulos; título de capítulo en fuente 34, encabezado "COURSE MANUAL \| Chapter N \| …" | 23 capítulos con página de inicio en el índice (pp. 6–7) |
| Ruido a quitar | encabezados y pies, ATLS-OE, índice alfabético (p. 356+) | bibliografía al final de cada capítulo, pies de foto ("Cortesía…"), palabras partidas en líneas sueltas |

No hace falta OCR. Estimado: ~900 fragmentos del ATLS y ~300 del manual del
HUV, frente a los 8.198 casos.

## Principio

Los libros **no reemplazan** la regla del proyecto, la amplían. **Casos y
libros pesan lo mismo** (decisión del 07/10): la respuesta se arma con las
dos fuentes, cada una con su espacio, y cualquiera de las dos basta para
responder. Hay dos tipos de afirmación y nunca se mezclan:

- **Casos:** "qué se hizo en casos parecidos". Lleva conteos y `[caso: ID]`.
- **Guía:** "qué recomienda el manual". Sin conteos, con `[guía: ID]` que
  apunta a libro, capítulo y página, y el fragmento visible en la app.

Por eso los libros van en **una colección aparte**, no mezclados con los
casos: los puntajes no son comparables, y `hallazgos.py`, la tabla y
`verificar.py` cuentan sobre casos ("2 de 5"). Si un fragmento de guía entrara
en la lista de casos, rompería todos los conteos.

---

## Decisiones (tomadas el 07/10/2026)

1. **Permiso de uso.** Los dos libros tienen derechos de autor (el ATLS es del
   American College of Surgeons). Uso local y académico, sin publicar el
   texto: el texto extraído y los índices quedan fuera de git.
2. **El ATLS queda en inglés.** bge-m3 es multilingüe: una consulta en español
   encuentra el pasaje en inglés por la vía densa (BM25 no). El modelo redacta
   en español y la app muestra el fragmento original. En las oraciones
   citadas al ATLS el verificador solo comprueba la cita y los números. No se
   traduce: con un modelo de 4B, una mala traducción se volvería "la guía
   dice".
3. **Mismo peso para casos y libros.** Consecuencias:
   - **Abstención por fuente:** se responde si pasa cualquiera de las dos
     guardas (casos o libros); se abstiene solo si no pasa ninguna. Con una
     sola fuente, la respuesta lo dice ("no hay casos similares en el
     registro" / "los manuales no tratan esta situación").
   - **Mismo espacio en el prompt:** presupuesto de tokens parecido para cada
     fuente (p. ej. 4 casos + 3 fragmentos, ~1.400 tokens cada lado), no el
     mismo número, porque un fragmento mide más que un caso.
   - **La prosa da las dos miradas:** lo que se hizo en casos similares y lo
     que recomiendan los manuales, sin que una quede como nota al pie de la
     otra. Si no coinciden, la respuesta lo dice; no elige un ganador.
   - **El respaldo determinista también tiene las dos secciones.**
   - **Las dos guardas se calibran con las mismas consultas fuera de
     dominio:** ninguna fuente puede responder sola a "cetoacidosis".

---

## Fases

### Fase 1 — Extracción y limpieza (hecha el 07/10/2026; falta la revisión a mano)

**Resultado:** `python scripts\extraer_guias.py --muestra 30` →
`guias/secciones.jsonl` con 926 secciones: ATLS 676 en 24 capítulos (1,03 M
caracteres) y HUV 250 en 23 capítulos (0,34 M). Descartado: bibliografía
(132 k + 65 k), flujogramas y rótulos sueltos (1,6 k + 7,5 k). Tests en
`tests/test_guias.py` (42).

Cómo quedó (`rag/guias.py`, reglas por libro en `guias/perfiles.yaml`):
- cada bloque recibe un rol por su fuente dominante: títulos (4 niveles por
  tamaño y color), cuerpo, pie ("Table 7-4:", "Tabla 1."), bibliografía (por
  el texto: en el ATLS viene en dos fuentes distintas) o material;
- orden de lectura: los bloques que cruzan el centro parten la página en
  franjas; en cada franja, columna izquierda y después derecha;
- el material va con su pie (tabla o figura); el que no tiene pie se agrupa
  en "Recuadro:" (NEXUS Chest, indicaciones de dren pleural, grados de
  trauma renal). **Los flujogramas se descartan** (se reconocen por "SI",
  "NO", "(+)", "¿…?"): sin las flechas no se sabe qué rama lleva a qué;
- tablas y recuadros van al final de su sección (si no, cortan oraciones);
- páginas impresas para citar (ATLS: PDF − 19), portadas con objetivos del
  ATLS fuera; capítulos no clínicos marcados (`clinico: false`): ATLS 2,
  16, 18, 19, 20 y HUV 1.

**Pendiente de la fase 1:** revisar a mano `guias/muestra_revision.md` (30
secciones) contra el PDF. Revisado por Claude: sin encabezados, sin guiones
blandos ni caracteres de Wingdings; las 11 "et al." que quedan son citas en
el texto, no bibliografía; los "hypo- and" son guiones suspendidos reales.

Plan original:
- Script `rag_pipeline/scripts/extraer_guias.py` con PyMuPDF (open source,
  local). Va a `requirements.txt`.
- Un perfil por libro (`rag_pipeline/guias/perfiles.yaml`): páginas a saltar
  (portada, colaboradores, índice alfabético, ATLS-OE) y patrones de
  encabezado y pie. Capítulos: el del HUV, desde su índice; el del ATLS, por
  el tamaño de fuente del título y el encabezado de página.
- Limpieza: unir palabras partidas y líneas sueltas, quitar la bibliografía y
  los pies de foto, conservar las tablas como texto (las del ATLS tienen
  umbrales: ABC score, PAS ≤ 90, FC ≥ 120).
- Salida: `rag_pipeline/guias/secciones.jsonl` (libro, capítulo, sección,
  páginas, texto). Fuera de git.
- **Verificación:** tests de la limpieza (sin encabezados, sin referencias,
  sin palabras cortadas) y revisión a mano de 30 secciones contra el PDF.

### Fase 2 — Fragmentación (en curso desde el 07/10/2026)

El usuario pidió comparar varias configuraciones y elegir con mediciones. Se
evalúa por SECCIÓN (cada fragmento hereda su `seccion_id`), así todas las
configuraciones se miden contra la misma anotación. Decide R@ppto: secciones
de grado 2 que caben en los 1.400 tokens del prompt. Verdad base: el conjunto
anotado decide; las ~100 preguntas sintéticas confirman la tendencia.
Pasos para la 1650: `CONTEXTO_SESION.md`, sección 00.

**Hecho:**
- `rag/fragmentar.py`: 7 configuraciones (`seccion`, `fijo-256`, `fijo-512`,
  `parrafos-350`, `parrafos-350-sin-ruta`, `oraciones-128` small-to-big,
  `semantico`). Con el tokenizador real, ninguna pasa su tope: entre 1.552
  (`fijo-512`) y 4.747 (`oraciones-128`) fragmentos.
- `eval/consultas_guias.yaml`: 51 consultas (25 de trauma, 16 de
  procedimiento, 10 fuera de dominio), con 156 secciones de grado 2 y 223 de
  grado 1, propuestas por Claude. **Falta que el usuario o el médico las
  valide.**
- `scripts/comparar_fragmentacion.py` (probado de punta a punta con
  `ENCODER=falso`) y `scripts/generar_preguntas_guias.py`.
- Tests: `test_fragmentar.py` y `test_comparar_fragmentacion.py`.
- **Ojo:** los IDs de sección (`ATLS-07-019`) salen del orden de extracción.
  Si se cambian las reglas de la fase 1 y se re-extrae, los IDs pueden
  correrse y la anotación queda desalineada. El script avisa si falta un ID
  anotado, pero no si apunta a otra sección.

**Falta (en la GTX 1650):**
1. `python scripts\generar_preguntas_guias.py` (Ollama abierto).
2. Cerrar Ollama y correr `python scripts\comparar_fragmentacion.py --rutas
   densa,bm25,rrf,rerank`. En el PC sin GPU, bge-m3 tarda decenas de
   minutos por configuración: no es viable allí.

Diseño original:
- Por sección, sin cruzar capítulos. De 250 a 450 tokens de bge-m3, con un
  párrafo de solapamiento. **Límite duro: 512**, el `max_length` del encoder;
  todo lo que pase de ahí se pierde sin aviso.
- Cada fragmento lleva su ruta de títulos al inicio ("ATLS › Chapter 7
  Disability › Cerebral contusion"): mejora la recuperación y le dice al
  modelo de qué trata.
- ID estable: `ATLS-07-012`, `HUV-11-004`. Metadatos: `fuente`, `idioma`,
  `capitulo`, `seccion`, `pagina_ini`, `pagina_fin`, `region_anatomica`
  (opcional, con el mismo léxico de los casos).
- **Verificación:** distribución de tamaños, 0 fragmentos > 512 tokens, cada
  página con texto cubierta.

### Fase 3 — Indexación
- Colección `guias` en el mismo Qdrant local (un solo cliente; ver trampa
  de Qdrant en `CONTEXTO_SESION.md` §6) y `indices/bm25_guias.pkl`.
- `python -m rag.index --fuente guias` (los casos no se reindexan).
  `meta.json` registra las dos fuentes.
- Config: `GUIAS_JSONL`, `COLECCION_GUIAS`, `TOP_K_GUIAS` (2–3), `GUIAS=1`
  para encender o apagar todo (ablación y demo).

### Fase 4 — Recuperación y evaluación
- `retrieve.buscar_guias(consulta, k)`: densa sobre las dos, BM25 solo sobre
  el manual del HUV (decisión 2a), fusión RRF; reranker cuando haya ≥ 7 GB.
  Umbral propio de relevancia: un fragmento flojo no entra.
- Abstención: `UMBRAL_GUIAS` para los libros, al lado de `UMBRAL_ABSTENCION`
  y `UMBRAL_RERANK` para los casos (decisión 3).
- Evaluación: agregar a `eval/consultas.yaml` el capítulo esperado de cada
  libro para las 25 consultas de trauma, más algunas consultas de
  procedimiento ("¿cuándo poner un tubo de tórax?"), que se responden solo con
  los libros. Medir Recall@3 por capítulo y comparar densa contra RRF.
  `calibrar_umbral.py` calibra las dos guardas: las 10 consultas fuera de
  dominio no deben pasar ninguna.

### Fase 5 — Generación y verificación
- El prompt tiene dos bloques del mismo tamaño, "Casos similares" y
  "Fragmentos de los manuales". La prosa separa "En casos similares…" de "El
  ATLS / el manual del HUV recomienda…" y dice si coinciden o no.
- Si solo pasó una fuente, el prompt lleva solo esa y la respuesta lo dice.
- `NUM_CTX`: 4 casos + 3 fragmentos (~2.800 tokens con las reglas) no caben
  en 3.072. Pasa a 4.096; medir VRAM y tiempo en la GTX 1650 y en el PC de 8
  GB. `TOP_K` de casos baja de 5 a 4: medir antes y después.
- `verificar.py`:
  - citas `[guía: ID]` existentes;
  - en una oración citada a una guía: los números deben aparecer en el
    fragmento y, si está en español, los términos del léxico también;
  - no se atribuye a un caso lo que solo dice la guía, ni al revés;
  - sin cambios en las reglas de casos.
- Respaldo determinista: si la prosa no pasa, se muestra lo de siempre más
  "Según las guías", con título y página de cada fragmento (sin parafrasear).

### Fase 6 — Aplicación
- Expander "Fragmentos de guía consultados", con libro, capítulo, página y
  texto.
- El aviso fijo nombra las fuentes: ATLS 11.ª ed. y el manual del HUV.
- `preparar_demo.py` y los botones de ejemplo al día.

### Fase 7 — Revisión médica
- Planilla con la prosa nueva (pendiente 1 de `CONTEXTO_SESION.md`) y una
  pregunta más en la rúbrica: ¿la guía citada corresponde a la situación?
- Cada error marcado se vuelve un test.

---

## Orden de trabajo propuesto

1. ~~Decisiones 1–3~~ (07/10).
2. Fases 1–2: el resultado se revisa a ojo antes de indexar nada.
3. Fases 3–4: se mide si la recuperación funciona **antes** de tocar el
   prompt.
4. Fases 5–6, y después 7 con el médico.
