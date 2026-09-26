# Pipeline de limpieza y anonimización — `trauma.csv`

Prepara el registro de trauma (27.157 registros) como corpus para un RAG clínico
en español: normaliza, marca calidad, anonimiza, enriquece con metadatos y
verifica el resultado.

## Instalación

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_lg   # solo para la capa C (opcional)
```

## Uso

```bash
python run_pipeline.py                  # pipeline completo (~25 s)
python -m src.ner_review --top 4000     # capa C: NER de revisión manual
python -m pytest tests/ -q              # 83 tests
```

## Estructura

```
config/
  gazetteers.yaml        listas curadas: instituciones, médicos, municipios, genéricos
  abreviaturas.yaml      diccionario de abreviaturas de urgencias/trauma
src/
  normalize.py           fase 1 — normalización estructural
  quality.py             fase 2 — flags de calidad
  anonymize.py           fase 3 — capas A y B de anonimización
  enrich.py              fase 4 — abreviaturas y metadatos derivados
  build_corpus.py        fase 5 — ensamblado y export
  verify.py              fase 6 — residuo, muestreo y reporte
  ner_review.py          capa C — NER de revisión (no reemplaza)
scripts/
  build_gazetteers.py    genera el borrador de gazetteers desde el propio corpus
outputs/
  corpus.parquet                 todo, con columnas originales, limpias y anonimizadas
  corpus_indexable.jsonl         solo los casos aptos para el RAG, listo para ingesta
  excluidos_rag.csv              qué quedó fuera del índice y por qué
  log_reemplazos.csv             auditoría: cada reemplazo con su span y original
  manifest.json                  hash del CSV, parámetros y métricas de la corrida
reports/
  reporte_anonimizacion.html     resumen visual
  residuo_phi.csv                lo que los detectores independientes encontraron
  muestra_revision_manual.csv    250 documentos estratificados para revisar a mano
  medicos_no_whitelist.csv       tokens tras Dr/Dra que no se reemplazaron
  hospital_tokens_anonimizados.csv  control de sobre-anonimización
```

## Decisiones de diseño

**Se generaliza, no se suprime.** `[FECHA]`, `[INSTITUCION]`, `[EDAD:80-89]`.
El placeholder conserva la estructura sintáctica de la frase (mejor embedding),
le dice al generador que ahí había un dato en vez de dejar un hueco que tiende
a rellenar, y deja el resultado auditable. Las edades siguen la regla de Safe
Harbor: `≥90` colapsa a `[EDAD:90+]`.

**Se marca, no se borra.** Ningún registro desaparece del `.parquet`. La columna
`apto_rag` decide qué entra al índice y se puede recalcular sin reprocesar.

**Solo casos completos entran al índice.** El RAG es un asistente de apoyo al
diagnóstico: un caso cortado o sin evolución es justo el que el generador tiende
a completar inventando el manejo o el desenlace. Por eso `apto_rag` exige los
tres campos llenos (motivo, examen físico, hospitalización) y ningún campo
truncado. Quedan **8.198 casos** de 22.176 indexables. Fuera: 2.865 por
truncado y 11.113 por algún campo vacío (casi siempre la hospitalización).
`--incluir-incompletos` reproduce el criterio anterior para la ablación.

**Truncado se detecta por el tope del export, no por la puntuación.** El
export cortó *examen físico* y *hospitalización* a 500 caracteres; el motivo
de consulta no tiene tope (llega a más de 5.000). Un campo de esos dos a ≥480
caracteres se trata como cortado: hay un pico anómalo en 495–499 que no llega
a 500 y, tan cerca del tope, no se puede saber si la nota seguía. Lejos del
tope solo cuenta una frase que termina en palabra funcional (`...AL
RECUPERARLA SE`). Terminar sin punto no es señal: `...SE DA SALIDA` es la
forma habitual de cerrar una nota, y la heurística anterior marcaba así ~2.200
casos completos.

**Reglas asimétricas para instituciones.** En este corpus "hospital" casi
siempre precede al nombre de una IPS, mientras que "clínica" es adjetivo la
mitad de las veces ("evolución clínica satisfactoria", 100+ apariciones). Por
eso `hospital` usa lista negra y `clínica` usa lista blanca. Sin esta asimetría
se pierde contenido clínico o se filtra PHI, según hacia dónde se incline la
regla única.

**El NER no reemplaza.** Sin filtros, `es_core_news_lg` produce 26.078
entidades únicas sobre 3.000 documentos: `HERIDA` como ORG, `TRAUMA` como LOC,
`TRANSITO` como PER. El modelo está entrenado sobre prosa periodística y el
texto clínico en mayúsculas lo descalabra. La capa C aplica tres filtros
(enmascarado de placeholders, rareza en el corpus, exclusión de documentos en
mayúsculas) y entrega una lista revisable a mano.

**Expansión aumentativa de abreviaturas.** `HPAF` → `HPAF (herida por arma de
fuego)`, no `herida por arma de fuego`. El chunk indexado matchea tanto la
consulta abreviada como la expandida, y no se pierde el término tal como lo
escribió el clínico. Solo la primera aparición por documento.

**Un registro = un documento.** Con mediana de 163 caracteres no hay nada que
partir por ventana deslizante. Los 30 documentos que superan 1.500 caracteres
quedan marcados con `doc_largo` para que el indexador decida.

## Verificación

El pipeline no confía en su propia anonimización: la mide. `src/verify.py`
define detectores de PHI **independientes** de los que hicieron el reemplazo, y
clasifica cada hallazgo contra las exclusiones declaradas en
`config/gazetteers.yaml`:

- `veredicto = esperado` → exclusión declarada (p. ej. "clínica satisfactoria")
- `veredicto = revisar` → residuo real

Estado actual: **69 hallazgos a revisar sobre 22.176 documentos (0,31 %)**, todos
del detector de instituciones y en su mayoría erratas de digitación de la
palabra "clínica". Cero residuo en fechas, horas, documentos y médicos.

`reports/hospital_tokens_anonimizados.csv` controla el error contrario —
sobre-anonimización — listando qué tokens se reemplazaron tras "hospital".

## Marco normativo

Ley 1581/2012 y Decreto 1377/2013 (tratamiento de datos sensibles de salud) y
Resolución 1995/1999 sobre historia clínica. HIPAA Safe Harbor no aplica
legalmente en Colombia pero se usó como checklist operativo de los 18
identificadores.

Este pipeline es una medida técnica de minimización, no un certificado de
anonimización. La revisión manual de `reports/muestra_revision_manual.csv` es
parte del procedimiento, no un extra.

## Para volver a curar los gazetteers

```bash
python scripts/build_gazetteers.py    # regenera el borrador desde el corpus
```

Sobrescribe `config/gazetteers.yaml`, así que conviene versionarlo antes: el
archivo actual está curado a mano y contiene decisiones que el script no puede
inferir.
