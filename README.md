# rag_trauma

Dos partes del mismo proyecto, cada una en su carpeta. Comparten el entorno
virtual `.venv` de la raíz.

```
rag_trauma/
  .venv/              entorno compartido
  requirements.txt    instala ambas partes (-r a cada carpeta)
  anonimizacion/      pipeline de limpieza y anonimización de trauma.csv
    run_pipeline.py, src/, config/, tests/, scripts/build_gazetteers.py
    data/             fuentes crudas, fuera de git: trauma.csv (PHI) y los dos
                      libros de trauma (ATLS 11.ª ed., manual del HUV)
    outputs/          corpus anonimizado  ← lo consume el RAG
    reports/          reportes de anonimización
  rag_pipeline/       prototipo RAG (índices, evaluación y chatbot)
    app.py, rag/, eval/, scripts/revisar_entorno.py
    indices/          BM25 + Qdrant local
    reports/          resultados de la evaluación
```

El único vínculo entre ambas es el corpus: `rag_pipeline/rag/config.py` lee
`../anonimizacion/outputs/corpus_indexable.jsonl` (sobrescribible con la
variable de entorno `CORPUS_JSONL`).

## Uso rápido

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt

cd anonimizacion
python run_pipeline.py            # genera outputs/corpus_indexable.jsonl
python -m pytest tests/ -q

cd ..\rag_pipeline
python -m rag.index               # construye los índices
python -m eval.evaluate           # compara rutas de recuperación
streamlit run app.py              # chatbot
```

Detalle de cada parte: `anonimizacion/README.md` y `rag_pipeline/SETUP.md`.
Estado del trabajo y pendientes: `CONTEXTO_SESION.md`.
