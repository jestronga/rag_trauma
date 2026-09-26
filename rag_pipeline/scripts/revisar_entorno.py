#!/usr/bin/env python3
"""Comprueba que el entorno está listo: VRAM, dependencias, índices y Ollama.

    python scripts/revisar_entorno.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, MAL, AVISO = "  ✓", "  ✗", "  !"


def main() -> int:
    problemas = 0

    print("GPU")
    try:
        import torch
        if torch.cuda.is_available():
            libre, total = torch.cuda.mem_get_info()
            gb = total / 1024 ** 3
            print(f"{OK} {torch.cuda.get_device_name(0)} · {gb:.1f} GB "
                  f"({libre / 1024 ** 3:.1f} GB libres)")
            if gb < 7:
                print(f"{AVISO} Menos de 7 GB: usa un generador de 3B  →  "
                      f"MODELO_LLM=qwen2.5:3b-instruct-q4_K_M")
            elif gb < 11:
                print(f"{AVISO} 8 GB: el reranker no cabe junto al generador. "
                      f"Déjalo apagado o en CPU (es el valor por defecto).")
        else:
            print(f"{AVISO} torch sin CUDA: la indexación irá por CPU "
                  f"(~40-60 min para 22k documentos)")
    except ImportError:
        print(f"{MAL} torch no está instalado")
        problemas += 1

    print("\nDependencias")
    for mod, paquete in [("qdrant_client", "qdrant-client"), ("rank_bm25", "rank_bm25"),
                         ("sentence_transformers", "sentence-transformers"),
                         ("streamlit", "streamlit"), ("yaml", "PyYAML")]:
        try:
            __import__(mod)
            print(f"{OK} {paquete}")
        except ImportError:
            print(f"{MAL} falta {paquete}  →  pip install {paquete}")
            problemas += 1

    print("\nCorpus e índices")
    from rag import config
    faltan = []
    for etiqueta, ruta in [("corpus", config.CORPUS_JSONL),
                           ("índice BM25", config.BM25_PATH),
                           ("índice denso", config.QDRANT_PATH)]:
        if Path(ruta).exists():
            print(f"{OK} {etiqueta}: {ruta}")
        else:
            print(f"{MAL} falta {etiqueta}: {ruta}")
            faltan.append(etiqueta)
            problemas += 1
    if "corpus" in faltan:
        print("     →  python ../anonimizacion/run_pipeline.py")
    if "índice denso" in faltan or "índice BM25" in faltan:
        print("     →  python -m rag.index")

    print("\nOllama")
    from rag.generate import ollama_disponible
    ok, msg = ollama_disponible()
    print(f"{OK if ok else MAL} {msg}")
    if not ok:
        problemas += 1

    print("\n" + ("Todo listo." if not problemas else f"{problemas} problema(s) por resolver."))
    return 0 if not problemas else 1


if __name__ == "__main__":
    raise SystemExit(main())
