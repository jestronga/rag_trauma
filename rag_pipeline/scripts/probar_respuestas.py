#!/usr/bin/env python3
"""Prueba de punta a punta de la generación: recupera, genera y revisa la respuesta.

    python scripts/probar_respuestas.py "herida por arma de fuego en tórax con hipoventilación"
    python scripts/probar_respuestas.py "consulta 1" "consulta 2" ...

Por cada consulta imprime el tiempo, las citas válidas e inválidas, la
respuesta completa y las "palabras sin raíz en los casos": palabras de 6 o más
letras de la respuesta cuyo inicio no aparece en ningún caso recuperado. La
mayoría son lenguaje común ("presentan", "similares"); un término clínico en
esa lista es una posible invención que hay que revisar.

Qdrant local admite un solo proceso: cierra el chatbot (streamlit) antes de
correrlo, o apunta a una copia de los índices con INDICE_DIR.
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import inspect  # noqa: E402

from rag import build_prompt, config, hallazgos  # noqa: E402
from rag.build_prompt import resumen_calculado  # noqa: E402
from rag.generate import responder  # noqa: E402

# El texto fijo de la respuesta (rótulos, términos del léxico, etiquetas) lo
# escribe el código: no cuenta como palabra nueva.
VOCABULARIO_CODIGO = " ".join([inspect.getsource(build_prompt), inspect.getsource(hallazgos),
                               *config.ETIQUETAS_MECANISMO.values(), *config.ETIQUETAS_REGION.values()])


def _norm(texto: str) -> str:
    return unicodedata.normalize("NFD", texto.lower()).encode("ascii", "ignore").decode()


def main(consultas: list[str]) -> None:
    print(f"modelo: {config.MODELO_LLM} · sintesis_llm: {config.SINTESIS_LLM}")
    for consulta in consultas:
        t0 = time.time()
        r = responder(consulta)
        print("=" * 90)
        print(f"CONSULTA: {consulta}")
        print(f"{time.time() - t0:.0f}s | confianza {r.confianza:.2f} | abstuvo {r.abstuvo} | "
              f"error {r.error}")
        print(f"citas válidas {r.citas_validas} | inválidas {r.citas_invalidas}")
        print("-" * 90)
        print(r.texto)
        if r.casos:
            fuente = _norm(" ".join(c.text for c in r.casos) + " " + consulta + " "
                           + VOCABULARIO_CODIGO + " " + resumen_calculado(r.casos))
            nuevas = sorted({w for w in re.findall(r"[a-z]{6,}", _norm(r.texto))
                             if w[:5] not in fuente})
            print("-" * 90)
            print("palabras sin raíz en los casos:", nuevas)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
