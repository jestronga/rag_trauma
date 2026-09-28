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

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from rag import config  # noqa: E402
from rag.generate import responder  # noqa: E402
from rag.verificar import palabras_sin_raiz  # noqa: E402


def main(consultas: list[str]) -> None:
    print(f"modelo: {config.MODELO_LLM} · salida: {config.SALIDA} · sintesis_llm: {config.SINTESIS_LLM}")
    for consulta in consultas:
        t0 = time.time()
        r = responder(consulta)
        print("=" * 90)
        print(f"CONSULTA: {consulta}")
        print(f"{time.time() - t0:.0f}s | confianza {r.confianza:.2f} | abstuvo {r.abstuvo} | "
              f"error {r.error}")
        print(f"citas válidas {r.citas_validas} | inválidas {r.citas_invalidas}")
        if r.prosa_verificada is not None:
            estado = "verificada" if r.prosa_verificada else "NO pasó → versión determinista"
            print(f"prosa del modelo: {estado}")
            for m in r.motivos:
                print(f"  · {m}")
        print("-" * 90)
        print(r.texto)
        if r.casos:
            print("-" * 90)
            print("palabras sin raíz en los casos:", palabras_sin_raiz(r.texto, r.casos, consulta))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
