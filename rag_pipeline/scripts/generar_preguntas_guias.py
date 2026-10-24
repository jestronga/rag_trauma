"""
Preguntas sintéticas para evaluar la recuperación de los libros (fase 2 de
PLAN_INDEXAR_LIBROS.md).

    python scripts/generar_preguntas_guias.py               # 100 preguntas, repartidas por libro
    python scripts/generar_preguntas_guias.py --n 10        # prueba
    python scripts/generar_preguntas_guias.py --libro ATLS10 --n 40 --agregar
                                    # suma 40 de un libro sin tocar las que ya hay

Toma secciones clínicas al azar, repartidas por igual entre los libros y,
dentro de cada libro, entre capítulos. Para cada una, el modelo local escribe
UNA pregunta clínica en español que esa sección responde. La relevante es la
sección de origen.

El riesgo de este método es que el modelo copie palabras del texto, y eso
favorece a BM25 y a los fragmentos chicos. Por eso se le pide registro de
médico y sus propias palabras, y se descartan las preguntas que repiten 5 o
más palabras seguidas de la sección. Aun así, la decisión se toma con el
conjunto anotado a mano (eval/consultas_guias.yaml); este solo confirma la
tendencia con más volumen.

Se guarda en eval/preguntas_sinteticas_guias.jsonl: todas las
configuraciones se miden con las mismas preguntas. Necesita Ollama.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from rag import config  # noqa: E402
from rag.generate import _ollama  # noqa: E402

SALIDA = RAIZ / "eval" / "preguntas_sinteticas_guias.jsonl"
MIN_CARACTERES = 400        # secciones más cortas no dan para una pregunta
MAX_CARACTERES = 3000       # lo que ve el modelo de cada sección
COPIA_MAX = 5               # palabras seguidas en común con la sección

SISTEMA = (
    "Eres un médico de urgencias que consulta un manual de trauma durante la "
    "atención de un paciente. Escribes en español, en registro clínico.")

INSTRUCCION = (
    "Lee este fragmento de un manual de trauma. Escribe UNA pregunta que un "
    "médico de urgencias haría al atender a un paciente y que este fragmento "
    "responde.\n"
    "- En español, aunque el fragmento esté en inglés.\n"
    "- Con tus propias palabras: no copies frases del fragmento.\n"
    "- Concreta y clínica, de una sola oración; sin mencionar el manual ni el "
    "capítulo.\n"
    'Responde solo con JSON: {"pregunta": "..."}\n\n'
    "FRAGMENTO:\n")

ESQUEMA = {"type": "object", "properties": {"pregunta": {"type": "string"}},
           "required": ["pregunta"]}


def _palabras(texto: str) -> list[str]:
    return re.findall(r"\w+", texto.lower())


def copia(pregunta: str, texto: str, n: int = COPIA_MAX) -> bool:
    """¿La pregunta repite n o más palabras seguidas del texto?"""
    p = _palabras(pregunta)
    ngramas = {tuple(p[i:i + n]) for i in range(len(p) - n + 1)}
    t = _palabras(texto)
    return any(tuple(t[i:i + n]) in ngramas for i in range(len(t) - n + 1))


def muestrear(secciones: list[dict], n: int, semilla: int = 7) -> list[dict]:
    """n secciones: mitad por libro y, dentro de cada libro, por turnos entre
    capítulos (así no domina un capítulo largo)."""
    rng = random.Random(semilla)
    por_libro: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for s in secciones:
        if s["clinico"] and len(s["texto"]) >= MIN_CARACTERES:
            por_libro[s["fuente"]][s["capitulo_num"]].append(s)
    salida = []
    libros = sorted(por_libro)
    for k, libro in enumerate(libros):
        cuota = n // len(libros) + (1 if k < n % len(libros) else 0)
        caps = por_libro[libro]
        for lista in caps.values():
            rng.shuffle(lista)
        orden = sorted(caps)
        rng.shuffle(orden)
        elegidas: list[dict] = []
        while len(elegidas) < cuota and any(caps.values()):
            for c in orden:
                if caps[c] and len(elegidas) < cuota:
                    elegidas.append(caps[c].pop())
        salida += elegidas
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera preguntas sintéticas desde las secciones de los libros")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--modelo", default=config.MODELO_LLM)
    ap.add_argument("--libro", default=None, help="solo secciones de este libro (campo fuente: ATLS10)")
    ap.add_argument("--agregar", action="store_true",
                    help="suma las preguntas al archivo en vez de reemplazarlo: las de antes "
                         "no cambian y todas las configuraciones se siguen midiendo igual")
    args = ap.parse_args()

    secciones = [json.loads(l) for l in config.GUIAS_SECCIONES.read_text(encoding="utf-8").splitlines()]
    previas = []
    if args.agregar and args.salida.exists():
        previas = [json.loads(l) for l in args.salida.read_text(encoding="utf-8").splitlines()]
        usadas = {p["seccion_id"] for p in previas}
        secciones = [s for s in secciones if s["id"] not in usadas]
    if args.libro:
        secciones = [s for s in secciones if s["fuente"] == args.libro]
        if not secciones:
            raise SystemExit(f"No hay secciones del libro {args.libro} en {config.GUIAS_SECCIONES}")
    elegidas = muestrear(secciones, args.n)
    preguntas, descartadas = list(previas), 0
    for k, s in enumerate(elegidas, 1):
        texto = s["texto"][:MAX_CARACTERES]
        pregunta = None
        for temperatura in (0.3, 0.7):                 # un reintento si copia
            mensajes = [{"role": "system", "content": SISTEMA},
                        {"role": "user", "content": INSTRUCCION + texto}]
            try:
                crudo = _ollama(mensajes, modelo=args.modelo, temperatura=temperatura,
                                max_tokens=120, formato=ESQUEMA)
                candidata = json.loads(crudo)["pregunta"].strip()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [{s['id']}] respuesta inválida: {e}")
                continue
            if candidata and not copia(candidata, s["texto"]):
                pregunta = candidata
                break
        if pregunta is None:
            descartadas += 1
            print(f"  [{s['id']}] descartada (copia o inválida)")
            continue
        preguntas.append({"id": f"sin-{len(preguntas) + 1:03d}", "texto": pregunta,
                          "seccion_id": s["id"], "fuente": s["fuente"],
                          "capitulo_num": s["capitulo_num"]})
        print(f"{k:3d}/{len(elegidas)} {s['id']}: {pregunta}")

    args.salida.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preguntas),
                           encoding="utf-8")
    por_libro = defaultdict(int)
    for p in preguntas:
        por_libro[p["fuente"]] += 1
    print(f"\n→ {args.salida}: {len(preguntas)} preguntas {dict(por_libro)} · {descartadas} descartadas")


if __name__ == "__main__":
    main()
