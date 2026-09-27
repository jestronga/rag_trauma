#!/usr/bin/env python3
"""Planilla para que un médico califique las respuestas, a ciegas: formato anterior vs. actual.

    python scripts/exportar_revision.py                  # 15 consultas de eval/consultas.yaml
    python scripts/exportar_revision.py --n 20 --base 5c3a035

Por cada consulta se recupera y se llama al modelo UNA vez; con esa misma
salida se arma la respuesta con el código de `--base` (formato y extractor
anteriores, tomados de git) y con el código actual. Así la única diferencia es
el formato y el extractor, no la recuperación ni la elección del modelo.

Genera en `eval/revision/`:
  · revision.html   lo que lee el médico: la consulta, las respuestas A y B en
                    orden aleatorio y el texto completo de los casos, para
                    comprobar cada conteo y cada cita;
  · planilla.csv    una fila por respuesta para calificar (rúbrica abajo);
  · clave.csv       qué respuesta es cuál. NO se le entrega al médico.

Rúbrica (1 = muy mal, 5 = muy bien), adaptada de la evaluación de Med-PaLM:
  conteos_y_citas   cada número y cada [caso: ID] coincide con el texto del caso
  sin_omisiones     no omite algo importante que está en los casos
  sin_riesgo        nada en la respuesta podría inducir a una decisión dañina
  utilidad          ayuda a decidir qué valorar o hacer
  se_lee_en_30s     sí / no

Qdrant local admite un solo proceso: cierra el chatbot antes de correrlo, o
apunta a una copia de los índices con INDICE_DIR.
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib
import random
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

from markdown_it import MarkdownIt  # noqa: E402

from rag import config  # noqa: E402
from rag.build_prompt import SISTEMA, construir_prompt  # noqa: E402
from rag.generate import _ollama, ensamblar  # noqa: E402
from rag.retrieve import buscar, confianza  # noqa: E402

COLUMNAS = ["consulta_id", "respuesta", "conteos_y_citas", "sin_omisiones", "sin_riesgo",
            "utilidad", "se_lee_en_30s", "comentario"]
MD = MarkdownIt("commonmark").enable("table")

CSS = """
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 16px;
       color: #222; background: #fafafa; line-height: 1.45; }
h2 { border-top: 2px solid #999; padding-top: 18px; margin-top: 40px; }
.respuestas { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 800px) { .respuestas { grid-template-columns: 1fr; } }
.resp { background: #fff; border: 1px solid #ccc; border-radius: 8px; padding: 12px; font-size: 14px;
        overflow-x: auto; }
.resp h3 { margin-top: 0; }
table { border-collapse: collapse; font-size: 12.5px; }
td, th { border: 1px solid #ddd; padding: 3px 6px; vertical-align: top; }
details { margin-top: 12px; }
.caso { background: #fff; border-left: 3px solid #999; padding: 6px 10px; margin: 8px 0;
        font-size: 13px; white-space: pre-wrap; }
.rubrica { background: #eef; padding: 10px 14px; border-radius: 8px; font-size: 14px; }
"""


def cargar_base(ref: str):
    """El paquete `rag` tal como estaba en `ref`, importable como `rag_base`."""
    destino = Path(tempfile.mkdtemp()) / "rag_base"
    destino.mkdir()
    nombres = subprocess.run(["git", "ls-tree", "--name-only", ref, "rag_pipeline/rag/"], cwd=RAIZ.parent,
                             capture_output=True, text=True, check=True).stdout.split()
    for ruta in nombres:
        if ruta.endswith(".py"):
            contenido = subprocess.run(["git", "show", f"{ref}:{ruta}"], cwd=RAIZ.parent,
                                       capture_output=True, check=True).stdout
            (destino / Path(ruta).name).write_bytes(contenido)
    sys.path.insert(0, str(destino.parent))
    return importlib.import_module("rag_base.generate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=15, help="consultas a exportar")
    ap.add_argument("--base", default="5c3a035",
                    help="commit con el formato anterior (5c3a035 = antes de la revisión del 26/09/2026)")
    ap.add_argument("--semilla", type=int, default=7)
    args = ap.parse_args()

    base = cargar_base(args.base)
    consultas = yaml.safe_load((RAIZ / "eval" / "consultas.yaml").read_text(encoding="utf-8"))["consultas"]
    azar = random.Random(args.semilla)
    salida = RAIZ / "eval" / "revision"
    salida.mkdir(parents=True, exist_ok=True)

    bloques, planilla, clave = [], [], []
    for c in consultas:
        if len(clave) >= 2 * args.n:
            break
        texto = c["texto"]
        if confianza(texto, None) < config.UMBRAL_ABSTENCION:
            continue                                   # se abstiene igual en ambos formatos
        casos = buscar(texto, k=config.TOP_K)
        if len(casos) < config.MIN_CASOS:
            continue
        print(f"{c['id']}: {texto}")
        salida_modelo = _ollama([{"role": "system", "content": SISTEMA},
                                 {"role": "user", "content": construir_prompt(texto, casos)}])
        versiones = {"anterior": base.ensamblar(salida_modelo, casos, texto),
                     "actual": ensamblar(salida_modelo, casos, texto)}
        orden = list(versiones)
        azar.shuffle(orden)
        columnas = []
        for letra, version in zip("AB", orden):
            clave.append({"consulta_id": c["id"], "respuesta": letra, "version": version})
            planilla.append({"consulta_id": c["id"], "respuesta": letra, **{k: "" for k in COLUMNAS[2:]}})
            columnas.append(f"<div class='resp'><h3>Respuesta {letra}</h3>{MD.render(versiones[version])}</div>")
        textos_casos = "".join(f"<div class='caso'><b>caso {x.record_id}</b>\n{html.escape(x.text)}</div>"
                               for x in casos)
        bloques.append(f"<h2>{c['id']} · {html.escape(texto)}</h2>"
                       f"<div class='respuestas'>{''.join(columnas)}</div>"
                       f"<details><summary>Texto completo de los {len(casos)} casos (para verificar)</summary>"
                       f"{textos_casos}</details>")

    rubrica = (__doc__.split("Rúbrica")[1].split("Qdrant")[0]).strip()
    (salida / "revision.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>Revisión de respuestas</title><style>{CSS}</style>"
        f"<h1>Revisión de respuestas del asistente</h1>"
        f"<div class='rubrica'><b>Cómo calificar</b> (en planilla.csv, una fila por respuesta)"
        f"<pre>{html.escape('Rúbrica ' + rubrica)}</pre>"
        f"Las respuestas A y B de cada consulta usan los mismos casos. Comentario libre: qué cambiaría.</div>"
        + "".join(bloques), encoding="utf-8")
    for nombre, filas, cols in (("planilla.csv", planilla, COLUMNAS),
                                ("clave.csv", clave, ["consulta_id", "respuesta", "version"])):
        with (salida / nombre).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(filas)
    print(f"\n{len(bloques)} consultas → {salida}\\revision.html, planilla.csv (para el médico) "
          f"y clave.csv (no entregar)")


if __name__ == "__main__":
    main()
