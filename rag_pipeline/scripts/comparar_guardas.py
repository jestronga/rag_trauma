"""
Guardas con señal de trauma (PLAN_INDEXAR_ATLS10.md, decisión 5).

Política medida: si la consulta nombra un mecanismo, una lesión o un
procedimiento de trauma (rag/senal.py), las guardas de hoy; si no, un umbral
más alto para los libros (UMBRAL_GUIAS) y otro para los casos (la
probabilidad del MIN_CASOS-ésimo caso, UMBRAL_RERANK). Sin señal no se
abstiene: hay preguntas legítimas sin esas palabras.

Lee de Qdrant con el ATLS 10 encendido y el reparto de la decisión 2 (cuota,
1.800 tokens), ruta rerank:
  · casos: eval/consultas.yaml (25 de trauma, 20 fuera de dominio);
  · libros: eval/consultas_guias.yaml (las mismas, más 16 de procedimiento) y
    las 127 preguntas sintéticas.

    python scripts/comparar_guardas.py          # con el chatbot cerrado
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import yaml                                                       # noqa: E402

from calibrar_guias import recall                                 # noqa: E402
from rag import config                                            # noqa: E402
from rag.retrieve import buscar_rerank, candidatos_guias, confianza, seleccionar_guias  # noqa: E402
from rag.senal import senal_trauma                                # noqa: E402

LIBROS_SIN_SENAL = [0.34, 0.45, 0.55, 0.65, 0.75]
CASOS_SIN_SENAL = [0.2, 0.4, 0.6, 0.8, 1.01]          # 1,01 = sin señal, los casos nunca responden
PRESUPUESTO = 1800


def main() -> None:
    config.ATLS10 = True
    leer = lambda f: yaml.safe_load((RAIZ / "eval" / f).read_text(encoding="utf-8"))["consultas"]  # noqa: E731
    casos_q = leer("consultas.yaml")
    libros_q = leer("consultas_guias.yaml")
    sint = [json.loads(l) for l in (RAIZ / "eval" / "preguntas_sinteticas_guias.jsonl")
            .read_text(encoding="utf-8").splitlines()]

    buscar_rerank("calentamiento", config.TOP_K)
    casos = []
    for c in casos_q:
        res = buscar_rerank(c["texto"], config.TOP_K)
        p = [r.relevancia for r in res]
        casos.append({"id": c["id"], "fuera": bool(c.get("espera_abstencion")),
                      "coseno": confianza(c["texto"]),
                      "decisiva": p[config.MIN_CASOS - 1] if len(p) >= config.MIN_CASOS else 0.0,
                      "senal": bool(senal_trauma(c["texto"]))})

    candidatos_guias("calentamiento", "rerank", por_libro=False)
    libros = []
    for c in [*libros_q, *({"id": p["id"], "texto": p["texto"], "categoria": "sintetica",
                            "relevantes": {p["seccion_id"]: 2}} for p in sint)]:
        cands = candidatos_guias(c["texto"], "rerank", por_libro=False)
        libros.append({"id": c["id"], "grupo": ("fuera" if c["categoria"] == "fuera_de_dominio" else
                                                 "proc" if c["categoria"] == "procedimiento" else
                                                 "sint" if c["categoria"] == "sintetica" else "trauma"),
                       "cands": cands, "relevantes": c.get("relevantes") or {},
                       "mejor": max((f.puntaje for f in cands), default=0.0),
                       "senal": bool(senal_trauma(c["texto"]))})

    n = lambda g: sum(f["grupo"] == g for f in libros)                     # noqa: E731
    print(f"Sin señal: trauma {sum(not f['senal'] for f in libros if f['grupo'] == 'trauma')}/{n('trauma')}"
          f" · proc {sum(not f['senal'] for f in libros if f['grupo'] == 'proc')}/{n('proc')}"
          f" · sintéticas {sum(not f['senal'] for f in libros if f['grupo'] == 'sint')}/{n('sint')}"
          f" · fuera {sum(not f['senal'] for f in libros if f['grupo'] == 'fuera')}/{n('fuera')}")

    print(f"\n=== Libros: con señal {config.UMBRAL_GUIAS}; sin señal, el umbral de la columna "
          f"(cuota, {PRESUPUESTO} tokens) ===")
    print(f"{'sin señal':>9} {'trauma sin libros':>17} {'proc sin libros':>15} {'sint. sin libros':>16} "
          f"{'R@ppto anot':>11} {'R@ppto sint':>11}  fuera que pasan")
    for u0 in LIBROS_SIN_SENAL:
        umbral = {f["id"]: config.UMBRAL_GUIAS if f["senal"] else u0 for f in libros}
        pasa = {f["id"]: f["mejor"] >= umbral[f["id"]] for f in libros}
        r_anot, r_sint = [], []
        for f in libros:
            # el umbral también decide qué libros tienen cuota
            sel = seleccionar_guias(f["cands"], umbral[f["id"]], PRESUPUESTO, cuota=True)
            rc = recall(sel, f["relevantes"])
            if rc is not None:
                (r_sint if f["grupo"] == "sint" else r_anot).append(rc)
        sin = lambda g: sum(not pasa[f["id"]] for f in libros if f["grupo"] == g)   # noqa: E731
        fuera = [f["id"] for f in libros if f["grupo"] == "fuera" and pasa[f["id"]]]
        print(f"{u0:>9.2f} {sin('trauma'):>12}/{n('trauma')} {sin('proc'):>10}/{n('proc')} "
              f"{sin('sint'):>10}/{n('sint')} {sum(r_anot) / len(r_anot):>11.3f} "
              f"{sum(r_sint) / len(r_sint):>11.3f}  {fuera}")

    dentro = [f for f in casos if not f["fuera"]]
    fuera = [f for f in casos if f["fuera"]]
    print(f"\n=== Casos: con señal UMBRAL_RERANK {config.UMBRAL_RERANK}; sin señal, el de la columna ===")
    print(f"{'sin señal':>9} {'trauma que se abstienen':>23}  fuera que responden")
    for u0 in CASOS_SIN_SENAL:
        responde = {f["id"]: f["coseno"] >= config.UMBRAL_ABSTENCION
                    and f["decisiva"] >= (config.UMBRAL_RERANK if f["senal"] else u0) for f in casos}
        print(f"{u0:>9.2f} {sum(not responde[f['id']] for f in dentro):>18}/{len(dentro)}  "
              f"{[f['id'] for f in fuera if responde[f['id']]]}")
    mejor_libro = {f["id"]: f["mejor"] for f in libros}
    print("\nde dominio SIN señal (lo que paga la guarda): casos prob. del 2.º caso; libros mejor fragmento")
    for f in casos:
        if not f["fuera"] and not f["senal"]:
            print(f"  {f['id']:7} casos {f['decisiva']:.3f} · libros {mejor_libro.get(f['id'], 0.0):.3f}")
    for f in libros:
        if f["grupo"] == "proc" and not f["senal"]:
            print(f"  {f['id']:7} libros {f['mejor']:.3f}")
    bajos = sorted(f["mejor"] for f in libros if f["grupo"] == "sint" and not f["senal"])
    print(f"  sintéticas sin señal ({len(bajos)}): mejor fragmento más bajos "
          + ", ".join(f"{x:.3f}" for x in bajos[:8]))
    print("\nfuera de dominio (casos: coseno, prob. del 2.º caso; libros: mejor fragmento):")
    for f in sorted(fuera, key=lambda f: -f["decisiva"]):
        print(f"  {f['id']:7} señal {'sí' if f['senal'] else 'no'} · casos {f['coseno']:.3f} / "
              f"{f['decisiva']:.3f} · libros {mejor_libro.get(f['id'], 0.0):.3f}")


if __name__ == "__main__":
    main()
