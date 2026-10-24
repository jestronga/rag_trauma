"""
Calibra la guarda de los libros (UMBRAL_GUIAS, UMBRAL_GUIAS_COSENO) y mide lo
que llega al prompt con retrieve.candidatos_guias + seleccionar_guias, leyendo
la colección de Qdrant (fase 4 de PLAN_INDEXAR_LIBROS.md).

Consultas: eval/consultas_guias.yaml (25 de trauma, 16 de procedimiento, 10
fuera de dominio) y, si existe, eval/preguntas_sinteticas_guias.jsonl.

Para cada umbral candidato:
  · R@ppto: secciones de grado 2 que llegan al prompt (una consulta en la que
    los libros se abstienen cuenta 0);
  · trauma / procedimiento sin libros: consultas que se quedan sin la mirada
    de los manuales (las de trauma igual pueden responderlas los casos);
  · fuera de dominio que pasan: deben ser 0 (decisión 3: ninguna fuente
    responde sola a "cetoacidosis").

Uso (con el chatbot cerrado, o INDICE_DIR apuntando a una copia):
    python scripts/calibrar_guias.py                  # rutas según la GPU
    python scripts/calibrar_guias.py --rutas densa    # como en la GTX 1650
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from comparar_fragmentacion import cargar_consultas               # noqa: E402
from rag import config                                            # noqa: E402
from rag.index import cargar_secciones                            # noqa: E402
from rag.retrieve import (_reranker_cabe, candidatos_guias,       # noqa: E402
                          seleccionar_guias, umbral_guias)

UMBRALES = {"rerank": [0.05, 0.1, 0.15, 0.18, 0.2, 0.25, 0.3, 0.34, 0.4],
            "densa": [0.5, 0.52, 0.55, 0.57, 0.59, 0.6, 0.62]}


def grupo(c: dict) -> str:
    if c["categoria"] == "fuera_de_dominio":
        return "fuera"
    if c["conjunto"] == "sintetico":
        return "sintetico"
    return "proc" if c["categoria"] == "procedimiento" else "trauma"


def libros_indexados() -> set[str]:
    """Los libros (`fuente`) que hay en la colección. secciones.jsonl puede
    traer uno extraído y anotado que todavía no se indexó (el ATLS 10 desde el
    23/10, PLAN_INDEXAR_ATLS10.md): sus secciones no se pueden recuperar y, si
    contaran, R@ppto bajaría sin que la búsqueda haya cambiado."""
    from rag.retrieve import _qdrant
    libros: set[str] = set()
    siguiente = None
    while True:
        puntos, siguiente = _qdrant().scroll(config.COLECCION_GUIAS, limit=1000, offset=siguiente,
                                             with_payload=["fuente"], with_vectors=False)
        libros |= {p.payload["fuente"] for p in puntos}
        if siguiente is None:
            return libros


def recall(seleccion, relevantes: dict[str, int]) -> float | None:
    dos = {s for s, g in relevantes.items() if g >= 2}
    if not dos:
        return None
    return len({f.seccion_id for f in seleccion} & dos) / len(dos)


def evaluar(filas: list[dict], umbral: float, presupuesto: int) -> dict:
    r = {"anotado": [], "sintetico": []}
    sin_libros = {"trauma": 0, "proc": 0, "fuera": 0}
    for f in filas:
        sel = seleccionar_guias(f["cands"], umbral, presupuesto)
        if f["grupo"] in sin_libros:
            sin_libros[f["grupo"]] += not sel
        rc = recall(sel, f["relevantes"])
        if rc is not None:
            r[f["conjunto"]].append(rc)
    media = {k: statistics.mean(v) if v else float("nan") for k, v in r.items()}
    return {"anot": media["anotado"], "sint": media["sintetico"], **sin_libros}


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibra la guarda de los libros")
    ap.add_argument("--rutas", default=None, help="rerank,densa (por defecto, según la GPU)")
    ap.add_argument("--presupuestos", default="1000,1400,1800")
    ap.add_argument("--conjunto", choices=["anotado", "sintetico", "ambos"], default="ambos")
    args = ap.parse_args()
    rutas = args.rutas.split(",") if args.rutas else (["rerank", "densa"] if _reranker_cabe() else ["densa"])

    libros = libros_indexados()
    consultas = cargar_consultas({s["id"]: s for s in cargar_secciones()}, args.conjunto, libros)
    n = {g: sum(grupo(c) == g for c in consultas) for g in ("trauma", "proc", "fuera")}
    print(f"{len(consultas)} consultas ({n['trauma']} de trauma, {n['proc']} de procedimiento, "
          f"{n['fuera']} fuera de dominio, el resto sintéticas) · rutas {rutas} · "
          f"presupuesto {config.PRESUPUESTO_GUIAS} tokens · libros {', '.join(sorted(libros))}")

    for ruta in rutas:
        candidatos_guias("calentamiento", ruta)              # carga los modelos
        filas, tiempos = [], []
        for c in consultas:
            t0 = time.perf_counter()
            cands = candidatos_guias(c["texto"], ruta)
            tiempos.append(time.perf_counter() - t0)
            filas.append({"id": c["id"], "texto": c["texto"], "grupo": grupo(c), "conjunto": c["conjunto"],
                          "relevantes": c["relevantes"], "cands": cands,
                          "mejor": max((f.puntaje for f in cands), default=0.0)})
        actual = umbral_guias(ruta)
        escala = "probabilidad del reranker" if ruta == "rerank" else "coseno"
        print(f"\n=== ruta {ruta} · guarda: {escala} del mejor fragmento · "
              f"{statistics.median(tiempos):.2f} s mediana por consulta ===")

        anotadas = sorted((f for f in filas if f["conjunto"] == "anotado"),
                          key=lambda f: (f["grupo"] == "fuera", f["mejor"]))
        print(f"\n{'id':8} {'mejor':>6}  consulta (las de trauma y procedimiento bajo "
              f"{actual}, y todas las fuera de dominio)")
        for f in anotadas:
            if f["grupo"] == "fuera" or f["mejor"] < actual:
                marca = "  ← fuera de dominio" if f["grupo"] == "fuera" else ""
                print(f"{f['id']:8} {f['mejor']:6.3f}  {f['texto'][:60]}{marca}")

        print(f"\n{'umbral':>6} {'R@ppto anot':>11} {'R@ppto sint':>11} {'trauma sin libros':>17} "
              f"{'proc sin libros':>15} {'fuera que pasan':>15}")
        for u in [0.0, *UMBRALES[ruta]]:
            e = evaluar(filas, u, config.PRESUPUESTO_GUIAS)
            marca = "  ← actual" if abs(u - actual) < 1e-9 else ""
            print(f"{u:6.2f} {e['anot']:11.3f} {e['sint']:11.3f} {e['trauma']:>12}/{n['trauma']} "
                  f"{e['proc']:>10}/{n['proc']} {n['fuera'] - e['fuera']:>10}/{n['fuera']}{marca}")

        fuera = [f["mejor"] for f in filas if f["grupo"] == "fuera"]
        if fuera:
            peor = max(fuera)
            minimo = min((u for u in UMBRALES[ruta] if u > peor), default=None)
            print(f"\nFuera de dominio máx. {peor:.3f}: el primer umbral que las frena a todas es "
                  f"{minimo if minimo is not None else '(ninguno de la lista)'}. Actual: {actual}")

        print(f"\nPresupuesto con el umbral actual ({actual}):")
        for p in [int(x) for x in args.presupuestos.split(",")]:
            e = evaluar(filas, actual, p)
            print(f"  {p:5d} tokens → R@ppto anot {e['anot']:.3f} · sint {e['sint']:.3f}")


if __name__ == "__main__":
    main()
