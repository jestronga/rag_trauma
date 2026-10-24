"""
Cómo repartir el prompt entre los libros y qué guarda frena las consultas
fuera de dominio, con el ATLS 10 en el índice (fase 4 de
PLAN_INDEXAR_ATLS10.md; decisiones 2 y 5).

Lee la colección de Qdrant con la ruta rerank y el ATLS 10 encendido (se fija
ATLS10=1 aquí; `--sin-atls10` da la referencia con dos libros). Las consultas
son las de calibrar_guias.py (eval/consultas_guias.yaml y las sintéticas).

Configuraciones, candidatos × selección (rag/retrieve.py):
  común            los 30 primeros de los libros juntos, presupuesto común (como hoy)
  cuota            los mismos candidatos; cada libro que pasa la guarda tiene su parte
  por libro        los 30 primeros de cada libro, presupuesto común
  por libro+cuota  las dos
cada una con PRESUPUESTO_GUIAS de 1.400 y 1.800 tokens, en el umbral de hoy.

Por configuración:
  · R@ppto (secciones de grado 2 que llegan al prompt), total y por libro: si
    un libro desplaza a otro, se ve en la columna del desplazado;
  · % de los fragmentos del prompt que son de cada libro;
  · consultas de trauma y de procedimiento sin libros, fuera de dominio que pasan;
  · latencia mediana de la búsqueda (por libro, el reranker ve 30 por libro).
Guardas: umbral del mejor fragmento × umbral del segundo mejor de OTRA
sección (la guarda de los casos ya pide dos casos; las fuera de dominio que se
cuelan suelen entrar por un solo pasaje).

    python scripts/comparar_libros.py                 # con el chatbot cerrado
    python scripts/comparar_libros.py --sin-atls10

Escribe reports/comparar_libros.csv (una fila por consulta y configuración).
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

import pandas as pd                                               # noqa: E402

from calibrar_guias import grupo, recall                          # noqa: E402
from comparar_fragmentacion import cargar_consultas               # noqa: E402
from rag import config                                            # noqa: E402
from rag.index import cargar_secciones                            # noqa: E402
from rag.retrieve import candidatos_guias, libros_activos, seleccionar_guias  # noqa: E402

SALIDA = RAIZ / "reports" / "comparar_libros.csv"
CONFIGS = [("común", False, False), ("cuota", False, True),
           ("por libro", True, False), ("por libro+cuota", True, True)]
GUARDA_MEJOR = [0.34, 0.40, 0.45, 0.50, 0.55]
GUARDA_SEGUNDO = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def segundo_mejor(cands) -> float:
    """Puntaje del mejor fragmento de una sección distinta de la del mejor."""
    if not cands:
        return 0.0
    mejor = max(cands, key=lambda f: f.puntaje)
    return max((f.puntaje for f in cands if f.seccion_id != mejor.seccion_id), default=0.0)


def recall_libro(seleccion, relevantes: dict[str, int], libro: str) -> float | None:
    return recall(seleccion, {s: g for s, g in relevantes.items() if s.split("-")[0] == libro})


def main() -> None:
    ap = argparse.ArgumentParser(description="Reparto del prompt entre los libros y guardas (fase 4)")
    ap.add_argument("--sin-atls10", action="store_true", help="referencia: solo ATLS 11 y HUV")
    ap.add_argument("--presupuestos", default="1400,1800")
    ap.add_argument("--salida", type=Path, default=SALIDA)
    args = ap.parse_args()

    config.ATLS10 = not args.sin_atls10
    libros = libros_activos()
    umbral = config.UMBRAL_GUIAS
    consultas = cargar_consultas({s["id"]: s for s in cargar_secciones()}, "ambos", set(libros))
    for c in consultas:
        c["grupo"] = grupo(c)
    n = {g: sum(c["grupo"] == g for c in consultas) for g in ("trauma", "proc", "fuera", "sintetico")}
    print(f"libros {', '.join(libros)} · ruta rerank · umbral {umbral} · {n['trauma']} de trauma, "
          f"{n['proc']} de procedimiento, {n['fuera']} fuera de dominio, {n['sintetico']} sintéticas")

    # candidatos: una búsqueda por consulta y modo
    cands: dict[bool, dict[str, list]] = {}
    for por_libro in (False, True):
        candidatos_guias("calentamiento", "rerank", por_libro=por_libro)
        tiempos, cands[por_libro] = [], {}
        for c in consultas:
            t0 = time.perf_counter()
            cands[por_libro][c["id"]] = candidatos_guias(c["texto"], "rerank", por_libro=por_libro)
            tiempos.append(time.perf_counter() - t0)
        print(f"  candidatos {'por libro' if por_libro else 'juntos'}: "
              f"{statistics.median(tiempos):.2f} s mediana, {max(tiempos):.2f} s peor")

    filas = []
    for nombre, por_libro, cuota in CONFIGS:
        for ppto in [int(p) for p in args.presupuestos.split(",")]:
            for c in consultas:
                cs = cands[por_libro][c["id"]]
                sel = seleccionar_guias(cs, umbral, ppto, cuota=cuota)
                fila = {"config": nombre, "presupuesto": ppto, "id": c["id"], "grupo": c["grupo"],
                        "conjunto": c["conjunto"], "mejor": max((f.puntaje for f in cs), default=0.0),
                        "segundo": segundo_mejor(cs), "n_frags": len(sel),
                        "tokens": sum(f.n_tokens_contexto for f in sel),
                        "R@ppto": recall(sel, c["relevantes"])}
                for libro in libros:
                    fila[f"R@ppto_{libro}"] = recall_libro(sel, c["relevantes"], libro)
                    fila[f"frags_{libro}"] = sum(f.fuente == libro for f in sel)
                filas.append(fila)
    df = pd.DataFrame(filas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False)

    # --- tabla por configuración ---------------------------------------------
    print("\n=== Configuraciones (umbral de hoy) ===")
    cols = ["R@ppto"] + [f"R@ppto_{l}" for l in libros]
    resumen = []
    for (nombre, ppto), g in df.groupby(["config", "presupuesto"], sort=False):
        anot = g[g["conjunto"] == "anotado"]
        sint = g[g["conjunto"] == "sintetico"]
        prompt = g[g["grupo"].isin(["trauma", "proc"])]
        total = prompt["n_frags"].sum() or 1
        r = {"config": nombre, "ppto": ppto}
        for col in cols:
            corto = col.replace("R@ppto", "anot") if col != "R@ppto" else "anot"
            r[corto] = anot[col].mean()
        r["sint"] = sint["R@ppto"].mean()
        for l in libros:
            r[f"sint_{l}"] = sint[f"R@ppto_{l}"].mean()
        for l in libros:
            r[f"%frags_{l}"] = 100 * prompt[f"frags_{l}"].sum() / total
        r["trauma sin libros"] = int((g[g["grupo"] == "trauma"]["n_frags"] == 0).sum())
        r["proc sin libros"] = int((g[g["grupo"] == "proc"]["n_frags"] == 0).sum())
        r["fuera pasan"] = int((g[g["grupo"] == "fuera"]["n_frags"] > 0).sum())
        resumen.append(r)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(resumen).round(3).to_string(index=False))

    # --- guardas ---------------------------------------------------------------
    print("\n=== Guardas: umbral del mejor × del segundo mejor de otra sección "
          "(trauma sin libros / proc sin libros / fuera que pasan) ===")
    for por_libro in (False, True):
        base = df[(df["config"] == ("por libro" if por_libro else "común"))
                  & (df["presupuesto"] == int(args.presupuestos.split(",")[0]))]
        base = base.set_index("id")
        print(f"\n-- candidatos {'por libro' if por_libro else 'juntos'} --")
        print("mejor \\ 2.º " + "".join(f"{u2:>14.2f}" for u2 in GUARDA_SEGUNDO))
        for u1 in GUARDA_MEJOR:
            celdas = []
            for u2 in GUARDA_SEGUNDO:
                pasa = (base["mejor"] >= u1) & (base["segundo"] >= u2)
                t = int((~pasa[base["grupo"] == "trauma"]).sum())
                p = int((~pasa[base["grupo"] == "proc"]).sum())
                f = int(pasa[base["grupo"] == "fuera"].sum())
                celdas.append(f"{t:>4}/{p:>2}/{f:>2}")
            print(f"{u1:>10.2f} " + "".join(f"{x:>14}" for x in celdas))
        fuera = base[base["grupo"] == "fuera"].sort_values("mejor", ascending=False).head(5)
        print("   fuera de dominio más altas (mejor, 2.º): " +
              ", ".join(f"{i} {r['mejor']:.3f}/{r['segundo']:.3f}" for i, r in fuera.iterrows()))
        dentro = base[base["grupo"].isin(["trauma", "proc"]) & (base["mejor"] >= umbral)]
        print(f"   de dominio que hoy pasan, 2.º más bajo: "
              + ", ".join(f"{i} {r['segundo']:.3f}" for i, r in
                          dentro.sort_values("segundo").head(5).iterrows()))
    print(f"\n→ {args.salida}")


if __name__ == "__main__":
    main()
