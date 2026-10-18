"""
Calibra UMBRAL_RERANK con las consultas de eval/consultas.yaml.

La regla de abstención con la ruta rerank es: se responde solo si al menos
MIN_CASOS casos superan UMBRAL_RERANK (y el coseno supera UMBRAL_ABSTENCION).
Por eso lo que decide es la probabilidad del caso número MIN_CASOS, no la del
mejor: con MIN_CASOS=2, la del segundo.

Para cada umbral candidato cuenta:
  · abstenciones falsas: consultas de trauma que se quedarían sin respuesta
  · respuestas falsas: consultas fuera de dominio que se responderían

y propone el punto medio del hueco entre ambos grupos, si lo hay.

Si los libros están indexados, agrega la guarda de los libros (columna
"libros": puntaje del mejor fragmento) y cuenta las dos guardas juntas, como
responde el sistema (decisión 3 de PLAN_INDEXAR_LIBROS.md): se responde si
pasa cualquiera; ninguna consulta fuera de dominio debe pasar ninguna. La
guarda de los libros se calibra aparte con scripts/calibrar_guias.py.

Uso (con el chatbot cerrado, o INDICE_DIR apuntando a una copia):
    python scripts/calibrar_umbral.py
    python scripts/calibrar_umbral.py --profundidad 20
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from rag import config                                   # noqa: E402
from rag.retrieve import (buscar_rerank, candidatos_guias, confianza,  # noqa: E402
                          dispositivo_reranker, hay_guias, ruta_guias_por_defecto,
                          umbral_guias)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibra el umbral del reranker")
    ap.add_argument("--profundidad", type=int, default=config.PROFUNDIDAD_RERANK)
    args = ap.parse_args()

    consultas = yaml.safe_load((RAIZ / "eval" / "consultas.yaml").read_text(encoding="utf-8"))["consultas"]
    n_min = config.MIN_CASOS

    filas, tiempos = [], []
    buscar_rerank("calentamiento", config.TOP_K, profundidad=args.profundidad)   # carga el modelo
    libros = hay_guias()
    ruta_libros = ruta_guias_por_defecto()
    for c in consultas:
        conf = confianza(c["texto"])
        t0 = time.perf_counter()
        res = buscar_rerank(c["texto"], config.TOP_K, profundidad=args.profundidad)
        tiempos.append(time.perf_counter() - t0)
        probas = [r.relevancia for r in res]
        decisiva = probas[n_min - 1] if len(probas) >= n_min else 0.0
        mejor_libro = (max((f.puntaje for f in candidatos_guias(c["texto"], ruta_libros)), default=0.0)
                       if libros else None)
        filas.append({"id": c["id"], "fuera": bool(c.get("espera_abstencion")),
                      "coseno": conf, "p1": probas[0] if probas else 0.0, "decisiva": decisiva,
                      "libros": mejor_libro, "texto": c["texto"]})

    print(f"Profundidad {args.profundidad} · {dispositivo_reranker()} · "
          f"rerank {sorted(tiempos)[len(tiempos) // 2]:.2f} s mediana, {max(tiempos):.2f} s peor\n")
    print(f"{'id':8} {'coseno':>6} {'p1':>6} {'p' + str(n_min):>6} {'libros':>6}  consulta")
    for f in sorted(filas, key=lambda f: (f["fuera"], f["decisiva"])):
        marca = "  ← fuera de dominio" if f["fuera"] else ""
        lib = f"{f['libros']:6.3f}" if libros else f"{'—':>6}"
        print(f"{f['id']:8} {f['coseno']:6.3f} {f['p1']:6.3f} {f['decisiva']:6.3f} {lib}  "
              f"{f['texto'][:60]}{marca}")

    def abstiene(f, t):
        return f["coseno"] < config.UMBRAL_ABSTENCION or f["decisiva"] < t

    print(f"\n{'umbral':>6} {'abst. falsas':>13} {'resp. falsas':>13}")
    dentro = [f for f in filas if not f["fuera"]]
    fuera = [f for f in filas if f["fuera"]]
    for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"{t:6.2f} {sum(abstiene(f, t) for f in dentro):>8}/{len(dentro)} "
              f"{sum(not abstiene(f, t) for f in fuera):>8}/{len(fuera)}")

    # solo cuentan las fuera de dominio que el coseno no detiene ya
    pasan_coseno = [f["decisiva"] for f in fuera if f["coseno"] >= config.UMBRAL_ABSTENCION]
    peor_fuera = max(pasan_coseno, default=0.0)
    mejor_dentro = min(f["decisiva"] for f in dentro)
    if mejor_dentro > peor_fuera:
        print(f"\nHueco entre {peor_fuera:.3f} (fuera) y {mejor_dentro:.3f} (dentro): "
              f"UMBRAL_RERANK ≈ {(peor_fuera + mejor_dentro) / 2:.2f}")
    else:
        print(f"\nSin hueco: una consulta de dominio ({mejor_dentro:.3f}) queda por debajo "
              f"de una fuera de dominio ({peor_fuera:.3f}). Revisar las filas de arriba.")
    print(f"(UMBRAL_RERANK actual: {config.UMBRAL_RERANK} · MIN_CASOS: {n_min} · "
          f"UMBRAL_ABSTENCION: {config.UMBRAL_ABSTENCION})")

    if libros:
        u_lib = umbral_guias(ruta_libros)

        def responde(f) -> tuple[bool, bool]:
            return not abstiene(f, config.UMBRAL_RERANK), f["libros"] >= u_lib

        print(f"\nLas dos guardas juntas (decisión 3), con los umbrales actuales "
              f"(libros: ruta {ruta_libros}, umbral {u_lib}):")
        for nombre, grupo in (("trauma", dentro), ("fuera de dominio", fuera)):
            r = [responde(f) for f in grupo]
            print(f"  {nombre:16} solo casos {sum(c and not l for c, l in r):>2} · solo libros "
                  f"{sum(l and not c for c, l in r):>2} · las dos {sum(c and l for c, l in r):>2} · "
                  f"ninguna {sum(not (c or l) for c, l in r):>2}  (de {len(grupo)})")
        pasan = [f["id"] for f in fuera if any(responde(f))]
        print("  → ninguna consulta fuera de dominio pasa." if not pasan else
              f"  → fuera de dominio que responderían: {pasan}")
    else:
        print("\n(Sin la colección de los libros: python -m rag.index --fuente guias)")


if __name__ == "__main__":
    main()
