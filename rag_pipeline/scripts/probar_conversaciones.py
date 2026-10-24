"""
Corre las conversaciones de eval/conversaciones.yaml por el pipeline completo
(generate.responder, con Ollama) con la ficha del paciente y con la reescritura
del modelo (PLAN_CHATBOT_CLINICO.md, fase 2).

Qué mide, por modo (CONVERSACION=ficha y =llm):
  ficha       (solo ficha) turnos con la ficha esperada y con el tipo esperado;
              conversaciones que mezclan dos pacientes (un paciente nuevo
              tomado como seguimiento) o que lo pierden (al revés)
  tiempo      segundos por turno de seguimiento, y cuánto de eso es la
              reescritura del modelo (solo llm)
  casos       en el último turno de cada conversación, el proxy de
              consultas.yaml (eval/evaluate.py): casos que coinciden en
              mecanismo y región (grado 2) o en uno de los dos (grado ≥ 1),
              con la ficha esperada de ese turno

Los turnos "caso NNNN" no llaman al modelo: solo cuenta si se reconocen.

Uso (con el chatbot cerrado, o INDICE_DIR apuntando a una copia):
    python scripts/probar_conversaciones.py                    # los dos conjuntos, los dos modos
    python scripts/probar_conversaciones.py --solo-ficha       # sin Ollama ni índices: la ficha
    python scripts/probar_conversaciones.py --conjuntos control --modos ficha
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from rag import config, generate                      # noqa: E402
from rag.ficha import RX_CASO, actualizar             # noqa: E402

CONVERSACIONES = RAIZ / "eval" / "conversaciones.yaml"
CONJUNTOS = {"desarrollo": "conversaciones", "control": "control"}


def esperada(f: dict) -> dict:
    """La ficha de conversaciones.yaml en la forma de Ficha.datos()."""
    return {"mecanismo": f.get("mecanismo"), "regiones": sorted(f.get("regiones") or []),
            "afirmados": sorted(f.get("afirmados") or []), "negados": sorted(f.get("negados") or []),
            "signos": {k: str(v) for k, v in (f.get("signos") or {}).items()},
            "grupo": f.get("grupo"), "estado": f.get("estado")}


def cargar(conjuntos: list[str]) -> list[dict]:
    datos = yaml.safe_load(CONVERSACIONES.read_text(encoding="utf-8"))
    return [{**c, "conjunto": nombre} for nombre in conjuntos for c in datos[CONJUNTOS[nombre]]]


def diferencias(obtenida: dict, esperada_: dict) -> str:
    return " · ".join(f"{k}: {obtenida[k]} ≠ {esperada_[k]}" for k in esperada_ if obtenida[k] != esperada_[k])


def solo_ficha(conversaciones: list[dict]) -> list[dict]:
    """La ficha y el tipo, sin buscar ni generar."""
    filas = []
    for c in conversaciones:
        ficha = None
        for i, t in enumerate(c["turnos"], 1):
            tipo, ficha = actualizar(ficha, t["texto"])
            obtenida, esp = ficha.datos(), esperada(t["ficha"])
            filas.append({"conjunto": c["conjunto"], "id": c["id"], "turno": i, "texto": t["texto"],
                          "tipo_esperado": t["tipo"], "tipo": tipo, "ficha_ok": obtenida == esp,
                          "diferencias": diferencias(obtenida, esp),
                          "consulta": ficha.consulta(t["texto"]) if tipo == "seguimiento" else t["texto"]})
    return filas


def relevancia(casos, ficha_esperada: dict) -> tuple[float | None, float | None]:
    """Fracción de casos con grado 2 y con grado ≥ 1 (eval/evaluate.py)."""
    from eval.evaluate import relevancia as grado
    if not casos:
        return None, None
    grados = [grado({"mecanismo": ficha_esperada.get("mecanismo"),
                     "regiones": ficha_esperada.get("regiones") or []}, c.metadata) for c in casos]
    return sum(g == 2 for g in grados) / len(grados), sum(g >= 1 for g in grados) / len(grados)


def correr(conversaciones: list[dict], modo: str) -> list[dict]:
    """Cada conversación por generate.responder, como la app."""
    config.CONVERSACION = modo
    reescribir = generate.reescribir
    tiempo_reescritura = []

    def cronometrada(*a, **k):
        t0 = time.perf_counter()
        try:
            return reescribir(*a, **k)
        finally:
            tiempo_reescritura.append(time.perf_counter() - t0)
    generate.reescribir = cronometrada

    filas = []
    try:
        for c in conversaciones:
            historial, ficha, anterior = [], None, None
            for i, t in enumerate(c["turnos"], 1):
                fila = {"conjunto": c["conjunto"], "id": c["id"], "turno": i, "modo": modo,
                        "texto": t["texto"], "tipo_esperado": t["tipo"], "ultimo": i == len(c["turnos"])}
                if RX_CASO.match(t["texto"]):                   # la app lo resume sin buscar
                    if modo == "ficha" and ficha is not None:   # y la ficha no cambia
                        obtenida, esp = ficha.datos(), esperada(t["ficha"])
                        fila.update({"ficha_ok": obtenida == esp, "diferencias": diferencias(obtenida, esp)})
                    filas.append({**fila, "tipo": "caso"})
                    historial += [{"role": "user", "content": t["texto"]},
                                  {"role": "assistant", "content": f"(resumen del caso {t['texto']})"}]
                    continue
                tiempo_reescritura.clear()
                t0 = time.perf_counter()
                r = generate.responder(t["texto"], historial=historial, ficha=ficha, anterior=anterior)
                segundos = time.perf_counter() - t0
                p2, p1 = relevancia(r.casos, t["ficha"])
                fila.update({
                    "tipo": r.tipo, "s_total": round(segundos, 2),
                    "s_reescritura": round(sum(tiempo_reescritura), 2),
                    "consulta": r.consulta_reescrita or t["texto"],
                    "casos": len(r.casos), "pasajes": len(r.guias), "abstuvo": r.abstuvo,
                    "prosa_verificada": r.prosa_verificada, "grado2": p2, "grado1": p1,
                    "error": r.error or "",
                })
                if r.ficha is not None:
                    obtenida, esp = r.ficha.datos(), esperada(t["ficha"])
                    fila.update({"ficha_ok": obtenida == esp, "diferencias": diferencias(obtenida, esp),
                                 "ficha": r.ficha.linea(), "cambio": r.cambio or ""})
                    ficha, anterior = r.ficha, r
                filas.append(fila)
                historial += [{"role": "user", "content": t["texto"]},
                              {"role": "assistant", "content": r.texto}]
                print(f"  {modo:5} {c['id']} t{i} {segundos:5.1f} s  {fila['consulta'][:90]}", flush=True)
    finally:
        generate.reescribir = reescribir
    return filas


def resumen_ficha(df: pd.DataFrame) -> None:
    for conjunto, g in df.groupby("conjunto", sort=False):
        mezclan = sorted({r.id for r in g.itertuples() if r.tipo_esperado == "nuevo" and r.tipo == "seguimiento"})
        pierden = sorted({r.id for r in g.itertuples() if r.tipo_esperado == "seguimiento" and r.tipo == "nuevo"})
        ok = g.ficha_ok.fillna(False).astype(bool)
        print(f"  {conjunto:10} tipo {int((g.tipo == g.tipo_esperado).sum())}/{len(g)} · ficha "
              f"{int(ok.sum())}/{len(g)} ({ok.mean():.0%}) · mezclan dos pacientes: {mezclan or 'ninguna'}"
              f" · pierden al paciente: {pierden or 'ninguna'}")
        for r in g[~ok].itertuples():
            print(f"      {r.id} t{r.turno} «{r.texto}»: {r.diferencias}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Conversaciones: ficha del paciente contra reescritura")
    ap.add_argument("--conjuntos", default="desarrollo,control")
    ap.add_argument("--modos", default="ficha,llm")
    ap.add_argument("--solo-ficha", action="store_true", help="sin Ollama ni índices")
    ap.add_argument("--salida", type=Path, default=RAIZ / "reports" / "probar_conversaciones.csv")
    args = ap.parse_args()
    conversaciones = cargar(args.conjuntos.split(","))
    print(f"{len(conversaciones)} conversaciones, {sum(len(c['turnos']) for c in conversaciones)} turnos")

    if args.solo_ficha:
        print("\nFicha (sin buscar):")
        resumen_ficha(pd.DataFrame(solo_ficha(conversaciones)))
        return

    filas = []
    for modo in args.modos.split(","):
        print(f"\n=== CONVERSACION={modo} ===")
        filas += correr(conversaciones, modo)
    df = pd.DataFrame(filas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False, encoding="utf-8")

    if "ficha" in df.modo.values:
        print("\nFicha (CONVERSACION=ficha):")
        resumen_ficha(df[df.modo == "ficha"])
    print(f"\n{'modo':6} {'conjunto':10} {'seguim. mediana':>15} {'peor':>6} {'reescritura':>11} "
          f"{'grado 2 últ.':>12} {'grado ≥1 últ.':>13} {'verificada':>10}")
    for (modo, conjunto), g in df.groupby(["modo", "conjunto"], sort=False):
        seg = g[(g.turno > 1) & g.s_total.notna()]
        ult = g[g.ultimo & g.grado2.notna()]
        ver = g[g.prosa_verificada.notna()]
        print(f"{modo:6} {conjunto:10} {statistics.median(seg.s_total):13.2f} s {seg.s_total.max():5.1f}s "
              f"{seg.s_reescritura.median():9.2f} s {ult.grado2.mean():12.2f} {ult.grado1.mean():13.2f} "
              f"{int(ver.prosa_verificada.astype(bool).sum()):>5}/{len(ver)}")
    print(f"\nCSV: {args.salida}")


if __name__ == "__main__":
    main()
