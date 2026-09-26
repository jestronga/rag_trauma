"""
Arnés de evaluación: compara las cuatro rutas sobre eval/consultas.yaml.

Relevancia PRE-ANOTADA por metadatos (ver la cabecera de consultas.yaml):

    2 = coincide mecanismo Y comparte al menos una región
    1 = coincide solo uno de los dos
    0 = ninguno

Qué mide y qué no
-----------------
Mide si una ruta trae casos del mecanismo y la región correctos. Es necesario
para que el caso sea relevante, pero no suficiente: dos heridas por arma de
fuego en tórax pueden ser clínicamente muy distintas. Sirve para ELEGIR RUTA,
no para afirmar utilidad clínica.

Por eso el script exporta `reports/validacion_manual.csv` con los top-5 de la
ruta ganadora, para que un humano marque relevante/no relevante sobre una
muestra y se pueda reportar la concordancia con la pre-anotación.

Métricas: nDCG@10 (relevancia graduada), Recall@k y MRR. Se reportan POR
CATEGORÍA, no promediadas: cada categoría mide un fenómeno distinto y el
promedio los oculta.

Uso:
    python -m eval.evaluate
    python -m eval.evaluate --rutas bm25,densa,rrf --k 10
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from rag import config                      # noqa: E402
from rag.retrieve import RUTAS, buscar, confianza   # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
CONSULTAS = RAIZ / "eval" / "consultas.yaml"


# ---------------------------------------------------------------------------
# Pre-anotación
# ---------------------------------------------------------------------------
def relevancia(consulta: dict, meta: dict) -> int:
    esperado_mec = consulta.get("mecanismo")
    esperadas_reg = set(consulta.get("regiones") or [])
    if not esperado_mec and not esperadas_reg:
        return 0                      # consultas fuera de dominio: nada es relevante

    coincide_mec = bool(esperado_mec) and meta.get("mecanismo_lesion") == esperado_mec
    coincide_reg = bool(esperadas_reg & set(meta.get("region_anatomica") or []))

    if consulta.get("pediatrico") and not meta.get("es_pediatrico"):
        return 0

    if coincide_mec and coincide_reg:
        return 2
    if coincide_mec or coincide_reg:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def dcg(ganancias: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(ganancias))


def ndcg_at(ganancias: list[int], k: int, ideal: list[int]) -> float:
    idcg = dcg(sorted(ideal, reverse=True)[:k])
    return dcg(ganancias[:k]) / idcg if idcg else 0.0


def mrr(ganancias: list[int]) -> float:
    for i, g in enumerate(ganancias, start=1):
        if g >= 2:
            return 1.0 / i
    return 0.0


def precision_at(ganancias: list[int], k: int) -> float:
    if not k:
        return 0.0
    return sum(1 for g in ganancias[:k] if g >= 2) / k


def evaluar(consultas: list[dict], rutas: list[str], k: int, profundidad: int):
    filas, detalle = [], []
    for c in consultas:
        fuera = bool(c.get("espera_abstencion"))
        # la confianza no depende de la ruta: es el coseno del mejor caso denso
        conf = confianza(c["texto"])
        for ruta in rutas:
            res = buscar(c["texto"], k=profundidad, ruta=ruta)
            ganancias = [relevancia(c, r.metadata) for r in res]
            # el ideal se aproxima con lo mejor visto por CUALQUIER ruta a esta
            # profundidad: sin qrels exhaustivos no hay ideal verdadero
            ideal = sorted(ganancias, reverse=True)

            filas.append({
                "id": c["id"],
                "categoria": c["categoria"],
                "ruta": ruta,
                "nDCG@10": ndcg_at(ganancias, 10, ideal),
                f"P@{k}": precision_at(ganancias, k),
                "MRR": mrr(ganancias),
                "confianza": conf,
                "abstendria": conf < config.UMBRAL_ABSTENCION,
                "espera_abstencion": fuera,
            })
            for pos, (r, g) in enumerate(zip(res[:k], ganancias[:k]), start=1):
                detalle.append({
                    "id": c["id"], "ruta": ruta, "pos": pos,
                    "record_id": r.record_id, "relevancia_auto": g,
                    "score_norm": round(r.score_norm, 4),
                    "mecanismo": r.metadata.get("mecanismo_lesion"),
                    "regiones": ",".join(r.metadata.get("region_anatomica") or []),
                    "texto": r.text[:300].replace("\n", " "),
                })
    return pd.DataFrame(filas), pd.DataFrame(detalle)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara las rutas de recuperación")
    ap.add_argument("--rutas", default="bm25,densa,rrf")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--profundidad", type=int, default=20)
    ap.add_argument("--salida", default="reports")
    args = ap.parse_args()

    rutas = [r.strip() for r in args.rutas.split(",") if r.strip()]
    desconocidas = [r for r in rutas if r not in RUTAS]
    if desconocidas:
        raise SystemExit(f"Rutas desconocidas: {desconocidas}. Opciones: {', '.join(RUTAS)}")

    consultas = yaml.safe_load(CONSULTAS.read_text(encoding="utf-8"))["consultas"]
    en_dominio = [c for c in consultas if not c.get("espera_abstencion")]
    fuera = [c for c in consultas if c.get("espera_abstencion")]
    print(f"Consultas: {len(en_dominio)} en dominio · {len(fuera)} fuera de dominio")
    print(f"Rutas: {', '.join(rutas)}\n")

    df, detalle = evaluar(consultas, rutas, args.k, args.profundidad)
    dentro = df[~df["espera_abstencion"]]

    print("=== Global (solo consultas en dominio) ===")
    glob = dentro.groupby("ruta")[["nDCG@10", f"P@{args.k}", "MRR"]].mean().round(3)
    print(glob.sort_values("nDCG@10", ascending=False).to_string())

    print("\n=== nDCG@10 por categoría ===")
    piv = dentro.pivot_table(index="categoria", columns="ruta", values="nDCG@10").round(3)
    print(piv.to_string())

    print("\n=== Abstención en consultas fuera de dominio ===")
    if fuera.__len__():
        ab = df[df["espera_abstencion"]].groupby("ruta").agg(
            se_abstiene=("abstendria", "sum"), n=("abstendria", "size"),
            confianza_media=("confianza", "mean")).round(3)
        print(ab.to_string())
        print(f"(umbral actual: {config.UMBRAL_ABSTENCION})")

    ganadora = glob["nDCG@10"].idxmax()
    print(f"\n→ Ruta con mejor nDCG@10: {ganadora}")

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida / "eval_rutas.csv", index=False, encoding="utf-8")
    detalle.to_csv(salida / "eval_detalle.csv", index=False, encoding="utf-8")

    # muestra para validación humana: top-5 de la ruta ganadora
    val = detalle[detalle["ruta"] == ganadora].copy()
    val["relevante_humano"] = ""      # columna a llenar a mano: 0, 1 o 2
    val.to_csv(salida / "validacion_manual.csv", index=False, encoding="utf-8")
    print(f"   {salida / 'eval_rutas.csv'}")
    print(f"   {salida / 'validacion_manual.csv'}  ← llenar 'relevante_humano'")


if __name__ == "__main__":
    main()
