"""
Compara configuraciones de fragmentación de los libros (fase 2 de
PLAN_INDEXAR_LIBROS.md): cuál recupera mejor las secciones correctas.

    python scripts/comparar_fragmentacion.py                        # todas las configuraciones
    python scripts/comparar_fragmentacion.py --configs seccion,parrafos-350
    python scripts/comparar_fragmentacion.py --rutas densa,bm25,rrf,rerank   # reranker: GPU, Ollama cerrado
    python scripts/comparar_fragmentacion.py --limite 150           # prueba rápida con pocas secciones
    ENCODER=falso python scripts/comparar_fragmentacion.py ...      # valida la tubería sin bge-m3

Se evalúa por SECCIÓN: cada fragmento hereda el id de su sección; los
fragmentos de una misma sección cuentan una vez. Así todas las
configuraciones se miden contra la misma anotación
(eval/consultas_guias.yaml y, si existe, eval/preguntas_sinteticas_guias.jsonl).

Métricas, sobre las secciones de grado 2:
  · R@3 y R@10, nDCG@5 y MRR: por posición;
  · R@ppto: recall con lo que cabe en PRESUPUESTO tokens de prompt (el
    contexto de cada fragmento, sin repetir). Es la que decide: un fragmento
    grande no gana solo por traer más texto.

Busca en memoria (numpy y rank_bm25), sin Qdrant: no choca con el chatbot
abierto. Los embeddings se guardan en guias/cache/ (se pagan una vez).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.evaluate import mrr, ndcg_at        # noqa: E402
from rag import config                        # noqa: E402
from rag import fragmentar as fr              # noqa: E402
from rag.corpus import tokenizar              # noqa: E402

CONSULTAS = RAIZ / "eval" / "consultas_guias.yaml"
SINTETICAS = RAIZ / "eval" / "preguntas_sinteticas_guias.jsonl"
CACHE = config.GUIAS_DIR / "cache"
SALIDA = RAIZ / "reports" / "comparar_fragmentacion.csv"
PRESUPUESTO = 1400          # tokens por fuente en el prompt (decisión 3 del plan)
PROFUNDIDAD = 100           # fragmentos que trae cada ruta antes de deduplicar


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_consultas(secciones: dict[str, dict], conjunto: str) -> list[dict]:
    consultas = []
    if conjunto in ("anotado", "ambos"):
        for c in yaml.safe_load(CONSULTAS.read_text(encoding="utf-8"))["consultas"]:
            rel = c.get("relevantes") or {}
            faltan = [s for s in rel if s not in secciones]
            if faltan:
                # los IDs cambian si se re-extraen los libros con otras reglas
                print(f"[aviso] {c['id']}: secciones anotadas que ya no existen: {faltan}")
            consultas.append({"id": c["id"], "texto": c["texto"], "categoria": c["categoria"],
                              "conjunto": "anotado",
                              "relevantes": {s: g for s, g in rel.items() if s in secciones}})
    if conjunto in ("sintetico", "ambos") and SINTETICAS.exists():
        for l in SINTETICAS.read_text(encoding="utf-8").splitlines():
            p = json.loads(l)
            if p["seccion_id"] in secciones:
                consultas.append({"id": p["id"], "texto": p["texto"], "categoria": f"sintetica-{p['fuente']}",
                                  "conjunto": "sintetico", "relevantes": {p["seccion_id"]: 2}})
    elif conjunto in ("sintetico", "ambos"):
        print(f"[aviso] no existe {SINTETICAS.name}: solo el conjunto anotado "
              f"(generarlo con scripts/generar_preguntas_guias.py)")
    return consultas


def recortar(secciones: list[dict], consultas: list[dict], limite: int) -> list[dict]:
    """Prueba rápida: las secciones relevantes de alguna consulta más otras al
    azar hasta `limite`."""
    if not limite:
        return secciones
    relevantes = {s for c in consultas for s in c["relevantes"]}
    propias = [s for s in secciones if s["id"] in relevantes][:limite]
    resto = [s for s in secciones if s["id"] not in relevantes]
    random.Random(7).shuffle(resto)
    return propias + resto[:max(0, limite - len(propias))]


# ---------------------------------------------------------------------------
# Embeddings con caché
# ---------------------------------------------------------------------------
class Embebedor:
    def __init__(self, dispositivo: str, precision: str, lote: int):
        from rag.encoder import cargar_encoder
        self.encoder = cargar_encoder(dispositivo, precision=precision)
        self.lote = lote
        self.etiqueta = f"{os.environ.get('ENCODER', 'hf')}-{config.MODELO_EMBEDDING.replace('/', '_')}"
        CACHE.mkdir(parents=True, exist_ok=True)

    def __call__(self, textos: list[str], nombre: str = "") -> np.ndarray:
        h = hashlib.sha1("\x1e".join(textos).encode("utf-8")).hexdigest()[:12]
        ruta = CACHE / f"{nombre or 'textos'}-{self.etiqueta}-{h}.npy"
        if ruta.exists():
            return np.load(ruta)
        t0 = time.time()
        v = self.encoder.encode(textos, batch_size=self.lote, normalize_embeddings=True,
                                show_progress_bar=len(textos) > 200, convert_to_numpy=True)
        v = np.asarray(v, dtype=np.float32)
        if nombre:
            np.save(ruta, v)
            print(f"    embeddings {nombre}: {v.shape} en {time.time() - t0:.0f}s")
        return v


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
def rrf(listas: list[list[int]], k: int = config.RRF_K) -> list[int]:
    puntaje: dict[int, float] = {}
    for lista in listas:
        for pos, i in enumerate(lista, 1):
            puntaje[i] = puntaje.get(i, 0.0) + 1.0 / (k + pos)
    return sorted(puntaje, key=puntaje.get, reverse=True)


def rankings(texto: str, qv: np.ndarray, frags: list[dict], matriz: np.ndarray, bm25,
             rutas: list[str]) -> tuple[dict[str, list[int]], dict[str, float]]:
    """Índices de fragmentos ordenados por cada ruta, y el puntaje del primero
    (coseno en densa, probabilidad en rerank): sirve para la abstención."""
    cos = matriz @ qv
    densa = list(np.argsort(-cos)[:PROFUNDIDAD])
    salida = {"densa": densa}
    tope = {"densa": float(cos[densa[0]])}
    if {"bm25", "rrf", "rerank"} & set(rutas):
        s = bm25.get_scores(tokenizar(texto))
        salida["bm25"] = [int(i) for i in np.argsort(-s)[:PROFUNDIDAD] if s[i] > 0]
        salida["rrf"] = rrf([densa, salida["bm25"]])[:PROFUNDIDAD]
    if "rerank" in rutas:
        from rag.retrieve import _probabilidades
        cand = salida["rrf"][:config.PROFUNDIDAD_RERANK]
        p = _probabilidades([(texto, frags[i]["texto"]) for i in cand])
        orden = [cand[j] for j in np.argsort(-p)]
        salida["rerank"] = orden + salida["rrf"][config.PROFUNDIDAD_RERANK:]
        tope["rerank"] = float(p.max())
    return {r: salida[r] for r in rutas}, tope


# ---------------------------------------------------------------------------
# Métricas por sección
# ---------------------------------------------------------------------------
def metricas(orden: list[int], frags: list[dict], relevantes: dict[str, int],
             contar, presupuesto: int) -> dict:
    secciones: list[str] = []
    for i in orden:
        s = frags[i]["seccion_id"]
        if s not in secciones:
            secciones.append(s)
    dos = {s for s, g in relevantes.items() if g >= 2}
    ganancias = [relevantes.get(s, 0) for s in secciones]

    # lo que entraría al prompt, en orden, sin repetir contextos (small-to-big
    # devuelve el mismo padre para varios hijos)
    vistos, tokens, cubiertas = set(), 0, set()
    for i in orden:
        ctx = frags[i]["texto_contexto"]
        if ctx in vistos:
            continue
        n = contar(ctx)
        if tokens + n > presupuesto:
            break
        vistos.add(ctx)
        tokens += n
        cubiertas.add(frags[i]["seccion_id"])

    def recall(conjunto: set[str], filtro: str | None = None) -> float:
        objetivo = {s for s in dos if filtro is None or s.startswith(filtro)}
        return len(conjunto & objetivo) / len(objetivo) if objetivo else float("nan")

    return {
        "R@3": recall(set(secciones[:3])),
        "R@10": recall(set(secciones[:10])),
        "nDCG@5": ndcg_at(ganancias, 5, list(relevantes.values())),
        "MRR": mrr(ganancias),
        "R@ppto": recall(cubiertas),
        "R@ppto_ATLS": recall(cubiertas, "ATLS-"),
        "R@ppto_HUV": recall(cubiertas, "HUV-"),
        "frags_ppto": len(vistos),
        "tokens_ppto": tokens,
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Compara configuraciones de fragmentación de los libros")
    ap.add_argument("--configs", default=",".join(fr.CONFIGS))
    ap.add_argument("--rutas", default="densa,bm25,rrf")
    ap.add_argument("--conjunto", choices=["anotado", "sintetico", "ambos"], default="ambos")
    ap.add_argument("--limite", type=int, default=0, help="usar solo N secciones (prueba rápida)")
    ap.add_argument("--presupuesto", type=int, default=PRESUPUESTO)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--precision", default="fp32", help="en CPU: fp32 o bf16 (en CUDA siempre fp16)")
    ap.add_argument("--lote", type=int, default=config.LOTE_INDEXACION)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi

    todas = [json.loads(l) for l in config.GUIAS_SECCIONES.read_text(encoding="utf-8").splitlines()]
    consultas = cargar_consultas({s["id"]: s for s in todas}, args.conjunto)
    secciones = recortar(todas, consultas, args.limite)
    ids = {s["id"] for s in secciones}
    for c in consultas:
        c["relevantes"] = {s: g for s, g in c["relevantes"].items() if s in ids}
    rutas = args.rutas.split(",")
    print(f"{len(secciones)} secciones · {len(consultas)} consultas · rutas {rutas}")

    contar = fr.contador_bge()
    embeber = Embebedor(args.dispositivo, args.precision, args.lote)
    qv = embeber([c["texto"] for c in consultas], "consultas")

    filas = []
    for nombre in args.configs.split(","):
        t0 = time.time()
        frags = fr.fragmentar(secciones, nombre, contar,
                              embeber=lambda t: embeber(t, "oraciones"))
        matriz = embeber([f["texto"] for f in frags], nombre)
        bm25 = BM25Okapi([tokenizar(f["texto"]) for f in frags])
        print(f"[{nombre}] {len(frags)} fragmentos ({time.time() - t0:.0f}s)")
        for c, v in zip(consultas, qv):
            ordenes, tope = rankings(c["texto"], v, frags, matriz, bm25, rutas)
            for ruta, orden in ordenes.items():
                fila = {"conjunto": c["conjunto"], "id": c["id"], "categoria": c["categoria"],
                        "config": nombre, "ruta": ruta, "puntaje_top": tope.get(ruta),
                        "n_relevantes": sum(g >= 2 for g in c["relevantes"].values())}
                if fila["n_relevantes"]:
                    fila |= metricas(orden, frags, c["relevantes"], contar, args.presupuesto)
                filas.append(fila)

    df = pd.DataFrame(filas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False)
    print(f"\n→ {args.salida}")
    resumen(df)


def resumen(df: pd.DataFrame) -> None:
    con = df[df["n_relevantes"] > 0]
    cols = ["R@ppto", "R@3", "R@10", "nDCG@5", "MRR", "frags_ppto"]
    for conjunto, g in con.groupby("conjunto"):
        print(f"\n=== {conjunto} ({g['id'].nunique()} consultas) · ordenado por R@ppto ===")
        t = g.groupby(["config", "ruta"])[cols].mean().round(3)
        print(t.sort_values("R@ppto", ascending=False).to_string())
        print("\n--- R@ppto por libro (secciones de grado 2 de cada libro) ---")
        print(g.groupby(["config", "ruta"])[["R@ppto_ATLS", "R@ppto_HUV"]].mean().round(3).to_string())
        if conjunto == "anotado":
            print("\n--- R@ppto por categoría (ruta rrf si está, si no densa) ---")
            ruta = "rrf" if "rrf" in set(g["ruta"]) else "densa"
            print(g[g["ruta"] == ruta].pivot_table(index="categoria", columns="config",
                                                     values="R@ppto").round(2).to_string())
    neg = df[(df["categoria"] == "fuera_de_dominio")]
    if len(neg):
        print("\n--- puntaje del mejor fragmento: trauma vs fuera de dominio (para UMBRAL_GUIAS) ---")
        dentro = df[(df["conjunto"] == "anotado") & (df["categoria"] != "fuera_de_dominio")]
        for ruta in sorted(set(neg["ruta"]) & {"densa", "rerank"}):
            for nombre in sorted(set(neg["config"])):
                a = dentro[(dentro["ruta"] == ruta) & (dentro["config"] == nombre)]["puntaje_top"]
                b = neg[(neg["ruta"] == ruta) & (neg["config"] == nombre)]["puntaje_top"]
                print(f"  {ruta:6s} {nombre:24s} trauma mín {a.min():.3f} · fuera de dominio máx {b.max():.3f}")


if __name__ == "__main__":
    main()
