"""
Compara generadores y modos de salida sobre eval/consultas.yaml, con tiempos
por etapa y el uso de los casos recuperados.

Por consulta, la recuperación se hace UNA vez y todos los modelos y modos ven
los mismos casos: la única diferencia es el generador.

Qué mide, por (modelo, modo):
  tiempos       coseno, recuperación, generación (carga, lectura del prompt,
                salida) y ensamblado, en segundos
  elección      propuestas, descartadas (sin respaldo, obvias o repetidas) y
                completadas por el código
  uso de casos  citados: casos que respaldan alguna sugerencia final;
                citados / recuperados, y la precisión de ambos grupos con la
                relevancia pre-anotada (mecanismo + región, eval/evaluate.py).
                Si los citados son más precisos que los recuperados, la
                elección favorece los casos pertinentes; si se citan pocos,
                sobran casos en el prompt (y tokens: pendiente 8).
  GPU           % del modelo en VRAM según /api/ps (lo que muestra `ollama ps`)
  prosa         (--modos prosa) si la prosa del modelo pasa rag/verificar.py y,
                si no, por qué regla; la prosa queda en el CSV para leerla

Las consultas fuera de dominio solo miden si el sistema se abstiene: eso no
depende del generador.

Uso (con el chatbot cerrado, o INDICE_DIR apuntando a una copia):
    python scripts/comparar_modelos.py                                   # modelo de config, JSON
    python scripts/comparar_modelos.py --modelos qwen3:4b-instruct,qwen3:8b
    python scripts/comparar_modelos.py --modos texto,json --n 5          # salida en texto vs JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.evaluate import relevancia                                        # noqa: E402
from rag import config                                                      # noqa: E402
from rag.generate import (                                                  # noqa: E402
    _ollama, eleccion_modelo, elegir_sugerencias, ensamblar, ensamblar_prosa, filtrar_relevantes,
    generar_eleccion, generar_prosa, porcentaje_gpu,
)
from rag.hallazgos import elegibles                                         # noqa: E402
from rag.retrieve import buscar, confianza, ruta_por_defecto                # noqa: E402
from rag.verificar import RX_CITA, verificar_prosa                          # noqa: E402

MODOS = {"texto": False, "json": True, "prosa": None}   # prosa: SALIDA=prosa


def recuperar(c: dict, ruta: str) -> dict:
    """Las dos guardas de responder(), con sus tiempos."""
    t0 = time.perf_counter()
    conf = confianza(c["texto"])
    t1 = time.perf_counter()
    casos = [] if conf < config.UMBRAL_ABSTENCION else filtrar_relevantes(
        buscar(c["texto"], k=config.TOP_K, ruta=ruta))
    t2 = time.perf_counter()
    return {"casos": casos, "confianza": conf, "abstiene": len(casos) < config.MIN_CASOS,
            "s_coseno": t1 - t0, "s_recuperacion": t2 - t1}


def descargar(modelo: str) -> None:
    """Saca el modelo de memoria: con él se va la caché de prompts de Ollama.

    Sin esto, repetir una consulta (o correr el script dos veces) "lee" el
    prompt en 1,6 s en vez de ~12: Ollama reutiliza lo que ya procesó. La
    recarga que provoca se mide aparte (s_carga) y no cuenta en s_total.
    """
    cuerpo = json.dumps({"model": modelo, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(f"{config.OLLAMA_URL}/api/generate", data=cuerpo,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()


def precision(casos, consulta: dict) -> float | None:
    return sum(relevancia(consulta, c.metadata) == 2 for c in casos) / len(casos) if casos else None


def medir(c: dict, rec: dict, estructurada: bool) -> dict:
    casos = rec["casos"]
    metricas: dict = {}
    t0 = time.perf_counter()
    salida = generar_eleccion(c["texto"], casos, estructurada=estructurada, metricas=metricas)
    t1 = time.perf_counter()
    ensamblar(salida, casos, c["texto"])
    t2 = time.perf_counter()

    validos, propuestas = eleccion_modelo(salida, elegibles(casos, c["texto"]))
    finales = elegir_sugerencias(salida, casos, c["texto"])
    ids_citados = {rid for e in finales for rid in e.casos}
    citados = [x for x in casos if x.record_id in ids_citados]
    return {
        "s_generacion": round(t1 - t0, 2),
        "s_carga": round(metricas.get("s_carga", 0.0), 2),
        "s_prompt": round(metricas.get("s_prompt", 0.0), 2),
        "s_salida": round(metricas.get("s_salida", 0.0), 2),
        "s_ensamblado": round(t2 - t1, 3),
        "tokens_prompt": metricas.get("tokens_prompt", 0),
        "tokens_salida": metricas.get("tokens_salida", 0),
        "propuestas": propuestas,
        "descartadas": propuestas - len(validos),
        "completadas": sum(1 for e in finales if e not in validos),
        "recuperados": len(casos),
        "citados": len(citados),
        "precision_recuperados": precision(casos, c),
        "precision_citados": precision(citados, c),
        "finales": " | ".join(e.termino for e in finales),
    }


# Motivos de rechazo de la prosa (rag/verificar.py), agrupados por regla
_TIPOS_MOTIVO = [("cita inventada", "no están entre los recuperados"),
                 ("término sin respaldo", "no consta en ningún caso"),
                 ("sin cita", "sin citar el caso"),
                 ("mal atribuido", "no lo documentan"),
                 ("conteo", ("no coincide", "se recuperaron", "«todos»", "«la mayoría»")),
                 ("palabra ajena", "no aparecen en los casos"),
                 ("orden", "da una orden"),
                 ("sin prosa", "no hay prosa")]


def tipo_motivo(motivo: str) -> str:
    for tipo, claves in _TIPOS_MOTIVO:
        if any(k in motivo for k in ((claves,) if isinstance(claves, str) else claves)):
            return tipo
    return "otro"


def medir_prosa(c: dict, rec: dict) -> dict:
    """La prosa del modelo, si pasa la verificación y por qué no."""
    casos = rec["casos"]
    metricas: dict = {}
    t0 = time.perf_counter()
    prosa = generar_prosa(c["texto"], casos, metricas=metricas)
    t1 = time.perf_counter()
    ok, motivos = verificar_prosa(prosa, casos, c["texto"])
    (ensamblar_prosa(prosa, casos) if ok else ensamblar("", casos, c["texto"]))
    t2 = time.perf_counter()

    ids = {m.group(1) for m in RX_CITA.finditer(prosa)}
    citados = [x for x in casos if x.record_id in ids]
    return {
        "s_generacion": round(t1 - t0, 2),
        "s_carga": round(metricas.get("s_carga", 0.0), 2),
        "s_prompt": round(metricas.get("s_prompt", 0.0), 2),
        "s_salida": round(metricas.get("s_salida", 0.0), 2),
        "s_ensamblado": round(t2 - t1, 3),
        "tokens_prompt": metricas.get("tokens_prompt", 0),
        "tokens_salida": metricas.get("tokens_salida", 0),
        "recuperados": len(casos),
        "citados": len(citados),
        "precision_recuperados": precision(casos, c),
        "precision_citados": precision(citados, c),
        "verificada": ok,
        "motivos": " || ".join(motivos),
        "tipos_motivo": " | ".join(sorted({tipo_motivo(m) for m in motivos})),
        "finales": "prosa" if ok else "determinista",
        "prosa": prosa,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara generadores y modos de salida")
    ap.add_argument("--modelos", default=config.MODELO_LLM, help="separados por coma")
    ap.add_argument("--modos", default="json", help="texto, json, prosa, separados por coma")
    ap.add_argument("--n", type=int, default=0, help="solo las primeras N consultas de dominio")
    ap.add_argument("--ruta", default=None, help="ruta de recuperación (por defecto la de config)")
    args = ap.parse_args()

    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]
    modos = [m.strip() for m in args.modos.split(",") if m.strip()]
    if set(modos) - set(MODOS):
        raise SystemExit(f"Modos desconocidos: {sorted(set(modos) - set(MODOS))}. Opciones: {', '.join(MODOS)}")
    ruta = args.ruta or ruta_por_defecto()
    consultas = yaml.safe_load((RAIZ / "eval" / "consultas.yaml").read_text(encoding="utf-8"))["consultas"]
    dentro = [c for c in consultas if not c.get("espera_abstencion")]
    fuera = [c for c in consultas if c.get("espera_abstencion")]
    if args.n:
        dentro = dentro[:args.n]

    print(f"Recuperando ({ruta}) · {len(dentro)} consultas de dominio, {len(fuera)} fuera …")
    # la primera búsqueda carga bge-m3 (y el reranker): no es un tiempo típico
    recuperar({"texto": "herida en mano"}, ruta)
    recs = {c["id"]: recuperar(c, ruta) for c in dentro + fuera}
    falsas = [c["id"] for c in dentro if recs[c["id"]]["abstiene"]]
    respondidas = [c["id"] for c in fuera if not recs[c["id"]]["abstiene"]]

    filas = []
    for modelo in modelos:
        config.MODELO_LLM = modelo            # _ollama lee el modelo de config
        # calentamiento: la primera llamada carga el modelo y no es un tiempo típico
        carga: dict = {}
        _ollama([{"role": "user", "content": "ok"}], max_tokens=1, metricas=carga)
        gpu = porcentaje_gpu(modelo)
        print(f"\n{modelo}: carga {carga.get('s_carga', 0):.1f} s · GPU {gpu}%")
        # modo por fuera y consulta por dentro: dos llamadas seguidas con la
        # misma consulta comparten el prefijo del prompt (sistema + casos) y
        # Ollama reutiliza su caché; la segunda "leía" el prompt en 12 s en vez
        # de 76. En el chatbot cada consulta llega sin caché.
        for modo in modos:
            for c in dentro:
                rec = recs[c["id"]]
                if rec["abstiene"]:
                    continue
                descargar(modelo)
                fila = {"modelo": modelo, "modo": modo, "id": c["id"], "categoria": c["categoria"],
                        "s_coseno": round(rec["s_coseno"], 3),
                        "s_recuperacion": round(rec["s_recuperacion"], 3),
                        **(medir_prosa(c, rec) if modo == "prosa" else medir(c, rec, MODOS[modo]))}
                fila["gpu_pct"] = porcentaje_gpu(modelo)     # tras cada recarga
                # como en el chatbot con el modelo ya cargado: sin la recarga forzada
                fila["s_total"] = round(fila["s_coseno"] + fila["s_recuperacion"]
                                        + fila["s_generacion"] - fila["s_carga"]
                                        + fila["s_ensamblado"], 2)
                filas.append(fila)
                if modo == "prosa":
                    detalle = fila["tipos_motivo"] or "verificada"
                else:
                    detalle = f"descartadas {fila['descartadas']}  → {fila['finales']}"
                print(f"  {c['id']:8} {modo:5} {fila['s_total']:6.1f} s  "
                      f"citados {fila['citados']}/{fila['recuperados']}  {detalle}")

    df = pd.DataFrame(filas)
    salida = RAIZ / "reports" / "comparar_modelos.csv"
    df.to_csv(salida, index=False, encoding="utf-8")

    print("\n=== Tiempos en segundos (mediana; 'peor' = el máximo) ===")
    t = df.groupby(["modelo", "modo"]).agg(
        total=("s_total", "median"), peor=("s_total", "max"),
        recuperacion=("s_recuperacion", "median"), prompt=("s_prompt", "median"),
        salida=("s_salida", "median"), tokens_prompt=("tokens_prompt", "median"),
        tokens_salida=("tokens_salida", "median"), gpu_pct=("gpu_pct", "first"))
    print(t.round(2).to_string())

    print("\n=== Elección y uso de los casos ===")
    for col in ("descartadas", "completadas"):      # no existen en el modo prosa
        if col not in df:
            df[col] = pd.NA
    u = df.groupby(["modelo", "modo"]).agg(
        descartadas=("descartadas", "sum"), completadas=("completadas", "sum"),
        citados=("citados", "sum"), recuperados=("recuperados", "sum"),
        prec_recuperados=("precision_recuperados", "mean"),
        prec_citados=("precision_citados", "mean"))
    u["citados/recuperados"] = u["citados"] / u["recuperados"]
    print(u.round(3).to_string())

    if "verificada" in df:
        pr = df[df["modo"] == "prosa"]
        print("\n=== Prosa del modelo ===")
        for modelo, g in pr.groupby("modelo"):
            tipos = pd.Series([t for ts in g["tipos_motivo"].fillna("") if ts for t in ts.split(" | ")])
            print(f"{modelo}: {int(g['verificada'].sum())} de {len(g)} verificadas"
                  + (f" · rechazos por: {tipos.value_counts().to_dict()}" if len(tipos) else ""))

    print(f"\nAbstención (no depende del modelo): {len(falsas)} de {len(dentro)} consultas de "
          f"dominio se abstienen{' (' + ', '.join(falsas) + ')' if falsas else ''}; "
          f"{len(respondidas)} de {len(fuera)} fuera de dominio responden"
          f"{' (' + ', '.join(respondidas) + ')' if respondidas else ''}.")
    print(f"Detalle: {salida}")


if __name__ == "__main__":
    main()
