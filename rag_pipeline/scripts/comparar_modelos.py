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
                si no, por qué regla; la prosa queda en el CSV para leerla.
                Con los libros (GUIAS, por defecto), qué fuentes pasaron su
                guarda y cuántos pasajes de los manuales cita la prosa

Las consultas fuera de dominio solo miden si el sistema se abstiene: eso no
depende del generador.

Uso (con el chatbot cerrado, o INDICE_DIR apuntando a una copia):
    python scripts/comparar_modelos.py                                   # modelo de config, JSON
    python scripts/comparar_modelos.py --modelos qwen3:4b-instruct,qwen3:8b
    python scripts/comparar_modelos.py --modos texto,json --n 5          # salida en texto vs JSON
    python scripts/comparar_modelos.py --modos prosa --consultas guias   # + las 16 de procedimiento
    python scripts/comparar_modelos.py --modos prosa --sin-guias         # solo casos (ablación)
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
    _ollama, armar_secciones, eleccion_modelo, elegir_sugerencias, ensamblar, fuentes_en_prosa,
    guarda_casos, generar_eleccion, generar_prosa, mensajes_prosa, porcentaje_gpu,
    prosa_por_fuente, usar_guias,
)
from rag.hallazgos import elegibles                                         # noqa: E402
from rag.retrieve import buscar, buscar_guias, confianza, ruta_por_defecto  # noqa: E402
from rag.verificar import RX_CITA, RX_CITA_GUIA, verificar_por_fuente       # noqa: E402

MODOS = {"texto": False, "json": True, "prosa": None}   # prosa: SALIDA=prosa


def recuperar(c: dict, ruta: str, con_guias: bool = False) -> dict:
    """Las guardas de responder(), con sus tiempos: la de los casos y, con los
    libros, la de los manuales (se responde si pasa cualquiera)."""
    t0 = time.perf_counter()
    conf = confianza(c["texto"])
    t1 = time.perf_counter()
    casos = [] if conf < config.UMBRAL_ABSTENCION else guarda_casos(
        c["texto"], buscar(c["texto"], k=config.TOP_K, ruta=ruta))[0]
    t2 = time.perf_counter()
    guias = buscar_guias(c["texto"]) if con_guias else []
    t3 = time.perf_counter()
    return {"casos": casos, "guias": guias, "con_guias": con_guias, "confianza": conf,
            "abstiene": not casos and not guias,
            "s_coseno": t1 - t0, "s_recuperacion": t2 - t1 + t3 - t2, "s_guias": t3 - t2}


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
                 ("sin prosa", "no hay prosa"),
                 # oraciones que citan los manuales
                 ("guía inventada", "fragmentos recuperados"),
                 ("mezcla fuentes", "mezcla citas"),
                 ("manual equivocado", "pasaje de otro manual"),
                 ("cifra", ("no está en la guía", "cifras sin citar")),
                 ("término fuera de la guía", "no consta en la guía"),
                 ("palabra fuera de la guía", "no aparecen en la guía")]


def tipo_motivo(motivo: str) -> str:
    for tipo, claves in _TIPOS_MOTIVO:
        if any(k in motivo for k in ((claves,) if isinstance(claves, str) else claves)):
            return tipo
    return "otro"


def medir_prosa(c: dict, rec: dict) -> dict:
    """La prosa del modelo, si pasa la verificación y por qué no."""
    casos, guias = rec["casos"], rec["guias"]
    metricas: dict = {}
    t0 = time.perf_counter()
    prosa = generar_prosa(c["texto"], casos, metricas=metricas, guias=guias)
    t1 = time.perf_counter()
    v = verificar_por_fuente(prosa, casos, c["texto"], guias)
    ok, motivos = v.pasa, v.motivos
    # lo que vería el médico con la verificación encendida (PROSA_SIEMPRE=0)
    armar_secciones(casos, c["texto"], guias, rec["con_guias"],
                    *prosa_por_fuente(v, casos, guias, siempre=False))
    t2 = time.perf_counter()

    ids = {m.group(1) for m in RX_CITA.finditer(prosa)}
    citados = [x for x in casos if x.record_id in ids]
    secciones = {f.seccion_id for f in guias}
    citadas = {m.group(1).upper() for m in RX_CITA_GUIA.finditer(prosa)} & secciones
    mensajes = mensajes_prosa(c["texto"], casos, guias) or []
    return {
        "fuentes": "+".join(n for n, hay in (("casos", casos), ("guias", guias)) if hay),
        # lo que ve el médico en prosa; el resto va en la versión determinista
        "en_prosa": "+".join(fuentes_en_prosa(v, casos, guias)),
        # para calibrar la estimación de tokens de generate._opciones_contexto
        "caracteres_prompt": sum(len(m["content"]) for m in mensajes),
        "pasajes": len(secciones),
        "pasajes_citados": len(citadas),
        "pasajes_citados_atls": sum(s.startswith("ATLS-") for s in citadas),
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
    ap.add_argument("--consultas", choices=["casos", "guias"], default="casos",
                    help="casos: eval/consultas.yaml · guias: eval/consultas_guias.yaml "
                         "(las mismas más 16 de procedimiento, que responden los manuales)")
    ap.add_argument("--sin-guias", action="store_true", help="solo casos, como antes de los libros")
    ap.add_argument("--salida", type=Path, default=RAIZ / "reports" / "comparar_modelos.csv")
    args = ap.parse_args()

    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]
    modos = [m.strip() for m in args.modos.split(",") if m.strip()]
    if set(modos) - set(MODOS):
        raise SystemExit(f"Modos desconocidos: {sorted(set(modos) - set(MODOS))}. Opciones: {', '.join(MODOS)}")
    ruta = args.ruta or ruta_por_defecto()
    con_guias = usar_guias(False if args.sin_guias else None)
    base = yaml.safe_load((RAIZ / "eval" / "consultas.yaml").read_text(encoding="utf-8"))["consultas"]
    consultas = base
    if args.consultas == "guias":
        # mismos id y texto; mecanismo y regiones (para la precisión) vienen de consultas.yaml
        por_id = {c["id"]: c for c in base}
        consultas = [{**por_id.get(c["id"], {}), **c} for c in
                     yaml.safe_load((RAIZ / "eval" / "consultas_guias.yaml").read_text(encoding="utf-8"))["consultas"]]
    es_fuera = lambda c: c.get("espera_abstencion") or c.get("categoria") == "fuera_de_dominio"   # noqa: E731
    dentro = [c for c in consultas if not es_fuera(c)]
    fuera = [c for c in consultas if es_fuera(c)]
    if args.n:
        dentro = dentro[:args.n]

    print(f"Recuperando ({ruta}{', con los manuales' if con_guias else ', solo casos'}) · "
          f"{len(dentro)} consultas de dominio, {len(fuera)} fuera …")
    # la primera búsqueda carga bge-m3 (y el reranker): no es un tiempo típico
    recuperar({"texto": "herida en mano"}, ruta, con_guias)
    recs = {c["id"]: recuperar(c, ruta, con_guias) for c in dentro + fuera}
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
                # sin casos, los modos texto y json no tienen qué elegir
                if rec["abstiene"] or (modo != "prosa" and not rec["casos"]):
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
    salida = args.salida
    salida.parent.mkdir(parents=True, exist_ok=True)
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
            if con_guias:
                for fuentes, h in g.groupby("fuentes"):
                    con_pasajes = h[h["pasajes"] > 0]
                    citan = (f" · citan algún pasaje {int((con_pasajes['pasajes_citados'] > 0).sum())} de "
                             f"{len(con_pasajes)} (del ATLS: {int((con_pasajes['pasajes_citados_atls'] > 0).sum())})"
                             if len(con_pasajes) else "")
                    en_prosa = h["en_prosa"].fillna("").replace("", "determinista").value_counts().to_dict()
                    print(f"  {fuentes or 'sin fuentes'}: {int(h['verificada'].sum())} de {len(h)} "
                          f"verificadas{citan} · en prosa: {en_prosa}")

    print(f"\nAbstención (no depende del modelo): {len(falsas)} de {len(dentro)} consultas de "
          f"dominio se abstienen{' (' + ', '.join(falsas) + ')' if falsas else ''}; "
          f"{len(respondidas)} de {len(fuera)} fuera de dominio responden"
          f"{' (' + ', '.join(respondidas) + ')' if respondidas else ''}.")
    print(f"Detalle: {salida}")


if __name__ == "__main__":
    main()
