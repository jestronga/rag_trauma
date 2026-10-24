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
    _ollama, armar_secciones, eleccion_modelo, elegir_sugerencias, ensamblar, es_presentacion,
    fuentes_en_prosa, guarda_casos, generar_eleccion, generar_prosa, mensajes_prosa, porcentaje_gpu,
    prosa_atls, prosa_por_fuente, usar_guias,
)
from rag.ficha import actualizar                                            # noqa: E402
from rag.hallazgos import elegibles                                         # noqa: E402
from rag.retrieve import buscar, buscar_guias, confianza, fragmentos_de, ruta_por_defecto  # noqa: E402
from rag.tarjetas import armar as armar_tarjeta, elegir as elegir_tarjeta   # noqa: E402
from rag.verificar import RX_CITA, RX_CITA_GUIA, verificar_por_fuente       # noqa: E402

# prosa: SALIDA=prosa; atls: SALIDA=prosa con ORDEN_ATLS (PLAN_CHATBOT_CLINICO.md, fase 4);
# tarjeta: como atls, pero con la tarjeta del mecanismo en lugar de la prosa de los
# manuales cuando la hay (TARJETAS, PLAN_RESPUESTA_IDEAL.md)
MODOS = {"texto": False, "json": True, "prosa": None, "atls": None, "tarjeta": None}


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
                 ("sin la edición", "sin decir la edición"),            # ATLS 10, decisión 3
                 ("cifra", ("no está en la guía", "cifras sin citar")),
                 ("término fuera de la guía", "no consta en la guía"),
                 ("palabra fuera de la guía", "no aparecen en la guía"),
                 # por letras (modo atls)
                 ("letra equivocada", ("es de la letra", "que es de la letra")),
                 ("fuente cruzada", ("cita un manual en el apartado", "no cita un manual")),
                 ("json inválido", "no es un JSON válido")]


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
        "pasajes_atls10": sum(s.startswith("ATLS10-") for s in secciones),
        "pasajes_citados_atls10": sum(s.startswith("ATLS10-") for s in citadas),
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


def medir_atls(c: dict, rec: dict) -> dict:
    """La prosa por letras (ORDEN_ATLS): si pasa, cuántas letras escribe de las
    pedidas, si cada texto es de su letra y si el JSON llega entero. Las
    preguntas de procedimiento no van por letras (fase 4): se miden como prosa."""
    if not es_presentacion(None, c["texto"]):
        return {**medir_prosa(c, rec), "orden": False}
    casos, guias = rec["casos"], rec["guias"]
    metricas: dict = {}
    t0 = time.perf_counter()
    p_casos, p_guias, ok, motivos, detalle = prosa_atls(c["texto"], casos, guias, siempre=True, metricas=metricas)
    t1 = time.perf_counter()
    textos = detalle["textos"]
    prosa = "\n\n".join(p for p in (p_casos, p_guias) if p)
    ids = {m.group(1) for m in RX_CITA.finditer(prosa)}
    citados = [x for x in casos if x.record_id in ids]
    secciones = {f.seccion_id for f in guias}
    citadas = {m.group(1).upper() for m in RX_CITA_GUIA.finditer(prosa)} & secciones
    return {
        "orden": True,
        "fuentes": "+".join(n for n, hay in (("casos", casos), ("guias", guias)) if hay),
        "en_prosa": "+".join(f for f in ("casos", "guias")
                             if any(t.fuente == f for t in textos) and all(not t.motivos for t in textos
                                                                           if t.fuente == f)),
        "pasajes": len(secciones), "pasajes_citados": len(citadas),
        "letras_pedidas": len(detalle["letras_casos"]) + len(detalle["letras_manuales"]),
        "letras_escritas": len(textos),
        "letras_verificadas": sum(not t.motivos for t in textos),
        "letras_equivocadas": sum(any("de la letra" in m for m in t.motivos) for t in textos),
        "json_invalido": detalle["json_invalido"],
        "s_generacion": round(t1 - t0, 2),
        "s_carga": round(metricas.get("s_carga", 0.0), 2),
        "s_prompt": round(metricas.get("s_prompt", 0.0), 2),
        "s_salida": round(metricas.get("s_salida", 0.0), 2),
        "s_ensamblado": 0.0,
        "tokens_prompt": metricas.get("tokens_prompt", 0),
        "tokens_salida": metricas.get("tokens_salida", 0),
        "recuperados": len(casos), "citados": len(citados),
        "precision_recuperados": precision(casos, c), "precision_citados": precision(citados, c),
        "verificada": ok,
        "motivos": " || ".join(motivos),
        "tipos_motivo": " | ".join(sorted({tipo_motivo(m) for m in motivos})),
        "finales": "prosa" if ok else "determinista",
        "prosa": prosa,
    }


def medir_tarjeta(c: dict, rec: dict) -> dict:
    """Como en la app con TARJETAS: si el mecanismo tiene tarjeta, el apartado
    de los manuales es la tarjeta (sus viñetas pasan el verificador antes, en
    tests/test_tarjetas.py) y el modelo escribe solo las letras de los casos.
    Sin tarjeta, lo mismo que el modo atls."""
    ficha = actualizar(None, c["texto"])[1]
    tarjeta = elegir_tarjeta(ficha, c["texto"], rec["con_guias"])
    if tarjeta is None or not es_presentacion(ficha, c["texto"]):
        return {**medir_atls(c, rec), "tarjeta": ""}
    casos = rec["casos"]
    t0 = time.perf_counter()
    texto, citas = armar_tarjeta(tarjeta, ficha)
    pasajes = fragmentos_de(citas)
    t1 = time.perf_counter()
    metricas: dict = {}
    p_casos, _, ok, motivos, detalle = prosa_atls(c["texto"], casos, [], siempre=True, metricas=metricas)
    t2 = time.perf_counter()
    textos = detalle["textos"]
    ids = {m.group(1) for m in RX_CITA.finditer(p_casos)}
    citados = [x for x in casos if x.record_id in ids]
    casos_ok = bool(textos) and all(not t.motivos for t in textos)
    return {
        "orden": True, "tarjeta": tarjeta.clave,
        "fuentes": "+".join(n for n, hay in (("casos", casos), ("tarjeta", True)) if hay),
        "en_prosa": "+".join(f for f, hay in (("casos", casos_ok), ("tarjeta", True)) if hay),
        "pasajes": len({f.seccion_id for f in rec["guias"]}), "pasajes_citados": 0,
        "pasajes_tarjeta": len({f.seccion_id for f in pasajes}),
        "vinetas_tarjeta": texto.count("\n- "),
        "letras_pedidas": len(detalle["letras_casos"]), "letras_escritas": len(textos),
        "letras_verificadas": sum(not t.motivos for t in textos),
        "letras_equivocadas": sum(any("de la letra" in m for m in t.motivos) for t in textos),
        "json_invalido": detalle["json_invalido"],
        "s_generacion": round(t2 - t1, 2),
        "s_carga": round(metricas.get("s_carga", 0.0), 2),
        "s_prompt": round(metricas.get("s_prompt", 0.0), 2),
        "s_salida": round(metricas.get("s_salida", 0.0), 2),
        "s_ensamblado": round(t1 - t0, 3),
        "tokens_prompt": metricas.get("tokens_prompt", 0),
        "tokens_salida": metricas.get("tokens_salida", 0),
        "recuperados": len(casos), "citados": len(citados),
        "precision_recuperados": precision(casos, c), "precision_citados": precision(citados, c),
        "verificada": ok,
        "motivos": " || ".join(motivos),
        "tipos_motivo": " | ".join(sorted({tipo_motivo(m) for m in motivos})),
        "finales": "prosa" if ok else "determinista",
        "prosa": f"{texto}\n\n{p_casos}",
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
                if rec["abstiene"] or (modo not in ("prosa", "atls", "tarjeta") and not rec["casos"]):
                    continue
                descargar(modelo)
                medida = (medir_prosa(c, rec) if modo == "prosa" else medir_atls(c, rec) if modo == "atls"
                          else medir_tarjeta(c, rec) if modo == "tarjeta" else medir(c, rec, MODOS[modo]))
                fila = {"modelo": modelo, "modo": modo, "id": c["id"], "categoria": c["categoria"],
                        "s_coseno": round(rec["s_coseno"], 3),
                        "s_recuperacion": round(rec["s_recuperacion"], 3), **medida}
                fila["gpu_pct"] = porcentaje_gpu(modelo)     # tras cada recarga
                # como en el chatbot con el modelo ya cargado: sin la recarga forzada
                fila["s_total"] = round(fila["s_coseno"] + fila["s_recuperacion"]
                                        + fila["s_generacion"] - fila["s_carga"]
                                        + fila["s_ensamblado"], 2)
                filas.append(fila)
                if modo in ("prosa", "atls", "tarjeta"):
                    detalle = fila["tipos_motivo"] or "verificada"
                    if fila.get("orden"):
                        detalle += f"  · letras {fila['letras_verificadas']}/{fila['letras_escritas']} de " \
                                   f"{fila['letras_pedidas']}"
                    if fila.get("tarjeta"):
                        detalle += f"  · tarjeta {fila['tarjeta']}"
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

    if "orden" in df and df["orden"].fillna(False).astype(bool).any():
        at = df[(df["modo"] == "atls") & df["orden"].fillna(False).astype(bool)]
        print("\n=== Por letras (ORDEN_ATLS), solo las presentaciones de un paciente ===")
        for modelo, g in at.groupby("modelo"):
            pr_mismas = df[(df["modo"] == "prosa") & df["id"].isin(g["id"]) & (df["modelo"] == modelo)]
            tipos = pd.Series([t for ts in g["tipos_motivo"].fillna("") if ts for t in ts.split(" | ")])
            print(f"{modelo}: {int(g['verificada'].sum())} de {len(g)} verificadas enteras "
                  f"(la prosa de siempre, en las mismas: {int(pr_mismas['verificada'].sum())} de "
                  f"{len(pr_mismas)}) · JSON inválido {int(g['json_invalido'].sum())} · letras "
                  f"verificadas {int(g['letras_verificadas'].sum())} de {int(g['letras_escritas'].sum())} "
                  f"escritas ({int(g['letras_pedidas'].sum())} pedidas) · de la letra equivocada "
                  f"{int(g['letras_equivocadas'].sum())} · mediana {g['s_total'].median():.1f} s, "
                  f"{g['tokens_salida'].median():.0f} tokens de salida"
                  + (f"\n  rechazos por: {tipos.value_counts().to_dict()}" if len(tipos) else ""))

    if "tarjeta" in df and (df["modo"] == "tarjeta").any():
        print("\n=== Con tarjeta (TARJETAS) contra por letras, en las mismas presentaciones ===")
        for modelo, g in df[(df["modo"] == "tarjeta") & df["orden"].fillna(False).astype(bool)].groupby("modelo"):
            con = g[g["tarjeta"].fillna("") != ""]
            at = df[(df["modo"] == "atls") & df["id"].isin(g["id"]) & (df["modelo"] == modelo)]

            def verificadas(h) -> int:
                return int(sum(len([f for f in str(e).split("+") if f]) for e in h["en_prosa"].fillna("")))

            def fuentes(h) -> int:
                return int(sum(len([f for f in str(e).split("+") if f]) for e in h["fuentes"].fillna("")))
            print(f"{modelo}: {len(con)} de {len(g)} presentaciones con tarjeta ({con['tarjeta'].value_counts().to_dict()})"
                  + f"\n  fuentes verificadas: con tarjeta {verificadas(g)} de {fuentes(g)} · por letras "
                    f"{verificadas(at)} de {fuentes(at)}"
                  + f"\n  mediana {g['s_total'].median():.1f} s (peor {g['s_total'].max():.1f}) · por letras "
                    f"{at['s_total'].median():.1f} s (peor {at['s_total'].max():.1f})"
                  + f"\n  tokens: prompt {g['tokens_prompt'].median():.0f} contra {at['tokens_prompt'].median():.0f}, "
                    f"salida {g['tokens_salida'].median():.0f} contra {at['tokens_salida'].median():.0f} · "
                    f"JSON inválido {int(g['json_invalido'].sum())}")

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
                             f"{len(con_pasajes)} (del ATLS: {int((con_pasajes['pasajes_citados_atls'] > 0).sum())}"
                             f", del ATLS 10: {int((con_pasajes['pasajes_citados_atls10'] > 0).sum())} de "
                             f"{int((con_pasajes['pasajes_atls10'] > 0).sum())} con pasajes suyos)"
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
