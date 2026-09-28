"""
Chequeo antes de una demostración: correrlo en el PC de la demo, con el
chatbot CERRADO (Qdrant local admite un solo proceso), 10 minutos antes.

    python scripts/preparar_demo.py

Qué revisa:
  1. GPU y configuración elegida: VRAM, ruta (en 4 GB debe ser rrf: el
     reranker no cabe junto al generador) y umbral de abstención.
  2. Ollama: que tenga el modelo, lo carga y mide cuánto quedó en la GPU
     (debe ser 100 %; si no, algo más ocupa la VRAM: cerrar el navegador
     con aceleración por hardware, juegos, etc.).
  3. Los botones de la demo (eval/consultas.yaml, campo `boton`) por el
     sistema completo, como en la app: tiempo, si se abstuvo (solo el de
     "fuera de alcance" debe hacerlo), si la prosa pasó la verificación y
     que no haya citas inválidas.

Al final deja el modelo cargado (KEEP_ALIVE) para que la primera pregunta de
la demo no espere la carga.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from rag import config                                                      # noqa: E402
from rag.build_prompt import SISTEMA_PROSA                                  # noqa: E402
from rag.generate import _ollama, ollama_disponible, porcentaje_gpu, responder  # noqa: E402
from rag.retrieve import dispositivo_reranker, ruta_por_defecto, vram_gb   # noqa: E402

OK, MAL, AVISO = "  [ok]", "  [!!]", "  [ ?]"


def main() -> int:
    problemas = 0
    print("1 · GPU y configuración")
    vram = vram_gb()
    ruta = ruta_por_defecto()
    print(f"{OK if vram else AVISO} VRAM {vram:.1f} GB" + ("" if vram else " (torch no ve una GPU NVIDIA)"))
    print(f"{OK} ruta {ruta} · reranker en {dispositivo_reranker()} · "
          f"umbral de abstención {config.UMBRAL_ABSTENCION} · salida {config.SALIDA} · "
          f"keep_alive {config.KEEP_ALIVE}")
    if 0 < vram < config.VRAM_MIN_RERANK and ruta == "rerank":
        print(f"{MAL} RUTA=rerank con {vram:.1f} GB: el reranker no cabe junto al generador. "
              f"Quitar la variable RUTA.")
        problemas += 1

    print("\n2 · Ollama")
    ok, msg = ollama_disponible()
    print(f"{OK if ok else MAL} {msg}")
    if not ok:
        return 1
    t0 = time.perf_counter()
    _ollama([{"role": "system", "content": SISTEMA_PROSA}, {"role": "user", "content": "ok"}],
            max_tokens=1, timeout=300)
    gpu = porcentaje_gpu()
    print(f"{OK} modelo cargado en {time.perf_counter() - t0:.0f} s")
    if vram and gpu is not None:
        bien = gpu >= 100
        print(f"{OK if bien else MAL} {gpu} % en GPU"
              + ("" if bien else " → algo más ocupa la VRAM (navegador, juegos). Cerrarlo y reintentar; "
                                 "si sigue, $env:NUM_GPU = \"-1\" (más lento)."))
        problemas += not bien

    print("\n3 · Botones de la demo (pipeline completo)")
    consultas = yaml.safe_load((RAIZ / "eval" / "consultas.yaml").read_text(encoding="utf-8"))["consultas"]
    for c in (c for c in consultas if c.get("boton")):
        espera = bool(c.get("espera_abstencion"))
        t0 = time.perf_counter()
        r = responder(c["texto"], reescribir_consulta=False)
        s = time.perf_counter() - t0
        fallos = []
        if r.error:
            fallos.append(r.error)
        if r.abstuvo != espera:
            fallos.append(f"{'no ' if espera else ''}se abstuvo (coseno {r.confianza:.3f})")
        if r.citas_invalidas:
            fallos.append(f"citas inválidas {r.citas_invalidas}")
        if r.abstuvo:
            detalle = f"se abstiene (coseno {r.confianza:.3f})"
        elif r.prosa_verificada:
            detalle = "prosa verificada"
        else:
            detalle = f"versión determinista ({r.motivos[0] if r.motivos else 'sin prosa'})"
        print(f"{MAL if fallos else OK} {c['boton']:28} {s:5.1f} s  {detalle}")
        for f in fallos:
            print(f"        {f}")
        problemas += bool(fallos)

    print(f"\n{'Todo listo' if not problemas else f'{problemas} problema(s): ver arriba'}. "
          f"Abrir el chatbot desde rag_pipeline\\: streamlit run app.py")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
