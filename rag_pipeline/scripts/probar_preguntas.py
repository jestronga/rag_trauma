"""
Mide las preguntas de rag/preguntas.py (PLAN_CHATBOT_CLINICO.md, fase 3) en
las consultas de eval/consultas.yaml, sin generar: solo recupera
(generate.recuperar), como respondería la app.

Qué mide:
  · en las de trauma: cuántas preguntas salen (2 como máximo) y cuáles;
  · ninguna por algo que la consulta ya dice (preguntas.ya_dicho);
  · para cada pregunta, responderla con cada opción (como un seguimiento de la
    ficha) y ver si cambian los casos o los pasajes: si ninguna opción cambia
    nada, la pregunta no aporta;
  · en las fuera de dominio: las preguntas de antes de responder (mecanismo y
    región) cuando las dos guardas se abstienen.

Escribe reports/probar_preguntas.csv con una columna vacía para el médico:
"¿es la pregunta que usted haría?".

Uso (con el chatbot cerrado, o INDICE_DIR apuntando a una copia):
    python scripts/probar_preguntas.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from rag import config                                 # noqa: E402
from rag.ficha import actualizar                       # noqa: E402
from rag.generate import recuperar                     # noqa: E402
from rag.preguntas import preguntar, ya_dicho          # noqa: E402


def huella(rec) -> tuple[frozenset, frozenset]:
    return (frozenset(c.record_id for c in rec.casos), frozenset(f.seccion_id for f in rec.fragmentos))


def main() -> None:
    ap = argparse.ArgumentParser(description="Preguntas: cuáles salen y si cambian lo recuperado")
    ap.add_argument("--salida", type=Path, default=RAIZ / "reports" / "probar_preguntas.csv")
    args = ap.parse_args()
    config.CONVERSACION, config.PREGUNTAS = "ficha", True
    consultas = yaml.safe_load((RAIZ / "eval" / "consultas.yaml").read_text(encoding="utf-8"))["consultas"]

    recuperar("calentamiento")
    filas, por_respuesta, claves = [], [], Counter()
    for c in consultas:
        _, ficha = actualizar(None, c["texto"])
        base = recuperar(c["texto"])
        abstuvo = not base.casos and not base.fragmentos
        preguntas = preguntar(ficha, base.casos, c["texto"], abstuvo)
        por_respuesta.append(len(preguntas))
        fuera = bool(c.get("espera_abstencion"))
        print(f"{c['id']:7} {'(abstiene) ' if abstuvo else ''}{[p.clave for p in preguntas]}  {c['texto'][:60]}")
        for p in preguntas:
            claves[p.clave] += 1
            cambian = []
            for o in p.opciones:
                if o.mensaje is None:
                    continue
                _, despues = actualizar(ficha, o.mensaje, forzar="seguimiento")
                rec = recuperar(despues.consulta(o.mensaje))
                casos_b, pasajes_b = huella(base)
                casos_d, pasajes_d = huella(rec)
                cambia = (casos_b != casos_d) or (pasajes_b != pasajes_d)
                cambian.append(cambia)
                filas.append({"id": c["id"], "consulta": c["texto"], "fuera_de_dominio": fuera,
                              "abstiene": abstuvo, "pregunta": p.texto, "clave": p.clave,
                              "opcion": o.etiqueta, "busqueda": despues.consulta(o.mensaje),
                              "casos_nuevos": len(casos_d - casos_b), "pasajes_nuevos": len(pasajes_d - pasajes_b),
                              "responde": bool(rec.casos or rec.fragmentos), "cambia": cambia,
                              "ya_dicho": ya_dicho(p, c["texto"]), "medico_la_haria": ""})
            print(f"        {p.clave:22} cambia con {sum(cambian)} de {len(cambian)} opciones")

    df = pd.DataFrame(filas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False, encoding="utf-8")

    trauma = df[~df.fuera_de_dominio]
    por_pregunta = trauma.groupby(["id", "clave"]).cambia.any()
    print(f"\nTrauma ({sum(not c.get('espera_abstencion') for c in consultas)} consultas): "
          f"{len(por_pregunta)} preguntas · máximo {max(por_respuesta)} por respuesta · "
          f"ya dichas {int(trauma.groupby(['id', 'clave']).ya_dicho.any().sum())}")
    print(f"  respuestas (opciones) que cambian lo recuperado: {int(trauma.cambia.sum())} de {len(trauma)} "
          f"({trauma.cambia.mean():.0%}) · preguntas con alguna opción que cambia: "
          f"{int(por_pregunta.sum())} de {len(por_pregunta)}")
    for clave, g in trauma.groupby("clave"):
        print(f"    {clave:24} {g.groupby('id').ngroups:>2} veces · cambian {int(g.cambia.sum())} de {len(g)} opciones")
    fuera = df[df.fuera_de_dominio]
    if len(fuera):
        pasan = sorted(set(fuera[fuera.responde].id))
        print(f"\nFuera de dominio: {fuera.groupby('id').ngroups} con preguntas de antes de responder · "
              f"responderían al contestarlas: {pasan or 'ninguna'}")
    print(f"\nCSV: {args.salida}")


if __name__ == "__main__":
    main()
