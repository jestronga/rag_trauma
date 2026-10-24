"""
El registro como epidemiología (PLAN_RESPUESTA_IDEAL.md, fase 2): por cada
tarjeta, cuántos casos del registro tienen ese mecanismo, qué regiones, qué
hallazgos y manejo, y qué desenlace.

    python scripts/patrones_registro.py

Escribe tarjetas/patrones_registro.json (solo conteos: ningún texto de casos)
y rag/tarjetas.linea_registro lo pone arriba de los casos. Corre en segundos y
sin modelos; hay que volver a correrlo si se regenera el corpus.

Las regiones y el desenlace los pone el pipeline de anonimización con reglas
(anonimizacion/src/enrich.py): el 35 % de los casos aptos no tiene desenlace
(pendiente 7 de CONTEXTO_SESION.md), por eso se cuenta solo entre los que lo
tienen. Los hallazgos y el manejo son el léxico de rag/hallazgos.py, con su
negación, el mismo que lee los casos recuperados.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config                                   # noqa: E402
from rag.corpus import cargar, normalizar                # noqa: E402
from rag.hallazgos import terminos                       # noqa: E402

# El peatón se reconoce por el texto: el clasificador lo deja en "accidente de
# tránsito" (291 de 324 el 24/10) o sin determinar (32).
RX_PEATON = re.compile(r"atropell|arrollad|peaton")
NOMBRES = {
    "peaton": "peatón atropellado",
    "ocupante": "accidente de tránsito sin peatón",
    "caida": "caída",
    "arma_de_fuego": "herida por arma de fuego",
    "cortopunzante": "herida por arma cortopunzante",
}
DESENLACES = ("hospitalizacion", "cirugia", "uci", "fallecimiento", "remision", "alta")


def cohorte(doc) -> str | None:
    if RX_PEATON.search(normalizar(doc.text)):
        return "peaton"
    mecanismo = doc.metadata.get("mecanismo_lesion")
    if mecanismo == "accidente_transito":
        return "ocupante"
    return mecanismo if mecanismo in NOMBRES else None


def main() -> None:
    docs = cargar()
    grupos: dict[str, list] = {k: [] for k in NOMBRES}
    for d in docs:
        if (k := cohorte(d)) is not None:
            grupos[k].append(d)
    salida = {}
    for clave, casos in grupos.items():
        regiones, elementos, desenlace = Counter(), Counter(), Counter()
        for d in casos:
            regiones.update(set(d.metadata.get("region_anatomica") or []))
            elementos.update({t for ts in terminos(d.text).values() for t in ts})
            desenlace[d.metadata.get("desenlace") or "no_determinado"] += 1
        conocido = sum(n for k, n in desenlace.items() if k != "no_determinado")
        salida[clave] = {
            "nombre": NOMBRES[clave],
            "n": len(casos),
            "regiones": [[config.ETIQUETAS_REGION.get(r, r).lower(), n] for r, n in regiones.most_common(6)],
            "hallazgos": [[t, n] for t, n in elementos.most_common(6)],
            "desenlace": {"conocido": conocido, **{k: desenlace[k] for k in DESENLACES}},
        }
        print(f"{clave:15s} {len(casos):5d} casos · {salida[clave]['regiones'][:3]} · "
              f"{salida[clave]['hallazgos'][:3]} · desenlace conocido {conocido}")
    config.PATRONES_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    config.PATRONES_REGISTRO.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n",
                                        encoding="utf-8")
    print(f"\n→ {config.PATRONES_REGISTRO}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
