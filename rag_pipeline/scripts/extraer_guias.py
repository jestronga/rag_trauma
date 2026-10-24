"""
Extrae las secciones de los libros de trauma (fase 1 de PLAN_INDEXAR_LIBROS.md).

    python scripts/extraer_guias.py                 # todos los libros
    python scripts/extraer_guias.py --libro huv     # uno (clave de guias/perfiles.yaml)
    python scripts/extraer_guias.py --muestra 30    # + muestra para revisar a mano

Lee los PDF de LIBROS_DIR (anonimizacion/data/) con los perfiles de
guias/perfiles.yaml y escribe guias/secciones.jsonl. Todo queda fuera de git:
los libros tienen derechos de autor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config  # noqa: E402
from rag.guias import Extractor, cargar_perfiles, cargar_transcripciones  # noqa: E402


def huella(registros: list[dict]) -> str:
    """IDs y títulos de las secciones: si cambia, los IDs anotados en
    eval/consultas_guias.yaml pueden apuntar a otras secciones."""
    ids = [r["id"] + "|" + " > ".join(r["titulos"]) for r in registros]
    return f"{len(ids)} {hashlib.sha1(chr(10).join(ids).encode()).hexdigest()[:12]}"


def main() -> None:
    perfiles = cargar_perfiles()
    ap = argparse.ArgumentParser(description="Extrae las secciones de los libros de trauma")
    ap.add_argument("--libro", choices=[*perfiles, "todos"], default="todos")
    ap.add_argument("--salida", type=Path, default=config.GUIAS_SECCIONES)
    ap.add_argument("--muestra", type=int, default=0,
                    help="escribe N secciones al azar en guias/muestra_revision.md")
    args = ap.parse_args()

    import pymupdf

    transcripciones = cargar_transcripciones()
    if not transcripciones:
        print(f"[aviso] falta {config.GUIAS_TRANSCRIPCIONES}: los flujogramas del HUV entran como "
              f"recuadros sueltos y faltan las tablas en imagen. Copiarlo junto a los PDF.")
    claves = list(perfiles) if args.libro == "todos" else [args.libro]
    registros = []
    for clave in claves:
        perfil = perfiles[clave]
        ruta = config.LIBROS_DIR / perfil["archivo"]
        if not ruta.exists():
            raise SystemExit(f"Falta {ruta}")
        extractor = Extractor(perfil, transcripciones.get(clave))
        with pymupdf.open(ruta) as doc:
            secciones = list(extractor.secciones(doc))
        registros += secciones
        largos = [len(s["texto"]) for s in secciones]
        caps = {s["capitulo_num"] for s in secciones}
        print(f"{perfil['fuente']}: {len(secciones)} secciones en {len(caps)} capítulos · "
              f"{sum(largos):,} caracteres (mediana {statistics.median(largos):.0f}, "
              f"máx. {max(largos):,})")
        st = extractor.stats
        print(f"    descartado: referencias {st['referencias']:,} · material sin pie "
              f"{st['material_sin_pie']:,} · pies de foto y dibujo {st['figuras']:,} · "
              f"otros {st['descartado']:,} caracteres")
        if st["transcritas"]:
            print(f"    transcritas a mano: {st['transcritas']} páginas (reemplazan "
                  f"{st['reemplazado']:,} caracteres de cajas sueltas)")
        sin_texto = sorted({c for c in (n for n, *_ in perfil["capitulos"])} - caps)
        if sin_texto:
            print(f"    ¡capítulos sin secciones!: {sin_texto}")
        print(f"    huella: {huella(secciones)}")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"→ {args.salida} ({len(registros)} secciones)")

    if args.muestra:
        random.seed(7)
        muestra = random.sample(registros, min(args.muestra, len(registros)))
        ruta = args.salida.parent / "muestra_revision.md"
        with ruta.open("w", encoding="utf-8") as f:
            f.write("# Muestra para revisar contra el PDF\n\n"
                    "Por cada sección: ¿falta texto, sobra texto (encabezados, "
                    "referencias, rótulos de figura), o hay palabras cortadas?\n\n")
            for r in sorted(muestra, key=lambda r: r["id"]):
                ruta_t = " › ".join(r["titulos"]) or "(sin título)"
                f.write(f"## {r['id']} — {r['fuente']} cap. {r['capitulo_num']}, "
                        f"p. {r['pagina_ini']}-{r['pagina_fin']} (PDF {r['pagina_pdf_ini']})\n\n"
                        f"**{ruta_t}**\n\n{r['texto']}\n\n---\n\n")
        print(f"→ {ruta}")


if __name__ == "__main__":
    main()
