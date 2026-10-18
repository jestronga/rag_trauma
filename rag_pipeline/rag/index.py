"""
Construcción de los índices.

  · denso   — bge-m3 (1024d, coseno) en Qdrant modo LOCAL embebido.
              Sin Docker y sin servicio: Qdrant escribe en `indices/qdrant/`.
              Para pasar a Qdrant servidor solo cambia el constructor del
              cliente; el resto del código no se entera.

  · disperso — BM25 Okapi sobre tokens normalizados sin tildes (solo casos).

Dos fuentes, cada una en su colección del mismo Qdrant: los casos
(COLECCION) y los libros de trauma (COLECCION_GUIAS, PLAN_INDEXAR_LIBROS.md
fase 3). Reindexar una fuente reemplaza solo su colección: la otra no se toca.
Los libros no llevan BM25 (fase 2: en una lista fusionada saca al ATLS del
presupuesto).

Uso:
    python -m rag.index                   # los casos (BM25 + denso)
    python -m rag.index --fuente guias    # los libros (antes: scripts/extraer_guias.py)
    python -m rag.index --fuente todas
    python -m rag.index --limite 2000     # subconjunto, para probar rápido
    python -m rag.index --solo bm25       # solo el índice disperso de los casos (segundos)
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from pathlib import Path

from . import config
from . import fragmentar as fr
from .corpus import Documento, cargar, tokenizar
from .encoder import cargar_encoder, dispositivo as _dispositivo


# ---------------------------------------------------------------------------
# Índice disperso
# ---------------------------------------------------------------------------
def construir_bm25(docs: list[Documento], salida: Path) -> None:
    from rank_bm25 import BM25Okapi

    corpus_tok = [tokenizar(d.text) for d in docs]
    bm25 = BM25Okapi(corpus_tok)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": [d.id for d in docs]}, f)


# ---------------------------------------------------------------------------
# Índice denso
# ---------------------------------------------------------------------------
def escribir_coleccion(nombre: str, vectores, payloads: list[dict],
                       indices: list[tuple[str, str]] = ()) -> None:
    """Crea la colección `nombre` (o la reemplaza si existe) con un punto por
    vector. Las demás colecciones del Qdrant local no se tocan: antes se
    borraba la carpeta entera, y reindexar los casos habría borrado los libros."""
    from qdrant_client import QdrantClient, models

    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    cliente = QdrantClient(path=str(config.QDRANT_PATH))
    try:
        if cliente.collection_exists(nombre):
            cliente.delete_collection(nombre)
        cliente.create_collection(
            collection_name=nombre,
            vectors_config=models.VectorParams(
                size=vectores.shape[1], distance=models.Distance.COSINE
            ),
        )
        # payload indexado: permite filtrar por metadatos antes de buscar (en
        # el modo local no tiene efecto; sirve al pasar a Qdrant servidor)
        for campo, tipo in indices:
            cliente.create_payload_index(nombre, field_name=campo,
                                         field_schema=models.PayloadSchemaType(tipo))
        puntos = [
            models.PointStruct(id=i, vector=vectores[i].tolist(), payload=payload)
            for i, payload in enumerate(payloads)
        ]
        for inicio in range(0, len(puntos), 512):
            cliente.upsert(nombre, points=puntos[inicio:inicio + 512])
    finally:
        cliente.close()


def _embeber(encoder, textos: list[str], lote: int):
    t0 = time.time()
    vectores = encoder.encode(
        textos,
        batch_size=lote,
        normalize_embeddings=True,     # coseno == producto punto
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"    embeddings: {vectores.shape} en {time.time() - t0:.0f}s")
    return vectores


def construir_denso(docs: list[Documento], lote: int = 32, dispositivo: str | None = None,
                    encoder=None) -> None:
    encoder = encoder or cargar_encoder(dispositivo)
    vectores = _embeber(encoder, [d.text for d in docs], lote)
    escribir_coleccion(
        config.COLECCION, vectores,
        [{"record_id": d.id, "text": d.text, **d.metadata} for d in docs],
        indices=[("mecanismo_lesion", "keyword"), ("region_anatomica", "keyword"),
                 ("desenlace", "keyword"), ("es_pediatrico", "bool"), ("truncado", "bool")],
    )


# ---------------------------------------------------------------------------
# Libros de trauma
# ---------------------------------------------------------------------------
def cargar_secciones(ruta: Path | None = None) -> list[dict]:
    ruta = Path(ruta or config.GUIAS_SECCIONES)
    if not ruta.exists():
        raise SystemExit(
            f"No encuentro las secciones de los libros en {ruta}.\n"
            "Extráelas con:  python scripts/extraer_guias.py  (los PDF van en anonimizacion/data/)"
        )
    with ruta.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def fragmentos_guias(secciones: list[dict], nombre: str, contar: fr.Contador,
                     embeber: fr.Embebedor | None = None) -> list[dict]:
    """Fragmentos de `nombre` con lo que la cita y la app necesitan de su
    sección (libro, capítulo, títulos). Falla si alguno pasa del límite de
    bge-m3: el encoder lo truncaría sin avisar y su final no se recuperaría."""
    por_id = {s["id"]: s for s in secciones}
    frags = fr.fragmentar(secciones, nombre, contar, embeber=embeber)
    largos = [f["id"] for f in frags if f["n_tokens"] > fr.LIMITE]
    if largos:
        raise SystemExit(f"{len(largos)} fragmentos pasan de {fr.LIMITE} tokens "
                         f"(p. ej. {largos[:3]}): revisar rag/fragmentar.py")
    for f in frags:
        s = por_id[f["seccion_id"]]
        f.update(libro=s["libro"], capitulo=s["capitulo"], titulos=s["titulos"])
    return frags


def payload_guia(frag: dict) -> dict:
    """`record_id` y `text` como en los casos; `text` es lo que se embebe (con
    la ruta de títulos) y `texto_contexto` lo que iría al prompt."""
    resto = {k: v for k, v in frag.items() if k not in ("id", "texto")}
    return {"record_id": frag["id"], "text": frag["texto"], **resto}


def resumen_guias(frags: list[dict], secciones: list[dict]) -> dict:
    """Lo que se imprime y queda en meta.json: tamaños por libro y secciones
    sin ningún fragmento (no deberían quedar: toda sección tiene texto o tabla)."""
    salida = {"n_fragmentos": len(frags), "n_secciones": len(secciones), "por_libro": {}}
    for fuente in sorted({f["fuente"] for f in frags}):
        tams = [f["n_tokens"] for f in frags if f["fuente"] == fuente]
        salida["por_libro"][fuente] = {
            "fragmentos": len(tams),
            "tablas": sum(f["tipo"] == "tabla" for f in frags if f["fuente"] == fuente),
            "tokens_mediana": int(statistics.median(tams)),
            "tokens_max": max(tams),
        }
    con_fragmentos = {f["seccion_id"] for f in frags}
    salida["secciones_sin_fragmentos"] = [s["id"] for s in secciones if s["id"] not in con_fragmentos]
    return salida


def construir_guias(lote: int = 32, dispositivo: str | None = None, limite: int = 0,
                    encoder=None, contar: fr.Contador | None = None) -> dict:
    secciones = cargar_secciones()
    if limite:
        secciones = secciones[:limite]
    encoder = encoder or cargar_encoder(dispositivo)
    contar = contar or fr.contador_bge()

    def embeber(textos: list[str]):      # solo lo usa la configuración "semantico"
        return encoder.encode(textos, batch_size=lote, normalize_embeddings=True,
                              convert_to_numpy=True)

    t0 = time.time()
    frags = fragmentos_guias(secciones, config.FRAGMENTACION_GUIAS, contar, embeber)
    resumen = resumen_guias(frags, secciones)
    print(f"    {config.FRAGMENTACION_GUIAS}: {len(frags):,} fragmentos de "
          f"{len(secciones)} secciones ({time.time() - t0:.0f}s)")
    for fuente, d in resumen["por_libro"].items():
        print(f"      {fuente}: {d['fragmentos']:,} ({d['tablas']} de tablas) · tokens "
              f"mediana {d['tokens_mediana']}, máx. {d['tokens_max']}")
    if resumen["secciones_sin_fragmentos"]:
        print(f"    [aviso] secciones sin fragmentos: {resumen['secciones_sin_fragmentos']}")

    vectores = _embeber(encoder, [f["texto"] for f in frags], lote)
    escribir_coleccion(config.COLECCION_GUIAS, vectores, [payload_guia(f) for f in frags],
                       indices=[("fuente", "keyword"), ("seccion_id", "keyword"),
                                ("clinico", "bool")])
    return resumen


# ---------------------------------------------------------------------------
# meta.json: una entrada por fuente
# ---------------------------------------------------------------------------
def actualizar_meta(fuente: str, datos: dict) -> None:
    """Registra `fuente` sin perder la otra. Nadie lee este archivo desde el
    código: sirve para saber qué se indexó, cuándo y con qué."""
    meta: dict = {}
    if config.META_PATH.exists():
        try:
            meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}
    if "fuentes" not in meta:
        # formato anterior al 17/10/2026: un solo bloque, el de los casos
        meta = {"fuentes": {"casos": meta} if meta else {}}
    meta["fuentes"][fuente] = {
        **datos,
        "modelo_embedding": config.MODELO_EMBEDDING,
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config.META_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye los índices de los casos y de los libros")
    ap.add_argument("--fuente", choices=["casos", "guias", "todas"], default="casos")
    ap.add_argument("--limite", type=int, default=0,
                    help="indexar solo N casos o N secciones de los libros")
    ap.add_argument("--solo", choices=["bm25", "denso"], default=None,
                    help="solo uno de los índices de los casos")
    ap.add_argument("--lote", type=int, default=config.LOTE_INDEXACION)
    ap.add_argument("--dispositivo", default=config.DISPOSITIVO_INDEXACION)
    args = ap.parse_args()
    if args.solo and args.fuente != "casos":
        ap.error("--solo es para los casos: los libros solo tienen índice denso")

    config.INDICE_DIR.mkdir(parents=True, exist_ok=True)
    encoder = None
    if args.solo != "bm25":
        print(f"[0] encoder {config.MODELO_EMBEDDING} "
              f"(dispositivo: {_dispositivo(args.dispositivo)})")
        encoder = cargar_encoder(args.dispositivo)

    if args.fuente in ("casos", "todas"):
        docs = cargar()
        if args.limite:
            docs = docs[: args.limite]
        print(f"[casos] corpus: {len(docs):,} documentos")
        if args.solo != "denso":
            t0 = time.time()
            construir_bm25(docs, config.BM25_PATH)
            print(f"    BM25 → {config.BM25_PATH}  ({time.time() - t0:.0f}s)")
        if args.solo != "bm25":
            construir_denso(docs, lote=args.lote, encoder=encoder)
            print(f"    Qdrant → {config.QDRANT_PATH} (colección {config.COLECCION!r})")
        actualizar_meta("casos", {"n_docs": len(docs), "coleccion": config.COLECCION,
                                  "corpus": str(config.CORPUS_JSONL)})

    if args.fuente in ("guias", "todas"):
        print(f"[guias] secciones: {config.GUIAS_SECCIONES}")
        resumen = construir_guias(lote=args.lote, limite=args.limite, encoder=encoder)
        print(f"    Qdrant → {config.QDRANT_PATH} (colección {config.COLECCION_GUIAS!r})")
        actualizar_meta("guias", {"coleccion": config.COLECCION_GUIAS,
                                  "secciones": str(config.GUIAS_SECCIONES),
                                  "fragmentacion": config.FRAGMENTACION_GUIAS, **resumen})

    print(f"[listo] → {config.META_PATH}")


if __name__ == "__main__":
    main()
