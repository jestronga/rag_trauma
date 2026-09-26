"""
Construcción de los dos índices.

  · denso   — bge-m3 (1024d, coseno) en Qdrant modo LOCAL embebido.
              Sin Docker y sin servicio: Qdrant escribe en `indices/qdrant/`.
              Para pasar a Qdrant servidor solo cambia el constructor del
              cliente; el resto del código no se entera.

  · disperso — BM25 Okapi sobre tokens normalizados sin tildes.

Uso:
    python -m rag.index                 # indexa todo
    python -m rag.index --limite 2000   # subconjunto, para probar rápido
    python -m rag.index --solo bm25     # solo el índice disperso (segundos)
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

from . import config
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
def construir_denso(docs: list[Documento], lote: int = 32, dispositivo: str | None = None) -> None:
    from qdrant_client import QdrantClient, models

    encoder = cargar_encoder(dispositivo)
    textos = [d.text for d in docs]

    t0 = time.time()
    vectores = encoder.encode(
        textos,
        batch_size=lote,
        normalize_embeddings=True,     # coseno == producto punto
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"    embeddings: {vectores.shape} en {time.time() - t0:.0f}s")

    config.QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if config.QDRANT_PATH.exists():
        shutil.rmtree(config.QDRANT_PATH)

    cliente = QdrantClient(path=str(config.QDRANT_PATH))
    cliente.create_collection(
        collection_name=config.COLECCION,
        vectors_config=models.VectorParams(
            size=vectores.shape[1], distance=models.Distance.COSINE
        ),
    )
    # payload indexado: permite filtrar por metadatos antes de buscar
    for campo, tipo in [
        ("mecanismo_lesion", models.PayloadSchemaType.KEYWORD),
        ("region_anatomica", models.PayloadSchemaType.KEYWORD),
        ("desenlace", models.PayloadSchemaType.KEYWORD),
        ("es_pediatrico", models.PayloadSchemaType.BOOL),
        ("truncado", models.PayloadSchemaType.BOOL),
    ]:
        cliente.create_payload_index(config.COLECCION, field_name=campo, field_schema=tipo)

    puntos = [
        models.PointStruct(
            id=i,
            vector=vectores[i].tolist(),
            payload={"record_id": d.id, "text": d.text, **d.metadata},
        )
        for i, d in enumerate(docs)
    ]
    for inicio in range(0, len(puntos), 512):
        cliente.upsert(config.COLECCION, points=puntos[inicio:inicio + 512])
    cliente.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye los índices denso y disperso")
    ap.add_argument("--limite", type=int, default=0, help="indexar solo N documentos")
    ap.add_argument("--solo", choices=["bm25", "denso"], default=None)
    ap.add_argument("--lote", type=int, default=config.LOTE_INDEXACION)
    ap.add_argument("--dispositivo", default=config.DISPOSITIVO_INDEXACION)
    args = ap.parse_args()

    docs = cargar()
    if args.limite:
        docs = docs[: args.limite]
    print(f"[0] corpus: {len(docs):,} documentos")

    config.INDICE_DIR.mkdir(parents=True, exist_ok=True)

    if args.solo != "denso":
        t0 = time.time()
        construir_bm25(docs, config.BM25_PATH)
        print(f"[1] BM25 → {config.BM25_PATH}  ({time.time() - t0:.0f}s)")

    if args.solo != "bm25":
        print(f"[2] denso con {config.MODELO_EMBEDDING} "
              f"(dispositivo: {_dispositivo(args.dispositivo)})")
        construir_denso(docs, lote=args.lote, dispositivo=args.dispositivo)
        print(f"    Qdrant → {config.QDRANT_PATH}")

    config.META_PATH.write_text(
        json.dumps(
            {
                "n_docs": len(docs),
                "modelo_embedding": config.MODELO_EMBEDDING,
                "coleccion": config.COLECCION,
                "corpus": str(config.CORPUS_JSONL),
                "generado": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[3] listo → {config.META_PATH}")


if __name__ == "__main__":
    main()
