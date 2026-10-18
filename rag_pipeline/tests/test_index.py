"""Indexación (rag/index.py): cada fuente en su colección, sin borrar la otra,
y los fragmentos de los libros con lo que la cita necesita. Con el encoder
hash, un contador falso (una palabra = un token) y un Qdrant en tmp_path."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rag import config, index
from rag.corpus import Documento
from rag.encoder import EncoderHash


def contar(texto: str) -> int:
    return len(texto.split())


def seccion(sid: str, parrafos, anexos=(), fuente="ATLS", clinico=True) -> dict:
    return {"id": sid, "fuente": fuente, "libro": f"Libro {fuente}", "idioma": "en",
            "capitulo_num": 7, "capitulo": "Disability", "clinico": clinico,
            "titulos": ["TRAUMATIC BRAIN INJURY"], "pagina_ini": 108, "pagina_fin": 109,
            "parrafos": list(parrafos), "anexos": list(anexos)}


@pytest.fixture
def qdrant_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(config, "META_PATH", tmp_path / "meta.json")
    return tmp_path


def _contar_puntos(nombre: str) -> int | None:
    from qdrant_client import QdrantClient
    cliente = QdrantClient(path=str(config.QDRANT_PATH))
    try:
        return cliente.count(nombre).count if cliente.collection_exists(nombre) else None
    finally:
        cliente.close()


def _vectores(n: int) -> np.ndarray:
    v = np.random.default_rng(0).normal(size=(n, 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
def test_reindexar_una_fuente_no_borra_la_otra(qdrant_tmp):
    index.escribir_coleccion("trauma", _vectores(3), [{"record_id": str(i)} for i in range(3)])
    index.escribir_coleccion("guias", _vectores(2), [{"record_id": f"g{i}"} for i in range(2)])
    # reindexar los casos (con otro tamaño) reemplaza solo su colección
    index.escribir_coleccion("trauma", _vectores(5), [{"record_id": str(i)} for i in range(5)])
    assert _contar_puntos("trauma") == 5
    assert _contar_puntos("guias") == 2


def test_construir_denso_de_casos_conserva_los_libros(qdrant_tmp):
    index.escribir_coleccion(config.COLECCION_GUIAS, _vectores(2), [{"record_id": "g"}] * 2)
    docs = [Documento("101", "herida por arma de fuego en tórax", {"desenlace": "alta"}),
            Documento("102", "caída de propia altura", {"desenlace": "cirugia"})]
    index.construir_denso(docs, encoder=EncoderHash())
    assert _contar_puntos(config.COLECCION) == 2
    assert _contar_puntos(config.COLECCION_GUIAS) == 2


def test_construir_guias_indexa_los_fragmentos_con_su_cita(qdrant_tmp, monkeypatch):
    ruta = qdrant_tmp / "secciones.jsonl"
    secciones = [seccion("ATLS-07-001", ["Word " * 300], anexos=["Table 7-1: GCS\nEye 4\nVerbal 5"]),
                 seccion("HUV-02-001", ["Palabra " * 40], fuente="HUV", clinico=False)]
    ruta.write_text("\n".join(json.dumps(s) for s in secciones), encoding="utf-8")
    monkeypatch.setattr(config, "GUIAS_SECCIONES", ruta)

    resumen = index.construir_guias(encoder=EncoderHash(), contar=contar)

    assert resumen["secciones_sin_fragmentos"] == []
    assert _contar_puntos(config.COLECCION_GUIAS) == resumen["n_fragmentos"]
    from qdrant_client import QdrantClient
    cliente = QdrantClient(path=str(config.QDRANT_PATH))
    try:
        puntos, _ = cliente.scroll(config.COLECCION_GUIAS, limit=100, with_payload=True)
    finally:
        cliente.close()
    pays = {p.payload["record_id"]: p.payload for p in puntos}
    primero = pays["ATLS-07-001-01"]
    # la ruta de títulos va al inicio de lo que se embebe
    assert primero["text"].startswith("ATLS › Disability › TRAUMATIC BRAIN INJURY\n")
    assert (primero["libro"], primero["capitulo"], primero["pagina_ini"]) == ("Libro ATLS", "Disability", 108)
    assert primero["seccion_id"] == "ATLS-07-001"
    # fijo-256: 300 palabras no caben en una ventana; la tabla va aparte
    assert sum(p["seccion_id"] == "ATLS-07-001" and p["tipo"] == "texto" for p in pays.values()) == 2
    assert any(p["tipo"] == "tabla" for p in pays.values())
    assert pays["HUV-02-001-01"]["clinico"] is False


def test_fragmentos_guias_falla_si_alguno_pasa_el_limite():
    largo = lambda t: 600 if "LARGA" in t else contar(t)      # noqa: E731
    with pytest.raises(SystemExit, match="pasan de"):
        index.fragmentos_guias([seccion("ATLS-07-001", ["LARGA"])], "fijo-256", largo)


def test_n_tokens_contexto_cuenta_lo_que_va_al_prompt():
    secciones = [seccion("ATLS-07-001", ["Word " * 300])]
    fijo = index.fragmentos_guias(secciones, "fijo-256", contar)
    assert all(f["n_tokens_contexto"] == f["n_tokens"] for f in fijo)
    # small-to-big: se embebe un hijo de ~128 y al prompt va el padre, más largo
    hijos = index.fragmentos_guias(secciones, "oraciones-128", contar)
    assert all(f["n_tokens_contexto"] == contar(f["texto_contexto"]) > f["n_tokens"] for f in hijos)


def test_cargar_secciones_sin_extraer_dice_que_correr(tmp_path):
    with pytest.raises(SystemExit, match="extraer_guias"):
        index.cargar_secciones(tmp_path / "no_existe.jsonl")


def test_actualizar_meta_conserva_la_otra_fuente_y_migra_el_formato_viejo(qdrant_tmp):
    viejo = {"n_docs": 8198, "coleccion": "trauma", "generado": "2026-10-17T21:19:58"}
    config.META_PATH.write_text(json.dumps(viejo), encoding="utf-8")
    index.actualizar_meta("guias", {"coleccion": "guias", "n_fragmentos": 10})
    meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    assert meta["fuentes"]["casos"]["n_docs"] == 8198
    assert meta["fuentes"]["guias"]["n_fragmentos"] == 10
    index.actualizar_meta("casos", {"n_docs": 5})
    meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    assert meta["fuentes"]["casos"]["n_docs"] == 5
    assert meta["fuentes"]["guias"]["n_fragmentos"] == 10
