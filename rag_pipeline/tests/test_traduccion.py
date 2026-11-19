"""El ATLS 11 se busca con la consulta en inglés (TRADUCIR_ATLS11, 18/11/2026):
consulta.en_ingles y su uso en retrieve.candidatos_guias, sin Ollama ni Qdrant."""
from __future__ import annotations

import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

from rag import config, consulta, retrieve
from rag.consulta import en_ingles, limpiar_traduccion


# ---------------------------------------------------------------------------
# limpiar_traduccion y en_ingles
# ---------------------------------------------------------------------------
def test_se_queda_con_la_primera_linea():
    assert limpiar_traduccion("HPAF en tórax", "Gunshot wound to the thorax") == "Gunshot wound to the thorax"
    # visto el 18/11 con proc-02: en vez de traducir, respondió la pregunta
    crudo = "Closed thoracostomy technique step-by-step:\n\n1. Position the patient supine"
    assert limpiar_traduccion("Técnica de toracostomía cerrada paso a paso", crudo) == \
        "Closed thoracostomy technique step-by-step"
    assert limpiar_traduccion("TCE leve", '"Mild traumatic brain injury"') == "Mild traumatic brain injury"


@pytest.mark.parametrize("crudo", ["", "   \n ", "¿?", "x" * 200, "Glasgow"])
def test_descarta_lo_que_no_es_traduccion(crudo):
    # vacía, sin letras, mucho más larga que la consulta, o la misma consulta
    assert limpiar_traduccion("Glasgow", crudo) is None


@pytest.fixture
def traductor(monkeypatch):
    monkeypatch.setattr(config, "TRADUCIR_ATLS11", True)
    llamadas = []

    def falso(texto):
        llamadas.append(texto)
        return "Gunshot wound to the abdomen"
    monkeypatch.setattr(consulta, "_pedir_traduccion", falso)
    return llamadas


def test_en_ingles_traduce_una_vez_por_consulta(traductor):
    assert en_ingles("HPAF en abdomen") == "Gunshot wound to the abdomen"
    assert en_ingles("HPAF en abdomen") == "Gunshot wound to the abdomen"
    assert traductor == ["HPAF en abdomen"]


def test_apagada_no_llama_al_modelo(traductor, monkeypatch):
    monkeypatch.setattr(config, "TRADUCIR_ATLS11", False)
    assert en_ingles("HPAF en abdomen") is None and traductor == []


def test_sin_ollama_no_traduce_y_reintenta(monkeypatch):
    monkeypatch.setattr(config, "TRADUCIR_ATLS11", True)
    intentos = []

    def caido(texto):
        intentos.append(texto)
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(consulta, "_pedir_traduccion", caido)
    assert en_ingles("HPAF en abdomen") is None
    assert en_ingles("HPAF en abdomen") is None
    assert len(intentos) == 2                     # la falla no se guarda


# ---------------------------------------------------------------------------
# candidatos_guias: el ATLS 11 con la consulta en inglés, los demás en español
# ---------------------------------------------------------------------------
def _punto(fid: str, score: float, fuente: str) -> SimpleNamespace:
    return SimpleNamespace(score=score, payload={
        "record_id": fid, "seccion_id": fid.rsplit("-", 1)[0], "fuente": fuente, "libro": fuente,
        "capitulo": "Cap", "titulos": ["T"], "pagina_ini": 1, "pagina_fin": 1, "tipo": "texto",
        "idioma": "en" if fuente == "ATLS" else "es", "text": f"texto {fid}",
        "texto_contexto": f"texto {fid}", "n_tokens": 200, "n_tokens_contexto": 200})


class _Qdrant:
    """Filtra por `must` (un libro) y `must_not` (libros fuera) y anota con
    qué vector se buscó cada vez."""
    def __init__(self, puntos):
        self.puntos, self.llamadas = puntos, []

    def query_points(self, coleccion, query, limit, with_payload, query_filter=None):
        puntos = self.puntos
        if query_filter is not None and query_filter.must:
            puntos = [p for p in puntos if p.payload["fuente"] == query_filter.must[0].match.value]
        if query_filter is not None and query_filter.must_not:
            fuera = set(query_filter.must_not[0].match.any)
            puntos = [p for p in puntos if p.payload["fuente"] not in fuera]
        self.llamadas.append((query[0], sorted({p.payload["fuente"] for p in puntos})))
        return SimpleNamespace(points=puntos[:limit])


@pytest.fixture
def libros(monkeypatch):
    falso = _Qdrant([_punto("HUV-11-007-01", 0.70, "HUV"), _punto("ATLS10-04-007-01", 0.65, "ATLS10"),
                     _punto("ATLS-05-012-01", 0.40, "ATLS")])
    monkeypatch.setattr(retrieve, "_qdrant", lambda: falso)
    monkeypatch.setattr(retrieve, "hay_guias", lambda: True)
    monkeypatch.setattr(retrieve, "libros_indexados", lambda: ("ATLS", "ATLS10", "HUV"))
    monkeypatch.setattr(config, "ATLS10", True)
    monkeypatch.setattr(config, "ABREV_LIBROS", False)
    monkeypatch.setattr(config, "MECANISMO_LIBROS", False)
    # el "vector" dice en qué idioma se buscó: 1 inglés, 0 español
    monkeypatch.setattr(retrieve, "_vector_consulta",
                        lambda texto: np.array([1.0 if texto.startswith("Gunshot") else 0.0]))
    pares = []

    def rerank(ps):
        pares.extend(ps)
        return np.array([0.9 if q.startswith("Gunshot") else 0.5 for q, _ in ps])
    monkeypatch.setattr(retrieve, "_probabilidades", rerank)
    return SimpleNamespace(qdrant=falso, pares=pares)


def test_el_atls11_se_busca_y_reordena_en_ingles(libros, traductor):
    frags = retrieve.candidatos_guias("HPAF en abdomen", "rerank", por_libro=False)
    # dos búsquedas: los libros en español con la consulta de siempre, el ATLS 11 traducida
    assert libros.qdrant.llamadas == [(0.0, ["ATLS10", "HUV"]), (1.0, ["ATLS"])]
    assert dict((t, q) for q, t in libros.pares) == {
        "texto HUV-11-007-01": "HPAF en abdomen", "texto ATLS10-04-007-01": "HPAF en abdomen",
        "texto ATLS-05-012-01": "Gunshot wound to the abdomen"}
    assert frags[0].fuente == "ATLS" and frags[0].busqueda == "Gunshot wound to the abdomen"
    assert {f.busqueda for f in frags[1:]} == {"HPAF en abdomen"}


def test_por_libro_tambien_traduce_solo_el_atls11(libros, traductor):
    retrieve.candidatos_guias("HPAF en abdomen", "rerank", por_libro=True)
    assert sorted(libros.qdrant.llamadas) == [(0.0, ["ATLS10"]), (0.0, ["HUV"]), (1.0, ["ATLS"])]


def test_sin_traduccion_una_sola_busqueda_como_antes(libros, monkeypatch):
    monkeypatch.setattr(config, "TRADUCIR_ATLS11", True)
    monkeypatch.setattr(consulta, "_pedir_traduccion", lambda texto: "")     # traducción inservible
    retrieve.candidatos_guias("HPAF en abdomen", "rerank", por_libro=False)
    assert libros.qdrant.llamadas == [(0.0, ["ATLS", "ATLS10", "HUV"])]
    assert {q for q, _ in libros.pares} == {"HPAF en abdomen"}


def test_la_densa_no_traduce(libros, traductor):
    retrieve.candidatos_guias("HPAF en abdomen", "densa", por_libro=False)
    assert traductor == [] and len(libros.qdrant.llamadas) == 1


def test_sin_el_atls11_activo_no_traduce(libros, traductor, monkeypatch):
    monkeypatch.setattr(retrieve, "libros_indexados", lambda: ("ATLS10", "HUV"))
    retrieve.candidatos_guias("HPAF en abdomen", "rerank", por_libro=False)
    assert traductor == []
