"""Recuperación en los libros (rag/retrieve.py, fase 4): guarda por el mejor
fragmento, presupuesto de tokens y reranker, con un Qdrant falso."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rag import retrieve
from rag.retrieve import FragmentoGuia, seleccionar_guias


def frag(fid: str, coseno: float, relevancia: float | None = None, tokens: int = 200,
         contexto: str | None = None) -> FragmentoGuia:
    return FragmentoGuia(id=fid, seccion_id=fid.rsplit("-", 1)[0], fuente="ATLS", libro="ATLS",
                         capitulo="Breathing", titulos=["HEMOTHORAX"], pagina_ini=80, pagina_fin=81,
                         tipo="texto", idioma="en", texto=f"texto {fid}",
                         texto_contexto=contexto or f"texto {fid}", n_tokens_contexto=tokens,
                         coseno=coseno, relevancia=relevancia)


# ---------------------------------------------------------------------------
# seleccionar_guias: guarda y presupuesto
# ---------------------------------------------------------------------------
def test_si_el_mejor_no_llega_al_umbral_los_libros_se_abstienen():
    assert seleccionar_guias([frag("A-01-001-01", 0.9, 0.15), frag("A-01-001-02", 0.8, 0.1)], 0.2, 1400) == []
    assert seleccionar_guias([], 0.2, 1400) == []


def test_si_el_mejor_pasa_entran_tambien_los_flojos_hasta_el_presupuesto():
    # la guarda es por el mejor: la continuación de una sección relevante puede
    # puntuar bajo leída sola y filtrarla recuperaba menos (0,297 contra 0,331)
    cands = [frag("A-01-001-01", 0.7, 0.9), frag("A-01-001-02", 0.6, 0.05)]
    assert [f.id for f in seleccionar_guias(cands, 0.2, 1400)] == ["A-01-001-01", "A-01-001-02"]


def test_el_presupuesto_corta_en_el_primero_que_no_cabe():
    # como R@ppto en comparar_fragmentacion.py: el tercero (100) cabría, pero no se salta
    cands = [frag("A-01-001-01", 0.7, 0.9, tokens=600), frag("A-01-002-01", 0.7, 0.8, tokens=600),
             frag("A-01-003-01", 0.7, 0.7, tokens=300), frag("A-01-004-01", 0.7, 0.6, tokens=100)]
    assert [f.id for f in seleccionar_guias(cands, 0.2, 1400)] == ["A-01-001-01", "A-01-002-01"]


def test_no_repite_un_contexto_ya_incluido():
    # small-to-big: varios hijos traen el mismo padre al prompt
    cands = [frag("A-01-001-01", 0.7, 0.9, contexto="padre"), frag("A-01-001-02", 0.7, 0.8, contexto="padre"),
             frag("A-01-002-01", 0.7, 0.7)]
    assert [f.id for f in seleccionar_guias(cands, 0.2, 1400)] == ["A-01-001-01", "A-01-002-01"]


def test_el_puntaje_es_la_probabilidad_si_hubo_reranker_y_si_no_el_coseno():
    assert frag("A-01-001-01", 0.61).puntaje == 0.61
    assert frag("A-01-001-01", 0.61, 0.05).puntaje == 0.05


# ---------------------------------------------------------------------------
# candidatos_guias y buscar_guias, con Qdrant y reranker falsos
# ---------------------------------------------------------------------------
def _punto(fid: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(score=score, payload={
        "record_id": fid, "seccion_id": fid.rsplit("-", 1)[0], "fuente": "ATLS", "libro": "ATLS",
        "capitulo": "Breathing", "titulos": ["HEMOTHORAX"], "pagina_ini": 80, "pagina_fin": 81,
        "tipo": "texto", "idioma": "en", "text": f"ATLS › Breathing › HEMOTHORAX\ntexto {fid}",
        "texto_contexto": f"ATLS › Breathing › HEMOTHORAX\ntexto {fid}", "n_tokens": 200,
        "n_tokens_contexto": 200, "clinico": True})


class _FalsoQdrant:
    def __init__(self, puntos):
        self.puntos = puntos
        self.coleccion = None

    def query_points(self, coleccion, query, limit, with_payload):
        self.coleccion = coleccion
        return SimpleNamespace(points=self.puntos[:limit])


@pytest.fixture
def qdrant(monkeypatch):
    falso = _FalsoQdrant([_punto("ATLS-05-013-01", 0.58), _punto("ATLS-05-031-02", 0.55),
                          _punto("ATLS-22-009-01", 0.50)])
    monkeypatch.setattr(retrieve, "_qdrant", lambda: falso)
    monkeypatch.setattr(retrieve, "hay_guias", lambda: True)
    monkeypatch.setattr(retrieve, "_vector_consulta", lambda consulta: np.zeros(4))
    # el reranker prefiere el tercero de la densa
    monkeypatch.setattr(retrieve, "_probabilidades", lambda pares: np.array([0.3, 0.1, 0.9])[:len(pares)])
    return falso


def test_rerank_reordena_y_conserva_el_coseno(qdrant):
    frags = retrieve.candidatos_guias("hemotórax por arma de fuego", "rerank")
    assert qdrant.coleccion == retrieve.config.COLECCION_GUIAS
    assert [f.id for f in frags] == ["ATLS-22-009-01", "ATLS-05-013-01", "ATLS-05-031-02"]
    assert (frags[0].relevancia, frags[0].coseno) == (0.9, 0.50)
    assert frags[0].pagina_ini == 80 and frags[0].texto.startswith("ATLS › Breathing")


def test_densa_no_llama_al_reranker(qdrant, monkeypatch):
    def no_llamar(pares):
        raise AssertionError("la ruta densa no usa el reranker")
    monkeypatch.setattr(retrieve, "_probabilidades", no_llamar)
    frags = retrieve.candidatos_guias("hemotórax por arma de fuego", "densa")
    assert [f.id for f in frags] == ["ATLS-05-013-01", "ATLS-05-031-02", "ATLS-22-009-01"]
    assert all(f.relevancia is None for f in frags)


def test_cada_ruta_usa_su_umbral(qdrant, monkeypatch):
    monkeypatch.setattr(retrieve.config, "UMBRAL_GUIAS", 0.2)
    monkeypatch.setattr(retrieve.config, "UMBRAL_GUIAS_COSENO", 0.60)
    # mejor coseno 0,58 < 0,60: con la densa, los libros se abstienen
    assert retrieve.buscar_guias("hemotórax", "densa") == []
    # mejor probabilidad 0,9 >= 0,2: con el reranker, responden
    assert len(retrieve.buscar_guias("hemotórax", "rerank")) == 3


def test_ruta_desconocida(qdrant):
    with pytest.raises(ValueError, match="desconocida"):
        retrieve.candidatos_guias("hemotórax", "rrf")


def test_sin_la_coleccion_dice_que_correr(monkeypatch):
    monkeypatch.setattr(retrieve, "hay_guias", lambda: False)
    with pytest.raises(SystemExit, match="--fuente guias"):
        retrieve.candidatos_guias("hemotórax", "densa")


@pytest.mark.parametrize("vram, ruta", [(8.0, "rerank"), (4.0, "densa"), (0.0, "densa")])
def test_ruta_de_libros_segun_vram(monkeypatch, vram, ruta):
    monkeypatch.setattr(retrieve.config, "RUTA_GUIAS", "auto")
    monkeypatch.setattr(retrieve, "vram_gb", lambda: vram)
    assert retrieve.ruta_guias_por_defecto() == ruta


def test_ruta_de_libros_fija_manda(monkeypatch):
    monkeypatch.setattr(retrieve.config, "RUTA_GUIAS", "densa")
    assert retrieve.ruta_guias_por_defecto() == "densa"
