"""Reranker: una sola sigmoide, y la relevancia filtra sin tocar el coseno."""
import numpy as np
import pytest

from rag import retrieve
from rag.generate import filtrar_relevantes
from rag.retrieve import Resultado


class _FalsoCrossEncoder:
    """Devuelve logits crudos solo si se le pide Identity, como sentence-transformers >= 4."""

    def predict(self, pares, activation_fn=None, **_):
        logits = np.array([2.0, -2.0][:len(pares)])
        return logits if activation_fn is not None else 1 / (1 + np.exp(-logits))


def test_probabilidades_aplica_una_sola_sigmoide(monkeypatch):
    monkeypatch.setattr(retrieve, "_reranker", lambda: _FalsoCrossEncoder())
    p = retrieve._probabilidades([("q", "a"), ("q", "b")])
    # con doble sigmoide saldría 0,70 y 0,53: todo apiñado sobre 0,5
    assert round(p[0], 3) == 0.881
    assert round(p[1], 3) == 0.119


def _caso(rid, relevancia):
    return Resultado(rid, "", {}, 0.0, 0.6, relevancia=relevancia)


def test_filtrar_relevantes_descarta_bajo_umbral():
    casos = [_caso("a", 0.9), _caso("b", 0.2), _caso("c", 0.5)]
    assert [c.record_id for c in filtrar_relevantes(casos, 0.5)] == ["a", "c"]


def test_filtrar_relevantes_no_toca_rutas_sin_reranker():
    casos = [_caso("a", None), _caso("b", None)]
    assert filtrar_relevantes(casos, 0.99) == casos


def test_rerank_conserva_el_coseno_en_score_norm():
    # SIMILITUD_BAJA y la columna "similitud" de la tabla leen score_norm como coseno
    c = _caso("a", 0.95)
    assert c.score_norm == 0.6 and c.relevancia == 0.95


@pytest.mark.parametrize("vram, ruta, dispositivo", [
    (8.0, "rerank", "cuda"),      # PC de 8 GB: el reranker cabe junto al 4B
    (4.0, "rrf", "cpu"),          # GTX 1650: no cabe, Ollama daría HTTP 500
    (0.0, "rrf", "cpu"),          # sin GPU NVIDIA
])
def test_ruta_auto_segun_vram(monkeypatch, vram, ruta, dispositivo):
    monkeypatch.setattr(retrieve.config, "RUTA_DEFECTO", "auto")
    monkeypatch.setattr(retrieve.config, "DISPOSITIVO_RERANKER", "auto")
    monkeypatch.setattr(retrieve, "vram_gb", lambda: vram)
    assert retrieve.ruta_por_defecto() == ruta
    assert retrieve.dispositivo_reranker() == dispositivo


def test_ruta_fija_manda(monkeypatch):
    monkeypatch.setattr(retrieve.config, "RUTA_DEFECTO", "densa")
    assert retrieve.ruta_por_defecto() == "densa"
