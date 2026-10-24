"""Señal de trauma en la consulta y las guardas que la usan (decisión 5 de
PLAN_INDEXAR_ATLS10.md): sin señal, los casos y los libros piden más."""
from __future__ import annotations

import pytest

from rag import config, generate, retrieve
from rag.retrieve import Resultado
from rag.senal import senal_trauma


@pytest.mark.parametrize("consulta, clases", [
    ("HPAF en abdomen", ["mecanismo:arma_de_fuego"]),
    ("paciente apuñalado en el pecho, herida penetrante", ["mecanismo:cortopunzante", "lesion"]),
    ("luxacion de hombro por caida", ["mecanismo:caida", "lesion"]),
    ("FX de radio distal con inmovilización", ["lesion"]),
    ("uso del torniquete en hemorragia de una extremidad", ["procedimiento"]),
    # las fuera de dominio que pasaban las guardas (23/10)
    ("sangrado vaginal en el primer trimestre de embarazo", []),
    ("accidente cerebrovascular isquémico con hemiparesia de inicio súbito", []),
    ("apendicitis aguda con dolor en fosa ilíaca derecha y fiebre", []),
    # límite de palabra: "riña" en "warfarina" y "urinaria", "ácido" en "cetoacidosis"
    ("ajuste de la dosis de warfarina en fibrilación auricular", []),
    ("delirium en adulto mayor con infección urinaria", []),
    ("manejo de cetoacidosis diabética en urgencias", []),
    ("quemadura por ácido en la mano", ["mecanismo:quemadura", "lesion"]),
    # legítimas sin señal: por eso sin señal no se abstiene, se pide más
    ("cuándo activar el protocolo de transfusión masiva", []),
])
def test_senal_trauma(consulta, clases):
    assert senal_trauma(consulta) == clases


def _caso(rid: str, relevancia: float) -> Resultado:
    return Resultado(rid, "texto", {}, score=0.7, score_norm=0.7, relevancia=relevancia)


@pytest.fixture
def con_senal(monkeypatch):
    monkeypatch.setattr(config, "SENAL_TRAUMA", True)
    monkeypatch.setattr(config, "UMBRAL_RERANK", 0.2)
    monkeypatch.setattr(config, "UMBRAL_RERANK_SIN_SENAL", 0.7)
    monkeypatch.setattr(config, "MIN_CASOS", 2)


def test_guarda_casos_sin_senal_pide_dos_casos_altos(con_senal):
    # neg-15 (ACV): el 2.º caso en 0,580 pasaba la guarda de 0,2
    casos = [_caso("1", 0.77), _caso("2", 0.58), _caso("3", 0.42)]
    acv = "accidente cerebrovascular isquémico con hemiparesia"
    assert generate.guarda_casos(acv, casos) == ([], casos)
    # con señal, la de siempre
    assert generate.guarda_casos("caída con TCE y hemiparesia", casos) == (casos, [])
    # sin señal pero con dos casos altos, pasa, y al prompt van también los de 0,2 o más
    altos = [_caso("1", 0.95), _caso("2", 0.80), _caso("3", 0.30), _caso("4", 0.10)]
    assert generate.guarda_casos("qemadura en cara", altos) == (altos[:3], [])


def test_guarda_casos_apagada_o_sin_reranker(con_senal, monkeypatch):
    casos = [_caso("1", 0.77), _caso("2", 0.58)]
    monkeypatch.setattr(config, "SENAL_TRAUMA", False)
    assert generate.guarda_casos("hipoglucemia en diabético", casos) == (casos, [])
    monkeypatch.setattr(config, "SENAL_TRAUMA", True)
    sin_reranker = [Resultado("1", "t", {}, 0.7, 0.7), Resultado("2", "t", {}, 0.7, 0.7)]
    assert generate.guarda_casos("hipoglucemia en diabético", sin_reranker) == (sin_reranker, [])


def test_umbral_de_los_libros_segun_la_senal(monkeypatch):
    monkeypatch.setattr(config, "SENAL_TRAUMA", True)
    monkeypatch.setattr(config, "UMBRAL_GUIAS", 0.34)
    monkeypatch.setattr(config, "UMBRAL_GUIAS_SIN_SENAL", 0.65)
    monkeypatch.setattr(config, "UMBRAL_GUIAS_COSENO", 0.60)
    # neg-12: un pasaje de embarazo en 0,541 pasaba la guarda de 0,34
    assert retrieve.umbral_guias_consulta("sangrado vaginal en el primer trimestre", "rerank") == 0.65
    assert retrieve.umbral_guias_consulta("manejo del neumotórax a tensión", "rerank") == 0.34
    assert retrieve.umbral_guias_consulta("sangrado vaginal", "densa") == 0.60     # el coseno no se midió
    monkeypatch.setattr(config, "SENAL_TRAUMA", False)
    assert retrieve.umbral_guias_consulta("sangrado vaginal", "rerank") == 0.34
