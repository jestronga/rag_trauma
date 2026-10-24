"""La consulta para buscar en los manuales: abreviaturas expandidas
(rag/consulta.py; PLAN_CHATBOT_CLINICO.md, fase 1). Lee el diccionario de la
anonimización, que está en git."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pytest
import yaml

from rag import config, retrieve
from rag.consulta import (LAS_USAN_LOS_MANUALES, MINIMO_EN_MANUALES, NO_EXPANDIR, agregar_mecanismo,
                          diccionario, expandir_abreviaturas, para_libros)
from rag.senal import MECANISMOS
from rag.hallazgos import normalizar
from rag.index import cargar_secciones


@pytest.mark.parametrize("consulta, buscada", [
    ("HPAF en cráneo con orificio de entrada parietal",
     "HPAF (herida por arma de fuego) en cráneo con orificio de entrada parietal"),
    ("FX de radio distal con inmovilización", "FX (fractura) de radio distal con inmovilización"),
    ("HPACP en zona II de cuello", "HPACP (herida por arma cortopunzante) en zona II de cuello"),
    ("paciente con IOT y manejo en UCI por trauma de tórax",
     "paciente con IOT (intubacion orotraqueal) y manejo en UCI por trauma de tórax"),
    # las que los manuales ya usan, igual
    ("TCE leve", "TCE leve"),
    ("protocolo E-FAST paso a paso en trauma", "protocolo E-FAST paso a paso en trauma"),
    ("tubo de tórax de 28 Fr y Rx AP de tórax", "tubo de tórax de 28 Fr y Rx AP de tórax"),
    # solo la primera vez
    ("fx de tibia y fx de peroné", "fx (fractura) de tibia y fx de peroné"),
    # sin abreviaturas, igual
    ("quemadura de segundo grado en miembro superior", "quemadura de segundo grado en miembro superior"),
    # ambiguas: no se expanden (pendiente 6; "ha" es el verbo)
    ("Tx renal", "Tx renal"),
    ("Qx de II y III grado", "Qx de II y III grado"),
    ("se ha caído de la escalera", "se ha caído de la escalera"),
])
def test_expandir_abreviaturas(consulta, buscada):
    assert expandir_abreviaturas(consulta) == buscada


@pytest.mark.parametrize("consulta, buscada", [
    ("sujeto baleado en extremidad inferior",
     "sujeto baleado en extremidad inferior (herida por arma de fuego)"),
    ("precipitación desde altura con trauma de columna",
     "precipitación desde altura con trauma de columna (caída)"),
    ("agresión física con objeto romo", "agresión física con objeto romo (trauma contundente)"),
    # ya nombrado, igual (también con otras tildes y mayúsculas)
    ("herida por arma de fuego en abdomen", "herida por arma de fuego en abdomen"),
    ("lactante con trauma craneal por Caida de la cama", "lactante con trauma craneal por Caida de la cama"),
    # sin señal de trauma, igual
    ("hipoglucemia en paciente diabético", "hipoglucemia en paciente diabético"),
])
def test_agregar_mecanismo(consulta, buscada):
    assert agregar_mecanismo(consulta) == buscada


def test_para_libros_junta_las_dos():
    assert para_libros("HPAF en cráneo", abreviaturas=False, mecanismo=False) == "HPAF en cráneo"
    # la expansión ya nombra el mecanismo: no se repite
    assert para_libros("HPAF en cráneo", abreviaturas=True, mecanismo=True) == \
        "HPAF (herida por arma de fuego) en cráneo"
    assert para_libros("pte baleado", abreviaturas=True, mecanismo=True) == \
        "pte (paciente) baleado (herida por arma de fuego)"


def test_el_diccionario_es_el_de_la_anonimizacion():
    dic = diccionario()
    assert dic["hpaf"] == "herida por arma de fuego" and dic["hpacp"] == "herida por arma cortopunzante"
    assert not (NO_EXPANDIR | LAS_USAN_LOS_MANUALES) & set(dic)
    assert "dr" not in dic                      # "excluidas" del propio archivo


def test_cada_mecanismo_tiene_nombre():
    from rag.consulta import NOMBRES_MECANISMO
    assert set(NOMBRES_MECANISMO) == set(MECANISMOS)


@pytest.mark.skipif(not config.GUIAS_SECCIONES.exists(), reason="sin guias/secciones.jsonl (fuera de git)")
def test_las_que_usan_los_manuales_siguen_al_dia():
    """Si se agrega o re-extrae un libro, la lista puede cambiar."""
    cuenta = Counter(normalizar(w) for s in cargar_secciones() for w in re.findall(r"\w+", s["texto"]))
    abreviaturas = yaml.safe_load(config.ABREVIATURAS_YAML.read_text(encoding="utf-8"))["abreviaturas"]
    usadas = {normalizar(str(k)) for k in abreviaturas if cuenta[normalizar(str(k))] >= MINIMO_EN_MANUALES}
    assert usadas - NO_EXPANDIR == LAS_USAN_LOS_MANUALES     # "ha", el verbo, ya está fuera


def test_solo_cambia_la_busqueda_de_los_manuales(monkeypatch):
    vistas = []
    monkeypatch.setattr(retrieve, "hay_guias", lambda: True)
    monkeypatch.setattr(retrieve, "_vector_consulta", lambda consulta: vistas.append(consulta) or np.zeros(4))

    class Vacio:
        def query_points(self, *a, **k):
            from types import SimpleNamespace
            return SimpleNamespace(points=[])
    monkeypatch.setattr(retrieve, "_qdrant", lambda: Vacio())
    monkeypatch.setattr(config, "MECANISMO_LIBROS", False)
    monkeypatch.setattr(config, "ABREV_LIBROS", True)
    retrieve.candidatos_guias("HPAF en abdomen", "densa", por_libro=False)
    monkeypatch.setattr(config, "ABREV_LIBROS", False)
    retrieve.candidatos_guias("HPAF en abdomen", "densa", por_libro=False)
    assert vistas == ["HPAF (herida por arma de fuego) en abdomen", "HPAF en abdomen"]
