"""Métricas de scripts/comparar_modelos.py: citados frente a recuperados y su precisión."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rag.retrieve import Resultado

RUTA = Path(__file__).resolve().parents[1] / "scripts" / "comparar_modelos.py"
_spec = importlib.util.spec_from_file_location("comparar_modelos", RUTA)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

CONSULTA = {"texto": "herida por arma de fuego en tórax con hipoventilación",
            "mecanismo": "arma_de_fuego", "regiones": ["torax"]}


def caso(rid: str, examen: str, evolucion: str, mecanismo: str = "arma_de_fuego") -> Resultado:
    texto = (f"Motivo de consulta: HPAF\n\nExamen físico: {examen}\n\n"
             f"Evolución y hospitalización: {evolucion}")
    meta = {"mecanismo_lesion": mecanismo, "region_anatomica": ["torax"], "desenlace": "alta"}
    return Resultado(rid, texto, meta, 0.7, 0.7)


@pytest.fixture
def rec():
    return {"casos": [
        caso("1", "hemotorax", "tubo de torax"),
        caso("2", "hemotorax", "toracotomia"),
        caso("3", "herida superficial", "sutura", mecanismo="cortopunzante"),
        caso("4", "sin hallazgos", "salida"),
    ]}


def test_citados_son_los_casos_de_las_sugerencias(rec, monkeypatch):
    monkeypatch.setattr(cm, "generar_eleccion",
                        lambda *a, **k: '{"sugerencias": ["hemotórax", "tubo de tórax", "toracotomía"]}')
    fila = cm.medir(CONSULTA, rec, estructurada=True)
    assert fila["recuperados"] == 4
    assert fila["citados"] == 2                      # casos 1 y 2
    assert fila["precision_citados"] == 1.0          # los dos son arma de fuego + tórax
    assert fila["precision_recuperados"] == 0.75     # el 3 es cortopunzante
    assert fila["descartadas"] == 0


def test_precision_sin_casos_es_none():
    assert cm.precision([], CONSULTA) is None
