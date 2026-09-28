"""Salida estructurada: el modelo solo puede elegir de la lista, y el código la lee.

Sin Ollama: `_ollama` se reemplaza por un doble que registra la petición.
"""
from __future__ import annotations

import json

import pytest

from rag import generate
from rag.generate import eleccion_modelo, elegir_sugerencias, esquema_sugerencias, generar_eleccion
from rag.hallazgos import elegibles
from rag.retrieve import Resultado

CONSULTA = "herida por arma de fuego en tórax con hipoventilación"


def caso(rid: str, examen: str, evolucion: str, desenlace: str = "alta") -> Resultado:
    texto = (f"Motivo de consulta: HPAF (herida por arma de fuego)\n\nExamen físico: {examen}\n\n"
             f"Evolución y hospitalización: {evolucion}")
    meta = {"mecanismo_lesion": "arma_de_fuego", "region_anatomica": ["torax"], "desenlace": desenlace}
    return Resultado(rid, texto, meta, score=0.7, score_norm=0.7)


@pytest.fixture
def casos() -> list[Resultado]:
    return [
        caso("101", "hipoventilacion derecha", "se pasa tubo de torax, drena 300cc. Salida"),
        caso("102", "hipoventilacion, hemotorax", "tubo de torax. Llevado a cirugia: toracotomia", "cirugia"),
        caso("103", "herida penetrante, hipoventilacion", "tubo de torax, observacion", "no_determinado"),
        caso("104", "herida superficial", "sutura, salida"),
    ]


def test_elegibles_quita_lo_que_la_consulta_ya_dice(casos):
    terminos = {e.termino for e in elegibles(casos, CONSULTA)}
    assert "hipoventilación" not in terminos
    assert "herida penetrante" not in terminos
    assert "hemotórax" in terminos


def test_esquema_restringe_a_la_lista(casos):
    elementos = elegibles(casos, CONSULTA)
    items = esquema_sugerencias(elementos)["properties"]["sugerencias"]
    assert items["items"]["enum"] == [e.termino for e in elementos]
    assert items["maxItems"] == min(4, len(elementos))
    assert items["minItems"] == min(3, len(elementos))


def test_esquema_con_un_solo_elemento_no_invita_a_repetir(casos):
    items = esquema_sugerencias(elegibles(casos, CONSULTA)[:1])["properties"]["sugerencias"]
    assert items["minItems"] == items["maxItems"] == 1


def test_eleccion_lee_json_en_orden_y_sin_repetidos(casos):
    elementos = elegibles(casos, CONSULTA)
    salida = json.dumps({"sugerencias": ["hemotórax", "toracotomía", "hemotórax"]})
    elegidos, propuestas = eleccion_modelo(salida, elementos)
    assert [e.termino for e in elegidos] == ["hemotórax", "toracotomía"]
    assert propuestas == 3


def test_eleccion_descarta_lo_que_no_esta_en_la_lista(casos):
    # un modelo sin gramática podría escribir algo sin respaldo
    elementos = elegibles(casos, CONSULTA)
    elegidos, propuestas = eleccion_modelo('{"sugerencias": ["tomografía de tórax"]}', elementos)
    assert elegidos == [] and propuestas == 1


def test_elegir_sugerencias_completa_desde_json(casos):
    elegidos = elegir_sugerencias('{"sugerencias": ["hemotórax"]}', casos, CONSULTA)
    assert elegidos[0].termino == "hemotórax"
    assert 3 <= len(elegidos) <= 4
    assert any(e.categoria == "Manejo" for e in elegidos)


def test_generar_eleccion_envia_esquema_y_lista_elegible(casos, monkeypatch):
    peticiones = []

    def falso_ollama(mensajes, formato=None, metricas=None, **_):
        peticiones.append((mensajes, formato))
        return '{"sugerencias": ["hemotórax"]}'

    monkeypatch.setattr(generate, "_ollama", falso_ollama)
    generar_eleccion(CONSULTA, casos, estructurada=True)
    mensajes, formato = peticiones[0]
    assert formato["properties"]["sugerencias"]["items"]["enum"]
    # el modelo no ve en la lista lo que no puede elegir
    assert "- hipoventilación:" not in mensajes[1]["content"]
    assert '{"sugerencias"' in mensajes[1]["content"]


def test_sin_elementos_no_se_llama_al_modelo(monkeypatch):
    monkeypatch.setattr(generate, "_ollama", lambda *a, **k: pytest.fail("no debía llamar a Ollama"))
    vacio = [caso("201", "sin hallazgos", "sin cambios")]
    assert generar_eleccion("consulta", vacio, estructurada=True) == ""


def test_modo_texto_sin_formato(casos, monkeypatch):
    peticiones = []
    monkeypatch.setattr(generate, "_ollama",
                        lambda mensajes, formato=None, **_: peticiones.append(formato) or "")
    generar_eleccion(CONSULTA, casos, estructurada=False)
    assert peticiones == [None]
