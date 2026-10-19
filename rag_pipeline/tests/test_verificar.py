"""Verificación de la prosa del modelo y el respaldo a la versión determinista.

Sin Ollama ni índices: los casos se arman a mano y `_ollama`, `confianza` y
`buscar` se reemplazan por dobles.
"""
from __future__ import annotations

import json

import pytest

from rag import config, generate
from rag.build_prompt import ROTULO_CASOS, ROTULO_SUGERENCIAS
from rag.retrieve import Resultado
from rag.verificar import palabras_sin_raiz, suavizar, verificar_prosa

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


BUENA = ("Conviene valorar hemotórax [caso: 102]. "
         "En casos similares se hizo tubo de tórax [caso: 101] [caso: 102] [caso: 103]. "
         "En uno de ellos se llegó a toracotomía [caso: 102].")


def test_prosa_respaldada_pasa(casos):
    ok, motivos = verificar_prosa(BUENA, casos, CONSULTA)
    assert ok, motivos


def test_lo_que_dice_la_consulta_no_necesita_cita(casos):
    ok, motivos = verificar_prosa("El paciente presenta hipoventilación. " + BUENA, casos, CONSULTA)
    assert ok, motivos


@pytest.mark.parametrize("prosa, motivo", [
    ("Conviene valorar hemotórax [caso: 999].", "no están entre los recuperados"),
    ("Conviene valorar neumotórax [caso: 101].", "no consta en ningún caso"),
    ("Conviene valorar hemotórax [caso: 101].", "casos que no lo documentan"),
    ("Conviene valorar hemotórax.", "sin citar el caso"),
    ("Se documentó hemotórax en dos de ellos [caso: 102].", "no coincide"),
    ("Hubo tubo de tórax en 3 de 5 casos [caso: 101].", "se recuperaron 4 casos"),
    ("En todos los casos se hizo tubo de tórax [caso: 101].", "todos"),
    ("Conviene solicitar gammagrafía [caso: 101].", "no aparecen en los casos"),
    ("Se debe pasar tubo de tórax [caso: 101].", "da una orden"),
    ("Hay que pasar tubo de tórax [caso: 101].", "da una orden"),
    ("Se registró hemotórax en dos casos [caso: 102].", "no coincide"),
    ("Evaluar la estabilidad articular [caso: 101].", "no aparecen en los casos"),
])
def test_prosa_sin_respaldo_falla(casos, prosa, motivo):
    ok, motivos = verificar_prosa(prosa, casos, CONSULTA)
    assert not ok
    assert any(motivo in m for m in motivos), motivos


def test_conteo_correcto_pasa(casos):
    ok, motivos = verificar_prosa("Hubo tubo de tórax en 3 de 4 casos [caso: 101] [caso: 102] [caso: 103].",
                                  casos, CONSULTA)
    assert ok, motivos


def test_conteo_en_n_casos_correcto_pasa(casos):
    ok, motivos = verificar_prosa("Se registró hemotórax en un caso [caso: 102].", casos, CONSULTA)
    assert ok, motivos


def test_suavizar_solo_cambia_se_debe_considerar():
    assert suavizar("Se debe considerar TAC. Luego se debe valorar Glasgow.") == \
        "Conviene considerar TAC. Luego conviene valorar Glasgow."
    assert suavizar("Hay que pasar tubo.") == "Hay que pasar tubo."


def test_prosa_vacia_falla(casos):
    assert verificar_prosa("  ", casos, CONSULTA)[0] is False


def test_palabras_sin_raiz(casos):
    assert palabras_sin_raiz("Conviene valorar hemotórax", casos, CONSULTA) == []
    assert palabras_sin_raiz("Solicitar gammagrafía", casos, CONSULTA) == ["gammagrafia"]


# --- responder: prosa verificada o respaldo determinista ------------------------------
@pytest.fixture
def entorno(casos, monkeypatch):
    monkeypatch.setattr(config, "SALIDA", "prosa")
    monkeypatch.setattr(config, "SINTESIS_LLM", False)
    # el respaldo determinista se prueba con PROSA_SIEMPRE apagado, sea cual
    # sea el valor por defecto (la demo lo usa encendido)
    monkeypatch.setattr(config, "PROSA_SIEMPRE", False)
    monkeypatch.setattr(generate, "confianza", lambda *a, **k: 0.9)
    monkeypatch.setattr(generate, "buscar", lambda *a, **k: casos)
    # un PC sin los libros indexados: la respuesta es la de solo casos
    # (con libros: tests/test_responder_guias.py)
    monkeypatch.setattr(generate, "hay_guias", lambda: False)

    def modelo(prosa: str):
        monkeypatch.setattr(generate, "_ollama", lambda *a, **k: json.dumps({"sugerencia": prosa}))
    return modelo


def test_responder_muestra_la_prosa_verificada(entorno):
    entorno(BUENA)
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is True and r.motivos == []
    # apartado de casos: título, la prosa y el respaldo, sin tabla
    assert r.texto.startswith(f"{ROTULO_CASOS}\n\n{BUENA}\n\n*Respaldo: 4 casos")
    assert "| Caso |" not in r.texto
    assert r.citas_invalidas == []


def test_responder_vuelve_a_la_version_determinista(entorno):
    entorno("Conviene valorar neumotórax y pedir gammagrafía [caso: 101].")
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is False and r.motivos
    assert "neumotórax" not in r.texto
    assert ROTULO_SUGERENCIAS in r.texto            # la respuesta armada desde los casos


def test_prosa_siempre_muestra_la_prosa_aunque_no_pase(entorno, monkeypatch):
    # demo (decisión del 18/10): todo en prosa; la verificación queda en motivos
    monkeypatch.setattr(config, "PROSA_SIEMPRE", True)
    entorno("Conviene valorar neumotórax y pedir gammagrafía [caso: 101].")
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is False and r.motivos
    assert r.texto.startswith(ROTULO_CASOS) and "neumotórax" in r.texto
    assert ROTULO_SUGERENCIAS not in r.texto


def test_salida_sugerencias_no_usa_prosa(entorno, monkeypatch):
    monkeypatch.setattr(config, "SALIDA", "sugerencias")
    monkeypatch.setattr(generate, "_ollama", lambda *a, **k: '{"sugerencias": ["hemotórax"]}')
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is None
    assert ROTULO_SUGERENCIAS in r.texto
