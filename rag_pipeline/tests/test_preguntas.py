"""Lo que el asistente pregunta (rag/preguntas.py; PLAN_CHATBOT_CLINICO.md,
fase 3). Sin Ollama ni índices."""
from __future__ import annotations

import pytest

from rag import config, generate, verificar
from rag.ficha import actualizar
from rag.hallazgos import normalizar
from rag.preguntas import MAX_PREGUNTAS, descartar, preguntar, ya_dicho
from rag.retrieve import Resultado


def claves(texto: str, casos=(), abstuvo: bool = False) -> list[str]:
    _, ficha = actualizar(None, texto)
    return [p.clave for p in preguntar(ficha, casos, texto, abstuvo)]


@pytest.mark.parametrize("texto, esperadas", [
    ("HPAF en hemitórax izquierdo", ["B", "C"]),                     # en el orden de la revisión primaria
    ("HPAF en hemitórax izquierdo con hipoventilación", ["C", "grupo"]),
    ("HPAF en tórax, hipotenso", ["B", "grupo"]),
    ("TCE por caída de moto en adulto", ["D"]),
    ("TCE con Glasgow 14 en adulto", []),
    ("quemadura por agua caliente en tronco de un niño", ["E-superficie", "E-profundidad"]),
    ("quemadura de segundo grado del 15 % de superficie corporal en un adulto", []),
    ("lesión ocular penetrante en adulto", []),                      # el ojo solo no pide la C
    ("¿cuándo se pone un tubo de tórax?", []),                        # no hay paciente
])
def test_que_se_pregunta(texto, esperadas, monkeypatch):
    # las reglas de la fase 3; con tarjeta se suman las suyas (tests/test_tarjetas.py)
    monkeypatch.setattr(config, "TARJETAS", False)
    assert claves(texto) == esperadas


def test_nunca_mas_de_dos_ni_lo_ya_dicho():
    for texto in ("HPAF en tórax y abdomen por caída", "quemadura en tórax por accidente de tránsito"):
        _, ficha = actualizar(None, texto)
        ps = preguntar(ficha, [], texto)
        assert len(ps) <= MAX_PREGUNTAS and not any(ya_dicho(p, texto) for p in ps)


def test_el_choque_vial_no_responde_la_c():
    assert "C" in claves("choque de moto con trauma de abdomen en adulto")


def test_la_respuesta_se_suma_a_la_ficha_y_no_se_vuelve_a_preguntar():
    _, ficha = actualizar(None, "HPAF en abdomen")
    c = next(p for p in preguntar(ficha, [], "HPAF en abdomen") if p.clave == "C")
    for opcion in c.opciones:
        if opcion.mensaje is None:
            continue
        _, despues = actualizar(ficha, opcion.mensaje, forzar="seguimiento")
        assert despues.estado in ("estable", "inestable")
        assert "C" not in [p.clave for p in preguntar(despues, [], opcion.mensaje)]


@pytest.mark.parametrize("mensaje, valor", [
    ("Glasgow 15", "15"), ("Glasgow 13-14", "13-14"), ("Glasgow 9-12", "9-12"), ("Glasgow 8 o menos", "8 o menos"),
])
def test_las_respuestas_de_glasgow_se_leen(mensaje, valor):
    _, ficha = actualizar(None, "TCE por caída")
    _, despues = actualizar(ficha, mensaje, forzar="seguimiento")
    assert despues.signos["Glasgow"] == valor and "D" not in [p.clave for p in preguntar(despues)]


def test_cada_opcion_de_grupo_y_quemadura_se_lee():
    _, ficha = actualizar(None, "quemadura por agua caliente en tronco")
    for p in preguntar(ficha, [], "quemadura por agua caliente en tronco") + preguntar(
            actualizar(None, "HPAF en abdomen, hipotenso")[1]):
        for o in p.opciones:
            if o.mensaje:
                _, despues = actualizar(ficha, o.mensaje, forzar="seguimiento")
                assert p.clave not in [q.clave for q in preguntar(despues)], (p.clave, o.mensaje)


def test_no_se_sabe_deja_de_preguntarlo():
    _, ficha = actualizar(None, "HPAF en abdomen")
    sin = descartar(ficha, "C")
    assert "C" not in [p.clave for p in preguntar(sin)] and "C" in [p.clave for p in preguntar(ficha)]


def _caso(rid: str, texto: str) -> Resultado:
    return Resultado(rid, f"Examen físico: {texto}\n\nEvolución y hospitalización: {texto}", {}, 0.7, 0.7)


def test_si_queda_lugar_lo_que_separa_los_casos():
    casos = [_caso("1", "hemotorax, tubo de torax"), _caso("2", "tubo de torax"), _caso("3", "hemotorax")]
    _, ficha = actualizar(None, "HPACP en tórax, hipoventilación, estable, adulto")
    ps = preguntar(ficha, casos, "HPACP en tórax, hipoventilación, estable, adulto")
    assert [p.clave for p in ps] == ["afinar:hemotórax"]
    assert [o.mensaje for o in ps[0].opciones] == ["con hemotórax", "sin hemotórax", None]


def test_antes_de_responder_solo_sin_mecanismo_ni_region():
    assert claves("paciente con trauma", abstuvo=True) == ["mecanismo", "region"]
    assert claves("herida en abdomen", abstuvo=True) == []
    # fuera de dominio: el mensaje de abstención, sin preguntas
    assert claves("convulsión febril en lactante", abstuvo=True) == []


def test_los_textos_no_son_vocabulario_del_verificador():
    # van en preguntas.py: build_prompt.py y hallazgos.py cuentan como vocabulario permitido
    assert "exposicion" not in verificar._VOCABULARIO_CODIGO
    assert "ventila bien" not in normalizar(verificar._VOCABULARIO_CODIGO)


def test_responder_trae_las_preguntas(monkeypatch):
    casos = [_caso("101", "hipotension, transfusion"), _caso("102", "hipotension, laparotomia")]
    monkeypatch.setattr(config, "CONVERSACION", "ficha")
    monkeypatch.setattr(config, "PREGUNTAS", True)
    monkeypatch.setattr(config, "GUIAS", False)
    monkeypatch.setattr(config, "SALIDA", "prosa")
    monkeypatch.setattr(config, "ORDEN_ATLS", False)
    monkeypatch.setattr(config, "SINTESIS_LLM", False)
    monkeypatch.setattr(generate, "confianza", lambda *a, **k: 0.9)
    monkeypatch.setattr(generate, "buscar", lambda *a, **k: casos)
    monkeypatch.setattr(generate, "guarda_casos", lambda consulta, casos, u: (casos, []))
    monkeypatch.setattr(generate, "_ollama", lambda *a, **k: '{"sugerencia": "Transfusión [caso: 101]."}')
    assert [p.clave for p in generate.responder("HPAF en abdomen").preguntas] == ["C", "grupo"]
    monkeypatch.setattr(config, "PREGUNTAS", False)
    assert generate.responder("HPAF en abdomen").preguntas == []
    # abstención sin mecanismo ni región: se pregunta antes de responder
    monkeypatch.setattr(config, "PREGUNTAS", True)
    monkeypatch.setattr(generate, "confianza", lambda *a, **k: 0.1)
    r = generate.responder("paciente con trauma")
    assert r.abstuvo and [p.clave for p in r.preguntas] == ["mecanismo", "region"]
    assert generate.responder("convulsión febril en lactante").preguntas == []
