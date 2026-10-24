"""La ficha del paciente (rag/ficha.py; PLAN_CHATBOT_CLINICO.md, fase 2).
Sin Ollama ni índices. Los ejemplos son propios: los de
eval/conversaciones.yaml se miden con scripts/probar_conversaciones.py."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rag import config, generate
from rag.ficha import Ficha, actualizar, que_cambio, tipo_de_mensaje
from rag.retrieve import FragmentoGuia, Resultado


def conversacion(*mensajes: str) -> list[tuple[str, Ficha]]:
    salida, ficha = [], None
    for m in mensajes:
        tipo, ficha = actualizar(ficha, m)
        salida.append((tipo, ficha))
    return salida


# ---------------------------------------------------------------------------
# Tipo de mensaje
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mensaje, tipo", [
    ("caso 4521", "caso"),
    ("#4521", "caso"),
    ("siguiente paciente, HPAF en muslo", "nuevo"),
    ("hombre de 40 años con trauma de pelvis", "nuevo"),          # presenta a alguien
    ("una paciente con herida en mano", "nuevo"),
    ("el paciente tiene Glasgow 12", "seguimiento"),               # el mismo
    ("paciente ahora con Glasgow 12", "seguimiento"),
    ("se cayó de un segundo piso", "nuevo"),                       # hecho con otro mecanismo
    ("¿y si se hubiera caído de un segundo piso?", "seguimiento"), # hipótesis: el mismo
    ("fue por arma cortopunzante", "seguimiento"),                 # el mismo mecanismo
    ("TA 100/60", "seguimiento"),
])
def test_tipo_de_mensaje(mensaje, tipo):
    _, previa = actualizar(None, "herida por arma cortopunzante en abdomen")
    assert tipo_de_mensaje(previa, mensaje) == tipo


def test_el_primer_mensaje_es_nuevo_y_las_regiones_solas_no_son_un_paciente():
    assert tipo_de_mensaje(None, "TA 100/60") == "nuevo"
    _, ficha = actualizar(None, "¿cuándo se pone un tubo de tórax?")
    assert ficha.regiones == ["torax"] and not ficha.es_paciente()
    assert tipo_de_mensaje(ficha, "TA 100/60") == "nuevo"


def test_la_ficha_de_entrada_no_se_toca():
    _, antes = actualizar(None, "HPAF en abdomen")
    copia = antes.datos()
    actualizar(antes, "hipotenso")
    assert antes.datos() == copia


def test_forzar_corrige_al_codigo():
    _, previa = actualizar(None, "HPAF en abdomen")
    tipo, ficha = actualizar(previa, "mujer de 30 años con FAST positivo", forzar="seguimiento")
    assert tipo == "seguimiento" and ficha.mecanismo == "arma_de_fuego" and ficha.grupo == "adulto"


# ---------------------------------------------------------------------------
# Lo que se lee
# ---------------------------------------------------------------------------
def test_negacion_y_despues_afirmacion():
    (_, f1), (_, f2) = conversacion("HPACP en abdomen sin irritación peritoneal",
                                    "ahora con irritación peritoneal")
    assert f1.negados == ["irritación peritoneal"] and not f1.afirmados
    assert f2.afirmados == ["irritación peritoneal"] and not f2.negados


def test_lo_que_se_pregunta_no_es_un_hecho():
    (_, _), (_, f2), (_, f3) = conversacion("HPAF en muslo", "¿le tomo radiografía?",
                                            "se tomó radiografía con fractura de fémur")
    assert "radiografía" not in f2.afirmados
    assert {"radiografía", "fractura"} <= set(f3.afirmados)


def test_una_hipotesis_suma_el_hallazgo_y_cambia_el_mecanismo():
    _, _ = actualizar(None, "x")
    (_, _), (tipo, f2) = conversacion("HPAF en muslo", "¿y si fuera por arma cortopunzante con hemorragia?")
    assert tipo == "seguimiento" and f2.mecanismo == "cortopunzante"
    assert "sangrado activo" in f2.afirmados and f2.regiones == ["miembro_inferior"]


def test_signos_con_su_valor():
    _, f = actualizar(None, "HPAF en tórax, TA: 85/55, FC 130, FR de 30, SpO2 de 90%, Glasgow 13/15")
    assert f.signos == {"TA": "85/55", "FC": "130", "FR": "30", "SatO2": "90", "Glasgow": "13"}
    _, f = actualizar(None, "HPAF en tórax, Glasgow 20")             # fuera de la escala
    assert "Glasgow" not in f.signos


def test_estado_por_la_ta_y_por_la_palabra():
    (_, f1), (_, f2) = conversacion("HPAF en abdomen, TA 80/40", "TA 120/80")
    assert f1.estado == "inestable" and f2.estado is None         # la TA lo puso, la TA lo quita
    (_, f1), (_, f2) = conversacion("HPAF en abdomen, inestable", "TA 120/80")
    assert f2.estado == "inestable"                               # lo dijo el médico
    (_, f1), = conversacion("HPAF en abdomen, no está estable")
    assert f1.estado == "inestable"


@pytest.mark.parametrize("texto, grupo", [
    ("caída en paciente de 70 años", "adulto mayor"),
    ("caída de 2 metros en un niño", "niño"),
    ("lactante de 10 meses con caída", "niño"),
    ("caída en gestante", "embarazada"),
    ("caída hace 5 años, ahora con dolor de cadera", None),
    ("caída en paciente de 16 años", "niño"),
])
def test_grupo(texto, grupo):
    assert actualizar(None, texto)[1].grupo == grupo


def test_el_choque_vial_no_es_shock():
    _, f = actualizar(None, "choque de carro con trauma en tórax")
    assert f.mecanismo == "accidente_transito" and "hipotensión o choque" not in f.afirmados
    _, f = actualizar(None, "HPAF en abdomen, en choque")
    assert "hipotensión o choque" in f.afirmados and f.estado == "inestable"


@pytest.mark.parametrize("texto, regiones", [
    ("HPAF en abdomen con Glasgow 9", ["abdomen"]),                 # el Glasgow no es la región
    ("TCE con anisocoria pupilar", ["craneo_encefalo"]),             # las pupilas tampoco
    ("HPACP en muslo con pulso radial débil", ["miembro_inferior"]),
    ("fractura de radio distal", ["miembro_superior"]),
    ("trauma de tórax, oxígeno por cánula nasal", ["torax"]),
])
def test_regiones(texto, regiones):
    assert actualizar(None, texto)[1].regiones == regiones


def test_mordida():
    assert actualizar(None, "mordida de perro en mano")[1].mecanismo == "mordedura"


# ---------------------------------------------------------------------------
# La búsqueda de un seguimiento
# ---------------------------------------------------------------------------
def test_la_busqueda_son_las_palabras_del_medico_sin_lo_negado():
    _, f = actualizar(None, "HPACP en zona II de cuello sin sangrado activo")
    _, f = actualizar(f, "FC 110")
    assert f.consulta("¿qué estudios pido?") == "HPACP en zona II de cuello, FC 110, qué estudios pido"


def test_lo_contradicho_no_se_busca():
    (_, _), (_, f) = conversacion("HPAF en abdomen, sin hipotensión", "ahora hipotenso")
    assert f.consulta("ahora hipotenso") == "HPAF en abdomen, hipotenso"
    # lo que se afirmó y después se negó: se corta desde el término, no la cláusula entera
    (_, _), (_, f) = conversacion("herida por arma de fuego en muslo con sangrado activo",
                                  "ya no hay sangrado activo")
    assert f.negados == ["sangrado activo"]
    assert f.consulta("¿qué hago?") == "herida por arma de fuego en muslo, qué hago"


def test_la_busqueda_lleva_el_mecanismo_de_la_ficha():
    (_, _), (_, f) = conversacion("herida por arma de fuego en muslo con sangrado activo",
                                  "¿y si fuera por arma cortopunzante?")
    assert f.consulta("") == "herida por arma cortopunzante en muslo con sangrado activo, por arma cortopunzante"
    # si la cláusula ya nombra el mecanismo de la ficha, queda igual
    _, f = actualizar(None, "agresión física con arma cortopunzante en tórax")
    assert f.consulta("") == "agresión física con arma cortopunzante en tórax"


def test_linea_de_la_ficha():
    (_, _), (_, f) = conversacion("HPAF en abdomen sin irritación peritoneal", "hipotenso, TA 80/50")
    assert f.linea() == ("Paciente: arma de fuego · abdomen · hipotensión o choque · sin irritación "
                         "peritoneal · TA 80/50 · inestable")
    assert actualizar(None, "¿cuándo se pone un tubo de tórax?")[1].linea() == ""


# ---------------------------------------------------------------------------
# Qué cambió
# ---------------------------------------------------------------------------
def _caso(rid: str, texto: str) -> Resultado:
    return Resultado(rid, f"Examen físico: {texto}\n\nEvolución y hospitalización: {texto}", {}, 0.7, 0.7)


def _respuesta(casos, guias=()):
    return SimpleNamespace(casos=list(casos), guias=list(guias), abstuvo=False)


def test_que_cambio():
    antes = _respuesta([_caso("1", "sutura"), _caso("2", "sutura")])
    ahora = _respuesta([_caso("1", "sutura"), _caso("3", "transfusion, sutura"), _caso("4", "transfusion")])
    (_, f1), (_, f2) = conversacion("HPAF en abdomen", "hipotenso")
    assert que_cambio(antes, ahora, f1, f2) == (
        "Con hipotensión o choque e inestable: entran 2 casos nuevos; aparece transfusión (2 de 3).")
    assert que_cambio(antes, antes, f2, f2) == "Respecto de la respuesta anterior: mismos casos."
    assert que_cambio(None, ahora, f1, f2) is None


# ---------------------------------------------------------------------------
# responder con CONVERSACION=ficha
# ---------------------------------------------------------------------------
@pytest.fixture
def pipeline(monkeypatch):
    casos = [_caso("101", "hipotension, transfusion"), _caso("102", "hipotension, laparotomia")]
    buscadas = []
    monkeypatch.setattr(config, "CONVERSACION", "ficha")
    monkeypatch.setattr(config, "SALIDA", "prosa")
    monkeypatch.setattr(config, "SINTESIS_LLM", False)
    monkeypatch.setattr(config, "GUIAS", False)
    monkeypatch.setattr(generate, "confianza", lambda consulta, *a, **k: buscadas.append(consulta) or 0.9)
    monkeypatch.setattr(generate, "buscar", lambda *a, **k: casos)
    monkeypatch.setattr(generate, "guarda_casos", lambda consulta, casos, u: (casos, []))
    monkeypatch.setattr(generate, "reescribir", lambda *a, **k: pytest.fail("con la ficha no se reescribe"))
    monkeypatch.setattr(generate, "_ollama", lambda *a, **k: json.dumps(
        {"sugerencia": "En casos similares se hizo transfusión [caso: 101]."}))
    return buscadas


def test_responder_con_ficha(pipeline):
    r1 = generate.responder("HPAF en abdomen", historial=[])
    assert r1.tipo == "nuevo" and r1.ficha.mecanismo == "arma_de_fuego" and not r1.otro_paciente
    assert r1.consulta_reescrita is None and r1.cambio is None
    r2 = generate.responder("¿y si está hipotenso?", historial=[{"role": "user", "content": "HPAF en abdomen"}],
                            ficha=r1.ficha, anterior=r1)
    assert r2.tipo == "seguimiento" and "hipotensión o choque" in r2.ficha.afirmados
    assert pipeline == ["HPAF en abdomen", "HPAF en abdomen, hipotenso"]
    assert r2.consulta_reescrita == "HPAF en abdomen, hipotenso"
    assert r2.cambio.startswith("Con hipotensión o choque e inestable: mismos casos")
    r3 = generate.responder("mujer de 70 años con fractura de cadera", ficha=r2.ficha, anterior=r2)
    assert r3.tipo == "nuevo" and r3.otro_paciente and r3.ficha.grupo == "adulto mayor"
    assert r3.ficha.mecanismo is None and r3.cambio is None


def test_con_llm_la_ficha_no_esta(pipeline, monkeypatch):
    monkeypatch.setattr(config, "CONVERSACION", "llm")
    r = generate.responder("HPAF en abdomen", historial=[])
    assert r.ficha is None and r.tipo is None
