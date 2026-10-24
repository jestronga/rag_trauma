"""La respuesta ordenada como la revisión primaria del ATLS (rag/prioridades.py
y generate.prosa_atls; PLAN_CHATBOT_CLINICO.md, fase 4). Sin Ollama ni índices."""
from __future__ import annotations

import json

import pytest

from rag import config, generate, verificar
from rag.build_prompt import construir_prompt_atls, formatear_por_letra
from rag.hallazgos import LEXICO, normalizar
from rag.prioridades import DESPUES, letra_pasaje, letra_termino, rotulo, sin_letra_lexico
from rag.retrieve import FragmentoGuia, Resultado

CONSULTA = "herida por arma de fuego en tórax con hipoventilación"


def caso(rid: str, examen: str, evolucion: str) -> Resultado:
    texto = (f"Motivo de consulta: HPAF (herida por arma de fuego)\n\nExamen físico: {examen}\n\n"
             f"Evolución y hospitalización: {evolucion}")
    return Resultado(rid, texto, {"mecanismo_lesion": "arma_de_fuego", "region_anatomica": ["torax"]}, 0.7, 0.7)


def frag(fid: str, texto: str, fuente: str = "HUV", capitulo_num: int = 11) -> FragmentoGuia:
    ruta = f"{fuente} › Capítulo › Sección"
    return FragmentoGuia(id=fid, seccion_id=fid.rsplit("-", 1)[0], fuente=fuente, libro=fuente,
                         capitulo="Capítulo", titulos=["Sección"], pagina_ini=10, pagina_fin=10,
                         tipo="texto", idioma="en" if fuente == "ATLS" else "es",
                         texto=f"{ruta}\n{texto}", texto_contexto=f"{ruta}\n{texto}", n_tokens_contexto=60,
                         coseno=0.6, relevancia=0.8, capitulo_num=capitulo_num)


@pytest.fixture
def casos() -> list[Resultado]:
    return [caso("101", "hipoventilacion derecha, hipotension", "tubo de torax. Transfusion. Sutura"),
            caso("102", "hipoventilacion, hemotorax", "tubo de torax, sutura"),
            caso("103", "herida penetrante", "sutura, salida")]


@pytest.fixture
def guias() -> list[FragmentoGuia]:
    return [frag("HUV-11-007-01", "Ante hemotórax se coloca tubo de tórax en el quinto espacio intercostal."),
            frag("ATLS-06-010-01", "Hypotension after injury is hemorrhage until proven otherwise.", "ATLS", 6),
            frag("ATLS-10-004-01", "Long bone fractures are splinted.", "ATLS", 10)]


# ---------------------------------------------------------------------------
# Las letras
# ---------------------------------------------------------------------------
def test_cada_termino_del_lexico_tiene_letra():
    assert sin_letra_lexico() == []
    assert letra_termino("hipoventilación") == "B" and letra_termino("transfusión") == "C"
    assert letra_termino("intubación / ventilación mecánica") == "A" and letra_termino("sutura") == DESPUES


def test_letra_del_pasaje():
    assert letra_pasaje("ATLS", 5, "anything") == "B"                     # el capítulo manda
    assert letra_pasaje("ATLS", 21, "hemothorax and chest tube") == DESPUES  # en inglés, sin términos
    assert letra_pasaje("HUV", 11, "hemotórax: tubo de tórax") == "B"     # en español, por los términos
    assert letra_pasaje("HUV", 15, "hemorragia intraabdominal") == "C"    # la hemorragia de un pasaje es C
    assert letra_pasaje("ATLS10", 3, "fractura") == "C"                    # capítulo de choque
    assert letra_pasaje("HUV", 19, "sutura") == DESPUES


def test_rotulos():
    assert rotulo("B") == "**B · Respiración.**"
    assert rotulo(DESPUES) == "**Después de la revisión primaria.**"


def test_los_rotulos_no_son_vocabulario_del_verificador():
    assert "exanguinante" not in verificar._VOCABULARIO_CODIGO


# ---------------------------------------------------------------------------
# El prompt
# ---------------------------------------------------------------------------
def test_prompt_por_letras(casos, guias):
    letras = generate.letras_atls(CONSULTA, casos, guias)
    assert letras.casos == ["B", "C", DESPUES] and letras.manuales == ["B", "C", DESPUES]
    lista = formatear_por_letra(letras.elementos, len(casos), letras.de_termino)
    assert lista.index("[B] Respiración") < lista.index("tubo de tórax") < lista.index("[C] Circulación")
    prompt = construir_prompt_atls(CONSULTA, casos, letras.elementos, guias, letras.de_termino, letras.de_seccion)
    assert "[guía: HUV-11-007]" in prompt and '"casos": {"B": "…", "C": "…", "después": "…"}' in prompt
    esquema = generate.esquema_atls(letras, True, True)
    assert esquema["properties"]["casos"]["required"] == ["B", "C", DESPUES]
    assert esquema["properties"]["casos"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# La verificación por letra
# ---------------------------------------------------------------------------
def test_lo_que_nombra_un_texto_es_de_su_letra(casos, guias):
    letras = generate.letras_atls(CONSULTA, casos, guias)
    datos = {"casos": {"B": "En casos similares se colocó tubo de tórax [caso: 101] [caso: 102].",
                       "C": "En casos similares se hizo tubo de tórax [caso: 101]."},
             "manuales": {"C": "El manual del HUV recomienda tubo de tórax ante hemotórax [guía: HUV-11-007].",
                          DESPUES: "El ATLS recomienda inmovilizar las fracturas [guía: ATLS-10-004]."}}
    textos = {(t.fuente, t.letra): t for t in generate.verificar_atls(datos, casos, CONSULTA, guias, letras)}
    assert textos[("casos", "B")].motivos == []
    assert "«tubo de tórax» es de la letra B, no de la C" in textos[("casos", "C")].motivos
    # lo que se hizo, de "después", pasa en una letra primaria
    datos = {"casos": {"C": "En casos con hipotensión se hizo transfusión y sutura [caso: 101]."}}
    assert generate.verificar_atls(datos, casos, CONSULTA, guias, letras)[0].motivos == []
    # pero una letra primaria tiene que nombrar algo suyo
    datos = {"casos": {"B": "En casos similares se hizo sutura [caso: 101] [caso: 102] [caso: 103]."}}
    assert generate.verificar_atls(datos, casos, CONSULTA, guias, letras)[0].motivos == \
        ["no nombra nada de la letra B"]
    # el texto de los manuales va en la letra del pasaje que cita (la decide el código)
    assert ("guias", "C") not in textos and textos[("guias", "B")].motivos == []
    assert textos[("guias", DESPUES)].motivos == []
    # si cita pasajes de dos letras primarias, no se mueve y no pasa
    datos = {"manuales": {"B": "El manual del HUV recomienda tubo de tórax [guía: HUV-11-007] y el ATLS "
                               "dice algo [guía: ATLS-06-010]."}}
    t = generate.verificar_atls(datos, casos, CONSULTA, guias, letras)[0]
    assert t.letra == "B" and "cita [guía: ATLS-06-010], que es de la letra C" in t.motivos


def test_un_texto_de_los_casos_no_cita_manuales(casos, guias):
    letras = generate.letras_atls(CONSULTA, casos, guias)
    datos = {"casos": {"B": "Tubo de tórax [guía: HUV-11-007]."}}
    t = generate.verificar_atls(datos, casos, CONSULTA, guias, letras)[0]
    assert "cita un manual en el apartado de los casos" in t.motivos


def test_el_rotulo_copiado_se_quita(casos, guias):
    letras = generate.letras_atls(CONSULTA, casos, guias)
    datos = {"casos": {"B": "B · Respiración: en casos similares se colocó tubo de tórax [caso: 101]."}}
    t = generate.verificar_atls(datos, casos, CONSULTA, guias, letras)[0]
    assert t.texto.startswith("En casos similares") and not t.motivos


# ---------------------------------------------------------------------------
# prosa_atls y responder
# ---------------------------------------------------------------------------
def _modelo(monkeypatch, salida: dict | str, llamadas: list | None = None):
    def falso(mensajes, formato=None, **_):
        if llamadas is not None:
            llamadas.append(formato)
        return salida if isinstance(salida, str) else json.dumps(salida)
    monkeypatch.setattr(generate, "_ollama", falso)


SALIDA = {"casos": {"B": "En casos similares se colocó tubo de tórax [caso: 101] [caso: 102].",
                    "C": "En casos similares se hizo tubo de tórax [caso: 101]."},
          "manuales": {"B": "El manual del HUV recomienda tubo de tórax ante hemotórax [guía: HUV-11-007]."}}


def test_la_letra_que_no_pasa_va_desde_su_fuente(casos, guias, monkeypatch):
    _modelo(monkeypatch, SALIDA)
    p_casos, p_guias, pasa, motivos, _ = generate.prosa_atls(CONSULTA, casos, guias, siempre=False)
    assert not pasa and any(m.startswith("C · ") for m in motivos)
    assert p_casos.startswith("**B · Respiración.** En casos similares se colocó tubo de tórax")
    assert "**C · Circulación.** En casos similares consta hipotensión o choque [caso: 101]" in p_casos
    assert p_guias.startswith("**B · Respiración.** El manual del HUV recomienda")
    # con PROSA_SIEMPRE, lo que escribió el modelo
    p_casos, _, _, _, _ = generate.prosa_atls(CONSULTA, casos, guias, siempre=True)
    assert "**C · Circulación.** En casos similares se hizo tubo de tórax" in p_casos


def test_la_oracion_cortada_por_el_tope_se_quita(casos, guias, monkeypatch):
    letras = generate.letras_atls(CONSULTA, casos, guias)
    assert generate.esquema_atls(letras, True, True)["properties"]["casos"]["properties"]["B"]["maxLength"] == \
        generate.MAX_CARACTERES_LETRA
    datos = {"casos": {"B": "En casos similares se colocó tubo de tórax [caso: 101]. Y también se hizo",
                       "C": "En casos similares se hizo transfusión en"}}
    textos = {t.letra: t for t in generate.verificar_atls(datos, casos, CONSULTA, guias, letras)}
    assert textos["B"].texto == "En casos similares se colocó tubo de tórax [caso: 101]." and not textos["B"].motivos
    assert textos["C"].texto == "" and textos["C"].motivos == ["la oración quedó cortada"]
    _modelo(monkeypatch, datos)
    p_casos, *_ = generate.prosa_atls(CONSULTA, casos, guias, siempre=True)
    assert "**C · Circulación.** En casos similares consta" in p_casos      # armada desde su fuente


def test_json_invalido_deja_la_version_determinista(casos, guias, monkeypatch):
    _modelo(monkeypatch, '{"casos": {"B": "En casos')
    p_casos, p_guias, pasa, motivos, detalle = generate.prosa_atls(CONSULTA, casos, guias, siempre=True)
    assert (p_casos, p_guias, pasa) == ("", "", False) and detalle["json_invalido"]


@pytest.fixture
def entorno(casos, guias, monkeypatch):
    monkeypatch.setattr(config, "SALIDA", "prosa")
    monkeypatch.setattr(config, "SINTESIS_LLM", False)
    monkeypatch.setattr(config, "PROSA_SIEMPRE", False)
    monkeypatch.setattr(config, "ORDEN_ATLS", True)
    monkeypatch.setattr(config, "GUIAS", True)
    monkeypatch.setattr(config, "PREGUNTAS", False)
    monkeypatch.setattr(generate, "confianza", lambda *a, **k: 0.9)
    monkeypatch.setattr(generate, "buscar", lambda *a, **k: casos)
    monkeypatch.setattr(generate, "guarda_casos", lambda consulta, casos, u: (casos, []))
    monkeypatch.setattr(generate, "hay_guias", lambda: True)
    monkeypatch.setattr(generate, "buscar_guias", lambda *a, **k: guias)


@pytest.mark.parametrize("conversacion", ["ficha", "llm"])
def test_responder_por_letras_solo_con_un_paciente(entorno, monkeypatch, conversacion):
    monkeypatch.setattr(config, "CONVERSACION", conversacion)
    formatos = []
    _modelo(monkeypatch, SALIDA, formatos)
    monkeypatch.setattr(config, "PROSA_SIEMPRE_ATLS", False)
    r = generate.responder(CONSULTA)
    assert r.orden_atls and "**B · Respiración.**" in r.texto and "casos" in formatos[-1]["properties"]
    # la C de los casos no pasa (tubo de tórax es de la B): va armada desde su fuente y se dice
    assert r.letras_armadas == ["C de los casos"] and "En casos similares consta" in r.texto
    monkeypatch.setattr(config, "PROSA_SIEMPRE_ATLS", True)
    r = generate.responder(CONSULTA)
    assert r.letras_armadas == [] and "En casos similares se hizo tubo de tórax" in r.texto
    # una pregunta de procedimiento: la prosa de siempre
    _modelo(monkeypatch, {"sugerencia": "El manual del HUV recomienda tubo de tórax [guía: HUV-11-007]."},
            formatos)
    r = generate.responder("¿cuándo está indicado colocar un tubo de tórax en trauma?")
    assert not r.orden_atls and "sugerencia" in formatos[-1]["properties"]
    assert "Respiración" not in r.texto


def test_sin_orden_atls_la_prosa_de_siempre(entorno, monkeypatch):
    monkeypatch.setattr(config, "ORDEN_ATLS", False)
    _modelo(monkeypatch, {"sugerencia": "En casos similares se colocó tubo de tórax [caso: 101]."})
    r = generate.responder(CONSULTA)
    assert not r.orden_atls and "Respiración" not in r.texto


def test_cada_termino_del_lexico_tiene_su_letra_en_la_tabla_del_plan():
    # las del plan, literal: si se cambian, que sea a propósito (decisión 4, el médico)
    plan = {"intubación / ventilación mecánica": "A", "hipoventilación": "B", "neumotórax": "B",
            "hemotórax": "B", "tubo de tórax": "B", "hipotensión o choque": "C",
            "hemoperitoneo / líquido libre": "C", "ecografía / FAST": "C", "transfusión": "C",
            "alteración del Glasgow": "D", "pérdida de conciencia": "D", "anisocoria": "D",
            "déficit neurológico": "D", "fractura": DESPUES, "sutura": DESPUES, "antibiótico": DESPUES}
    assert {t: letra_termino(t) for t in plan} == plan
    assert {normalizar(t) for _, t, _ in LEXICO} >= {normalizar(t) for t in plan}
