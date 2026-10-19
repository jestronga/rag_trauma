"""Los libros como segunda fuente (fase 5 de PLAN_INDEXAR_LIBROS.md): el
verificador con [guía: ID], el prompt con dos bloques y responder con las dos
guardas. Sin Ollama ni índices: casos y fragmentos armados a mano."""
from __future__ import annotations

import json

import pytest

from rag import config, generate
from rag.build_prompt import (
    AVISO_DOS_FUENTES, AVISO_SOLO_CASOS, AVISO_SOLO_GUIAS, INSTRUCCION_PROSA_GUIAS, INSTRUCCION_PROSA_SOLO_GUIAS,
    ROTULO_CASOS, ROTULO_MANUALES, ROTULO_SUGERENCIAS, SISTEMA_PROSA, SISTEMA_PROSA_GUIAS, agrupar_guias,
    aviso_pasajes, cabecera_guia, construir_prompt, construir_prompt_guias, texto_guia,
)
from rag.hallazgos import elegibles
from rag.retrieve import FragmentoGuia, Resultado
from rag.verificar import cifras, verificar_prosa

CONSULTA = "herida por arma de fuego en tórax con hipoventilación"
RUTA_HUV = "HUV › Trauma torácico penetrante › TRATAMIENTO › Neumotórax a tensión"
RUTA_ATLS = "ATLS › Breathing and Ventilation › HEMOTHORAX ASSESSMENT"


def caso(rid: str, examen: str, evolucion: str, desenlace: str = "alta") -> Resultado:
    texto = (f"Motivo de consulta: HPAF (herida por arma de fuego)\n\nExamen físico: {examen}\n\n"
             f"Evolución y hospitalización: {evolucion}")
    meta = {"mecanismo_lesion": "arma_de_fuego", "region_anatomica": ["torax"], "desenlace": desenlace}
    return Resultado(rid, texto, meta, score=0.7, score_norm=0.7)


def frag(fid: str, texto: str, fuente: str = "HUV", pagina: int = 153, tipo: str = "texto") -> FragmentoGuia:
    ruta = RUTA_HUV if fuente == "HUV" else RUTA_ATLS
    titulos = ruta.split(" › ")[2:]
    return FragmentoGuia(id=fid, seccion_id=fid.rsplit("-", 1)[0], fuente=fuente, libro=fuente,
                         capitulo=ruta.split(" › ")[1], titulos=titulos, pagina_ini=pagina,
                         pagina_fin=pagina, tipo=tipo, idioma="es" if fuente == "HUV" else "en",
                         texto=f"{ruta}\n{texto}", texto_contexto=f"{ruta}\n{texto}",
                         n_tokens_contexto=60, coseno=0.6, relevancia=0.8,
                         capitulo_num=11 if fuente == "HUV" else 5)


@pytest.fixture
def casos() -> list[Resultado]:
    return [
        caso("101", "hipoventilacion derecha", "se pasa tubo de torax, drena 300cc. Salida"),
        caso("102", "hipoventilacion, hemotorax", "tubo de torax. Llevado a cirugia: toracotomia", "cirugia"),
        caso("103", "herida penetrante, hipoventilacion", "tubo de torax, observacion", "no_determinado"),
        caso("104", "herida superficial", "sutura, salida"),
    ]


@pytest.fixture
def guias() -> list[FragmentoGuia]:
    return [
        frag("HUV-11-007-01", "El neumotórax a tensión se trata con descompresión inmediata y luego "
                              "tubo de tórax en el quinto espacio intercostal."),
        frag("ATLS-05-013-01", "A massive hemothorax is the accumulation of more than 1,500 mL of "
                               "blood in the chest cavity.", fuente="ATLS", pagina=70),
        frag("HUV-11-007-02", "luego tubo de tórax en el quinto espacio intercostal. Se toma "
                              "radiografía de control a las 6 horas."),
    ]


CASOS_OK = ("Conviene valorar hemotórax [caso: 102]. "
            "En casos similares se hizo tubo de tórax [caso: 101] [caso: 102] [caso: 103].")
HUV_OK = "El manual del HUV recomienda descompresión inmediata y luego tubo de tórax [guía: HUV-11-007]."
ATLS_OK = "El ATLS recomienda considerar hemotórax masivo cuando se acumulan más de 1500 mL [guía: ATLS-05-013]."


# ---------------------------------------------------------------------------
# Verificador
# ---------------------------------------------------------------------------
def test_prosa_con_las_dos_fuentes_pasa(casos, guias):
    ok, motivos = verificar_prosa(f"{CASOS_OK} {HUV_OK} {ATLS_OK}", casos, CONSULTA, guias)
    assert ok, motivos


def test_cita_de_guia_inventada(casos, guias):
    ok, motivos = verificar_prosa("El ATLS recomienda una revisión [guía: ATLS-99-001].", casos, CONSULTA, guias)
    assert not ok and any("ATLS-99-001" in m for m in motivos)


def test_se_acepta_el_id_del_fragmento(casos, guias):
    ok, motivos = verificar_prosa(HUV_OK.replace("HUV-11-007]", "HUV-11-007-02]"), casos, CONSULTA, guias)
    assert ok, motivos


def test_no_mezcla_casos_y_manuales_en_una_oracion(casos, guias):
    prosa = "Se hizo tubo de tórax [caso: 101] como recomienda el manual del HUV [guía: HUV-11-007]."
    ok, motivos = verificar_prosa(prosa, casos, CONSULTA, guias)
    assert not ok and any("mezcla" in m for m in motivos)


def test_las_cifras_tienen_que_estar_en_la_guia_citada(casos, guias):
    ok, motivos = verificar_prosa(ATLS_OK.replace("1500", "1000"), casos, CONSULTA, guias)
    assert not ok and any("«1000»" in m for m in motivos)


def test_formas_de_un_numero():
    assert "1500" in cifras("more than 1,500 mL") and "0.5" in cifras("0,5 mg")


def test_atls_solo_revisa_cita_cifras_y_tono(casos, guias):
    # en español, con palabras que no están en el texto en inglés (decisión 2)
    prosa = "El ATLS recomienda vigilar la acumulación de más de 1500 mililitros [guía: ATLS-05-013]."
    ok, motivos = verificar_prosa(prosa, casos, CONSULTA, guias)
    assert ok, motivos


def test_huv_termino_que_no_esta_en_el_fragmento(casos, guias):
    ok, motivos = verificar_prosa("El manual del HUV recomienda toracotomía [guía: HUV-11-007].",
                                  casos, CONSULTA, guias)
    assert not ok and any("«toracotomía» no consta en la guía citada" in m for m in motivos)


def test_huv_palabra_que_no_esta_en_el_fragmento(casos, guias):
    ok, motivos = verificar_prosa("El manual del HUV recomienda gammagrafía [guía: HUV-11-007].",
                                  casos, CONSULTA, guias)
    assert not ok and any("gammagrafia" in m for m in motivos)


def test_lo_que_dice_la_guia_no_se_atribuye_a_un_caso(casos, guias):
    # radiografía está en el fragmento del HUV, no en ningún caso
    ok, motivos = verificar_prosa("En casos similares se hizo radiografía [caso: 101].", casos, CONSULTA, guias)
    assert not ok and any("«radiografía» no consta en ningún caso" in m for m in motivos)


def test_el_manual_nombrado_es_el_citado(casos, guias):
    # medido el 17/10: "El ATLS recomienda… [guía: HUV-04-013]"
    prosa = "El ATLS recomienda descompresión inmediata y luego tubo de tórax [guía: HUV-11-007]."
    ok, motivos = verificar_prosa(prosa, casos, CONSULTA, guias)
    assert not ok and any("atribuye al ATLS" in m for m in motivos)


def test_orden_en_una_oracion_de_guia(casos, guias):
    ok, motivos = verificar_prosa("El manual del HUV dice que se debe pasar tubo de tórax [guía: HUV-11-007].",
                                  casos, CONSULTA, guias)
    assert not ok and any("da una orden" in m for m in motivos)


def test_con_fragmentos_una_cifra_sin_cita_no_pasa(casos, guias):
    prosa = "Conviene repetir el control a las 6 horas."
    assert any("cifras sin citar" in m for m in verificar_prosa(prosa, casos, CONSULTA, guias)[1])
    # sin fragmentos la regla no existe: las de los casos no cambian
    assert not any("cifras sin citar" in m for m in verificar_prosa(prosa, casos, CONSULTA)[1])


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def test_texto_guia_une_ventanas_sin_ruta_ni_solape(guias):
    grupos = agrupar_guias(guias)
    assert list(grupos) == ["HUV-11-007", "ATLS-05-013"]          # orden del mejor fragmento
    texto = texto_guia(grupos["HUV-11-007"])
    assert "›" not in texto
    assert texto.count("quinto espacio intercostal") == 1
    assert texto.endswith("a las 6 horas.")


def test_las_tablas_van_aparte():
    fs = [frag("HUV-11-007-01", "Texto de la sección."),
          frag("HUV-11-007-02", "Tabla 1. Indicaciones\nFila uno", tipo="tabla")]
    assert texto_guia(fs) == "Texto de la sección.\n\nTabla 1. Indicaciones\nFila uno"


def test_cabecera_con_capitulo_y_paginas(guias):
    assert cabecera_guia(agrupar_guias(guias)["HUV-11-007"]) == (
        "manual del HUV · cap. 11 Trauma torácico penetrante › TRATAMIENTO › Neumotórax a tensión · p. 153")
    dos = [frag("HUV-11-007-01", "a", pagina=153), frag("HUV-11-007-02", "b", pagina=154)]
    assert cabecera_guia(dos).endswith("pp. 153–154")


def test_prompt_con_las_dos_fuentes(casos, guias):
    p = construir_prompt_guias(CONSULTA, casos, elegibles(casos, CONSULTA), guias)
    assert "CASOS SIMILARES DEL REGISTRO (4)" in p and "ELEMENTOS DOCUMENTADOS" in p
    assert "FRAGMENTOS DE LOS MANUALES (2)" in p
    assert "--- [guía: HUV-11-007] manual del HUV" in p
    assert "--- [guía: ATLS-05-013] ATLS 11.ª ed." in p and "(en inglés) ---" in p
    assert p.endswith(INSTRUCCION_PROSA_GUIAS)


def test_prompt_solo_manuales(guias):
    p = construir_prompt_guias(CONSULTA, [], [], guias)
    assert "CASOS SIMILARES" not in p and p.endswith(INSTRUCCION_PROSA_SOLO_GUIAS)


def test_solo_casos_el_prompt_no_cambia(casos):
    mensajes = generate.mensajes_prosa(CONSULTA, casos, [])
    assert mensajes[0]["content"] == SISTEMA_PROSA
    assert mensajes[1]["content"] == construir_prompt(CONSULTA, casos, elegibles(casos, CONSULTA), prosa=True)


def test_con_casos_sin_elementos_no_se_llama_al_modelo(guias):
    sin_nada = [caso("201", "paciente estable", "salida"), caso("202", "estable", "salida")]
    assert generate.mensajes_prosa(CONSULTA, sin_nada, guias) is None


def test_solo_manuales_usa_el_sistema_de_dos_fuentes(guias):
    mensajes = generate.mensajes_prosa(CONSULTA, [], guias)
    assert mensajes[0]["content"] == SISTEMA_PROSA_GUIAS


# ---------------------------------------------------------------------------
# responder: las dos guardas
# ---------------------------------------------------------------------------
@pytest.fixture
def entorno(casos, guias, monkeypatch):
    monkeypatch.setattr(config, "SALIDA", "prosa")
    monkeypatch.setattr(config, "SINTESIS_LLM", False)
    monkeypatch.setattr(config, "PROSA_SIEMPRE", False)
    monkeypatch.setattr(config, "GUIAS", True)
    monkeypatch.setattr(generate, "confianza", lambda *a, **k: 0.9)
    monkeypatch.setattr(generate, "buscar", lambda *a, **k: casos)
    monkeypatch.setattr(generate, "hay_guias", lambda: True)
    monkeypatch.setattr(generate, "buscar_guias", lambda *a, **k: guias)
    llamadas: list[list[dict]] = []

    def modelo(prosa: str):
        def falso(mensajes, **_):
            llamadas.append(mensajes)
            return json.dumps({"sugerencia": prosa})
        monkeypatch.setattr(generate, "_ollama", falso)
    modelo.llamadas = llamadas
    return modelo


def seccion(r, fuente: str) -> str:
    """El texto del apartado `fuente` de la respuesta ("" si no está)."""
    return next((s.texto for s in r.secciones if s.fuente == fuente), "")


def test_las_dos_fuentes(entorno, guias):
    entorno(f"{CASOS_OK} {HUV_OK}")
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is True, r.motivos
    assert entorno.llamadas[0][0]["content"] == SISTEMA_PROSA_GUIAS
    assert [s.fuente for s in r.secciones] == ["casos", "guias", "pie"]
    # casos: título, prosa y respaldo; sin tabla (el detalle, en el desplegable)
    casos = seccion(r, "casos")
    assert casos.startswith(f"{ROTULO_CASOS}\n\n{CASOS_OK}")
    assert "| Caso |" not in casos and "Respaldo: 4 casos" in casos
    # manuales: título y prosa; los pasajes, en el desplegable de la app
    assert seccion(r, "guias") == f"{ROTULO_MANUALES}\n\n{HUV_OK}"
    # que pueden no coincidir lo dice el código, no el modelo (decisión 3)
    assert seccion(r, "pie") == AVISO_DOS_FUENTES and r.texto.endswith(AVISO_DOS_FUENTES)
    assert r.texto == "\n\n".join(s.texto for s in r.secciones)
    assert r.guias == guias and r.citas_invalidas == []
    assert "HUV-11-007" in r.citas_validas


def test_solo_manuales_si_los_casos_no_pasan(entorno, monkeypatch):
    monkeypatch.setattr(generate, "confianza", lambda *a, **k: 0.3)
    entorno(HUV_OK)
    r = generate.responder(CONSULTA)
    assert not r.abstuvo and r.casos == [] and r.prosa_verificada is True, r.motivos
    assert seccion(r, "casos") == f"{ROTULO_CASOS}\n\n{AVISO_SOLO_GUIAS}"
    assert seccion(r, "guias").startswith(f"{ROTULO_MANUALES}\n\n{HUV_OK}")
    assert "pie" not in [s.fuente for s in r.secciones]


def test_solo_casos_si_los_manuales_no_pasan(entorno, monkeypatch):
    monkeypatch.setattr(generate, "buscar_guias", lambda *a, **k: [])
    entorno(CASOS_OK)
    r = generate.responder(CONSULTA)
    assert entorno.llamadas[0][0]["content"] == SISTEMA_PROSA       # el prompt de siempre
    assert r.prosa_verificada is True
    assert seccion(r, "guias") == f"{ROTULO_MANUALES}\n\n{AVISO_SOLO_CASOS}"


def test_se_abstiene_si_no_pasa_ninguna(entorno, monkeypatch):
    monkeypatch.setattr(generate, "confianza", lambda *a, **k: 0.3)
    monkeypatch.setattr(generate, "buscar_guias", lambda *a, **k: [])
    r = generate.responder(CONSULTA)
    assert r.abstuvo and r.secciones == []
    assert r.texto.startswith("No encontré casos en el registro ni pasajes de los manuales")


def test_si_falla_la_parte_de_los_manuales_los_casos_quedan_en_prosa(entorno, guias):
    entorno(f"{CASOS_OK} El manual del HUV recomienda toracotomía [guía: HUV-11-007].")
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is False and r.fuentes_en_prosa == ["casos"]
    assert seccion(r, "casos").startswith(f"{ROTULO_CASOS}\n\n{CASOS_OK}")
    assert ROTULO_SUGERENCIAS not in r.texto                  # los casos no van en determinista
    # los manuales, sin prosa: una línea que remite al desplegable
    assert seccion(r, "guias") == f"{ROTULO_MANUALES}\n\n{aviso_pasajes(guias)}"
    assert "Hay 2 pasajes" in aviso_pasajes(guias)
    assert "recomienda toracotomía" not in r.texto


def test_si_falla_la_parte_de_los_casos_los_manuales_quedan_en_prosa(entorno):
    entorno(f"Conviene valorar neumotórax [caso: 101]. {HUV_OK}")
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is False and r.fuentes_en_prosa == ["guias"]
    assert ROTULO_SUGERENCIAS in seccion(r, "casos")          # los casos, en determinista
    assert seccion(r, "guias").startswith(f"{ROTULO_MANUALES}\n\n{HUV_OK}")
    assert "valorar neumotórax" not in r.texto


def test_si_falla_todo_las_dos_en_determinista(entorno, guias):
    entorno("Conviene valorar neumotórax [caso: 101]. El manual del HUV recomienda toracotomía "
            "[guía: HUV-11-007].")
    r = generate.responder(CONSULTA)
    assert r.fuentes_en_prosa == []
    assert ROTULO_SUGERENCIAS in seccion(r, "casos")
    assert seccion(r, "guias") == f"{ROTULO_MANUALES}\n\n{aviso_pasajes(guias)}"


def test_sin_oraciones_de_los_manuales_remite_al_desplegable(entorno, guias):
    # la prosa pasa, pero no dijo nada de los manuales: el apartado no queda vacío
    entorno(CASOS_OK)
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is True and r.fuentes_en_prosa == ["casos"]
    assert seccion(r, "guias") == f"{ROTULO_MANUALES}\n\n{aviso_pasajes(guias)}"


def test_prosa_siempre_las_dos_partes_en_prosa(entorno, monkeypatch):
    # la demo (18/10): todo en prosa, aunque la parte de los manuales no pase
    monkeypatch.setattr(config, "PROSA_SIEMPRE", True)
    entorno(f"{CASOS_OK} El manual del HUV recomienda toracotomía [guía: HUV-11-007].")
    r = generate.responder(CONSULTA)
    assert r.prosa_verificada is False and r.fuentes_en_prosa == ["casos", "guias"]
    assert "recomienda toracotomía" in seccion(r, "guias")
    assert "| Caso |" not in seccion(r, "casos") and "Pasajes consultados" not in seccion(r, "guias")


def test_sin_guias_solo_el_apartado_de_casos(entorno, monkeypatch):
    def no_llamar(*a, **k):
        raise AssertionError("con GUIAS=0 no se buscan los libros")
    monkeypatch.setattr(config, "GUIAS", False)
    monkeypatch.setattr(generate, "buscar_guias", no_llamar)
    entorno(CASOS_OK)
    r = generate.responder(CONSULTA)
    assert r.guias == [] and r.prosa_verificada is True
    assert [s.fuente for s in r.secciones] == ["casos"] and AVISO_SOLO_CASOS not in r.texto


def test_verificar_citas_de_guias(casos, guias):
    validas, invalidas = generate.verificar_citas(
        "Algo [caso: 101]. Otro [guía: HUV-11-007]. Otro más [guía: ATLS-99-001].", casos, guias)
    assert validas == ["101", "HUV-11-007"] and invalidas == ["ATLS-99-001"]
