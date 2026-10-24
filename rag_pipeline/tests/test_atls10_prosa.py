"""El ATLS 10.ª ed. en el prompt, el verificador y la respuesta (fase 5 de
PLAN_INDEXAR_ATLS10.md; decisiones 1, 3 y 4). Sin Ollama ni índices."""
from __future__ import annotations

from rag import build_prompt, config, generate
from rag.build_prompt import (
    AVISO_ATLS10, AVISO_EDICIONES, INSTRUCCION_PROSA_GUIAS, INSTRUCCION_PROSA_GUIAS_ATLS10,
    INSTRUCCION_PROSA_SOLO_GUIAS, INSTRUCCION_PROSA_SOLO_GUIAS_ATLS10, SISTEMA_PROSA_GUIAS,
    SISTEMA_PROSA_GUIAS_ATLS10, cabecera_guia, construir_prompt_guias,
)
from rag.retrieve import FragmentoGuia
from rag.verificar import verificar_prosa

CONSULTA = "herida por arma de fuego en tórax con hipoventilación"


def frag(fid: str, texto: str, fuente: str = "ATLS10", capitulo: str = "Trauma torácico",
         capitulo_num: int = 4, titulos: list[str] | None = None) -> FragmentoGuia:
    titulos = titulos or ["Encuesta primaria: lesiones potencialmente mortales", "Tensión neumotoraxica"]
    ruta = " › ".join([fuente, capitulo, *titulos])
    return FragmentoGuia(id=fid, seccion_id=fid.rsplit("-", 1)[0], fuente=fuente, libro=fuente,
                         capitulo=capitulo, titulos=titulos, pagina_ini=66, pagina_fin=66, tipo="texto",
                         idioma="en" if fuente == "ATLS" else "es", texto=f"{ruta}\n{texto}",
                         texto_contexto=f"{ruta}\n{texto}", n_tokens_contexto=60, coseno=0.6,
                         relevancia=0.8, capitulo_num=capitulo_num)


ATLS10 = frag("ATLS10-04-007-01", "El neumotórax a tensión requiere descompresión inmediata con aguja "
                                  "en el quinto espacio intercostal.")
ATLS11 = frag("ATLS-05-013-01", "A massive hemothorax is the accumulation of more than 1,500 mL of blood.",
              fuente="ATLS", capitulo="Breathing", capitulo_num=5, titulos=["HEMOTHORAX ASSESSMENT"])
HUV = frag("HUV-11-007-01", "El neumotórax a tensión se trata con descompresión inmediata.",
           fuente="HUV", capitulo="Trauma torácico penetrante", capitulo_num=11, titulos=["TRATAMIENTO"])


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------
def test_las_variantes_con_el_atls10_cambian_lo_que_deben():
    for original, variante in ((SISTEMA_PROSA_GUIAS, SISTEMA_PROSA_GUIAS_ATLS10),
                               (INSTRUCCION_PROSA_GUIAS, INSTRUCCION_PROSA_GUIAS_ATLS10),
                               (INSTRUCCION_PROSA_SOLO_GUIAS, INSTRUCCION_PROSA_SOLO_GUIAS_ATLS10)):
        assert variante != original and "10.ª edición del ATLS" in variante
    assert build_prompt._TRES_MANUALES in SISTEMA_PROSA_GUIAS_ATLS10
    assert build_prompt._COMO_EMPIEZA_TRES in INSTRUCCION_PROSA_GUIAS_ATLS10
    assert build_prompt._COMO_EMPIEZA_TRES in INSTRUCCION_PROSA_SOLO_GUIAS_ATLS10
    assert build_prompt._NO_COMPARES_TRES in INSTRUCCION_PROSA_GUIAS_ATLS10
    assert build_prompt._TONO_TRES in INSTRUCCION_PROSA_GUIAS_ATLS10
    assert build_prompt._TONO_TRES in INSTRUCCION_PROSA_SOLO_GUIAS_ATLS10


def test_el_prompt_cambia_solo_con_pasajes_del_atls10():
    sin = construir_prompt_guias(CONSULTA, [], [], [ATLS11, HUV])
    assert INSTRUCCION_PROSA_SOLO_GUIAS in sin and "10.ª" not in sin
    con = construir_prompt_guias(CONSULTA, [], [], [ATLS10, ATLS11])
    assert INSTRUCCION_PROSA_SOLO_GUIAS_ATLS10 in con
    # la cabecera del pasaje dice la edición y el origen (decisión 1)
    assert f"--- [guía: ATLS10-04-007] {config.NOMBRES_LIBROS['ATLS10']} · cap. 4 Trauma torácico" in con
    assert generate.mensajes_prosa(CONSULTA, [], [ATLS10])[0]["content"] == SISTEMA_PROSA_GUIAS_ATLS10
    assert generate.mensajes_prosa(CONSULTA, [], [HUV])[0]["content"] == SISTEMA_PROSA_GUIAS


def test_cabecera_de_un_apendice_sin_el_numero_interno():
    ocular = frag("ATLS10-14-014-01", "texto", capitulo="Apéndice A: Trauma ocular", capitulo_num=14,
                  titulos=["Las lesiones oculares específicas", "Globos de abiertas"])
    cab = cabecera_guia([ocular])
    assert "cap. 14" not in cab and " · Apéndice A: Trauma ocular › Las lesiones oculares" in cab
    assert cabecera_guia([ATLS10]).split(" · ")[1].startswith("cap. 4 Trauma torácico")


# ---------------------------------------------------------------------------
# verificador: regla 8 por edición (decisión 3) y las demás con el ATLS 10
# ---------------------------------------------------------------------------
def _verificar(prosa: str, guias=(ATLS10, ATLS11, HUV)):
    return verificar_prosa(prosa, [], CONSULTA, list(guias))


def test_la_oracion_dice_la_edicion_del_atls10():
    ok, motivos = _verificar("La 10.ª edición del ATLS recomienda descompresión inmediata con aguja "
                             "[guía: ATLS10-04-007].")
    assert ok, motivos
    ok, motivos = _verificar("El ATLS recomienda descompresión inmediata con aguja [guía: ATLS10-04-007].")
    assert not ok and any("sin decir la edición" in m for m in motivos)


def test_la_edicion_nombrada_es_la_citada():
    ok, motivos = _verificar("La 10.ª edición del ATLS recomienda considerar hemotórax masivo "
                             "[guía: ATLS-05-013].")
    assert not ok and any("atribuye a la 10.ª edición" in m for m in motivos)
    ok, motivos = _verificar("El ATLS 11.ª ed. recomienda descompresión inmediata con aguja "
                             "[guía: ATLS10-04-007].")
    assert not ok and any("atribuye a la 11.ª edición" in m for m in motivos)


def test_la_edicion_no_cuenta_como_cifra():
    # "11" y "10" no están en el pasaje, y no por eso es una cifra inventada
    ok, motivos = _verificar("El ATLS 11.ª ed. recomienda considerar hemotórax masivo cuando se acumulan "
                             "más de 1500 mL [guía: ATLS-05-013].")
    assert ok, motivos
    ok, motivos = _verificar("La 10.ª edición del ATLS recomienda descompresión con aguja en el sexto "
                             "espacio intercostal, a 6 cm [guía: ATLS10-04-007].")
    assert not ok and any("«6» no está" in m for m in motivos)


def test_el_atls10_se_revisa_como_el_huv():
    # en español: términos del léxico y vocabulario contra el pasaje (regla 10)
    ok, motivos = _verificar("La 10.ª edición del ATLS recomienda toracotomía inmediata "
                             "[guía: ATLS10-04-007].")
    assert not ok and any("toracotomía" in m for m in motivos)


def test_sin_el_atls10_la_regla_es_la_de_antes():
    ok, motivos = _verificar("El ATLS recomienda descompresión inmediata [guía: HUV-11-007].", guias=(HUV,))
    assert not ok and any("atribuye al ATLS un pasaje de otro manual" in m for m in motivos)


# ---------------------------------------------------------------------------
# respuesta: el aviso de las ediciones (decisiones 1 y 4)
# ---------------------------------------------------------------------------
def _apartado_manuales(guias) -> str:
    secciones = generate.armar_secciones([], CONSULTA, guias, con_guias=True, prosa_guias="Prosa.")
    return next(s.texto for s in secciones if s.fuente == "guias")


def test_aviso_de_las_ediciones():
    assert _apartado_manuales([ATLS10, ATLS11]).endswith(AVISO_EDICIONES)
    assert _apartado_manuales([ATLS10, HUV]).endswith(AVISO_ATLS10)
    texto = _apartado_manuales([ATLS11, HUV])
    assert AVISO_ATLS10 not in texto and AVISO_EDICIONES not in texto
