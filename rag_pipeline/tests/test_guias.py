"""Extracción de los libros de trauma (rag/guias.py), sin los PDF.

Los bloques imitan lo medido en los PDF el 07/10/2026: fuentes, colores y
posiciones de páginas concretas (citadas en cada test).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag import guias
from rag.guias import Bloque

PERFILES = guias.cargar_perfiles()
ATLS, HUV = PERFILES["atls"], PERFILES["huv"]

BLANCO, AZUL, NEGRO_ATLS, NEGRO = 0xFFFFFF, 0x223A73, 0x231F20, 0x000000


def bloque(texto, x0=36, y0=100, x1=299, y1=None, fuente="Minion3-Regular", tam=10.0,
           color=NEGRO_ATLS, rol="") -> Bloque:
    lineas = texto.split("\n")
    return Bloque(x0, y0, x1, y1 if y1 is not None else y0 + 12 * len(lineas),
                  lineas, fuente, tam, color, rol)


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lineas, corta, esperado", [
    (["Balanced resuscitation before defin­", "itive control"], False,
     "Balanced resuscitation before definitive control"),          # ATLS p. 332
    (["Barrios desorgani-", "zados"], True, "Barrios desorganizados"),  # HUV p. 12
    (["bleeding from intra-", "abdominal sources"], False, "bleeding from intra-abdominal sources"),
    (["within the first 24-", "hour period"], False, "within the first 24-hour period"),
    (["the IMIST-", "AMBO handover"], False, "the IMIST-AMBO handover"),
    (["Three concepts:", "1.\t Treat the greatest threat", "2.\t Do not delay"], False,
     "Three concepts:\n1. Treat the greatest threat\n2. Do not delay"),
    (["\x03Table 6-3 lists", "clinical factors"], False, "Table 6-3 lists clinical factors"),
    (["Homicide ", "Epidemic ", "in ", "Cali"], True, "Homicide Epidemic in Cali"),  # HUV p. 21
])
def test_unir_lineas(lineas, corta, esperado):
    assert guias.unir_lineas(lineas, guion_corta_palabra=corta) == esperado


def test_unir_celdas_separa_celdas_y_une_continuaciones():
    # Tabla 7-4 del ATLS (p. 125) y Tabla 7-3 (viñeta partida en dos líneas)
    assert guias.unir_celdas(["Eye Opening (E)", "Spontaneous", "4"]) == "Eye Opening (E) | Spontaneous | 4"
    assert guias.unir_celdas(["• Acutely altered level of consciousness",
                              "(e.g., GCS <15, potential substance intoxication)"]) == \
        "• Acutely altered level of consciousness (e.g., GCS <15, potential substance intoxication)"


def test_continua_solo_si_la_oracion_quedo_abierta():
    assert guias.continua("creates injury to one side of the", "spinal cord.")
    assert not guias.continua("The airway is assessed first.", "then breathing")
    assert not guias.continua("creates injury to one side of the", "Spinal cord")


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("b, perfil, rol", [
    # ATLS: encabezado y número de página (márgenes)
    (bloque("COURSE MANUAL | Chapter 7 |Disability", y0=14, y1=25, fuente="OpenSans-Light", tam=8), ATLS, "margen"),
    (bloque("100", y0=758, y1=771, fuente="OpenSans-Light"), ATLS, "margen"),
    # Títulos por color y tamaño (p. 119)
    (bloque("INTRODUCTION", fuente="OpenSans-Bold", tam=12, color=BLANCO), ATLS, "titulo1"),
    (bloque("TRAUMATIC BRAIN INJURY", fuente="OpenSans-Bold", tam=12, color=AZUL), ATLS, "titulo2"),
    (bloque("PATHOLOGIC ANATOMY OF COMMON BRAIN INJURIES", fuente="OpenSans-Bold", tam=10,
            color=0x5B7785), ATLS, "titulo3"),
    (bloque("Extradural Hematoma", fuente="OpenSans-BoldItalic", tam=10, color=0x0A294F), ATLS, "titulo4"),
    # Encabezado de tabla en blanco 12 pero en minúsculas: no es título (p. 106)
    (bloque("Clinical Data Consistent with Pelvic Fracture", fuente="OpenSans-Bold", tam=12,
            color=BLANCO), ATLS, "material"),
    (bloque("Extradural hematoma (EDH) is a collection of blood"), ATLS, "cuerpo"),
    (bloque("Table 7-4: Glasgow Coma Scale and Scores.", fuente="OpenSans-Bold", tam=9, color=AZUL), ATLS, "pie"),
    (bloque("Box 22-4: Assessment of Blood Consumption", fuente="OpenSans-Bold", tam=9), ATLS, "pie"),
    # REFERENCES en Arial (p. 81) y en OpenSans (p. 36): se reconoce por el texto
    (bloque("REFERENCES", fuente="Arial-BoldMT", tam=12, color=AZUL), ATLS, "referencias"),
    (bloque("REFERENCES", fuente="OpenSans-Bold", tam=12, color=0x0A294F), ATLS, "referencias"),
    # Viñetas de KEY LEARNING POINTS (p. 36) y un rótulo en la misma fuente sin viñeta
    (bloque("•\t\tThe Primary Survey requires the clinician", fuente="OpenSans-Bold", tam=9), ATLS, "cuerpo"),
    (bloque("Component", fuente="OpenSans-Bold", tam=9), ATLS, "material"),
    # HUV (pp. 11, 12, 151, 270)
    (bloque("INTRODUCCIÓN", fuente="Montserrat-Bold", tam=16, color=0x00BAE3), HUV, "titulo1"),
    (bloque("● Radiografía de Tórax (Rx Tórax)", fuente="Tahoma-Bold", tam=12, color=0x00AEEF), HUV, "titulo2"),
    (bloque("1. La Violencia Juvenil, un problema de salud", fuente="FuturaBT-BookItalic", tam=13,
            color=0x6B6B6B), HUV, "titulo3"),
    (bloque("Es la imagen más utilizada", fuente="Tahoma", tam=12, color=NEGRO), HUV, "cuerpo"),
    (bloque("Tabla 1. Factores de riesgo", fuente="Tahoma", tam=12, color=NEGRO), HUV, "pie"),
    (bloque("LECTURAS RECOMENDADAS", fuente="Montserrat-Bold", tam=16, color=0x00BAE3), HUV, "referencias"),
    (bloque("Lecturas Recomendadas", fuente="Montserrat-Bold", tam=16, color=0x00BAE3), HUV, "referencias"),
    (bloque("2. www.seguridadjusticiaypaz.org 2019", fuente="FuturaBT-BookItalic", tam=12, color=NEGRO),
     HUV, "descartar"),
])
def test_clasificar(b, perfil, rol):
    assert guias.clasificar(b, perfil) == rol


# ---------------------------------------------------------------------------
# Maquetación
# ---------------------------------------------------------------------------
def test_orden_lectura_dos_columnas_con_tramo_a_todo_el_ancho():
    izq_arriba = bloque("izq 1", x0=36, y0=100, x1=299)
    der_arriba = bloque("der 1", x0=315, y0=90, x1=578)
    ancho = bloque("todo el ancho", x0=36, y0=300, x1=578)
    izq_abajo = bloque("izq 2", x0=36, y0=400, x1=299)
    der_abajo = bloque("der 2", x0=315, y0=380, x1=578)
    orden = guias.orden_lectura([der_abajo, ancho, der_arriba, izq_abajo, izq_arriba], 612)
    assert [b.lineas[0] for b in orden] == ["izq 1", "der 1", "todo el ancho", "izq 2", "der 2"]


def test_pie_arriba_recoge_su_tabla_por_filas():
    # ATLS p. 125: Table 7-4 en la columna derecha, celdas debajo
    pie = bloque("Table 7-4: Glasgow Coma Scale and Scores.", x0=315, y0=228, x1=573, rol="pie")
    celdas = [bloque("Eye Opening (E)", x0=325, y0=378, x1=420, rol="material"),
              bloque("Spontaneous", x0=429, y0=378, x1=500, rol="material"),
              bloque("4", x0=537, y0=378, x1=541, rol="material"),
              bloque("To sound", x0=429, y0=396, x1=500, rol="material")]
    otra_columna = bloque("A text in the left column", x0=36, y0=400, x1=299, rol="cuerpo")
    salida, perdido = guias.agrupar_pies([pie, *celdas, otra_columna], "arriba")
    assert pie.anexo == "Eye Opening (E) | Spontaneous\nTo sound"     # "4" es un rótulo suelto
    assert perdido == 0
    assert all(b.rol != "material" for b in salida)


def test_pie_abajo_recoge_la_tabla_de_encima():
    # HUV p. 12: "Tabla 1." va debajo de la tabla
    pie = bloque("Tabla 1. Factores de riesgo y protectores", x0=36, y0=723, x1=432, rol="pie")
    celda = bloque("Presencia de armas en el colegio", x0=171, y0=537, x1=345, rol="material")
    guias.agrupar_pies([pie, celda], "abajo")
    assert "Presencia de armas" in pie.anexo


def test_titulo_dentro_de_una_figura_es_material():
    # ATLS p. 46: "TQEH ICU INTUBATION CHECKLIST" dentro de la Figure 2-1
    pie = bloque("Figure 2-1: TQEH ICU Intubation Checklist.", x0=36, y0=100, x1=299, rol="pie")
    rotulo = bloque("TQEH ICU INTUBATION CHECKLIST", x0=60, y0=150, x1=280, rol="titulo1")
    item = bloque("Is the airway plan agreed by the team?", x0=60, y0=200, x1=280, rol="material")
    salida, _ = guias.agrupar_pies([pie, rotulo, item], "arriba")
    assert rotulo not in salida
    assert "TQEH ICU INTUBATION CHECKLIST" in pie.anexo


def test_titulo_despues_del_texto_no_es_de_la_figura():
    pie = bloque("Figure 6-6: Changes in Pelvic Volume", x0=315, y0=40, x1=568, rol="pie")
    texto = bloque("Pelvic fractures may cause hemorrhage.", x0=315, y0=150, x1=578, rol="cuerpo")
    titulo = bloque("PELVIC STABILIZATION", x0=315, y0=200, x1=500, rol="titulo3")
    rotulo = bloque("Uninjured pelvic volume", x0=433, y0=300, x1=560, rol="material")
    salida, _ = guias.agrupar_pies([pie, texto, titulo, rotulo], "arriba")
    assert titulo in salida and titulo.rol == "titulo3"


def test_recuadros_conserva_listas_y_descarta_flujogramas():
    # HUV p. 156: lista de indicaciones; p. 160: flujograma
    lista = [bloque("Indicaciones en trauma para colocación de dren pleural", y0=100, rol="material"),
             bloque("1. Neumotórax a tensión | 2. Neumotórax comunicante", y0=118, rol="material")]
    flujo = [bloque("¿Penetra?", x0=400, y0=500, x1=480, rol="material"),
             bloque("Egreso con indicaciones", x0=400, y0=520, x1=560, rol="material"),
             bloque("NO", x0=400, y0=540, x1=420, rol="material"),
             bloque("Indicación tubo de tórax", x0=400, y0=560, x1=560, rol="material")]
    cajas, perdido = guias.recuadros(lista + flujo)
    assert len(cajas) == 1 and "Neumotórax a tensión" in cajas[0].anexo
    assert perdido > 0


def test_es_flujograma():
    assert guias.es_flujograma(["Hemo / Neumotórax", "SI"])
    assert guias.es_flujograma(["Síntomas", "(-)"])
    assert not guias.es_flujograma(["Edad mayor de 60 años", "Dolor torácico", "Isquemia"])


# ---------------------------------------------------------------------------
# Extractor completo sobre un libro falso
# ---------------------------------------------------------------------------
def _span(texto, fuente, tam, color):
    return {"text": texto, "font": fuente, "size": tam, "color": color, "flags": 0}


def _pagina(bloques: list[tuple]) -> SimpleNamespace:
    """bloques: (texto, bbox, fuente, tam, color)."""
    dic = {"blocks": [{"type": 0, "bbox": bbox,
                       "lines": [{"spans": [_span(l, fuente, tam, color)]} for l in texto.split("\n")]}
                      for texto, bbox, fuente, tam, color in bloques]}
    return SimpleNamespace(get_text=lambda modo: dic, rect=SimpleNamespace(width=612))


def _perfil(**cambios) -> dict:
    p = dict(ATLS, capitulos=[[7, "Disability", 1], [8, "Exposure", 3]], fin=3,
             saltar=[], saltar_portada=False)
    p.update(cambios)
    return p


IZQ, DER = (36, 100, 299, 140), (315, 100, 578, 140)


def test_extractor_secciones_titulos_paginas_y_referencias():
    cuerpo = ("Minion3-Regular", 10, NEGRO_ATLS)
    doc = [
        _pagina([("INTRODUCTION", (36, 60, 200, 75), "OpenSans-Bold", 12, BLANCO),
                 ("TBI and SCI are leading causes of", IZQ, *cuerpo),
                 ("death and disability.", DER, *cuerpo)]),
        _pagina([("TRAUMATIC BRAIN INJURY", (36, 60, 300, 75), "OpenSans-Bold", 12, AZUL),
                 ("Secondary injury follows hypoxia.", IZQ, *cuerpo),
                 ("REFERENCES", (315, 60, 400, 75), "Arial-BoldMT", 12, AZUL),
                 ("1. Smith J, et al. J Trauma. 2020.", DER, *cuerpo)]),
        _pagina([("CHAPTER STATEMENT", (36, 60, 200, 75), "OpenSans-Bold", 12, BLANCO),
                 ("Exposure is the E in xABCDE.", IZQ, *cuerpo),
                 ("Table 8-1: Heat loss.", (315, 60, 578, 75), "OpenSans-Bold", 9, AZUL),
                 ("Conduction | Convection", (315, 90, 578, 102), "OpenSans-Medium", 8, NEGRO_ATLS)]),
    ]
    ex = guias.Extractor(_perfil(desfase_pagina=19))
    secciones = list(ex.secciones(doc))

    assert [s["id"] for s in secciones] == ["ATLS-07-001", "ATLS-07-002", "ATLS-08-001"]
    intro, tbi, cap8 = secciones
    # el párrafo cortado entre columnas se une
    assert intro["texto"] == "TBI and SCI are leading causes of death and disability."
    assert intro["titulos"] == ["INTRODUCTION"]
    # el título azul cuelga del blanco; la bibliografía no entra
    assert tbi["titulos"] == ["INTRODUCTION", "TRAUMATIC BRAIN INJURY"]
    assert "Smith" not in tbi["texto"]
    assert ex.stats["referencias"] > 0
    # página impresa = PDF - desfase
    assert (tbi["pagina_ini"], tbi["pagina_pdf_ini"]) == (2 - 19, 2)
    # la tabla va al final de su sección, con su contenido
    assert cap8["texto"].endswith("Table 8-1: Heat loss.\nConduction | Convection")
    assert cap8["capitulo"] == "Exposure" and cap8["clinico"] is True


def test_extractor_salta_portada_y_marca_capitulos_no_clinicos():
    cuerpo = ("Minion3-Regular", 10, NEGRO_ATLS)
    doc = [
        _pagina([("Objectives text that is not body.", IZQ, "OpenSans-Medium", 9, NEGRO_ATLS),
                 ("Body text on the cover page.", DER, *cuerpo)]),
        _pagina([("Team body text.", IZQ, *cuerpo)]),
        _pagina([("Exposure cover page.", IZQ, *cuerpo)]),
        _pagina([("Exposure body text.", IZQ, *cuerpo)]),
    ]
    perfil = _perfil(saltar_portada=True, no_clinicos=[7], fin=4)
    secciones = list(guias.Extractor(perfil).secciones(doc))
    textos = " ".join(s["texto"] for s in secciones)
    assert "cover page" not in textos
    assert [(s["texto"], s["clinico"]) for s in secciones] == [
        ("Team body text.", False), ("Exposure body text.", True)]


def test_perfiles_capitulos_consecutivos():
    for perfil in PERFILES.values():
        caps = guias.capitulos(perfil)
        assert [c.numero for c in caps] == list(range(1, len(caps) + 1))
        assert all(c.pdf_ini <= c.pdf_fin for c in caps)
        assert caps[-1].pdf_fin == perfil["fin"]
