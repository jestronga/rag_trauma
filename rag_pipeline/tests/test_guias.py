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
    # Pies de foto y dibujo del HUV (pp. 55, 140, 97): se descartan
    (bloque("Foto. E-FAST inicial de paciente HUV", fuente="FuturaBT-LightItalic", tam=11, color=NEGRO),
     HUV, "figura"),
    (bloque("Dibujos. Técnica de toracostomía", fuente="FuturaBT-LightItalic", tam=11, color=NEGRO),
     HUV, "figura"),
    (bloque("Foto Control del sangrado externo.", fuente="FuturaBT-LightItalic", tam=11, color=NEGRO),
     HUV, "figura"),
    # ...pero no el título "IMAGENES DIAGNÓSTICAS" (p. 198) ni el rótulo de un flujograma (p. 269)
    (bloque("IMAGENES DIAGNÓSTICAS", fuente="Montserrat-Bold", tam=16, color=0x00BAE3), HUV, "titulo1"),
    (bloque("Imagenes: AngioTAC", fuente="Montserrat-SemiBold", tam=9, color=BLANCO), HUV, "material"),
    # El ATLS no tiene la regla: sus figuras van con su pie ("Figure 7-1:")
    (bloque("Foto of a patient", fuente="OpenSans-Medium", tam=8), ATLS, "material"),
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


def test_pie_de_foto_partido_en_dos_bloques():
    # HUV p. 97: la segunda línea del pie es otro bloque, pegado debajo
    fuente = dict(fuente="FuturaBT-LightItalic", tam=11.0, color=NEGRO)
    pie = bloque("Foto. TAC trauma renal manejo conservador y", x0=72, y0=400, x1=540, y1=415,
                 rol="figura", **fuente)
    sigue = bloque("hemostáticos y trauma hepático. Cortesía grupo de CTE-FVL", x0=72, y0=413, x1=500,
                   y1=426, rol="material", **fuente)
    lejos = bloque("Otro rótulo en la misma fuente", x0=72, y0=480, x1=500, rol="material", **fuente)
    otra_fuente = bloque("Una celda", x0=72, y0=428, x1=300, fuente="Tahoma", tam=12, rol="material")
    guias.continuar_figuras([pie, sigue, lejos, otra_fuente])
    assert (sigue.rol, lejos.rol, otra_fuente.rol) == ("figura", "material", "material")


def test_pie_de_foto_no_se_lleva_la_lista_de_encima():
    # HUV p. 59: "Indicaciones para realizar TAC corporal Total" (sin pie)
    # encima del pie de una foto, en la misma columna
    lista = [bloque("Indicaciones para realizar TAC corporal Total", y0=100, rol="material"),
             bloque("Herida múltiple por AF | Herida múltiple por ACP", y0=118, rol="material")]
    foto = bloque("Foto. E-FAST inicial de paciente HUV", y0=300, rol="figura")
    salida, _ = guias.agrupar_pies(lista + [foto], "abajo")
    assert foto.anexo == ""
    assert [b.rol for b in salida if "TAC corporal" in b.anexo] == ["recuadro"]


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


# ---------------------------------------------------------------------------
# Transcripciones a mano (flujogramas y tablas en imagen del HUV)
# ---------------------------------------------------------------------------
TITULO_HUV = ("Montserrat-Bold", 16, 0x00BAE3)
CUERPO_HUV = ("Tahoma", 12, NEGRO)
CAJA = ("Montserrat-SemiBold", 9, BLANCO)


def _perfil_huv(**cambios) -> dict:
    p = dict(HUV, capitulos=[[15, "Trauma abdominal penetrante", 1]], fin=2)
    p.update(cambios)
    return p


def _transcripcion(tipo, texto, ancla=None):
    return {"tipo": tipo, "texto": texto, "ancla": ancla or guias.ANCLAS[tipo]}


FLUJO = "Flujograma 2: trauma abdominal penetrante (transcripción del diagrama del libro).\nAgónico: toracotomía."


def test_flujograma_transcrito_reemplaza_las_cajas_sueltas():
    # HUV p. 98: tras el título, recuadros y pasos leídos como cuerpo
    doc = [
        _pagina([("TRATAMIENTO", (36, 60, 300, 75), *TITULO_HUV),
                 ("El texto del capítulo sigue aquí.", (36, 100, 578, 140), *CUERPO_HUV),
                 ("Flujograma 2", (36, 200, 200, 215), *TITULO_HUV),
                 ("Control de hemorragia externa (Compresible).", (150, 300, 500, 330), *CUERPO_HUV),
                 ("Agónico", (60, 400, 120, 410), *CAJA),
                 ("Toractomía de Resucitación", (60, 500, 200, 520), *CAJA)]),
        _pagina([("Más texto del capítulo después del diagrama.", IZQ, *CUERPO_HUV)]),
    ]
    ex = guias.Extractor(_perfil_huv(), {1: _transcripcion("flujograma", FLUJO)})
    tratamiento, flujo = list(ex.secciones(doc))
    assert tratamiento["texto"] == "El texto del capítulo sigue aquí."
    assert flujo["titulos"] == ["Flujograma 2"]
    # la transcripción entera y nada de las cajas; la página siguiente sigue igual
    assert flujo["anexos"] == [FLUJO]
    assert flujo["parrafos"] == ["Más texto del capítulo después del diagrama."]
    assert "Toractomía" not in flujo["texto"] and "Compresible" not in flujo["texto"]
    assert ex.stats["reemplazado"] > 0 and ex.stats["transcritas"] == 1


def test_tabla_transcrita_reemplaza_pie_y_material():
    # HUV p. 120: la tabla no tiene texto; solo el pie "Tabla. Manejo medico inicial"
    tabla = "Tabla: manejo médico inicial (transcripción de la tabla del libro).\n4. Toxoide tetánico x 0.5 cc."
    doc = [_pagina([("CASOS ESPECIALES", (36, 60, 300, 75), *TITULO_HUV),
                    ("Texto antes de la tabla.", (36, 100, 578, 140), *CUERPO_HUV),
                    ("Celda suelta", (100, 600, 300, 612), *CAJA),
                    ("Tabla. Manejo medico inicial", (150, 700, 450, 712), *CUERPO_HUV)])]
    ex = guias.Extractor(_perfil_huv(fin=1), {1: _transcripcion("tabla", tabla, r"^Tabla\. Manejo")})
    (s,) = list(ex.secciones(doc))
    assert s["anexos"] == [tabla]
    assert s["parrafos"] == ["Texto antes de la tabla."]


def test_transcripcion_sin_ancla_falla():
    # si cambian las reglas y el título deja de serlo, no se pierde el diagrama sin aviso
    doc = [_pagina([("Solo texto.", IZQ, *CUERPO_HUV)])]
    ex = guias.Extractor(_perfil_huv(fin=1), {1: _transcripcion("flujograma", FLUJO)})
    with pytest.raises(ValueError, match="ancla"):
        list(ex.secciones(doc))


def test_transcripcion_en_pagina_que_no_se_extrae_falla():
    doc = [_pagina([("Solo texto.", IZQ, *CUERPO_HUV)])]
    ex = guias.Extractor(_perfil_huv(fin=1), {7: _transcripcion("flujograma", FLUJO)})
    with pytest.raises(ValueError, match=r"\[7\]"):
        list(ex.secciones(doc))


def test_cargar_transcripciones(tmp_path):
    assert guias.cargar_transcripciones(tmp_path / "no_existe.yaml") == {}
    ruta = tmp_path / "t.yaml"
    ruta.write_text("huv:\n"
                    "  - {pagina: 120, tipo: tabla, ancla: '^Tabla\\. Manejo', texto: \"Tabla: x\\n1. y\\n\"}\n"
                    "  - {pagina: 201, tipo: flujograma, texto: 'Flujograma 1: z', notas: duda}\n",
                    encoding="utf-8")
    t = guias.cargar_transcripciones(ruta)["huv"]
    assert t[120]["texto"] == "Tabla: x\n1. y" and t[120]["ancla"] == r"^Tabla\. Manejo"
    assert t[201]["ancla"] == guias.ANCLAS["flujograma"]
    ruta.write_text("huv:\n  - {pagina: 1, tipo: figura, texto: x}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tipo"):
        guias.cargar_transcripciones(ruta)
    ruta.write_text("huv:\n  - {pagina: 1, tipo: tabla, texto: x}\n  - {pagina: 1, tipo: tabla, texto: y}\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="misma página"):
        guias.cargar_transcripciones(ruta)


@pytest.mark.skipif(not guias.config.GUIAS_TRANSCRIPCIONES.exists(),
                    reason="el archivo de transcripciones está fuera de git (junto a los PDF)")
def test_transcripciones_reales_bien_formadas():
    """Lo que rag/fragmentar.py necesita: primera línea = título de una sola
    oración que dice que es transcripción; una fila por línea, ninguna vacía."""
    for libro, por_pagina in guias.cargar_transcripciones().items():
        paginas = {p for c in guias.capitulos(PERFILES[libro]) for p in range(c.pdf_ini, c.pdf_fin + 1)}
        for pagina, t in por_pagina.items():
            titulo, *filas = t["texto"].split("\n")
            assert pagina in paginas, pagina
            assert "(transcripción de" in titulo and titulo.endswith(")."), pagina
            assert filas and all(f.strip() for f in filas), pagina


# ---------------------------------------------------------------------------
# ATLS 10.ª ed. en español (PLAN_INDEXAR_ATLS10.md): traducción automática,
# una sola familia tipográfica, roles por color. Medido sobre el PDF el 23/10.
# ---------------------------------------------------------------------------
ATLS10 = PERFILES["atls10"]
ARIAL, ARIAL_B = "ArialUnicodeMS", "ArialUnicodeMS-Bold"
CUERPO10, ROJO_TIT, ROJO, OCRE, VERDE, GRIS, ROTULO = (
    0x383837, 0xAC1A25, 0xAD1D23, 0xC98505, 0x138E76, 0x616060, 0x171615)


@pytest.mark.parametrize("b, rol", [
    # encabezado de página (PDF 61)
    (bloque("CAPÍTULO 1 Evaluación y Tratamiento inicial", y0=39, y1=48, fuente=ARIAL, tam=8,
            color=ROTULO), "margen"),
    # títulos: blanco sobre franja, rojo, ocre y verde en negrita (PDF 105, 102)
    (bloque("El tratamiento inicial del shock hemorrágico", fuente=ARIAL, tam=11.2, color=BLANCO), "titulo1"),
    (bloque("examen físico", fuente=ARIAL, tam=10.4, color=ROJO_TIT), "titulo2"),
    (bloque("presión intracraneal", fuente=ARIAL, tam=10.0, color=ROJO), "titulo2"),
    (bloque("Airway y respiración", fuente=ARIAL_B, tam=7.6, color=OCRE), "titulo3"),
    (bloque("choque neurogénica", fuente=ARIAL_B, tam=7.0, color=VERDE), "titulo3"),
    # en blanco como los títulos, pero son de su tabla o recuadro
    (bloque("trampa \nprevención", fuente=ARIAL, tam=10.4, color=BLANCO), "material"),
    (bloque("trabajo en equipo", fuente=ARIAL, tam=9.6, color=BLANCO), "material"),
    # el pie de tabla va por el color: "TABLA 3-1 esboza…" también abre párrafos
    (bloque("Tabla 3-1 signos y síntomas de hemorragia por clase", fuente=ARIAL, tam=8.8,
            color=BLANCO), "pie"),
    (bloque("TABLA 3-1 esboza la pérdida estimada de sangre", fuente=ARIAL, tam=7.0,
            color=CUERPO10), "cuerpo"),
    # cuerpo resaltado en rojo; leyenda de figura en rojo y su continuación en gris
    (bloque("Un neumotórax simple puede ser convertido", fuente=ARIAL, tam=7.1, color=ROJO), "cuerpo"),
    (bloque("FIGURA 1-4 técnica de limitación de movimiento", fuente=ARIAL, tam=5.4, color=ROJO), "leyenda"),
    (bloque("Cuando se retira el collar cervical", fuente=ARIAL, tam=6.0, color=GRIS), "leyenda"),
    # rótulos y créditos de las figuras; "A B C" traducido (PDF 160, 88)
    (bloque("Libélula Media Group", fuente=ARIAL, tam=6.4, color=ROTULO), "descartar"),
    (bloque("UN segundo do", fuente=ARIAL, tam=7.0, color=CUERPO10), "descartar"),
    (bloque("bibliografía", fuente=ARIAL, tam=12.8, color=ROJO_TIT), "referencias"),
    # la traducción achica franjas a 8,4 (PDF 258) y cambia la palabra de los
    # pies: "mesa" (Table), "caja de correos" (Box), "Panorámica" (PDF 230, 364, 167)
    (bloque("Trauma de la cabeza", fuente=ARIAL, tam=8.4, color=BLANCO), "titulo1"),
    (bloque("mesa tasas de fluidos de resucitación 9-1 de quemaduras", fuente=ARIAL, tam=7.5,
            color=BLANCO), "pie"),
    (bloque("caja de correos 4 consejos para comunicarse con", fuente=ARIAL, tam=8.7, color=BLANCO), "pie"),
    (bloque("Panorámica general de 6-3 manejo de la lesión cerebral", fuente=ARIAL, tam=8.8,
            color=BLANCO), "pie"),
    (bloque("Parte 1: habilidades básicas de las vías respiratorias", fuente=ARIAL, tam=8.4,
            color=BLANCO), "titulo1"),
])
def test_clasificar_atls10(b, rol):
    assert guias.clasificar(b, ATLS10) == rol


def _pagina_con_cajas(lineas: list[tuple]) -> SimpleNamespace:
    """Un bloque con una caja por renglón: (texto, bbox)."""
    caja = (min(c[0] for _, c in lineas), min(c[1] for _, c in lineas),
            max(c[2] for _, c in lineas), max(c[3] for _, c in lineas))
    dic = {"blocks": [{"type": 0, "bbox": caja,
                       "lines": [{"bbox": c, "spans": [_span(t, ARIAL, 7.0, CUERPO10)]} for t, c in lineas]}]}
    return SimpleNamespace(get_text=lambda modo: dic, rect=SimpleNamespace(width=603))


def test_deduplicar_lineas_superpuestas():
    # PDF 32: la misma línea ×3 en la misma caja; PDF 103: las copias difieren
    # en milésimas y round() las separaba (81,501 → 82, 81,499 → 81)
    pagina = _pagina_con_cajas([
        ("los segundo pico se produce en ", (73, 137, 319, 146)),
        ("los segundo pico se produce en", (73, 137, 319, 146)),
        ("los segundo pico se produce en ", (73.2, 137, 319, 146)),
        ("un El exceso de base es", (81.501, 683, 530.7, 690)),
        ("un El exceso de base es", (81.499, 683, 530.7, 690)),
        ("los segundo pico se produce en ", (73, 160, 319, 169)),     # otra caja: otra línea
    ])
    (b,) = guias.bloques_de_pagina(pagina, deduplicar=True)
    assert [l.strip() for l in b.lineas] == ["los segundo pico se produce en", "un El exceso de base es",
                                             "los segundo pico se produce en"]
    assert len(b.cajas) == 3
    (sin,) = guias.bloques_de_pagina(pagina)
    assert len(sin.lineas) == 6


def test_corregir_con_los_reemplazos_del_perfil():
    reemplazos = guias.Extractor(ATLS10).reemplazos
    corregir = lambda l: guias.corregir(l, reemplazos)
    # la viñeta traducida como "norte", en medio y al final del bloque
    assert corregir("en el campo ( norte FIGURA 1-1 ).") == "en el campo ( FIGURA 1-1 )."
    assert corregir("se muestra en ( norte") == "se muestra en ("
    # la letra capitular traducida aparte (PDF 57, 159)
    assert corregir("W gallina que tratan a pacientes heridos, médicos ") == \
        "Cuando tratan a pacientes heridos, los médicos "
    assert corregir("MARIDO lesiones EAD se encuentran entre los más comunes") == \
        "Las lesiones de la cabeza se encuentran entre los más comunes"
    # la letra solo se arregla al empezar el renglón de la capitular
    assert corregir("la palabra MARIDO lesiones EAD") == "la palabra MARIDO lesiones EAD"
    # las partes de una figura (PDF 100)
    assert corregir("A. el pecho; SEGUNDO. el abdomen; DO. la pelvis; y RE. los huesos") == \
        "A. el pecho; B. el abdomen; C. la pelvis; y D. los huesos"


def test_pagina_impresa_por_tramos():
    # medido con los números de los encabezados: coinciden en las 362 páginas
    assert [guias.pagina_impresa(ATLS10, p) for p in (57, 74, 77, 128, 356, 360, 436, 459)] == \
        [4, 21, 23, 73, 300, 303, 378, 401]
    assert guias.pagina_impresa(ATLS, 119) == 100           # desfase fijo, como antes


def test_capitulos_con_ultima_pagina_explicita():
    # el apéndice G termina antes del índice alfabético (PDF 435-449)
    caps = {c.numero: c for c in guias.capitulos(ATLS10)}
    assert (caps[20].pdf_ini, caps[20].pdf_fin) == (392, 434)
    assert (caps[21].pdf_ini, caps[23].pdf_fin) == (450, 464)
    assert caps[13].pdf_fin == 312 and not caps[18].clinico and caps[16].clinico


def test_equivalencias_con_el_atls11():
    # cada capítulo del ATLS 10 apunta a capítulos que existen en el ATLS 11
    caps10 = {c.numero for c in guias.capitulos(ATLS10)}
    caps11 = {c.numero for c in guias.capitulos(ATLS)}
    eq = ATLS10["equivalente_11"]
    assert set(eq) == caps10
    assert all(set(v) <= caps11 for v in eq.values())


def test_tabla_trampa_prevencion_por_columnas():
    # PDF 61: cada renglón es un bloque; con _filas se intercalaban las celdas
    b = lambda texto, *caja: Bloque(*caja, lineas=[texto], cajas=[caja])
    partes = [
        Bloque(333, 85, 458, 99, lineas=["trampa ", "prevención"], cajas=[(333, 85, 368, 99), (407, 85, 458, 99)]),
        b("Falla en el ", 333, 113, 364, 122), b("equipo", 333, 125, 352, 134),
        b("• El equipo de pruebas con regularidad. ", 407, 115, 493, 121),
        Bloque(407, 125, 527, 134, lineas=["•", "Asegurarse que el equipo y baterías de repuesto "],
               cajas=[(407, 125, 410, 134), (416, 127, 527, 134)]),
        b("están fácilmente disponibles. ", 416, 139, 483, 146),
        b("intubación sin ", 333, 161, 374, 170), b("éxito  ", 333, 173, 350, 182),
        Bloque(407, 161, 528, 170, lineas=["•", "Identificar a los pacientes con anatomía de la vía"],
               cajas=[(407, 161, 410, 170), (416, 163, 528, 170)]),
        b("aérea difícil. ", 416, 175, 445, 182),
    ]
    assert guias._filas_por_columnas(partes).split("\n") == [
        "trampa | prevención",
        "Falla en el equipo | • El equipo de pruebas con regularidad. • Asegurarse que el equipo y "
        "baterías de repuesto están fácilmente disponibles.",
        "intubación sin éxito | • Identificar a los pacientes con anatomía de la vía aérea difícil.",
    ]


def test_tabla_con_filas_enteras_por_columnas():
    # Tabla 3-1 (PDF 103): un bloque por fila, una caja por celda
    fila = lambda y, *celdas: Bloque(81, y, 400, y + 7, lineas=list(celdas),
                                     cajas=[(x, y, x + 40, y + 7) for x in (81, 200, 280)][:len(celdas)])
    texto = guias._filas_por_columnas([fila(481, "Ritmo cardiaco ", "↔", "↔ / ↑ "),
                                       fila(505, "Presión sanguínea ", "↔", "↔")])
    assert texto == "Ritmo cardiaco | ↔ | ↔ / ↑\nPresión sanguínea | ↔ | ↔"


def test_pie_partido_en_dos_bloques():
    # PDF 210: "Tabla 8-1 deformidades luxación de la articulación" / "comunes";
    # sin unirlos, "comunes" (blanco, 8,6) se leería como título
    pie = bloque("Tabla 8-1 deformidades luxación de la articulación", y0=86, y1=96,
                 fuente=ARIAL, tam=8.6, color=BLANCO, rol="pie")
    sigue = bloque("comunes", y0=98, y1=108, fuente=ARIAL, tam=8.6, color=BLANCO, rol="titulo1")
    lejos = bloque("Otro título", y0=200, y1=210, fuente=ARIAL, tam=11.2, color=BLANCO, rol="titulo1")
    cuerpo = bloque("Texto del cuerpo debajo", y0=100, y1=110, color=CUERPO10, rol="cuerpo")
    salida = guias.unir_pies_partidos([pie, sigue, lejos, cuerpo])
    assert salida == [pie, lejos, cuerpo]
    assert pie.texto() == "Tabla 8-1 deformidades luxación de la articulación comunes"


def test_sin_repeticiones():
    # PDF 179: la traducción repite la oración en otra caja
    o = "antibióticos de amplio espectro profilácticos son recomendados en la lesión penetrante."
    assert guias.sin_repeticiones(f"Antes. {o} ( {o} ( {o} ( Gestión de pene- trar.") == \
        f"Antes. {o} ( Gestión de pene- trar."
    # las cortas no se tocan
    assert guias.sin_repeticiones("Sí. Sí. Sí.") == "Sí. Sí. Sí."


def test_extractor_atls10_titulo_partido_y_leyenda_en_dos_bloques():
    perfil = dict(ATLS10, capitulos=[[3, "Choque", 1]], fin=1, saltar=[], recortar_arriba={})
    doc = [_pagina([
        ("El shock hemorrágico", (72, 60, 300, 75), ARIAL, 11.2, BLANCO),
        # PDF 104: un título de dos renglones en dos bloques
        ("15% a 30% la pérdida de volumen de sangre: Hemorragia", (72, 80, 300, 90), ARIAL_B, 7.0, OCRE),
        ("Clase II", (72, 92, 120, 100), ARIAL_B, 7.0, OCRE),
        ("Los signos clínicos de la hemorragia de clase II incluyen taquicardia.", (72, 110, 300, 140),
         ARIAL, 7.0, CUERPO10),
        ("Libélula Media Group", (100, 200, 200, 208), ARIAL, 6.4, ROTULO),
        # PDF 100: la leyenda en rojo y su continuación en gris, en otro bloque
        ("norte FIGURA 3-2 El uso de ultrasonidos (FAST)", (72, 300, 300, 308), ARIAL, 5.4, ROJO),
        ("para buscar la causa del shock.", (72, 310, 300, 318), ARIAL, 6.0, GRIS),
    ])]
    (s,) = list(guias.Extractor(perfil).secciones(doc))
    assert s["id"] == "ATLS10-03-001"
    assert s["titulos"] == ["El shock hemorrágico", "15% a 30% la pérdida de volumen de sangre: Hemorragia Clase II"]
    assert s["parrafos"] == ["Los signos clínicos de la hemorragia de clase II incluyen taquicardia."]
    assert s["anexos"] == ["FIGURA 3-2 El uso de ultrasonidos (FAST) para buscar la causa del shock."]
    assert s["pagina_ini"] == 1 - 53
