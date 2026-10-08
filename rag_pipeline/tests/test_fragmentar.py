"""Fragmentación de los libros (rag/fragmentar.py), con un contador falso:
una palabra = un token. Así los tests no cargan el tokenizador de bge-m3."""
from __future__ import annotations

import numpy as np
import pytest

from rag import fragmentar as fr


def contar(texto: str) -> int:
    return len(texto.split())


def seccion(parrafos, anexos=(), sid="ATLS-07-019", titulos=("DISABILITY", "SCI SYNDROMES")) -> dict:
    return {"id": sid, "fuente": "ATLS", "idioma": "en", "capitulo_num": 7,
            "capitulo": "Disability", "clinico": True, "titulos": list(titulos),
            "pagina_ini": 108, "pagina_fin": 110, "parrafos": list(parrafos), "anexos": list(anexos)}


def parrafo(n: int, palabra: str = "word") -> str:
    """Un párrafo de n palabras en oraciones de 10."""
    oraciones = []
    for i in range(0, n, 10):
        k = min(10, n - i)
        oraciones.append(" ".join([palabra.capitalize()] + [palabra] * (k - 1)) + ".")
    return " ".join(oraciones)


# ---------------------------------------------------------------------------
# Oraciones
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("texto, esperado", [
    ("Shock is common. Treat it early.", ["Shock is common.", "Treat it early."]),
    ("As shown by Smith et al. Bleeding was controlled.", ["As shown by Smith et al. Bleeding was controlled."]),
    ("See Fig. 3 for details. Then act.", ["See Fig. 3 for details.", "Then act."]),
    ("Se usa, p. ej. La sonda Foley. Luego se fija.", ["Se usa, p. ej. La sonda Foley.", "Luego se fija."]),
    ("¿Hay sangrado? ¿Penetra la herida?", ["¿Hay sangrado?", "¿Penetra la herida?"]),
    ("Criterios:\n• Edad mayor de 60 años\n• Dolor torácico", ["Criterios:", "• Edad mayor de 60 años", "• Dolor torácico"]),
    ("PAS de 90 mm Hg. FC de 120.", ["PAS de 90 mm Hg.", "FC de 120."]),
])
def test_oraciones(texto, esperado):
    assert fr.oraciones(texto) == esperado


# ---------------------------------------------------------------------------
# Piezas
# ---------------------------------------------------------------------------
def test_por_palabras_respeta_presupuesto_y_solape():
    texto = " ".join(f"w{i}" for i in range(100))
    ventanas = fr._por_palabras(texto, 30, contar, solape=5)
    assert all(contar(v) <= 30 for v in ventanas)
    for a, b in zip(ventanas, ventanas[1:]):
        assert a.split()[-5:] == b.split()[:5]
    assert ventanas[-1].split()[-1] == "w99"


def test_empacar_solapa_un_parrafo():
    unidades = [parrafo(40, f"p{i}") for i in range(5)]
    grupos = fr._empacar(unidades, 100, contar, solape=1)
    assert all(contar(g) <= 100 for g in grupos)
    for a, b in zip(grupos, grupos[1:]):
        assert a.split("\n\n")[-1] == b.split("\n\n")[0]


def test_tabla_grande_se_parte_por_filas_repitiendo_el_pie():
    filas = [f"Row {i} | value {i}" for i in range(40)]
    anexo = "Table 7-4: Glasgow Coma Scale.\n" + "\n".join(filas)
    partes = fr._tabla(anexo, 50, contar)
    assert len(partes) > 1
    assert all(p.startswith("Table 7-4: Glasgow Coma Scale.\n") for p in partes)
    assert all(contar(p) <= 50 for p in partes)
    filas_salida = [l for p in partes for l in p.split("\n")[1:]]
    assert filas_salida == filas


def test_tabla_con_pie_largo_repite_solo_el_titulo():
    # ATLS Figure 1-2: pie de ~90 tokens; repetido entero dejaba sin lugar a las filas
    pie = "Figure 1-2: Field Triage Guideline. " + " ".join(["Use of the guideline helps."] * 15)
    filas = [f"• Criterion {i} with some detail" for i in range(30)]
    partes = fr._tabla(pie + "\n" + "\n".join(filas), 60, contar)
    assert all(p.startswith("Figure 1-2: Field Triage Guideline.\n") for p in partes)
    assert all(contar(p) <= 60 for p in partes)
    assert len(partes) <= 8
    assert set(" ".join(partes).split()) >= set((pie + " " + " ".join(filas)).split())


def test_partes_seccion_parejas():
    parrafos = [parrafo(100, f"p{i}") for i in range(6)]          # 600 tokens
    partes = fr._partes_seccion(parrafos, 500, contar)
    assert len(partes) == 2
    assert all(contar(p) <= 500 for p in partes)
    assert abs(contar(partes[0]) - contar(partes[1])) <= 100


# ---------------------------------------------------------------------------
# Configuraciones
# ---------------------------------------------------------------------------
SECCIONES = [
    seccion([parrafo(120, "alpha"), parrafo(300, "beta"), parrafo(40, "gamma")],
            anexos=["Table 7-1: Lesions.\nA | 1\nB | 2"]),
    seccion([parrafo(30, "delta")], sid="HUV-11-011", titulos=("TRATAMIENTO",)),
    seccion([], anexos=["Recuadro:\nIndicaciones de dren pleural"], sid="HUV-11-012"),
]


def embeber_falso(textos: list[str]) -> np.ndarray:
    """Vector por palabra dominante: oraciones de la misma palabra se parecen."""
    vocab = sorted({t.split()[0].lower().strip(".") for t in textos})
    m = np.zeros((len(textos), len(vocab)))
    for i, t in enumerate(textos):
        m[i, vocab.index(t.split()[0].lower().strip("."))] = 1
    return m


@pytest.mark.parametrize("nombre", list(fr.CONFIGS))
def test_configuracion_respeta_limites_y_secciones(nombre):
    frags = fr.fragmentar(SECCIONES, nombre, contar, embeber=embeber_falso)
    maximo = fr.CONFIGS[nombre]["maximo"]
    ids_seccion = {s["id"] for s in SECCIONES}
    assert frags, nombre
    for f in frags:
        assert f["n_tokens"] <= maximo, (nombre, f["id"], f["n_tokens"])
        assert contar(f["texto_contexto"]) <= fr.LIMITE
        assert f["seccion_id"] in ids_seccion
        assert f["id"].startswith(f["seccion_id"] + "-")
    # cada palabra de cada sección aparece en algún fragmento de esa sección
    for s in SECCIONES:
        propios = " ".join(f["texto"] for f in frags if f["seccion_id"] == s["id"])
        for p in s["parrafos"] + s["anexos"]:
            assert set(p.split()) <= set(propios.split()), (nombre, s["id"])
    # las tablas van en su propio fragmento, enteras
    tablas = [f for f in frags if f["tipo"] == "tabla"]
    assert any(f["texto"].endswith("Table 7-1: Lesions.\nA | 1\nB | 2") for f in tablas)
    # no hay sección vacía de fragmentos
    assert {f["seccion_id"] for f in frags} == ids_seccion


@pytest.mark.parametrize("nombre, con_ruta", [("parrafos-350", True), ("parrafos-350-sin-ruta", False)])
def test_ruta_de_titulos(nombre, con_ruta):
    frags = fr.fragmentar(SECCIONES, nombre, contar)
    primero = frags[0]["texto"]
    assert primero.startswith("ATLS › Disability › DISABILITY › SCI SYNDROMES\n") == con_ruta


def test_small_to_big_indexa_chico_y_devuelve_el_padre():
    frags = fr.fragmentar(SECCIONES, "oraciones-128", contar)
    texto = [f for f in frags if f["tipo"] == "texto" and f["seccion_id"] == "ATLS-07-019"]
    assert all(f["n_tokens"] <= 128 for f in texto)
    plano = lambda t: " ".join(t.split())                          # noqa: E731
    assert all(plano(f["texto"].split("\n", 1)[1]) in plano(f["texto_contexto"]) for f in texto)
    assert any(contar(f["texto_contexto"]) > 128 for f in texto)


def test_semantico_corta_entre_temas():
    # temas de 60 tokens: por encima del mínimo de 50 de la configuración
    s = seccion([parrafo(60, "alpha") + " " + parrafo(60, "beta")])
    frags = fr.fragmentar([s], "semantico", contar, embeber=embeber_falso)
    cuerpos = [f["texto"].split("\n", 1)[1] for f in frags]
    assert cuerpos == [parrafo(60, "alpha"), parrafo(60, "beta")]


def test_semantico_une_temas_demasiado_cortos():
    pares = [("Alpha one.", 0.0), ("Beta two.", 0.9), ("Gamma three words here.", 0.9)]
    assert fr.segmentar(pares, 0.5, 100, 3, contar) == ["Alpha one. Beta two.", "Gamma three words here."]


def test_semantico_sin_embebedor_falla():
    with pytest.raises(ValueError):
        fr.fragmentar(SECCIONES, "semantico", contar)
