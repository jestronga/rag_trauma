"""
Tests de la respuesta que lee el médico (build_prompt + generate.ensamblar).

Casos sintéticos, sin Ollama ni índices. Lo que se prueba es que cada número
y cada cita salgan de los casos, y que el formato tenga lo que se le promete
al médico: conclusión primero, tabla, sugerencias con contexto, límites.
Correr desde rag_pipeline/:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config  # noqa: E402
from rag.build_prompt import (  # noqa: E402
    ROTULO_AFINAR, ROTULO_SINTESIS, ROTULO_SUGERENCIAS, TITULO_SUGERENCIAS, conclusion_calculada,
    contexto_elemento, limites_calculados, para_afinar, tabla_casos,
)
from rag.generate import elegir_sugerencias, ensamblar, verificar_citas  # noqa: E402
from rag.hallazgos import extraer, resaltar_html  # noqa: E402
from rag.retrieve import Resultado  # noqa: E402

CONSULTA = "herida por arma de fuego en tórax con hipoventilación"


def caso(rid: str, examen: str, evolucion: str, desenlace: str = "alta", sim: float = 0.7) -> Resultado:
    texto = (f"Motivo de consulta: HPAF (herida por arma de fuego)\n\nExamen físico: {examen}\n\n"
             f"Evolución y hospitalización: {evolucion}")
    meta = {"mecanismo_lesion": "arma_de_fuego", "region_anatomica": ["torax"], "desenlace": desenlace}
    return Resultado(rid, texto, meta, score=sim, score_norm=sim)


@pytest.fixture
def casos() -> list[Resultado]:
    return [
        caso("101", "hipoventilacion derecha", "se pasa tubo de torax, drena 300cc. Salida", "alta", 0.71),
        caso("102", "hipoventilacion, hemotorax", "tubo de torax. Llevado a cirugia: toracotomia", "cirugia", 0.69),
        caso("103", "herida penetrante, hipoventilacion", "tubo de torax, observacion", "no_determinado", 0.68),
        caso("104", "herida superficial", "sutura, salida", "alta", 0.66),
    ]


# --- conclusión --------------------------------------------------------------
def test_conclusion_va_primero_con_conteos_y_respaldo(casos):
    texto = conclusion_calculada(casos, CONSULTA)
    assert texto.startswith(ROTULO_SINTESIS)
    assert "**tubo de tórax** (3 de 4)" in texto
    assert "similitud 0,66–0,71" in texto


def test_conclusion_cuenta_la_escalada_con_su_cita_y_hallazgos(casos):
    texto = conclusion_calculada(casos, CONSULTA)
    assert "1 de 4 fue operado o tuvo indicación quirúrgica [caso: 102]" in texto
    assert "hemotórax" in texto
    # la cirugía ya la cuenta la escalada: no se repite en "lo más documentado"
    assert "**cirugía**" not in texto


def test_fallecimiento_pesa_mas_que_cirugia(casos):
    casos[3].metadata["desenlace"] = "fallecimiento"
    assert "1 de 4 falleció [caso: 104]" in conclusion_calculada(casos, CONSULTA)


def test_similitud_baja_se_advierte(casos):
    assert "Similitud baja" not in conclusion_calculada(casos, CONSULTA)
    for c in casos:
        c.score_norm = config.SIMILITUD_BAJA - 0.05
    assert "Similitud baja" in conclusion_calculada(casos, CONSULTA)


def test_cirugia_no_se_sugiere_dos_veces():
    # mismo caso con osteosíntesis: "cirugía" y "osteosíntesis" serían la misma sugerencia
    casos = [caso("201", "fractura de femur", "reduccion abierta y osteosintesis", "cirugia"),
             caso("202", "fractura de femur", "ferula, salida")]
    salida_modelo = f"{TITULO_SUGERENCIAS}\n- Considerar cirugía\n- Considerar osteosíntesis"
    terminos_elegidos = [e.termino for e in elegir_sugerencias(salida_modelo, casos, "fractura de fémur")]
    assert "osteosíntesis / reducción" in terminos_elegidos and "cirugía" not in terminos_elegidos


def test_caso_solo_por_bm25_se_declara(casos):
    casos[0].score_norm = 0.0
    assert "1 hallado(s) solo por palabras clave" in conclusion_calculada(casos, CONSULTA)


# --- tabla ---------------------------------------------------------------------
def test_tabla_una_fila_por_caso_con_cita(casos):
    tabla = tabla_casos(casos, CONSULTA).splitlines()
    assert len(tabla) == 2 + len(casos)
    assert all(f"[caso: {c.record_id}]" in fila for c, fila in zip(casos, tabla[2:]))


def test_tabla_resalta_lo_que_dice_la_consulta_y_nombra_bien_el_desenlace(casos):
    tabla = tabla_casos(casos, CONSULTA)
    assert "**hipoventilación**" in tabla
    fila_102 = next(f for f in tabla.splitlines() if "[caso: 102]" in f)
    assert "operado" in fila_102 and "toracotomía" in fila_102
    assert "cirugía," not in fila_102      # sobra si consta el procedimiento
    assert "sin clasificar" in next(f for f in tabla.splitlines() if "[caso: 103]" in f)


# --- para considerar -----------------------------------------------------------
def test_contexto_de_un_manejo(casos):
    tubo = next(e for e in extraer(casos) if e.termino == "tubo de tórax")
    linea = contexto_elemento(tubo, casos, CONSULTA)
    assert linea.startswith("- **Considerar tubo de tórax** — se hizo en 3 de 4 casos similares  \n")
    assert "hemotórax 1" in linea
    assert "hipoventilación" not in linea          # ya lo dijo la consulta
    assert "herida penetrante" not in linea        # obvio en una herida por arma
    assert linea.index("alta") < linea.index("sin clasificar")
    assert linea.endswith("[caso: 101] [caso: 102] [caso: 103]")


def test_cirugia_dice_hecha_o_indicada(casos):
    cx = next(e for e in extraer(casos) if e.termino == "cirugía")
    assert "se hizo o se indicó en 1 de 4" in contexto_elemento(cx, casos, CONSULTA)


def test_sugerencias_sin_hallazgos_obvios(casos):
    salida_modelo = f"{TITULO_SUGERENCIAS}\n- Valorar herida penetrante [caso: 103]\n- Valorar hipoventilación"
    elegidos = elegir_sugerencias(salida_modelo, casos, CONSULTA)
    assert {e.termino for e in elegidos}.isdisjoint({"herida penetrante", "hipoventilación"})
    assert any(e.categoria == "Manejo" for e in elegidos)


# --- para afinar y límites -----------------------------------------------------
def test_para_afinar_lista_lo_que_diferencia_los_casos(casos):
    texto = para_afinar(casos, CONSULTA)
    assert texto.startswith(ROTULO_AFINAR)
    assert "hemotórax (1 de 4)" in texto
    assert "hipoventilación" not in texto and "herida penetrante" not in texto


def test_para_afinar_vacio_si_nada_difiere():
    iguales = [caso(str(i), "hipoventilacion", "tubo de torax") for i in range(3)]
    assert para_afinar(iguales, CONSULTA) == ""


def test_limites_son_de_estos_casos(casos):
    casos[0] = caso("101", "TA 120/80 FC 90, glasgow 15, hipoventilacion", "tubo de torax")
    texto = limites_calculados(casos)
    assert "1 de 4 sin desenlace clasificado" in texto
    assert "signos vitales anotados en 1 de 4" in texto
    assert "Glasgow en 1 de 4" in texto
    assert config.COBERTURA_TEMPORAL not in texto   # el aviso general va fijo en la app


# --- respuesta completa --------------------------------------------------------
def test_ensamblar_orden_y_citas_validas(casos):
    salida_modelo = f"{TITULO_SUGERENCIAS}\n- Considerar tubo de tórax [caso: 101]"
    texto = ensamblar(salida_modelo, casos, CONSULTA)
    assert texto.startswith(ROTULO_SUGERENCIAS)       # lo primero es la asistencia
    orden = [texto.index(r) for r in (ROTULO_SUGERENCIAS, ROTULO_SINTESIS, "| Caso |", "*Límites")]
    assert orden == sorted(orden)
    validas, invalidas = verificar_citas(texto, casos)
    assert invalidas == [] and set(validas) == {"101", "102", "103", "104"}


def test_ensamblar_ignora_citas_inventadas_por_el_modelo(casos):
    salida_modelo = f"{TITULO_SUGERENCIAS}\n- Considerar tubo de tórax [caso: 99999]"
    assert "99999" not in ensamblar(salida_modelo, casos, CONSULTA)


# --- resaltado en las tarjetas -------------------------------------------------
def test_resaltar_marca_la_consulta_y_escapa_html():
    texto = "Examen físico: HIPOVENTILACIÓN derecha, glasgow <15>"
    salida = resaltar_html(texto, CONSULTA)
    assert "<mark>HIPOVENTILACIÓN</mark>" in salida
    assert "&lt;15&gt;" in salida


def test_resaltar_palabras_de_la_consulta_por_raiz():
    texto = "HPACP en región abdominal, dolor a la palpación. Paciente estable"
    salida = resaltar_html(texto, "herida cortopunzante en abdomen con dolor a la palpación")
    assert "<mark>abdominal</mark>" in salida and "<mark>palpación</mark>" in salida
    assert "<mark>Paciente</mark>" not in salida        # palabras vacías no se marcan


def test_sugerencias_en_orden_diagnostico(casos):
    # el modelo elige primero el manejo; se muestran: valorar, estudiar, hacer
    salida_modelo = f"{TITULO_SUGERENCIAS}\n- Considerar tubo de tórax\n- Valorar hemotórax"
    texto = ensamblar(salida_modelo, casos, CONSULTA)
    assert texto.index("**Valorar hemotórax**") < texto.index("**Considerar tubo de tórax**")


def test_siempre_hay_un_hallazgo_a_valorar(casos):
    # el modelo elige solo manejo: se agrega el hallazgo de los casos que escalaron
    salida_modelo = (f"{TITULO_SUGERENCIAS}\n- Considerar tubo de tórax\n- Considerar sutura"
                     f"\n- Considerar observación\n- Considerar cirugía")
    elegidos = elegir_sugerencias(salida_modelo, casos, CONSULTA)
    assert len(elegidos) <= 4
    assert [e.termino for e in elegidos if e.categoria == "Hallazgos"] == ["hemotórax"]  # el del operado
    assert any(e.categoria == "Manejo" for e in elegidos)


def test_para_afinar_no_repite_lo_sugerido(casos):
    assert "hemotórax" not in para_afinar(casos, CONSULTA, ya_sugeridos={"hemotórax"})
