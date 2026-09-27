"""
Tests del extractor de hallazgos, estudios y manejo (rag/hallazgos.py).

Lo que extrae este módulo son los conteos que lee el médico ("tubo de tórax en
4 de 5 casos"): un falso positivo aquí es una afirmación clínica falsa. Los
textos son fragmentos reales del corpus (casos citados en cada test).
Correr desde rag_pipeline/:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.hallazgos import secciones, terminos  # noqa: E402


def caso(motivo: str, examen: str, evolucion: str) -> str:
    """Mismo formato que el texto indexado."""
    return (f"Motivo de consulta: {motivo}\n\nExamen físico: {examen}\n\n"
            f"Evolución y hospitalización: {evolucion}")


def manejo(texto: str) -> list[str]:
    return terminos(texto)["Manejo"]


def hallazgos(texto: str) -> list[str]:
    return terminos(texto)["Hallazgos"]


# --- secciones ---------------------------------------------------------------
def test_secciones_separa_los_tres_campos():
    s = secciones(caso("ME CAI", "DOLOR EN MUÑECA", "SE INMOVILIZA"))
    assert s == {"Motivo de consulta": "ME CAI", "Examen físico": "DOLOR EN MUÑECA",
                 "Evolución y hospitalización": "SE INMOVILIZA"}


def test_consulta_sin_secciones_se_lee_entera():
    assert "tubo de tórax" in manejo("herida en tórax con tubo de tórax")


# --- negación ----------------------------------------------------------------
def test_negacion_a_la_izquierda():
    assert "sangrado activo" not in hallazgos(caso("ME CORTE", "HERIDA SIN SANGRADO ACTIVO", "SUTURA"))


def test_negacion_a_la_derecha():
    t = caso("HPACP", "ABDOMEN BLANDO, BLUMBERG NEGATIVO", "OBSERVACION")
    assert "irritación peritoneal" not in hallazgos(t)


def test_no_a_la_derecha_no_niega():
    # "no desplazada" describe la fractura, no la niega
    t = caso("ME CAI", "FRACTURA NO DESPLAZADA DE RADIO", "FERULA")
    assert "fractura" in hallazgos(t)


# --- remisión: solo la de salida ---------------------------------------------
def test_remitido_al_llegar_no_es_remision():
    # caso 32652
    t = caso("remitido por HPAF (herida por arma de fuego)", "hipoventilacion izquierda",
             "salida con recomendaciones, signos de alarma.")
    assert "remisión" not in manejo(t)


def test_remiten_desde_otra_institucion_no_es_remision():
    # caso 46956
    t = caso("Fractura de los huesos propios de la nariz",
             "Consulta al [INSTITUCION] evidenciando fractura por lo cual remiten",
             "se realiza reduccion cerrada")
    assert "remisión" not in manejo(t)


@pytest.mark.parametrize("evolucion", [
    "se remite a nivel II, h. [INSTITUCION] por falta de disponibilidad",   # caso 33095
    "se decide remitir a nivel III",
    "se da remision a nivel II",
])
def test_remision_de_salida_si_cuenta(evolucion):
    assert "remisión" in manejo(caso("HPAF", "hipoventilacion", evolucion))


# --- cirugía: operado o con indicación, no interconsulta ----------------------
@pytest.mark.parametrize("evolucion", [
    "paciente valorado por cx plastica quien considera salida",             # caso 38452
    "evaluado por neurocirugia se considera dar salida",                    # caso 34998
    "se da salida con cita control por cirugía plástica",
    "PROYECTIL EXTRA-ARTICULAR EL CUAL NO REQUIERE MANEJO QUIRURGICO",      # caso 31868
])
def test_interconsulta_o_negada_no_es_cirugia(evolucion):
    assert "cirugía" not in manejo(caso("ME CAI", "DOLOR", evolucion))


@pytest.mark.parametrize("evolucion", [
    "PROCEDIMIENTO QX (quirurgico): se liga epiplocele y se corta",         # caso 15775
    "HALLAZGOS OPERATORIOS: HEMOPERITONEO DE APROX. 1000CC",                # caso 25851
    "es llevado a cx para realizar ventana pericardica",                    # caso 37621
    "SE LE REALIZA LAPARATOMIA EXPLORATORIA DON NO SE EVIDENCIA PERFORACION",  # caso 27722
    "PACIENTE QUIEN REQUIERE CX DE TOBILLO POR PRESENTAR FX",               # caso 31005
])
def test_cirugia_hecha_o_indicada(evolucion):
    assert "cirugía" in manejo(caso("HERIDA", "DOLOR", evolucion))


def test_laparatomia_mal_escrita():
    t = caso("HERIDA EN ABDOMEN", "BLUMBER POSITIVO", "SE LE REALIZA LAPARATOMIA EXPLORATORIA")
    assert "laparotomía / laparoscopia" in manejo(t)


def test_manejo_en_el_motivo_no_cuenta():
    # lo que se hizo antes de llegar no es manejo de este hospital
    t = caso("remitido con tubo de torax", "hipoventilacion", "se da salida")
    assert "tubo de tórax" not in manejo(t)


# --- hallazgos nuevos --------------------------------------------------------
@pytest.mark.parametrize("examen", [
    "ABDOMEN CON DOLOR A LA PALPACION CON BLUMBER POSITIVO",                # caso 27722
    "abdomen agdo BLOMBERG +",                                              # caso 3377
    "presenta abdomen agudo",
])
def test_irritacion_peritoneal(examen):
    assert "irritación peritoneal" in hallazgos(caso("HPACP", examen, "OBSERVACION"))


def test_hemoperitoneo_y_liquido_libre():
    t = caso("HACP", "ECO FAST QUE EVIDENCIA LIQUIDO LIBRE EN LA CAVIDAD", "laparotomia")  # caso 57373
    assert "hemoperitoneo / líquido libre" in hallazgos(t)
