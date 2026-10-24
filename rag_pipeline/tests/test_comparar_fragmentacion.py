"""Métricas de scripts/comparar_fragmentacion.py y muestreo de
scripts/generar_preguntas_guias.py, sin modelos."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _modulo(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, SCRIPTS / f"{nombre}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cf = _modulo("comparar_fragmentacion")
gp = _modulo("generar_preguntas_guias")


def contar(texto: str) -> int:
    return len(texto.split())


def frag(sid: str, texto: str, contexto: str | None = None) -> dict:
    return {"seccion_id": sid, "texto": texto, "texto_contexto": contexto or texto}


# ---------------------------------------------------------------------------
# comparar_fragmentacion
# ---------------------------------------------------------------------------
def test_cargar_consultas_solo_con_los_libros_indexados(tmp_path, monkeypatch, capsys):
    # ATLS 10 (23/10): anotado y extraído antes de indexarse; calibrar_guias.py
    # no debe contar sus secciones (no se pueden recuperar) ni avisar por ellas
    anot = tmp_path / "c.yaml"
    anot.write_text("consultas:\n  - id: q1\n    texto: x\n    categoria: lexica\n    relevantes:\n"
                    "      ATLS-05-012: 2\n      ATLS10-04-007: 2\n", encoding="utf-8")
    sint = tmp_path / "s.jsonl"
    sint.write_text('{"id": "sin-1", "texto": "y", "seccion_id": "ATLS10-04-007", "fuente": "ATLS10"}\n'
                    '{"id": "sin-2", "texto": "z", "seccion_id": "ATLS-05-012", "fuente": "ATLS"}\n',
                    encoding="utf-8")
    monkeypatch.setattr(cf, "CONSULTAS", anot)
    monkeypatch.setattr(cf, "SINTETICAS", sint)
    secciones = {"ATLS-05-012": {"fuente": "ATLS"}, "ATLS10-04-007": {"fuente": "ATLS10"}}
    todas = cf.cargar_consultas(secciones, "ambos")
    assert todas[0]["relevantes"] == {"ATLS-05-012": 2, "ATLS10-04-007": 2} and len(todas) == 3
    solo = cf.cargar_consultas(secciones, "ambos", {"ATLS", "HUV"})
    assert solo[0]["relevantes"] == {"ATLS-05-012": 2}
    assert [c["id"] for c in solo] == ["q1", "sin-2"]
    assert "aviso" not in capsys.readouterr().out



def test_metricas_deduplican_por_seccion():
    frags = [frag("ATLS-05-012", "a " * 10), frag("ATLS-05-012", "b " * 10),
             frag("HUV-11-007", "c " * 10), frag("X-01-001", "d " * 10)]
    rel = {"ATLS-05-012": 2, "HUV-11-007": 2, "ATLS-05-028": 1}
    m = cf.metricas([0, 1, 2, 3], frags, rel, contar, presupuesto=1000)
    # secciones en orden: ATLS-05-012, HUV-11-007, X-01-001 → las dos de grado 2 en el top 3
    assert m["R@3"] == 1.0
    assert m["MRR"] == 1.0
    assert m["R@ppto_ATLS"] == 1.0 and m["R@ppto_HUV"] == 1.0


def test_presupuesto_corta_y_no_repite_contextos():
    padre = "p " * 50
    frags = [frag("A-01-001", "hijo uno", padre), frag("A-01-001", "hijo dos", padre),
             frag("B-01-001", "otro", "q " * 50), frag("C-01-001", "tercero", "r " * 50)]
    rel = {"C-01-001": 2}
    m = cf.metricas([0, 1, 2, 3], frags, rel, contar, presupuesto=120)
    # el padre repetido cuenta una vez: entran A y B (100 tokens); C ya no cabe
    assert m["frags_ppto"] == 2 and m["tokens_ppto"] == 100
    assert m["R@ppto"] == 0.0
    assert math.isnan(m["R@ppto_HUV"])


def test_rrf_premia_lo_que_ambas_rutas_traen():
    assert cf.rrf([[1, 2, 3], [3, 4, 1]])[:2] == [1, 3]


# ---------------------------------------------------------------------------
# generar_preguntas_guias
# ---------------------------------------------------------------------------
def test_copia_detecta_cinco_palabras_seguidas():
    texto = "Hemothoraces with estimated volume more than 300 mL are evacuated with a tube."
    assert gp.copia("¿Hemothoraces with estimated volume more than 300 mL?", texto)
    assert not gp.copia("¿Cuándo se drena un hemotórax con tubo de tórax?", texto)


def test_muestrear_reparte_por_libro_y_capitulo():
    secciones = [{"id": f"{f}-{c:02d}-{i:03d}", "fuente": f, "capitulo_num": c, "clinico": True,
                  "texto": "x" * 500}
                 for f in ("ATLS", "HUV") for c in range(1, 6) for i in range(10)]
    secciones.append({"id": "HUV-01-999", "fuente": "HUV", "capitulo_num": 1, "clinico": False,
                      "texto": "x" * 500})
    secciones.append({"id": "HUV-02-999", "fuente": "HUV", "capitulo_num": 2, "clinico": True,
                      "texto": "corta"})
    elegidas = gp.muestrear(secciones, 20)
    assert len(elegidas) == 20
    assert sum(s["fuente"] == "ATLS" for s in elegidas) == 10
    # 10 por libro en 5 capítulos: 2 por capítulo
    for f in ("ATLS", "HUV"):
        caps = [s["capitulo_num"] for s in elegidas if s["fuente"] == f]
        assert sorted(caps) == sorted(list(range(1, 6)) * 2)
    assert all(s["clinico"] and len(s["texto"]) >= gp.MIN_CARACTERES for s in elegidas)
    assert len({s["id"] for s in elegidas}) == 20
