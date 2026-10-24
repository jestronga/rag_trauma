"""Las tarjetas por mecanismo (rag/tarjetas.py; PLAN_RESPUESTA_IDEAL.md). Sin
Ollama ni índices; la revisión contra el texto de los libros, solo si está
guias/secciones.jsonl (fuera de git: los libros tienen derechos de autor)."""
from __future__ import annotations

import json
import re

import pytest

from rag import config, generate, retrieve, tarjetas
from rag.ficha import actualizar
from rag.hallazgos import normalizar
from rag.preguntas import preguntar, ya_dicho
from rag.prioridades import LETRAS
from rag.retrieve import FragmentoGuia, Resultado
from rag.verificar import _RX_ORDEN, RX_CITA_GUIA, oraciones

CLAVES = {"peaton", "ocupante", "caida", "arma_de_fuego", "cortopunzante"}
RX_ID = re.compile(r"^(ATLS|ATLS10|HUV)-\d{2}-\d{3}$")


def ficha(*mensajes: str):
    f = None
    for m in mensajes:
        _, f = actualizar(f, m)
    return f


# ---------------------------------------------------------------------------
# El archivo
# ---------------------------------------------------------------------------
def test_las_cinco_tarjetas_con_todas_las_letras():
    cargadas = tarjetas.cargar()
    assert set(cargadas) == CLAVES                     # decisión 4 del usuario
    for t in cargadas.values():
        assert t.titulo and t.lectura and t.patron
        assert all(t.letras[l] for l in LETRAS), t.clave   # una acción por letra, y "después"


def test_cada_vineta_es_una_oracion_citada_que_orienta():
    for t in tarjetas.cargar().values():
        for parte, v in t.vinetas():
            assert v.citas and all(RX_ID.match(c) for c in v.citas), v.texto
            assert len(oraciones(v.texto)) == 1 and ";" not in v.texto and not v.texto.endswith(".")
            assert not RX_CITA_GUIA.search(v.texto)       # la cita la pone el código
            assert not _RX_ORDEN.search(normalizar(v.texto)), v.texto
            assert v.si is None or v.si in tarjetas.CONDICIONES


@pytest.mark.skipif(not config.GUIAS_SECCIONES.exists(), reason="sin el texto de los libros")
def test_cada_vineta_pasa_el_verificador_contra_su_pasaje():
    secciones = {}
    for linea in config.GUIAS_SECCIONES.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            s = json.loads(linea)
            secciones[s["id"]] = s
    assert tarjetas.revisar(secciones) == []


def test_revisar_atrapa_lo_que_no_dice_el_pasaje():
    secciones = {"HUV-02-003": {"fuente": "HUV", "idioma": "es", "texto": "Coloque un torniquete."}}
    t = tarjetas.Tarjeta("x", "x", [tarjetas.Vineta("Torniquete en la pierna con 5 cm de margen", ["HUV-02-003"]),
                                    tarjetas.Vineta("Torniquete", ["HUV-99-999"])], {})
    fallas = {texto: motivos for _, _, texto, motivos in tarjetas.revisar(secciones, {"x": t})}
    assert any("5" in m for m in fallas["Torniquete en la pierna con 5 cm de margen"])
    assert fallas["Torniquete"] == ["no existe la sección HUV-99-999"]


# ---------------------------------------------------------------------------
# Qué tarjeta y cuándo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("texto, clave", [
    ("hombre arrollado por vehiculo automotor", "peaton"),
    ("peatón atropellado por moto", "peaton"),
    ("accidente de moto con trauma en tórax", "ocupante"),
    ("caída de 6 metros", "caida"),
    ("HPAF en abdomen", "arma_de_fuego"),
    ("herida por arma cortopunzante en tórax", "cortopunzante"),
    ("quemadura por agua caliente", "quemadura"),           # sin tarjeta: como antes
])
def test_la_clave_sale_del_mecanismo(texto, clave):
    assert tarjetas.clave(ficha(texto)) == clave


def test_el_peaton_y_las_ruedas_no_cambian_el_mecanismo():
    f = ficha("hombre arrollado por vehiculo automotor")
    assert f.mecanismo == "accidente_transito" and f.variantes == ["peaton"]
    assert f.datos()["mecanismo"] == "accidente_transito" and "variantes" not in f.datos()
    f = ficha("hombre arrollado por vehiculo automotor", "las ruedas le pasaron por encima")
    assert f.variantes == ["peaton", "aplastamiento"] and "las ruedas le pasaron por encima" in f.linea()
    f = ficha("hombre arrollado por vehiculo automotor", "no, no le pasó por encima, fue solo el impacto")
    assert f.variantes == ["peaton", "impacto"]
    # la respuesta del botón cambia la anterior, y es un seguimiento aunque se escriba a mano
    _, f = actualizar(ficha("peatón atropellado"), "fue solo el impacto")
    assert f.variantes == ["peaton", "impacto"]
    tipo, f = actualizar(f, "las ruedas le pasaron por encima")
    assert tipo == "seguimiento" and f.variantes == ["peaton", "aplastamiento"]
    # lo que se pregunta no es un hecho
    assert ficha("peatón atropellado, ¿le pasó por encima?").variantes == ["peaton"]


@pytest.mark.parametrize("mensaje, especifica", [
    ("¿qué dice el ATLS del FAST?", True),
    ("¿le hago FAST?", True),
    ("HPAF en tórax, ¿cuándo poner un tubo de tórax?", True),
    ("¿qué hago?", False),
    ("paciente atropellado, ¿cuál es el manejo?", False),
    ("hombre arrollado por vehiculo automotor", False),
])
def test_una_pregunta_concreta_no_lleva_tarjeta(mensaje, especifica):
    assert tarjetas.pregunta_especifica(mensaje) is especifica


def test_elegir(monkeypatch):
    monkeypatch.setattr(config, "TARJETAS", True)
    f = ficha("HPAF en abdomen")
    assert tarjetas.elegir(f, "HPAF en abdomen", True).clave == "arma_de_fuego"
    assert tarjetas.elegir(f, "HPAF en abdomen", False) is None        # sin los manuales
    assert tarjetas.elegir(f, "¿qué dice el ATLS del FAST?", True) is None
    assert tarjetas.elegir(ficha("¿cuándo se pone un tubo de tórax?"), "x", True) is None   # sin paciente
    monkeypatch.setattr(config, "TARJETAS", False)
    assert tarjetas.elegir(f, "HPAF en abdomen", True) is None


# ---------------------------------------------------------------------------
# La tarjeta ajustada a la ficha
# ---------------------------------------------------------------------------
def test_sin_datos_van_las_dos_ramas_con_su_rotulo(monkeypatch):
    monkeypatch.setattr(retrieve, "libros_apagados", lambda: set())       # con el ATLS 10
    t = tarjetas.cargar()["peaton"]
    texto, citas = tarjetas.armar(t, ficha("hombre arrollado por vehiculo automotor"))
    assert texto.startswith("### Protocolo inicial: peatón atropellado")
    assert "*Si está estable:*" in texto and "*Si está inestable:*" in texto
    assert "*Si las ruedas le pasaron por encima:*" in texto
    assert all(f"**{l} · " in texto for l in "xABCDE") and "**Después de la revisión primaria**" in texto
    assert texto.endswith("**Datos que cambiarían la orientación:** la edad, los signos vitales, el Glasgow, "
                          "el tiempo desde el trauma y si las ruedas le pasaron por encima.")
    assert set(citas) == {c for _, v in t.vinetas() for c in v.citas}


def test_con_datos_la_rama_que_corresponde_y_los_datos_en_su_letra():
    t = tarjetas.cargar()["peaton"]
    f = ficha("peatón atropellado, TA 80/50, Glasgow 7, adulto", "las ruedas le pasaron por encima")
    texto, _ = tarjetas.armar(t, f)
    assert "**C · Circulación** · TA 80/50 · inestable" in texto and "**D · Déficit neurológico** · Glasgow 7" in texto
    assert "*Está inestable:*" in texto and "estable:*" not in texto.replace("*Está inestable:*", "")
    assert "*Las ruedas le pasaron por encima:*" in texto
    assert texto.endswith("**Datos que cambiarían la orientación:** el tiempo desde el trauma.")
    f = ficha("peatón atropellado, hemodinámicamente estable", "fue solo el impacto")
    texto, _ = tarjetas.armar(t, f)
    assert "*Está estable:*" in texto and "inestable" not in texto and "aplastamiento" not in texto.lower()


def test_la_region_decide_las_vinetas_de_cuello_y_torax():
    t = tarjetas.cargar()["arma_de_fuego"]
    sin_region, _ = tarjetas.armar(t, ficha("herida por arma de fuego"))
    assert "*Si hay heridas en el tórax:*" in sin_region and "*Si hay heridas en el cuello:*" in sin_region
    abdomen, _ = tarjetas.armar(t, ficha("HPAF en abdomen"))
    assert "tórax:*" not in abdomen and "cuello:*" not in abdomen
    cuello, _ = tarjetas.armar(t, ficha("HPAF en cuello"))
    assert "*Heridas en el cuello:*" in cuello


def test_sin_el_atls10_no_van_sus_vinetas(monkeypatch):
    monkeypatch.setattr(retrieve, "libros_apagados", lambda: {"ATLS10"})
    texto, citas = tarjetas.armar(tarjetas.cargar()["peaton"], ficha("peatón atropellado"))
    assert "ATLS10" not in texto and not any(c.startswith("ATLS10") for c in citas)
    assert "**x · Hemorragia exanguinante**" in texto


def test_la_linea_del_registro(tmp_path):
    ruta = tmp_path / "patrones.json"
    ruta.write_text(json.dumps({"peaton": {
        "nombre": "peatón atropellado", "n": 1324,
        "regiones": [["miembro inferior", 662], ["cráneo / encéfalo", 331]],
        "hallazgos": [["fractura", 331]],
        "desenlace": {"conocido": 1000, "hospitalizacion": 200, "cirugia": 1, "uci": 0, "fallecimiento": 0}}}),
        encoding="utf-8")
    assert tarjetas.linea_registro("peaton", ruta) == (
        "**En el registro** hay 1.324 casos de peatón atropellado: las regiones más afectadas son miembro "
        "inferior (50 %) y cráneo / encéfalo (25 %), y lo más documentado es fractura (25 %). De los 1.000 "
        "con desenlace registrado, 200 se hospitalizaron y 1 fue a cirugía.")
    assert tarjetas.linea_registro("caida", ruta) == ""


# ---------------------------------------------------------------------------
# Las preguntas de la tarjeta
# ---------------------------------------------------------------------------
def test_preguntas_con_tarjeta(monkeypatch):
    monkeypatch.setattr(config, "TARJETAS", True)
    texto = "hombre arrollado por vehiculo automotor"
    ps = preguntar(ficha(texto), [], texto)
    assert [p.clave for p in ps] == ["aplastamiento", "C"]
    assert [o.etiqueta for o in ps[0].opciones] == ["Impacto", "Le pasaron por encima", "No se sabe"]
    # la respuesta del botón la lee la ficha, y ya no se pregunta
    f = ficha(texto, ps[0].opciones[1].mensaje)
    assert "aplastamiento" in f.variantes and "aplastamiento" not in [p.clave for p in preguntar(f, [], "")]
    assert ya_dicho(ps[0], "peatón atropellado, las ruedas le pasaron por encima")
    # una caída de moto con TCE: la C, porque decide las ramas de la tarjeta
    assert [p.clave for p in preguntar(ficha("TCE por caída de moto en adulto"), [], "")] == ["C", "D"]
    monkeypatch.setattr(config, "TARJETAS", False)
    assert [p.clave for p in preguntar(ficha(texto), [], texto)] == ["D", "grupo"]


# ---------------------------------------------------------------------------
# responder
# ---------------------------------------------------------------------------
def _caso(rid: str, texto: str) -> Resultado:
    return Resultado(rid, f"Motivo de consulta: HPAF\n\nExamen físico: {texto}\n\nEvolución y hospitalización: "
                          f"{texto}", {"mecanismo_lesion": "arma_de_fuego", "region_anatomica": ["abdomen"]},
                     0.7, 0.7)


def _frag(sid: str, texto: str = "texto del pasaje") -> FragmentoGuia:
    fuente = sid.split("-")[0]
    return FragmentoGuia(id=f"{sid}-01", seccion_id=sid, fuente=fuente, libro=fuente, capitulo="Capítulo",
                         titulos=["Sección"], pagina_ini=1, pagina_fin=1, tipo="texto",
                         idioma="es" if fuente != "ATLS" else "en", texto=texto, texto_contexto=texto,
                         n_tokens_contexto=10, coseno=0.0)


@pytest.fixture
def entorno(monkeypatch):
    casos = [_caso("101", "hipotension, laparotomia"), _caso("102", "hipotension, transfusion")]
    formatos: list = []
    monkeypatch.setattr(config, "SALIDA", "prosa")
    monkeypatch.setattr(config, "SINTESIS_LLM", False)
    monkeypatch.setattr(config, "CONVERSACION", "ficha")
    monkeypatch.setattr(config, "ORDEN_ATLS", True)
    monkeypatch.setattr(config, "PROSA_SIEMPRE_ATLS", True)
    monkeypatch.setattr(config, "TARJETAS", True)
    monkeypatch.setattr(config, "GUIAS", True)
    monkeypatch.setattr(config, "PREGUNTAS", False)
    monkeypatch.setattr(generate, "confianza", lambda *a, **k: 0.9)
    monkeypatch.setattr(generate, "buscar", lambda *a, **k: casos)
    monkeypatch.setattr(generate, "guarda_casos", lambda consulta, c, u: (c, []))
    monkeypatch.setattr(generate, "hay_guias", lambda: True)
    monkeypatch.setattr(generate, "buscar_guias", lambda *a, **k: [_frag("HUV-15-004"), _frag("HUV-21-002")])
    monkeypatch.setattr(generate, "fragmentos_de", lambda secciones: [_frag(s) for s in secciones])

    def falso(mensajes, formato=None, **_):
        formatos.append(formato)
        if "manuales" in (formato or {}).get("properties", {}):
            return json.dumps({"manuales": {"C": "El manual del HUV recomienda FAST [guía: HUV-21-002]."}})
        return json.dumps({"casos": {"C": "En casos similares se hizo transfusión [caso: 102]."}})
    monkeypatch.setattr(generate, "_ollama", falso)
    return formatos


def test_con_tarjeta_el_protocolo_va_primero_y_el_modelo_escribe_solo_los_casos(entorno):
    r = generate.responder("HPAF en abdomen")
    assert r.tarjeta == "arma_de_fuego" and [s.fuente for s in r.secciones] == ["tarjeta", "casos", "pie"]
    assert r.secciones[0].texto.startswith("### Protocolo inicial: herida por arma de fuego")
    assert set(entorno[-1]["properties"]) == {"casos"}          # el modelo no escribe sobre los manuales
    assert "En casos similares se hizo transfusión [caso: 102]" in r.texto
    assert "**En el registro** hay" in r.secciones[1].texto or not config.PATRONES_REGISTRO.exists()
    # las citas de la tarjeta son válidas: sus pasajes vienen en pasajes_tarjeta
    assert r.citas_invalidas == [] and {f.seccion_id for f in r.pasajes_tarjeta} >= {"HUV-04-006", "HUV-02-003"}
    assert {f.seccion_id for f in r.guias} == {"HUV-15-004", "HUV-21-002"}     # la búsqueda, intacta


def test_una_pregunta_concreta_va_con_la_prosa_de_los_manuales(entorno):
    r1 = generate.responder("HPAF en abdomen")
    r2 = generate.responder("¿qué dice el ATLS del FAST?", ficha=r1.ficha, anterior=r1)
    assert r2.tarjeta is None and "manuales" in entorno[-1]["properties"]
    assert [s.fuente for s in r2.secciones][0] == "casos" and r2.pasajes_tarjeta == []


def test_un_seguimiento_con_datos_rearma_la_tarjeta(entorno):
    r1 = generate.responder("HPAF en abdomen")
    r2 = generate.responder("hipotenso o en choque", ficha=r1.ficha, anterior=r1)
    assert r2.tarjeta == "arma_de_fuego" and "*Está inestable:*" in r2.secciones[0].texto
    assert "*Si está estable:*" not in r2.secciones[0].texto


def test_sin_tarjetas_todo_como_antes(entorno, monkeypatch):
    monkeypatch.setattr(config, "TARJETAS", False)
    r = generate.responder("HPAF en abdomen")
    assert r.tarjeta is None and r.pasajes_tarjeta == [] and "tarjeta" not in [s.fuente for s in r.secciones]
    assert "manuales" in entorno[-1]["properties"]
