"""
Tarjetas por mecanismo (PLAN_RESPUESTA_IDEAL.md, fase 1): el protocolo inicial
de cada mecanismo, escrito una vez desde el texto de los pasajes que cita
(tarjetas/tarjetas.yaml) y ajustado a la ficha del paciente con reglas.

Casi toda una buena respuesta a "hombre arrollado por vehículo automotor" es
igual para todo peatón: la revisión primaria, los estudios, las lesiones que
se esperan. Es justo lo que peor le sale al 4B en cada consulta (los manuales
pasaban enteros la verificación en 4 de 20 respuestas, 24/10), y aquí no cuesta
tiempo: el modelo escribe solo sobre los casos. Cada viñeta cita su pasaje y
pasa el verificador de la prosa contra él (tests/test_tarjetas.py); el médico
las valida en la fase 7 de los libros (decisión 2 del usuario: se muestran
antes).

Lo que hace la ficha con una tarjeta:
  · las viñetas con condición (`si`) se filtran: sin dato van con su rótulo
    ("Si está inestable: …"), con el dato en contra no van, con el dato a favor
    van con el rótulo en presente ("Está inestable: …");
  · la letra con datos en la ficha los muestra ("C · Circulación · TA 85/55");
  · al final, los datos que cambiarían la orientación y que la ficha no tiene.

No hay tarjeta si la consulta pregunta algo concreto ("¿qué dice el ATLS del
FAST?", "¿le hago FAST?"): eso lo responde la prosa sobre los pasajes que
encontró la búsqueda, como antes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from . import config
from .ficha import Ficha, _analizable, describir_signo, leer, unir
from .hallazgos import menciones
from .prioridades import DESPUES, LETRAS, NOMBRES
from .senal import PROCEDIMIENTOS

PRIMARIAS = [l for l in LETRAS if l != DESPUES]
# Los apartados de la tarjeta además de las letras, en el orden en que se muestran
TITULOS = {"patron": "Lesiones que se esperan", "ademas": "Además"}
# Los datos de la ficha que se muestran junto a cada letra
_SIGNOS_LETRA = {"B": ("FR", "SatO2"), "C": ("TA", "FC"), "D": ("Glasgow",)}


@dataclass
class Vineta:
    texto: str
    citas: list[str]
    si: str | None = None

    def con_citas(self) -> str:
        return f"{self.texto} " + " ".join(f"[guía: {c}]" for c in self.citas)


@dataclass
class Tarjeta:
    clave: str
    titulo: str
    lectura: list[Vineta]
    letras: dict[str, list[Vineta]]          # x…E y "después", con la parte común primero
    patron: list[Vineta] = field(default_factory=list)
    ademas: list[Vineta] = field(default_factory=list)

    def partes(self) -> list[tuple[str, list[Vineta]]]:
        """(apartado, viñetas) en el orden en que se muestran."""
        return ([("lectura", self.lectura)] + [(l, self.letras.get(l, [])) for l in LETRAS]
                + [("patron", self.patron), ("ademas", self.ademas)])

    def vinetas(self) -> list[tuple[str, Vineta]]:
        return [(parte, v) for parte, vs in self.partes() for v in vs]


def _vinetas(datos: list[dict] | None) -> list[Vineta]:
    return [Vineta(d["texto"].strip(), list(d["cita"]), d.get("si")) for d in datos or []]


@lru_cache(maxsize=4)
def cargar(ruta: Path | str | None = None) -> dict[str, Tarjeta]:
    """clave -> tarjeta, con la revisión primaria común sumada a sus letras."""
    ruta = Path(ruta or config.TARJETAS_YAML)
    if not ruta.exists():
        return {}
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    clave_letra = {**{l: l for l in PRIMARIAS}, "despues": DESPUES}
    comun = {clave_letra[k]: _vinetas(v) for k, v in (datos.get("comun") or {}).items()}
    salida = {}
    for clave, t in (datos.get("tarjetas") or {}).items():
        propias = {clave_letra[k]: _vinetas(v) for k, v in (t.get("letras") or {}).items()}
        letras = {l: comun.get(l, []) + propias.get(l, []) for l in LETRAS}
        salida[clave] = Tarjeta(clave, t["titulo"], _vinetas(t.get("lectura")), letras,
                                _vinetas(t.get("patron")), _vinetas(t.get("ademas")))
    return salida


def clave(ficha: Ficha | None) -> str | None:
    """La tarjeta que corresponde al mecanismo de la ficha (exista o no)."""
    if ficha is None or not ficha.mecanismo:
        return None
    if ficha.mecanismo == "accidente_transito":
        return "peaton" if "peaton" in ficha.variantes else "ocupante"
    return ficha.mecanismo


_RX_FUENTE = re.compile(r"\b(atls|huv|manual(es)?|guias?|libros?)\b")
_RX_PROCEDIMIENTO = re.compile("|".join(PROCEDIMIENTOS))


def pregunta_especifica(mensaje: str) -> bool:
    """¿El mensaje pregunta algo concreto? Un término del léxico, un
    procedimiento o un manual en lo que pregunta: "¿le hago FAST?", "¿qué dice
    el ATLS del FAST?". "¿Qué hago?" no: ahí va la tarjeta."""
    for p in leer(mensaje).preguntas:
        n = _analizable(p)
        afirmados, negados = menciones(n)
        if afirmados or negados or _RX_PROCEDIMIENTO.search(n) or _RX_FUENTE.search(n):
            return True
    return False


def elegir(ficha: Ficha | None, mensaje: str, con_guias: bool) -> Tarjeta | None:
    """La tarjeta para este mensaje, o None: sin TARJETAS, sin los manuales,
    sin paciente, sin tarjeta para su mecanismo o con una pregunta concreta."""
    if not config.TARJETAS or not con_guias or ficha is None or not ficha.es_paciente():
        return None
    tarjeta = cargar().get(clave(ficha))
    if tarjeta is None or pregunta_especifica(mensaje):
        return None
    return tarjeta


# ---------------------------------------------------------------------------
# Condiciones (`si`): None = la ficha no lo sabe
# ---------------------------------------------------------------------------
def _estado(ficha: Ficha, cual: str) -> bool | None:
    return None if ficha.estado is None else ficha.estado == cual


def _glasgow(ficha: Ficha) -> int | None:
    valor = ficha.signos.get("Glasgow")
    return int(re.match(r"\d+", valor).group()) if valor else None


def _tce(ficha: Ficha) -> bool | None:
    dicho = _analizable(" ".join(ficha.clausulas))
    if "craneo_encefalo" in ficha.regiones or re.search(r"\btce\b|\btec\b|craneoencefal", dicho):
        return True
    return None


def _grupo(ficha: Ficha, cual: str) -> bool | None:
    return None if ficha.grupo is None else ficha.grupo == cual


def _variante(ficha: Ficha) -> bool | None:
    if "aplastamiento" in ficha.variantes:
        return True
    return False if "impacto" in ficha.variantes else None


def _region(ficha: Ficha, region: str) -> bool | None:
    if region in ficha.regiones:
        return True
    return False if ficha.regiones else None


# condición -> (qué dice la ficha, rótulo sin el dato, rótulo con el dato)
CONDICIONES = {
    "estable": (lambda f: _estado(f, "estable"), "Si está estable", "Está estable"),
    "inestable": (lambda f: _estado(f, "inestable"), "Si está inestable", "Está inestable"),
    "tce": (_tce, "Si hay TCE", "Con TCE"),
    "adulto_mayor": (lambda f: _grupo(f, "adulto mayor"), "Si es adulto mayor", "Es adulto mayor"),
    "aplastamiento": (_variante, "Si las ruedas le pasaron por encima", "Las ruedas le pasaron por encima"),
    "region:torax": (lambda f: _region(f, "torax"), "Si hay heridas en el tórax", "Heridas en el tórax"),
    "region:cuello": (lambda f: _region(f, "cuello"), "Si hay heridas en el cuello", "Heridas en el cuello"),
}


def _con_rotulo(v: Vineta, ficha: Ficha) -> str | None:
    """El texto de la viñeta con su rótulo, o None si la ficha dice lo contrario."""
    if v.si is None:
        return v.con_citas()
    dice, sin_dato, con_dato = CONDICIONES[v.si]
    cumple = dice(ficha)
    if cumple is False:
        return None
    return f"*{con_dato if cumple else sin_dato}:* {v.con_citas()}"


def _apagada(v: Vineta) -> bool:
    """Cita un libro que no se busca (el ATLS 10 con ATLS10=0)."""
    from .retrieve import libros_apagados
    return any(c.split("-")[0] in libros_apagados() for c in v.citas)


def _datos_letra(letra: str, ficha: Ficha) -> str:
    partes = [describir_signo(s, ficha.signos[s]) for s in _SIGNOS_LETRA.get(letra, ()) if s in ficha.signos]
    if letra == "C" and ficha.estado:
        partes.append(ficha.estado)
    return "".join(f" · {p}" for p in partes)


_RX_TIEMPO = re.compile(r"\bhace\s+(\d+|una?|media)\b|\b\d+\s*(h|hrs?|horas?|min|minutos)\b")


def faltan(tarjeta: Tarjeta, ficha: Ficha) -> list[str]:
    """Lo que la ficha no tiene y cambiaría la orientación (como la lista del
    final de la respuesta ideal): solo lo que la tarjeta usa."""
    salida = []
    if ficha.grupo is None:
        salida.append("la edad")
    if not (ficha.estado or {"TA", "FC"} & ficha.signos.keys()):
        salida.append("los signos vitales")
    if "Glasgow" not in ficha.signos and "D" not in ficha.sin_dato:
        salida.append("el Glasgow")
    usa_tiempo = any("tranex" in v.texto.lower() for _, v in tarjeta.vinetas())
    if usa_tiempo and ficha.estado != "estable" and not _RX_TIEMPO.search(_analizable(" ".join(ficha.clausulas))):
        salida.append("el tiempo desde el trauma")
    if any(v.si == "aplastamiento" for _, v in tarjeta.vinetas()) and _variante(ficha) is None:
        salida.append("si las ruedas le pasaron por encima")
    if any(v.si and v.si.startswith("region:") for _, v in tarjeta.vinetas()) and not ficha.regiones:
        salida.append("la localización de las heridas")
    return salida


def armar(tarjeta: Tarjeta, ficha: Ficha) -> tuple[str, list[str]]:
    """(el apartado en Markdown, las secciones que cita). Lo arma el código: el
    modelo no interviene."""
    bloques = [f"### Protocolo inicial: {tarjeta.titulo}"]
    citas: list[str] = []
    for parte, vinetas in tarjeta.partes():
        lineas = []
        for v in vinetas:
            if _apagada(v):
                continue
            texto = _con_rotulo(v, ficha)
            if texto is not None:
                lineas.append(f"- {texto}")
                citas += v.citas
        if not lineas:
            continue
        if parte == "lectura":
            bloques.append("\n".join(lineas))
        elif parte in TITULOS:
            bloques.append(f"**{TITULOS[parte]}**\n" + "\n".join(lineas))
        else:
            rotulo = NOMBRES[parte] if parte == DESPUES else f"{parte} · {NOMBRES[parte]}"
            bloques.append(f"**{rotulo}**{_datos_letra(parte, ficha)}\n" + "\n".join(lineas))
    if falta := faltan(tarjeta, ficha):
        bloques.append(f"**Datos que cambiarían la orientación:** {unir(falta)}.")
    return "\n\n".join(bloques), list(dict.fromkeys(citas))


# ---------------------------------------------------------------------------
# El registro como epidemiología (fase 2): conteos de scripts/patrones_registro.py
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _patrones(ruta: Path | str | None = None) -> dict:
    ruta = Path(ruta or config.PATRONES_REGISTRO)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def _pct(n: int, total: int) -> str:
    return f"{round(100 * n / total)} %"


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


# desenlace -> (singular, plural)
_DESENLACES = (("hospitalizacion", "se hospitalizó", "se hospitalizaron"),
               ("cirugia", "fue a cirugía", "fueron a cirugía"),
               ("uci", "fue a UCI", "fueron a UCI"),
               ("fallecimiento", "falleció", "fallecieron"))


def linea_registro(clave_tarjeta: str, ruta: Path | str | None = None) -> str:
    """"En el registro hay 324 casos de peatón atropellado…": todos los casos del
    registro con ese mecanismo, no los recuperados. Vacía sin el archivo."""
    p = _patrones(ruta).get(clave_tarjeta)
    if not p or not p.get("n"):
        return ""
    n = p["n"]
    texto = f"**En el registro** hay {_miles(n)} casos de {p['nombre']}"
    regiones = [f"{r} ({_pct(k, n)})" for r, k in p.get("regiones", [])[:4]]
    hallazgos = [f"{t} ({_pct(k, n)})" for t, k in p.get("hallazgos", [])[:3]]
    if regiones:
        texto += f": las regiones más afectadas son {unir(regiones)}"
    if hallazgos:
        texto += f"{', y' if regiones else ':'} lo más documentado es {unir(hallazgos)}"
    texto += "."
    d = p.get("desenlace") or {}
    partes = [f"{d[k]} {uno if d[k] == 1 else varios}" for k, uno, varios in _DESENLACES if d.get(k)]
    if d.get("conocido") and partes:
        texto += f" De los {_miles(d['conocido'])} con desenlace registrado, {unir(partes)}."
    return texto


# ---------------------------------------------------------------------------
# Revisión de las tarjetas contra sus pasajes (tests/test_tarjetas.py)
# ---------------------------------------------------------------------------
def revisar(secciones: dict[str, dict], tarjetas: dict[str, Tarjeta] | None = None) -> list[tuple[str, str, str, list[str]]]:
    """[(tarjeta, apartado, texto, motivos)] de cada viñeta que no pasa el
    verificador de la prosa contra el texto de las secciones que cita
    (`secciones`: id -> registro de guias/secciones.jsonl)."""
    from .retrieve import FragmentoGuia
    from .verificar import oraciones, verificar_por_fuente

    def fragmento(sid: str) -> FragmentoGuia:
        s = secciones[sid]
        return FragmentoGuia(id=f"{sid}-01", seccion_id=sid, fuente=s["fuente"], libro=s.get("libro", ""),
                             capitulo=s.get("capitulo", ""), titulos=list(s.get("titulos", [])),
                             pagina_ini=s.get("pagina_ini"), pagina_fin=s.get("pagina_fin"), tipo="texto",
                             idioma=s.get("idioma", "es"), texto=s["texto"], texto_contexto=s["texto"],
                             n_tokens_contexto=0, coseno=0.0, capitulo_num=s.get("capitulo_num"))

    salida, vistas = [], set()
    for t in (tarjetas or cargar()).values():
        for parte, v in t.vinetas():
            if (v.texto, tuple(v.citas)) in vistas:          # las de la parte común, una vez
                continue
            vistas.add((v.texto, tuple(v.citas)))
            motivos = [f"no existe la sección {c}" for c in v.citas if c not in secciones]
            if not motivos:
                ver = verificar_por_fuente(v.con_citas(), [], "", [fragmento(c) for c in v.citas])
                motivos = ver.motivos_guias + ver.motivos_casos
                if len(oraciones(v.texto)) != 1:
                    motivos.append("no es una sola oración")
            if v.si is not None and v.si not in CONDICIONES:
                motivos.append(f"condición desconocida: {v.si}")
            if motivos:
                salida.append((t.clave, parte, v.texto, motivos))
    return salida


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    secs = {}
    for linea in config.GUIAS_SECCIONES.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            s = json.loads(linea)
            secs[s["id"]] = s
    fallas = revisar(secs)
    for clave_t, parte, texto, motivos in fallas:
        print(f"[{clave_t} · {parte}] {texto}\n    → {' | '.join(motivos)}")
    total = len({(v.texto, tuple(v.citas)) for t in cargar().values() for _, v in t.vinetas()})
    print(f"\n{total - len(fallas)} de {total} viñetas pasan el verificador.")
