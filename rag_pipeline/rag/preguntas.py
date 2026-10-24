"""
Lo que el asistente pregunta (PLAN_CHATBOT_CLINICO.md, fase 3): reglas sobre
la ficha del paciente (rag/ficha.py), en el orden de la revisión primaria.

Un clínico no responde igual ante un paciente estable que ante uno en choque,
y los manuales tampoco. Si la ficha no lo dice, se pregunta. Primero se
responde con lo que hay y, debajo, van hasta dos preguntas con botones (decisión
1 del plan). Un clic suma el dato a la ficha como un seguimiento y responde de
nuevo; "No se sabe" no responde de nuevo: solo deja de preguntarlo
(Ficha.sin_dato).

  | Falta                    | Cuándo                                              |
  | B · ventilación          | región tórax                                        |
  | C · estado hemodinámico  | arma, herida penetrante, sangrado, o tórax, abdomen, |
  |                          | pelvis o cuello                                     |
  | D · Glasgow              | cráneo, caída o tránsito                            |
  | E · superficie, profundidad | quemadura                                        |
  | grupo                    | no se dijo la edad                                  |
  | lo que separa los casos  | si queda lugar (como build_prompt.para_afinar)      |

Nunca se pregunta lo que la ficha ya tiene ni lo que la consulta implica
(hallazgos.obvios), ni nada si no hay un paciente ("¿cuándo poner un tubo de
tórax?"). Antes de responder se pregunta solo cuando no hay con qué buscar: las
dos guardas se abstienen y la consulta nombra una lesión de trauma, pero no el
mecanismo ni la región; entonces, el mecanismo y la región.

Los textos van aquí y no en build_prompt.py ni en hallazgos.py: lo escrito en
esos dos archivos cuenta como vocabulario permitido para el verificador
(verificar._VOCABULARIO_CODIGO).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from . import config
from .consulta import NOMBRES_MECANISMO
from .ficha import HIPOTENSION, Ficha, _analizable
from .hallazgos import _LEXICO, extraer, menciones, normalizar, obvios

MAX_PREGUNTAS = 2


@dataclass
class Opcion:
    etiqueta: str
    mensaje: str | None          # lo que se suma como seguimiento; None = "No se sabe"


@dataclass
class Pregunta:
    clave: str                   # "B", "C", "D", "E-superficie", …, "afinar:<término>"
    texto: str
    opciones: list[Opcion]


NO_SE_SABE = Opcion("No se sabe", None)

_B = Pregunta("B", "B · Respiración — ¿cómo ventila?", [
    Opcion("Ventila bien", "ventilación normal, sin hipoventilación"),
    Opcion("Hipoventilación", "hipoventilación"),
    Opcion("Dificultad respiratoria", "dificultad respiratoria"),
])
_C = Pregunta("C", "C · Circulación — ¿cómo está hemodinámicamente?", [
    Opcion("Estable", "hemodinámicamente estable"),
    Opcion("Hipotenso o en choque", "hipotenso o en choque"),
    NO_SE_SABE,
])
_D = Pregunta("D", "D · Neurológico — ¿qué Glasgow tiene?", [
    Opcion("15", "Glasgow 15"),
    Opcion("13–14", "Glasgow 13-14"),
    Opcion("9–12", "Glasgow 9-12"),
    Opcion("8 o menos", "Glasgow 8 o menos"),
])
_E_SUPERFICIE = Pregunta("E-superficie", "E · Exposición — ¿qué superficie corporal está quemada?", [
    Opcion("Menos del 10 %", "superficie corporal quemada menor del 10 %"),
    Opcion("10 a 20 %", "superficie corporal quemada del 10 al 20 %"),
    Opcion("Más del 20 %", "superficie corporal quemada mayor del 20 %"),
    NO_SE_SABE,
])
_E_PROFUNDIDAD = Pregunta("E-profundidad", "E · Exposición — ¿qué profundidad tiene la quemadura?", [
    Opcion("Primer grado", "quemadura de primer grado"),
    Opcion("Segundo grado", "quemadura de segundo grado"),
    Opcion("Tercer grado", "quemadura de tercer grado"),
    NO_SE_SABE,
])
_GRUPO = Pregunta("grupo", "Paciente — ¿es niño, adulto, adulto mayor o embarazada?", [
    Opcion("Niño", "niño"), Opcion("Adulto", "adulto"),
    Opcion("Adulto mayor", "adulto mayor"), Opcion("Embarazada", "embarazada"),
])
_MECANISMO = Pregunta("mecanismo", "Si es un trauma — ¿cuál fue el mecanismo?", [
    Opcion(config.ETIQUETAS_MECANISMO[k], NOMBRES_MECANISMO[k])
    for k in ("arma_de_fuego", "cortopunzante", "accidente_transito", "caida", "contundente", "quemadura")
])
_FRASE_REGION = {"craneo_encefalo": "en cráneo"}
_REGION = Pregunta("region", "Si es un trauma — ¿en qué región?", [
    Opcion(etiqueta, _FRASE_REGION.get(k, f"en {etiqueta.lower()}"))
    for k, etiqueta in config.ETIQUETAS_REGION.items()
])

# Lo que ya responde cada pregunta, en lo que dijo el médico (Ficha.clausulas)
_RX_VENTILACION = re.compile(r"ventila|dificultad respiratoria|disnea|murmullo|respira")
_RX_HEMODINAMIA = re.compile(r"hemodinamic|choque|shock|hipotens|normotens|\bestable\b|inestab")
_RX_GLASGOW = re.compile(r"glasgow|\bgcs\b")
_RX_SUPERFICIE = re.compile(r"\bscq\b|\bsct\b|superficie corporal|\d+\s*%\s*(de\s+)?(la\s+)?(superficie|scq|sct)")
_RX_PROFUNDIDAD = re.compile(r"grado|espesor|superficial|profund")
_RX_QUEMADURA = re.compile(r"quemad")

_PENETRANTES = {"arma_de_fuego", "cortopunzante"}
_REGIONES_C = {"torax", "abdomen", "pelvis", "cuello"}


def _frase(termino: str) -> str | None:
    """Cómo decir que sí a un término para que la ficha lo lea: el término o
    una de sus variantes ("hemoperitoneo / líquido libre" → "hemoperitoneo").
    None si ninguna lo nombra: "alteración del Glasgow" (ya lo pregunta D)."""
    patron = next(p for _, t, p in _LEXICO if t == termino)
    candidatas = [termino, *termino.split(" / "), *termino.split(" o ")]
    return next((c for c in candidatas if patron.search(normalizar(c))), None)


def _afinar(ficha: Ficha, casos, consulta: str, ya: set[str]) -> list[Pregunta]:
    """Los hallazgos que separan los casos recuperados (presentes en algunos y
    no en todos) y la ficha no tiene: "¿tiene hemotórax?"."""
    n = len(casos)
    conocidos = set(ficha.afirmados) | set(ficha.negados) | obvios(_analizable(consulta)) | ya
    salida = []
    for e in extraer(casos):
        if e.categoria != "Hallazgos" or not 0 < len(e.casos) < n or e.termino in conocidos:
            continue
        frase = _frase(e.termino)
        if frase and f"afinar:{e.termino}" not in ficha.sin_dato:
            salida.append(Pregunta(f"afinar:{e.termino}",
                                   f"Los casos se diferencian ({len(e.casos)} de {n}) — ¿tiene {e.termino}?",
                                   [Opcion("Sí", f"con {frase}"), Opcion("No", f"sin {frase}"), NO_SE_SABE]))
    return salida


def preguntar(ficha: Ficha | None, casos=(), consulta: str = "", abstuvo: bool = False) -> list[Pregunta]:
    """Hasta MAX_PREGUNTAS, en el orden de la revisión primaria."""
    if ficha is None:
        return []
    if abstuvo:
        # Solo si nombra una lesión de trauma ("paciente con trauma"): medido el
        # 24/10, sin esto 17 de las 20 fuera de dominio preguntaban el
        # mecanismo ("cuál es la capital de Colombia") y la región no cambiaba nada.
        if not ficha.lesion or ficha.mecanismo or ficha.regiones:
            return []
        return [p for p in (_MECANISMO, _REGION) if p.clave not in ficha.sin_dato]
    if not ficha.es_paciente():
        return []
    dicho = _analizable(" ".join(ficha.clausulas + [consulta]))     # "choque de moto" no es shock
    conocidos = set(ficha.afirmados) | set(ficha.negados) | obvios(_analizable(consulta))
    regiones = set(ficha.regiones)
    quemadura = ficha.mecanismo == "quemadura" or bool(_RX_QUEMADURA.search(dicho))
    candidatas = []
    if "torax" in regiones and not ({"hipoventilación", "neumotórax", "hemotórax"} & conocidos
                                    or _RX_VENTILACION.search(dicho)):
        candidatas.append(_B)
    if ((ficha.mecanismo in _PENETRANTES or regiones & _REGIONES_C
         or {"herida penetrante", "sangrado activo"} & set(ficha.afirmados))
            and not (regiones and regiones <= {"ojo"})          # el trauma ocular aislado, no
            and not (ficha.estado or "TA" in ficha.signos or HIPOTENSION in conocidos
                     or _RX_HEMODINAMIA.search(dicho))):
        candidatas.append(_C)
    if (("craneo_encefalo" in regiones or ficha.mecanismo in ("caida", "accidente_transito"))
            and not ("Glasgow" in ficha.signos or _RX_GLASGOW.search(dicho))):
        candidatas.append(_D)
    if quemadura and not _RX_SUPERFICIE.search(dicho):
        candidatas.append(_E_SUPERFICIE)
    if quemadura and not _RX_PROFUNDIDAD.search(dicho):
        candidatas.append(_E_PROFUNDIDAD)
    if not ficha.grupo:
        candidatas.append(_GRUPO)
    candidatas = [p for p in candidatas if p.clave not in ficha.sin_dato]
    if len(candidatas) < MAX_PREGUNTAS and casos:
        # lo que ya pregunta una letra no se pregunta otra vez como diferencia
        ya = ({"hipoventilación"} if _B in candidatas else set()) | (
            {HIPOTENSION} if _C in candidatas else set())
        candidatas += _afinar(ficha, casos, consulta, ya)
    return candidatas[:MAX_PREGUNTAS]


def descartar(ficha: Ficha, clave: str) -> Ficha:
    """"No se sabe": una ficha que ya no lo pregunta (la de entrada no se toca)."""
    nueva = copy.deepcopy(ficha)
    if clave not in nueva.sin_dato:
        nueva.sin_dato.append(clave)
    return nueva


def ya_dicho(pregunta: Pregunta, consulta: str) -> bool:
    """¿La consulta ya respondía la pregunta? Para medir (debe ser siempre
    False): lo que la ficha no leyó pero el texto sí dice."""
    n = _analizable(consulta)
    rx = {"B": _RX_VENTILACION, "C": _RX_HEMODINAMIA, "D": _RX_GLASGOW,
          "E-superficie": _RX_SUPERFICIE, "E-profundidad": _RX_PROFUNDIDAD}.get(pregunta.clave)
    if rx is not None:
        return bool(rx.search(n))
    if pregunta.clave.startswith("afinar:"):
        afirmados, negados = menciones(n)
        return pregunta.clave.split(":", 1)[1] in afirmados + negados
    return False
