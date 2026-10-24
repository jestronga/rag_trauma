"""
La ficha del paciente (PLAN_CHATBOT_CLINICO.md, fase 2): lo que se sabe del
paciente de la conversación, armado por el código con reglas, sin el modelo.

Con CONVERSACION=llm, cada seguimiento ("¿y si está hipotenso?") pasa por
generate.reescribir: una llamada más al modelo, que puede cambiar términos, y
nadie ve qué entendió. Con CONVERSACION=ficha, cada mensaje es de un tipo:
  · caso: un número de caso (RX_CASO), como hoy; no toca la ficha;
  · nuevo: el primero; uno que dice "otro paciente"; uno que presenta a un
    paciente ("mujer de 75 años con…"); un hecho con otro mecanismo;
  · seguimiento: lo demás, que se suma a la ficha.

Qué se lee de cada mensaje:
  · mecanismo: los patrones de rag/senal.py (los de enrich.py, sin "choque")
    y unos pocos del registro clínico ("accidente de moto");
  · regiones: las de anonimizacion/src/enrich.py (REGIONES), copiadas (no se
    importa código de esa carpeta) y adaptadas: sin "glasgow" ni "pupila", que
    son del examen neurológico y no de la región, sin "radial" (el pulso) y sin
    la "cánula nasal" ni la "colisión frontal";
  · afirmados y negados: el léxico de hallazgos.py con su negación, lo mismo
    que lee los casos. Lo que se pregunta ("¿le hago FAST?") no es un hecho:
    sus estudios y su manejo no entran. Lo que se plantea ("¿y si está
    hipotenso?") sí;
  · signos con su valor (TA, FC, FR, SatO2, Glasgow), grupo y estado.

La búsqueda de un seguimiento (Ficha.consulta) son las palabras del médico,
no un texto armado con las etiquetas: las cláusulas del paciente sin lo negado
(el embedding no entiende el "sin"), sin lo que un mensaje posterior contradijo
y con el mecanismo de la ficha, más lo que pregunta el mensaje de ahora. El
primer mensaje de cada paciente se busca tal cual, como hoy.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from . import config
from .consulta import NOMBRES_MECANISMO
from .hallazgos import _LEXICO, CATEGORIA, _normalizar_alineado, extraer, menciones, normalizar
from .senal import LESIONES, PROCEDIMIENTOS
from .senal import MECANISMOS as _MECANISMOS_SENAL

# Un número de caso, solo ("caso 3377", "#3377"): el resumen de ese caso.
RX_CASO = re.compile(r"^\s*(?:caso|id|registro)?\s*#?\s*(\d{3,8})\s*$", re.IGNORECASE)

MECANISMOS = {k: list(v) for k, v in _MECANISMOS_SENAL.items()}
MECANISMOS["accidente_transito"] += [r"\bmoto\b", r"motociclist", r"\bpeaton",
                                     r"accidente\s+de\s+(carro|bus|bicicleta)"]# "choque de moto", "choque entre vehículos": la colisión, no el shock
_RX_CHOQUE_VIAL = re.compile(r"\b(choques?)(?=\s+(de|entre|contra|frontal|vehicular|con)\b)")

REGIONES = {
    "craneo_encefalo": [r"\btce\b", r"\btec\b", r"craneo", r"craneal", r"cefal", r"encefalic", r"cuero\s+cabelludo",
                        r"parietal", r"occipital", r"(?<!colision )(?<!impacto )frontal",
                        r"\btemporal(?!mente)"],
    "cara": [r"\bcara\b", r"facial", r"(?<!canula )nasal", r"piramide", r"mandibul", r"maxilar",
             r"labio", r"menton", r"orbit"],
    "ojo": [r"\bocular\b", r"\bojos?\b", r"\bod\b", r"\boi\b", r"cornea", r"conjuntiv", r"escler"],
    "cuello": [r"cuello", r"cervical", r"traquea", r"laring"],
    "torax": [r"torax", r"toracic", r"costal", r"pulmon", r"precordial", r"esternal", r"\beic\b"],
    "abdomen": [r"abdomen", r"abdominal", r"\babd\b", r"higado", r"hepatic", r"bazo", r"esplenic",
                r"intestin", r"laparotom", r"epigastric", r"flanco"],
    "pelvis": [r"pelvis", r"pelvic", r"cadera", r"isqui", r"pubi", r"iliac", r"genital", r"perine"],
    "columna": [r"columna", r"vertebr", r"lumbar", r"dorsal", r"medula"],
    "miembro_superior": [r"miembro\s+superior", r"\bmss?[di]\b", r"\bmmss\b", r"hombro", r"brazo",
                         r"codo", r"muneca", r"\bmano\b", r"dedo(?!\s+del\s+pie)", r"clavicula",
                         r"humero", r"\bradio\b", r"cubit"],
    "miembro_inferior": [r"miembro\s+inferior", r"\bmi[di]\b", r"\bmmii\b", r"muslo", r"rodilla",
                         r"pierna", r"tobillo", r"\bpie\b", r"femur", r"tibia", r"peron", r"calcaneo"],
}
_MEC = {k: [re.compile(p) for p in ps] for k, ps in MECANISMOS.items()}
_REG = {k: [re.compile(p) for p in ps] for k, ps in REGIONES.items()}
_LESION = re.compile("|".join(LESIONES))      # "TCE", "trauma", "fractura"… (rag/senal.py)
# Una pregunta general, de procedimiento ("¿cuándo está indicado colocar un tubo
# de tórax en trauma?", las proc-* de consultas_guias.yaml), no presenta a un
# paciente aunque nombre una lesión: no lleva preguntas (fase 3) ni orden xABCDE
# (fase 4).
_RX_GENERAL = re.compile(
    r"^(como|cuando|que|cual(es)?|indicaciones|criterios|tecnica|protocolo|dosis|uso|manejo"
    r"|prevencion|estabilizacion)\b|paso a paso|\bindicad[oa]s?\b|\bcriterios\b|\btecnica\b")
_PROCEDIMIENTO = re.compile("|".join(PROCEDIMIENTOS))

_VALOR = r"\s*(?:de\s+|:\s*|=\s*|en\s+|al?\s+)?"
SIGNOS = {
    "TA": re.compile(rf"\b(?:t\.?a\.?|p\.?a\.?|tension arterial|presion arterial){_VALOR}(\d{{2,3}}\s*/\s*\d{{2,3}})\b"),
    "FC": re.compile(rf"\b(?:f\.?c\.?|frecuencia cardiaca|pulso){_VALOR}(\d{{2,3}})\b"),
    "FR": re.compile(rf"\b(?:f\.?r\.?|frecuencia respiratoria){_VALOR}(\d{{1,2}})\b"),
    "SatO2": re.compile(rf"\b(?:sat\.?\s*o2|spo2|saturacion(?:\s+de\s+oxigeno)?|saturando){_VALOR}(\d{{2,3}})\s*%?"),
    # también un rango, como lo responden los botones de rag/preguntas.py
    "Glasgow": re.compile(rf"\b(?:glasgow|gcs){_VALOR}(\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}}|\s+o\s+menos)?)"
                          rf"(?:\s*/\s*15)?\b"),
}

_RX_EMBARAZO = re.compile(r"embaraz|gestante|semanas de gestacion")
_RX_ANOS = re.compile(r"(?<!hace )\b(\d{1,3})\s*anos\b")
_RX_MESES = re.compile(r"\bde\s+(\d{1,2})\s+meses\b|\b(\d{1,2})\s+meses\s+de\s+edad\b")
_RX_NINO = re.compile(r"\bnin[oa]s?\b|lactante|pediatric|\bbebes?\b|recien nacid|neonat|menor de edad"
                      r"|adolescente|\bescolar\b|preescolar")
_RX_MAYOR = re.compile(r"adult[oa]s? mayor|ancian[oa]|geriatric|tercera edad")
_RX_ADULTO = re.compile(r"\badult[oa]s?\b(?!\s+mayor)|\bjoven(es)?\b")

# la hemodinámica, no la mecánica ("fractura de pelvis con inestabilidad", proc-11)
_RX_INESTABLE = re.compile(r"(?<!pelvis )(?<!pelvica )(?<!fractura )\binestable\b|inestabilidad hemodinamica"
                           r"|\b(no|nunca)\s+(esta\s+|se encuentra\s+)?estable\b")
_RX_ESTABLE = re.compile(r"\bestable\b|normotens|hemodinamicamente normal")
HIPOTENSION = "hipotensión o choque"

# Presenta a un paciente: empieza por quién es, sin artículo definido ("el
# paciente está…" es el mismo), y sigue con su descripción.
_RX_PRESENTA = re.compile(
    r"^(?:(?:un|una)\s+)?(?:paciente|pcte|pte|hombre|mujer|varon|masculino|femenino|nin[oa]|lactante"
    r"|adolescente|ancian[oa]|adult[oa]|senora?|joven|gestante|embarazada|motociclista|peaton|conductor"
    r"|sujeto|individuo|politraumatizad[oa])\b"
    r"(?=\s*(?:de\s+\d|con\b|que\b|por\b|en\b|masculino|femenino|joven|adult|ancian|mayor|\d|,|$))")
_RX_DICE_NUEVO = re.compile(r"\b(otro|nuevo|siguiente)\s+(paciente|caso)\b|\b(paciente|caso)\s+nuevo\b")
_RX_PREGUNTA = re.compile(r"^(que|cual(es)?|cuando|como|cuant[oa]s?|donde|por que|para que)\b")
# sobre el texto normalizado; el "si" condicional, aparte y con tildes: "sí" no lo es
_RX_HIPOTESIS = re.compile(r"^(y|e)\s+(en|con|por|sin|si|para|cuando|ante|de)\b|que pasa si\b|en caso de")
_RX_SI = re.compile(r"\bsi\b")
# los dos puntos cortan "otro paciente: …", no "TA: 85/55"
_RX_CLAUSULA = re.compile(r"\s*,\s*|\s*:(?!\s*\d)\s*|\s+(?:y|e|pero|aunque)\s+", re.IGNORECASE)
_RX_CONECTOR = re.compile(r"^(?:(?:y|e|pero|adem[aá]s|ahora|entonces)\s+)?(?:(?:si|qu[eé] pasa si|en caso de que?)\s+"
                          r"(?:(?:fuera|fuese|est[aá]|es|tiene|presenta|hay|llega)\b)?\s*)?", re.IGNORECASE)
_RX_NEGACION = re.compile(r"\b(sin|no|niega|descart\w*|ausencia de|ni)\b")
_MAX_CLAUSULAS = 12


@dataclass
class Lectura:
    """Lo que dice un mensaje, antes de sumarlo a la ficha."""
    mecanismo: str | None = None
    mecanismo_es_hecho: bool = False
    regiones: list[str] = field(default_factory=list)
    afirmados: list[str] = field(default_factory=list)
    negados: list[str] = field(default_factory=list)
    signos: dict[str, str] = field(default_factory=dict)
    grupo: str | None = None
    estado: str | None = None
    hechos: list[str] = field(default_factory=list)       # cláusulas de hechos e hipótesis
    preguntas: list[str] = field(default_factory=list)    # cláusulas de lo que se pregunta
    presenta: bool = False
    dice_nuevo: bool = False
    lesion: bool = False                                   # nombra una lesión de trauma
    general: bool = False                                  # una pregunta de procedimiento


@dataclass
class Ficha:
    mecanismo: str | None = None
    regiones: list[str] = field(default_factory=list)
    afirmados: list[str] = field(default_factory=list)
    negados: list[str] = field(default_factory=list)
    signos: dict[str, str] = field(default_factory=dict)
    grupo: str | None = None
    estado: str | None = None
    estado_por_ta: bool = False       # el estado salió de la TA: otra TA lo cambia
    clausulas: list[str] = field(default_factory=list)   # las palabras del médico, para buscar
    sin_dato: list[str] = field(default_factory=list)    # "No se sabe": no se vuelve a preguntar (preguntas.py)
    lesion: bool = False                                   # nombró una lesión ("TCE severo")
    general: bool = False                                  # empezó con una pregunta de procedimiento

    def es_paciente(self) -> bool:
        """¿Hay un paciente? Las regiones solas no bastan: "¿cuándo poner un
        tubo de tórax?" deja tórax y no presenta a nadie; "TCE severo", sí. Una
        pregunta de procedimiento ("manejo del neumotórax a tensión") tampoco,
        salvo que traiga mecanismo, signos, grupo o estado."""
        if self.mecanismo or self.signos or self.grupo or self.estado:
            return True
        hallazgos = [t for t in self.afirmados + self.negados if CATEGORIA[t] == "Hallazgos"]
        return not self.general and bool(hallazgos or self.lesion)

    def datos(self) -> dict:
        """Lo que se compara con eval/conversaciones.yaml."""
        return {"mecanismo": self.mecanismo, "regiones": sorted(self.regiones),
                "afirmados": sorted(self.afirmados), "negados": sorted(self.negados),
                "signos": dict(self.signos), "grupo": self.grupo, "estado": self.estado}

    def partes(self) -> list[str]:
        """Los datos de la ficha, para la línea "Paciente: …" y la de qué cambió."""
        salida = []
        if self.mecanismo:
            salida.append(config.ETIQUETAS_MECANISMO.get(self.mecanismo, self.mecanismo).lower())
        salida += [config.ETIQUETAS_REGION.get(r, r).lower() for r in self.regiones]
        salida += self.afirmados + [f"sin {t}" for t in self.negados]
        salida += [describir_signo(k, v) for k, v in self.signos.items()]
        salida += [x for x in (self.grupo, self.estado) if x]
        return salida

    def linea(self) -> str:
        return "Paciente: " + " · ".join(self.partes()) if self.es_paciente() else ""

    def consulta(self, mensaje: str) -> str:
        """La búsqueda de un seguimiento: lo que se sabe del paciente, en las
        palabras del médico, más lo que pregunta `mensaje` (ya sumado a la ficha)."""
        partes = [p for p in (self._para_buscar(c) for c in self.clausulas) if p]
        if len(partes) > _MAX_CLAUSULAS:              # la presentación y lo último
            partes = partes[:4] + partes[-(_MAX_CLAUSULAS - 4):]
        partes += leer(mensaje).preguntas
        vistas, salida = set(), []
        for p in partes:
            if normalizar(p) not in vistas:
                vistas.add(normalizar(p))
                salida.append(p)
        return ", ".join(salida)

    def _para_buscar(self, clausula: str) -> str:
        """La cláusula sin lo negado ni lo que un mensaje posterior negó, y con
        el mecanismo de la ficha si nombraba otro."""
        afirmados, negados = menciones(_analizable(clausula))
        quitar = negados + [t for t in afirmados if t in self.negados]
        if quitar:
            clausula = _sin(clausula, quitar)
        if self.mecanismo:
            clausula = _con_mecanismo(clausula, self.mecanismo)
        return clausula.strip(" ,;")


def describir_signo(signo: str, valor: str) -> str:
    return f"{signo} {valor}{' %' if signo == 'SatO2' else ''}"


def _analizable(texto: str) -> str:
    """El texto en que se buscan los patrones: sin tildes, en minúsculas y con
    la colisión ("choque de moto") separada del shock."""
    return _RX_CHOQUE_VIAL.sub("colision", normalizar(texto))


def _alineado(texto: str) -> str:
    """Como _analizable, pero del mismo largo que `texto`, para cortar o
    reemplazar en el original lo encontrado: la colisión se tapa."""
    return _RX_CHOQUE_VIAL.sub(lambda m: "_" * len(m.group(0)), _normalizar_alineado(texto))


_RX_COLA = re.compile(r"(\s+(con|y|e|de|del|en|por|sin|ni|o|ya))+\s*$", re.IGNORECASE)


def _sin(clausula: str, terminos: list[str]) -> str:
    """Corta la cláusula donde empieza lo que no se busca: lo negado, desde su
    negación ("HPACP en zona II de cuello sin sangrado activo" → "HPACP en zona
    II de cuello"), y lo que un mensaje posterior negó, desde el término ("HPAF
    en muslo con sangrado activo" → "HPAF en muslo"). Un resto sin palabras de 4
    letras ("ya") no se busca."""
    alineado = _alineado(clausula)
    corte = len(clausula)
    for _, term, patron in _LEXICO:
        if term in terminos and (m := patron.search(alineado)):
            ini = max(0, m.start() - 40)
            claves = list(_RX_NEGACION.finditer(alineado[ini:m.start()]))
            corte = min(corte, (ini + claves[-1].start()) if claves else m.start())
    resto = _RX_COLA.sub("", clausula[:corte].rstrip(" ,;"))
    return resto if re.search(r"\w{4,}", resto) else ""


def _con_mecanismo(clausula: str, mecanismo: str) -> str:
    """Si la cláusula nombra otro mecanismo y no el de la ficha (lo cambió una
    hipótesis: "¿y si fuera por arma cortopunzante?"), pone el de la ficha en su
    lugar. "Agresión física con arma cortopunzante" queda igual."""
    alineado = _alineado(clausula)
    if any(p.search(alineado) for p in _MEC[mecanismo]):
        return clausula
    tramos = sorted((m.start(), m.end()) for k, ps in _MEC.items() if k != mecanismo
                    for p in ps for m in p.finditer(alineado))
    nombre = NOMBRES_MECANISMO[mecanismo]
    salida, pos = [], 0
    for ini, fin in tramos:
        if ini < pos:
            continue
        while fin < len(alineado) and alineado[fin].isalnum():       # la palabra entera
            fin += 1
        previo = alineado[:ini].rstrip()
        corto = nombre.removeprefix("herida por ") if previo.endswith("herida por") else nombre
        salida += [clausula[pos:ini], corto]
        pos = fin
    return "".join(salida) + clausula[pos:]


def _oraciones(mensaje: str) -> list[tuple[str, str]]:
    """[(oración, clase)] con clase "hecho", "pregunta" o "hipotesis"."""
    salida = []
    for o in re.split(r"(?<=[.;!?])\s+|\s*(?=¿)", mensaje.strip()):
        texto = o.strip(" ¿?¡!.;")
        if not texto:
            continue
        n = normalizar(texto)
        if _RX_HIPOTESIS.search(n) or _RX_SI.search(texto.lower()):
            clase = "hipotesis"
        elif "?" in o or "¿" in o or _RX_PREGUNTA.search(n):
            clase = "pregunta"
        else:
            clase = "hecho"
        salida.append((texto, clase))
    return salida


def _primero(patrones: dict[str, list[re.Pattern]], texto: str) -> list[str]:
    return [k for k, ps in patrones.items() if any(p.search(texto) for p in ps)]


def _grupo(n: str) -> str | None:
    if _RX_EMBARAZO.search(n):
        return "embarazada"
    if m := _RX_ANOS.search(n):
        edad = int(m.group(1))
        return "niño" if edad < 18 else "adulto mayor" if edad >= 65 else "adulto"
    if _RX_MESES.search(n) or _RX_NINO.search(n):
        return "niño"
    if _RX_MAYOR.search(n):
        return "adulto mayor"
    if _RX_ADULTO.search(n):
        return "adulto"
    return None


def leer(mensaje: str) -> Lectura:
    """Lo que dice un mensaje, por reglas."""
    lec = Lectura()
    oraciones = _oraciones(mensaje)
    if oraciones:
        primera, clase = oraciones[0]
        lec.presenta = clase == "hecho" and bool(_RX_PRESENTA.search(normalizar(primera)))
    lec.dice_nuevo = bool(_RX_DICE_NUEVO.search(normalizar(mensaje)))
    texto = normalizar(mensaje.strip(" ¿?¡!"))
    lec.general = bool(_RX_GENERAL.search(texto) or _PROCEDIMIENTO.search(texto))
    explicito = None
    for oracion, clase in oraciones:
        for c in (c for c in _RX_CLAUSULA.split(oracion) if c and c.strip()):
            c = _RX_CONECTOR.sub("", c.strip()).strip()
            n = _analizable(c)
            if not c or _RX_DICE_NUEVO.fullmatch(n):        # "otro paciente" no se busca
                continue
            (lec.preguntas if clase == "pregunta" else lec.hechos).append(c)
            for k in _primero(_MEC, n)[:1]:
                if lec.mecanismo is None or (clase == "hecho" and not lec.mecanismo_es_hecho):
                    lec.mecanismo, lec.mecanismo_es_hecho = k, clase == "hecho"
            lec.regiones += [r for r in _primero(_REG, n) if r not in lec.regiones]
            lec.lesion = lec.lesion or bool(_LESION.search(n))
            afirmados, negados = menciones(n)
            if clase == "pregunta":           # "¿le hago FAST?": no es un hecho
                afirmados = [t for t in afirmados if CATEGORIA[t] == "Hallazgos"]
                negados = [t for t in negados if CATEGORIA[t] == "Hallazgos"]
            lec.afirmados += [t for t in afirmados if t not in lec.afirmados]
            lec.negados += [t for t in negados if t not in lec.negados and t not in lec.afirmados]
            for signo, rx in SIGNOS.items():
                if m := rx.search(n):
                    valor = re.sub(r"\s*([/–-])\s*", r"\1", m.group(1)).replace("–", "-")
                    if signo != "Glasgow" or 3 <= int(re.match(r"\d+", valor).group()) <= 15:
                        lec.signos[signo] = valor
            lec.grupo = _grupo(n) or lec.grupo
            if _RX_INESTABLE.search(n) or HIPOTENSION in afirmados:
                explicito = "inestable"
            elif _RX_ESTABLE.search(n) or HIPOTENSION in negados:
                explicito = "estable"
    lec.estado = explicito
    return lec


def _sumar(f: Ficha, lec: Lectura, presentacion: bool) -> None:
    if lec.mecanismo:
        f.mecanismo = lec.mecanismo
    f.regiones += [r for r in lec.regiones if r not in f.regiones]
    for t in lec.afirmados:
        if t in f.negados:
            f.negados.remove(t)
        if t not in f.afirmados:
            f.afirmados.append(t)
    for t in lec.negados:
        if t in f.afirmados:
            f.afirmados.remove(t)
        if t not in f.negados:
            f.negados.append(t)
    f.signos.update(lec.signos)
    f.grupo = lec.grupo or f.grupo
    f.lesion = f.lesion or lec.lesion
    if lec.estado:
        f.estado, f.estado_por_ta = lec.estado, False
    elif "TA" in lec.signos:
        if int(lec.signos["TA"].split("/")[0]) < 90:
            f.estado, f.estado_por_ta = "inestable", True
        elif f.estado_por_ta:
            f.estado, f.estado_por_ta = None, False
    # la presentación entra entera, con lo que pregunta; los seguimientos, sus hechos
    f.clausulas += (lec.hechos + lec.preguntas) if presentacion else lec.hechos
    if presentacion:
        f.general = lec.general


def tipo_de_mensaje(ficha: Ficha | None, mensaje: str, lec: Lectura | None = None) -> str:
    if RX_CASO.match(mensaje):
        return "caso"
    lec = lec or leer(mensaje)
    if ficha is None or not ficha.es_paciente() or lec.dice_nuevo or lec.presenta:
        return "nuevo"
    if lec.mecanismo_es_hecho and ficha.mecanismo and lec.mecanismo != ficha.mecanismo:
        return "nuevo"
    return "seguimiento"


def actualizar(ficha: Ficha | None, mensaje: str, forzar: str | None = None) -> tuple[str, Ficha]:
    """(tipo, ficha después del mensaje). La ficha de entrada no se toca.
    `forzar` ("nuevo" o "seguimiento") corrige al código: el botón de la app."""
    lec = leer(mensaje)
    tipo = forzar or tipo_de_mensaje(ficha, mensaje, lec)
    if tipo == "caso":
        return tipo, copy.deepcopy(ficha) if ficha else Ficha()
    nueva = Ficha() if tipo == "nuevo" or ficha is None else copy.deepcopy(ficha)
    _sumar(nueva, lec, tipo == "nuevo")
    return tipo, nueva


def unir(partes: list[str]) -> str:
    """"a, b y c"; "y" pasa a "e" antes de i- o hi- ("hipotensión e inestable")."""
    if len(partes) < 2:
        return "".join(partes)
    ultima = normalizar(partes[-1])
    y = "e" if re.match(r"h?i(?![aeou])", ultima) else "y"
    return f"{', '.join(partes[:-1])} {y} {partes[-1]}"


def novedades(antes: Ficha | None, despues: Ficha) -> list[str]:
    """Lo que el mensaje sumó a la ficha, en el orden de Ficha.partes."""
    previas = set(antes.partes()) if antes else set()
    return [p for p in despues.partes() if p not in previas]


def que_cambio(anterior, actual, ficha_antes: Ficha | None, ficha_despues: Ficha | None) -> str | None:
    """Una línea armada por el código que compara la respuesta con la anterior:
    "Con hipotensión o choque: entran 3 casos nuevos; aparece transfusión (2 de
    5)". Sale de los IDs y de hallazgos.extraer: no la escribe el modelo.
    `anterior` y `actual` son generate.Respuesta."""
    if anterior is None or actual.abstuvo or (not anterior.casos and not anterior.guias):
        return None
    lo_nuevo = novedades(ficha_antes, ficha_despues) if ficha_despues else []
    frases = []
    ids_antes = {c.record_id for c in anterior.casos}
    n_nuevos = sum(c.record_id not in ids_antes for c in actual.casos)
    if actual.casos:
        frases.append(f"entra{'n' if n_nuevos != 1 else ''} {n_nuevos} caso{'s' if n_nuevos != 1 else ''} "
                      f"nuevo{'s' if n_nuevos != 1 else ''}" if n_nuevos else "mismos casos")
        terminos_antes = {e.termino for e in extraer(anterior.casos)}
        aparecen = [e for e in extraer(actual.casos) if e.termino not in terminos_antes][:3]
        if aparecen:
            frases.append("aparece " + unir([f"{e.termino} ({len(e.casos)} de {len(actual.casos)})"
                                             for e in aparecen]))
    secciones_antes = {f.seccion_id for f in anterior.guias}
    n_pasajes = len({f.seccion_id for f in actual.guias} - secciones_antes)
    if actual.guias:
        frases.append(f"{n_pasajes} pasaje{'s' if n_pasajes != 1 else ''} nuevo{'s' if n_pasajes != 1 else ''} "
                      f"de los manuales" if n_pasajes else "mismos pasajes de los manuales")
    if not frases:
        return None
    inicio = f"Con {unir(lo_nuevo)}" if lo_nuevo else "Respecto de la respuesta anterior"
    return f"{inicio}: {'; '.join(frases)}."
