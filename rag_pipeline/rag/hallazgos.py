"""
Extracción determinista de lo documentado en los casos recuperados.

Un modelo de 3B, puesto a "sugerir", inventa: "estudios de imagen (4 de 5)"
cuando ningún caso menciona imágenes. Aquí se extrae con reglas qué hallazgos,
estudios y manejo constan en cada caso, con sus identificadores, y el prompt
le pide al modelo que sugiera SOLO a partir de esta lista. El modelo redacta;
el respaldo lo pone el código.

Negación: en el registro, "sangrado activo" aparece casi siempre como "SIN
sangrado activo". Una mención precedida de sin / no / niega / descarta… no
cuenta (un NegEx mínimo sobre una ventana corta a la izquierda). Tampoco una
seguida de "negativo" o "(-)": "Blumberg negativo".

Por sección: estudios y manejo se buscan solo en el examen físico y la
evolución. El motivo de consulta cuenta lo que pasó ANTES de llegar: "remitido
por HPAF" (645 casos) se contaba como remisión hecha aquí.

El léxico se armó contando frecuencias sobre el corpus apto_rag; cada patrón
aparece en al menos ~15 documentos. Cambios revisados contra el corpus
(26/09/2026): "cirugía" contaba las interconsultas ("valorado por cirugía
plástica", "por neurocirugía") y ahora exige que se haya operado o indicado
operar; "laparatomía" mal escrita no se detectaba; Blumberg no contaba como
irritación peritoneal.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from .retrieve import Resultado

# (categoría, término mostrado, patrón sobre texto normalizado: minúsculas y sin tildes)
LEXICO: list[tuple[str, str, str]] = [
    # --- hallazgos ---
    ("Hallazgos", "hipoventilación", r"hipoventila|murmullo vesicular disminuido|\bmv disminuido"),
    ("Hallazgos", "hemotórax", r"hemotorax"),
    ("Hallazgos", "neumotórax", r"neumotorax"),
    ("Hallazgos", "herida penetrante", r"(?<!no )penetrante"),
    ("Hallazgos", "sangrado activo", r"sangrado activo|hemorragia"),
    ("Hallazgos", "hipotensión o choque", r"hipotens|choque|shock"),
    ("Hallazgos", "fractura", r"fractura|\bfx\b"),
    ("Hallazgos", "déficit neurológico", r"deficit neuro|focalizacion|focalizado"),
    ("Hallazgos", "compromiso vascular", r"pulsos? (distales )?(ausentes|disminuid)|lesion vascular|isquemi"),
    ("Hallazgos", "lesión tendinosa o nerviosa", r"tendin|tendon|lesion nerviosa|nervio"),
    ("Hallazgos", "irritación peritoneal",
     r"irritacion peritoneal|abdomen agu?do|defensa abdominal|bl[uo]mberg?\w*|signo de rebote"),
    ("Hallazgos", "hemoperitoneo / líquido libre",
     r"hemoperitoneo|liquidos? libres?|liquidos? en (la )?cavidad"),
    ("Hallazgos", "alteración del Glasgow", r"glasgow (de )?([3-9]|1[0-4])\b"),
    # TCE: sin estos, una consulta de trauma craneoencefálico salía sin hallazgos
    ("Hallazgos", "pérdida de conciencia", r"perdida (de |del )?(la )?(conciencia|conocimiento)|\bpdc\b"),
    ("Hallazgos", "amnesia", r"amnesia"),
    ("Hallazgos", "emesis", r"vomit\w*|emesis"),
    ("Hallazgos", "cefalea", r"cefalea"),
    ("Hallazgos", "otorragia / otoliquia", r"otorragia|otoliquia|rinoliquia"),
    ("Hallazgos", "convulsión", r"convulsion(o|es)?\b|convulsivo"),
    ("Hallazgos", "anisocoria", r"anisocoria"),
    # --- estudios ---
    ("Estudios", "radiografía", r"radiograf|\brx\b"),
    ("Estudios", "TAC", r"\btac\b|tomografia|\bscanner\b"),
    ("Estudios", "ecografía / FAST", r"ecograf|\bfast\b"),
    ("Estudios", "angiografía / doppler", r"angiograf|angiotac|doppler"),
    # --- manejo ---
    ("Manejo", "tubo de tórax", r"tubo (de|a|en) torax|toracostomia|pleurostomia|sonda de torax"),
    ("Manejo", "toracotomía", r"toracotomia"),
    ("Manejo", "laparotomía / laparoscopia", r"lapar[ao]tomia|laparoscopi"),
    # Operado o con indicación de operar. "cirugía" a secas no sirve: la
    # mayoría de menciones son interconsultas ("valorado por cx plástica").
    ("Manejo", "cirugía",
     r"(llevad[oa]|lleva|pasa|pasad[oa]|ingresa|sube) a (sala de )?(cirugia|cx|qx|quirofano)"
     r"|sala de operaciones|procedimientos? (qx|quiru?rgicos?)"
     r"|hallazgos (qx|operatorios|quiru?rgicos|intraoperatorios)"
     r"|intraoperatori|intervencion quirurgica|intervenid[oa] quirurgica"
     r"|(se )?(realiza|realizo|practica|programa) (una )?(cirugia|cx)\b|cirugia sin complicaciones"
     r"|p(os|ost) ?(operatori|quirurgic)|manejo (qx|quirurgico)|conducta quirurgica"
     r"|requiere (de )?(cx|qx|cirugia)|programa\w* (cx|cirugia)"
     r"|lapar[ao]tomia|laparoscopi|toracotomia|osteosintesis|reduccion abierta"),
    # "-rafia" = reparación (arteriorrafia, blefarorrafia), pero no "-grafía":
    # radiografía, tomografía y ecografía contaban como sutura en 1.209 de
    # 2.426 casos (medido el 28/09/2026 sobre los 8.198 aptos).
    ("Manejo", "sutura", r"sutur|(?<!g)rafia\b"),
    ("Manejo", "lavado / desbridamiento", r"lavado|desbrid"),
    # quemaduras: sin estos, una consulta de quemadura salía sin manejo
    ("Manejo", "curación", r"curacion\w*|nitrofurazona|sulfadiazina|silvadene"),
    ("Manejo", "escarectomía / injerto", r"escarectomia|escarotomia|injerto"),
    ("Manejo", "líquidos endovenosos", r"\blev\b|liquidos endovenosos|reanimacion hidrica|parkland"),
    ("Manejo", "osteosíntesis / reducción", r"osteosintesis|reduccion abierta|reduccion cerrada|fijador"),
    ("Manejo", "inmovilización / férula", r"ferula|inmoviliz|\byeso\b|cabestrillo|valva"),
    ("Manejo", "antibiótico", r"antibiotic|cefalotina|cefazolina|clindamicina|ampicilina|dicloxacilina"),
    ("Manejo", "analgesia", r"analges|dipirona|acetaminofen|tramadol|morfina|diclofenaco"),
    ("Manejo", "profilaxis antitetánica", r"antitetan|toxoide|tetanol"),
    ("Manejo", "transfusión", r"transfu|globulos rojos|hemoderivad"),
    ("Manejo", "intubación / ventilación mecánica", r"intubac|ventilacion mecanica|\biot\b"),
    ("Manejo", "UCI", r"\buci\b|cuidados intensivos"),
    # Solo la remisión HACIA otro nivel. "remitido por/de…" es cómo llegó.
    ("Manejo", "remisión", r"\bse (le )?remit\w*|\bremitir\w*|\bremitio\b|(?<!por )\bremision\b"),
    ("Manejo", "observación", r"observacion"),
]
_LEXICO = [(cat, term, re.compile(rx)) for cat, term, rx in LEXICO]
# Procedimientos que "cirugía" ya incluye: si consta el concreto, "cirugía" sobra.
PROCEDIMIENTOS_QUIRURGICOS = ("toracotomía", "laparotomía / laparoscopia", "osteosíntesis / reducción")
CATEGORIAS = ("Hallazgos", "Estudios", "Manejo")

_RX_NEGACION = re.compile(r"\b(sin|no|niega|negativ[oa]s?|descart\w*|ausencia de|ni)\b[^.;\n]{0,25}$")
# A la derecha solo marcadores inequívocos: "no" a la derecha falla con
# "fractura no desplazada".
_RX_NEGACION_DER = re.compile(r"^\s*:?\s*(\(\s*-\s*\)|-\s*(?=[,.;\n]|$)|negativ\w*|neg\b|ausentes?\b)")

# Secciones del texto indexado (las escribe el pipeline de anonimización).
MOTIVO, EXAMEN, EVOLUCION = "Motivo de consulta", "Examen físico", "Evolución y hospitalización"
_RX_SECCION = re.compile(
    rf"({MOTIVO}|{EXAMEN}|{EVOLUCION}):\s*(.*?)(?=\n\n(?:{MOTIVO}|{EXAMEN}|{EVOLUCION}):|\Z)",
    re.DOTALL,
)


def secciones(texto: str) -> dict[str, str]:
    """{sección: contenido en una línea}. Vacío si el texto no trae secciones (una consulta)."""
    return {m.group(1): " ".join(m.group(2).split()) for m in _RX_SECCION.finditer(texto)}


def normalizar(texto: str) -> str:
    return unicodedata.normalize("NFD", texto.lower()).encode("ascii", "ignore").decode()


def _afirmado(texto: str, patron: re.Pattern) -> bool:
    """Al menos una mención que no esté negada."""
    for m in patron.finditer(texto):
        if _RX_NEGACION.search(texto[max(0, m.start() - 40):m.start()]):
            continue
        if _RX_NEGACION_DER.search(texto[m.end():m.end() + 15]):
            continue
        return True
    return False


@dataclass
class Elemento:
    categoria: str
    termino: str
    casos: list[str]


def terminos(texto: str) -> dict[str, list[str]]:
    """Términos del léxico afirmados en un texto, por categoría.

    Hallazgos: en todo el texto. Estudios y manejo: sin el motivo de consulta.
    Si el texto no trae secciones (una consulta del usuario), se usa entero.
    """
    completo = normalizar(texto)
    secs = secciones(texto)
    atencion = normalizar(" \n ".join(v for k, v in secs.items() if k != MOTIVO)) if secs else completo
    salida: dict[str, list[str]] = {cat: [] for cat in CATEGORIAS}
    for cat, term, patron in _LEXICO:
        if _afirmado(completo if cat == "Hallazgos" else atencion, patron):
            salida[cat].append(term)
    return salida


# "herida penetrante" no le dice nada a quien consulta por una herida por arma.
_RX_HERIDA = re.compile(r"herida|\bh(p)?a(f|cp|c)\b|arma|pu[nñ]al|disparo|tiro")


def obvios(consulta: str) -> set[str]:
    """Hallazgos que la consulta ya dice o implica: mostrarlos como sugerencia
    o como diferencia entre casos es ruido."""
    if not consulta:
        return set()
    salida = set(terminos(consulta)["Hallazgos"])
    if _RX_HERIDA.search(normalizar(consulta)):
        salida.add("herida penetrante")
    return salida


def extraer(casos: list[Resultado], max_por_categoria: int = 6) -> list[Elemento]:
    """Elementos documentados en los casos, del más al menos frecuente."""
    por_caso = {c.record_id: terminos(c.text) for c in casos}
    elementos = []
    for cat, term, _ in _LEXICO:
        ids = [rid for rid, ts in por_caso.items() if term in ts[cat]]
        if ids:
            elementos.append(Elemento(cat, term, ids))
    salida = []
    for cat in CATEGORIAS:
        de_cat = sorted((e for e in elementos if e.categoria == cat), key=lambda e: -len(e.casos))
        salida.extend(de_cat[:max_por_categoria])
    return salida


def elegibles(casos: list[Resultado], consulta: str = "") -> list[Elemento]:
    """Lo que se puede sugerir: los elementos de los casos, menos el ruido.

      · un hallazgo que la consulta ya dice o implica ("valorar hipoventilación"
        a quien preguntó por hipoventilación);
      · "cirugía" en los mismos casos que un procedimiento concreto: sería la
        misma sugerencia dos veces.

    Es la lista que ve el modelo y el `enum` de su salida estructurada: lo que
    no está aquí, el modelo no lo puede elegir.
    """
    ya_dicho = obvios(consulta)
    todos = extraer(casos)
    concretos = [set(e.casos) for e in todos if e.termino in PROCEDIMIENTOS_QUIRURGICOS]
    return [e for e in todos
            if not (e.categoria == "Hallazgos" and e.termino in ya_dicho)
            and not (e.termino == "cirugía" and set(e.casos) in concretos)]


def _normalizar_alineado(texto: str) -> str:
    """Como normalizar(), pero carácter a carácter: misma longitud que el
    original, para marcar en el texto original lo encontrado en el normalizado."""
    return "".join((normalizar(ch) or " ")[:1] for ch in texto)


_VACIAS = {"para", "como", "caso", "casos", "desde", "sobre", "entre", "hacia", "segun", "tiene",
           "paciente", "pacientes", "presenta", "manejo", "consulta", "con", "sin", "por", "del"}


def _patrones_consulta(consulta: str) -> list[re.Pattern]:
    """Qué buscar de la consulta en un caso: los términos del léxico que menciona
    (con todas sus variantes: "HPAF" y "herida por arma de fuego") y sus
    palabras por raíz ("abdomen" encuentra "abdominal")."""
    buscados = {t for ts in terminos(consulta).values() for t in ts}
    patrones = [patron for _, term, patron in _LEXICO if term in buscados]
    for palabra in set(re.findall(r"[a-z]{4,}", normalizar(consulta))) - _VACIAS:
        raiz = palabra[:max(5, len(palabra) - 3)]
        patrones.append(re.compile(rf"\b{raiz}" if len(palabra) > 4 else rf"\b{palabra}\b"))
    return patrones


def resaltar_html(texto: str, consulta: str) -> str:
    """Texto escapado para HTML con <mark> en lo que coincide con la consulta:
    por qué este caso se parece a lo que se preguntó."""
    alineado = _normalizar_alineado(texto)
    tramos = sorted((m.start(), m.end()) for patron in (_patrones_consulta(consulta) if consulta else [])
                    for m in patron.finditer(alineado))
    salida, pos = [], 0
    for ini, fin in tramos:
        if ini < pos:            # solapado con el anterior
            continue
        while fin < len(alineado) and alineado[fin].isalnum():   # "hipoventila" → la palabra entera
            fin += 1
        salida += [html.escape(texto[pos:ini]), f"<mark>{html.escape(texto[ini:fin])}</mark>"]
        pos = fin
    salida.append(html.escape(texto[pos:]))
    return "".join(salida)


def elemento_mencionado(linea: str, elementos: list[Elemento]) -> Elemento | None:
    """El elemento de la lista al que se refiere una línea escrita por el modelo.

    Compara sin tildes y acepta cualquiera de las variantes del término
    ("laparotomía / laparoscopia" casa con "laparoscopia exploratoria").
    Si hay varios, el de término más largo: "tubo de tórax" antes que "tórax".
    """
    t = normalizar(linea)
    candidatos = []
    for e in elementos:
        variantes = [normalizar(v).strip() for v in e.termino.split("/")]
        if any(v and v in t for v in variantes):
            candidatos.append(e)
    return max(candidatos, key=lambda e: len(e.termino), default=None)


def formatear(elementos: list[Elemento], n_casos: int) -> str:
    if not elementos:
        return "(ningún hallazgo, estudio ni manejo reconocible en los casos)"
    lineas = []
    for cat in CATEGORIAS:
        de_cat = [e for e in elementos if e.categoria == cat]
        if not de_cat:
            continue
        lineas.append(f"{cat}:")
        for e in de_cat:
            citas = " ".join(f"[caso: {rid}]" for rid in e.casos)
            lineas.append(f"- {e.termino}: {len(e.casos)} de {n_casos} casos {citas}")
    return "\n".join(lineas)
