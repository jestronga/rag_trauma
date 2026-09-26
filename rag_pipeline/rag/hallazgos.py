"""
Extracción determinista de lo documentado en los casos recuperados.

Un modelo de 3B, puesto a "sugerir", inventa: "estudios de imagen (4 de 5)"
cuando ningún caso menciona imágenes. Aquí se extrae con reglas qué hallazgos,
estudios y manejo constan en cada caso, con sus identificadores, y el prompt
le pide al modelo que sugiera SOLO a partir de esta lista. El modelo redacta;
el respaldo lo pone el código.

Negación: en el registro, "sangrado activo" aparece casi siempre como "SIN
sangrado activo". Una mención precedida de sin / no / niega / descarta… no
cuenta (un NegEx mínimo sobre una ventana corta a la izquierda).

El léxico se armó contando frecuencias sobre el corpus apto_rag; cada patrón
aparece en al menos ~15 documentos.
"""
from __future__ import annotations

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
    ("Hallazgos", "irritación peritoneal", r"irritacion peritoneal|abdomen agudo|defensa abdominal"),
    ("Hallazgos", "alteración del Glasgow", r"glasgow (de )?([3-9]|1[0-4])\b"),
    # --- estudios ---
    ("Estudios", "radiografía", r"radiograf|\brx\b"),
    ("Estudios", "TAC", r"\btac\b|tomografia|\bscanner\b"),
    ("Estudios", "ecografía / FAST", r"ecograf|\bfast\b"),
    ("Estudios", "angiografía / doppler", r"angiograf|angiotac|doppler"),
    # --- manejo ---
    ("Manejo", "tubo de tórax", r"tubo (de|a|en) torax|toracostomia|pleurostomia|sonda de torax"),
    ("Manejo", "toracotomía", r"toracotomia"),
    ("Manejo", "laparotomía / laparoscopia", r"laparotomia|laparoscopi"),
    ("Manejo", "cirugía", r"cirugia|quirurgic|\bqx\b|\bcx\b"),
    ("Manejo", "sutura", r"sutur|rafia\b"),
    ("Manejo", "lavado / desbridamiento", r"lavado|desbrid"),
    ("Manejo", "osteosíntesis / reducción", r"osteosintesis|reduccion abierta|reduccion cerrada|fijador"),
    ("Manejo", "inmovilización / férula", r"ferula|inmoviliz|\byeso\b|cabestrillo|valva"),
    ("Manejo", "antibiótico", r"antibiotic|cefalotina|cefazolina|clindamicina|ampicilina|dicloxacilina"),
    ("Manejo", "analgesia", r"analges|dipirona|acetaminofen|tramadol|morfina|diclofenaco"),
    ("Manejo", "profilaxis antitetánica", r"antitetan|toxoide|tetanol"),
    ("Manejo", "transfusión", r"transfu|globulos rojos|hemoderivad"),
    ("Manejo", "intubación / ventilación mecánica", r"intubac|ventilacion mecanica|\biot\b"),
    ("Manejo", "UCI", r"\buci\b|cuidados intensivos"),
    ("Manejo", "remisión", r"\bremit|remision"),
    ("Manejo", "observación", r"observacion"),
]
_LEXICO = [(cat, term, re.compile(rx)) for cat, term, rx in LEXICO]
CATEGORIAS = ("Hallazgos", "Estudios", "Manejo")

_RX_NEGACION = re.compile(r"\b(sin|no|niega|negativ[oa]s?|descart\w*|ausencia de|ni)\b[^.;\n]{0,25}$")


def normalizar(texto: str) -> str:
    return unicodedata.normalize("NFD", texto.lower()).encode("ascii", "ignore").decode()


def _afirmado(texto: str, patron: re.Pattern) -> bool:
    """Al menos una mención que no esté negada."""
    for m in patron.finditer(texto):
        if not _RX_NEGACION.search(texto[max(0, m.start() - 40):m.start()]):
            return True
    return False


@dataclass
class Elemento:
    categoria: str
    termino: str
    casos: list[str]


def terminos(texto: str) -> dict[str, list[str]]:
    """Términos del léxico afirmados en un texto, por categoría."""
    t = normalizar(texto)
    salida: dict[str, list[str]] = {cat: [] for cat in CATEGORIAS}
    for cat, term, patron in _LEXICO:
        if _afirmado(t, patron):
            salida[cat].append(term)
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
