"""
Verificación de la sugerencia en prosa que escribe el modelo.

El modelo redacta; el código decide si lo redactado se puede mostrar. Una
prosa que no pasa TODAS las reglas se descarta y la respuesta vuelve a la
versión determinista (generate.ensamblar). Mejor una respuesta armada por el
código que una frase fluida sin respaldo: en rag_trauma el texto libre ya
inventó "estudios de imagen 4 de 5" y atribuyó una laparoscopia al caso
equivocado.

Reglas, por oración:
  1. citas      todo [caso: ID] está entre los casos recuperados
  2. términos   cada término del léxico (hallazgos.py) que nombra consta en
                algún caso recuperado o en la consulta
  3. atribución si nombra un término que no viene de la consulta, cita al
                menos un caso que lo documente
  4. conteos    "dos de ellos", "3 de 5", "todos los casos", "la mayoría" solo
                si el número coincide con cuántos casos documentan algo que la
                oración nombra (un término o una etiqueta de mecanismo, región
                o desenlace)
  5. tono       orienta ("conviene valorar"), no ordena ("se debe", "hay que")
  6. vocabulario ninguna palabra de 6 o más letras cuya raíz no aparezca en los
                casos, la consulta, el vocabulario del código o PALABRAS_COMUNES:
                atrapa lo que el léxico no conoce ("ecografía FAST")
"""
from __future__ import annotations

import inspect
import re
from collections import Counter

from . import build_prompt, config, hallazgos
from .hallazgos import normalizar, obvios, terminos
from .retrieve import Resultado

# `[caso: 12345]` es el formato pedido; los modelos pequeños a veces escriben
# `[12345]`. Si no se reconoce, una cita inventada pasaría sin marcarse.
RX_CITA = re.compile(r"\[(?:caso[:\s]+)?([0-9]{1,8})\]", re.IGNORECASE)

_RX_ORACION = re.compile(r"(?<=[.;!?])\s+|\n+")

_NUMEROS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
            "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
_NUM = r"\d+|" + "|".join(_NUMEROS)
# "dos de ellos", "3 de 5", "uno de los casos"
_RX_CONTEO = re.compile(rf"\b({_NUM})\s+de\s+(los|las|estos|estas|ellos|ellas|{_NUM})\b")
# "se registró en dos casos", "en un caso de herida en cuello"
# (no el total de "3 de 4 casos", que ya revisa _RX_CONTEO)
_RX_CONTEO_CASOS = re.compile(rf"(?<!de )\b({_NUM})\s+casos?\b")
_RX_TODOS = re.compile(r"\b(todos|todas)\s+(los|las|estos|estas|ellos|ellas)\b")
_RX_MAYORIA = re.compile(r"\bla mayoria\b")
# Orientación, no órdenes: es apoyo a la decisión, no una indicación. Medido
# (28/09): "Se debe realizar radiografía de cadera" pasaba todo lo demás.
_RX_ORDEN = re.compile(r"\b(se )?deben?\b|\bhay que\b|\bes (necesario|obligatorio|imperativo)\b"
                       r"|\besta indicad[oa]s?\b")

# Palabras de redacción que no nombran nada clínico. Se amplía midiendo los
# falsos rechazos (scripts/comparar_modelos.py --modos prosa); nunca con
# términos clínicos: esos tienen que salir de los casos.
PALABRAS_COMUNES = """
similares similar presentan presento presentaron presentado documento documentaron
documentado documentada requirieron requirio requiere requerir conviene considerar
valorar evaluar evaluacion orientacion orientar sugiere sugieren sugerencia pacientes
ademas tambien respecto relacion asociado asociada asociados segun clinico clinica
cuadro posible posibles realizo realizaron realizado realizada practico practicaron
indicado indicada indicacion manejado manejados consulta hallazgos estudios estudio
identificado identificaron registro registrados mediante durante despues inicial
inicialmente presencia signos descartar vigilar vigilancia seguimiento necesidad
necesario necesaria ameritaron amerito recibieron recibio tratados tratado tratamiento
procedimiento procedimientos observaron observado encontro encontraron evidencio
evidenciaron casos referencia paciente lesiones lesion traumatismo mecanismo
estado ocurrio ocurrieron coloco colocaron colocado afectacion tejido tejidos
administro administraron administrado atender atendio atendieron
""".split()
# Medido (28/09, 25 prosas): la última línea eran falsos rechazos. "proximal" y
# "estabilidad" NO van aquí: eran detalles que el modelo agregó sin que
# estuvieran en los casos ("región proximal", "evaluar la estabilidad").


def _numero(token: str) -> int | None:
    return int(token) if token.isdigit() else _NUMEROS.get(token)


def _documentados(casos: list[Resultado]) -> dict[str, set[str]]:
    """record_id -> términos del léxico afirmados en el caso (todas las categorías)."""
    return {c.record_id: {t for ts in terminos(c.text).values() for t in ts} for c in casos}


def _etiquetas(casos: list[Resultado]) -> Counter:
    """Etiquetas de metadatos que una oración puede contar: "arma de fuego", "tórax", "operado"."""
    conteo: Counter = Counter()
    for c in casos:
        meta = c.metadata
        vistas = set()
        mec = meta.get("mecanismo_lesion")
        if mec:
            vistas.add(normalizar(config.ETIQUETAS_MECANISMO.get(mec, mec)))
        for r in meta.get("region_anatomica") or []:
            vistas.add(normalizar(config.ETIQUETAS_REGION.get(r, r)))
        vistas.add(normalizar(build_prompt.desenlace_visible(c)))
        conteo.update(vistas)
    return conteo


_VOCABULARIO_CODIGO = normalizar(" ".join([
    inspect.getsource(build_prompt), inspect.getsource(hallazgos),
    *config.ETIQUETAS_MECANISMO.values(), *config.ETIQUETAS_REGION.values(),
    *config.ETIQUETAS_DESENLACE.values(), *PALABRAS_COMUNES,
]))


def palabras_sin_raiz(texto: str, casos: list[Resultado], consulta: str = "") -> list[str]:
    """Palabras de 6 o más letras cuyas primeras 5 no aparecen en los casos, la
    consulta ni el vocabulario del código. Un término clínico en esta lista es
    una posible invención."""
    fuente = normalizar(" ".join(c.text for c in casos) + " " + consulta) + " " + _VOCABULARIO_CODIGO
    return sorted({w for w in re.findall(r"[a-z]{6,}", normalizar(texto)) if w[:5] not in fuente})


# "Se debe considerar X" es la forma en que el 4B dice "considerar X" (8 de 25
# prosas, 28/09): se reescribe a "Conviene considerar X". Cambia el tono, no el
# contenido. Cualquier otra orden ("hay que", "debe realizarse") se rechaza.
_RX_SE_DEBE = re.compile(r"\b([Ss])e debe (considerar|evaluar|valorar)\b")


def suavizar(prosa: str) -> str:
    return _RX_SE_DEBE.sub(lambda m: ("C" if m.group(1) == "S" else "c") + "onviene " + m.group(2), prosa)


def oraciones(texto: str) -> list[str]:
    return [o.strip() for o in _RX_ORACION.split(texto) if o.strip()]


def verificar_prosa(prosa: str, casos: list[Resultado], consulta: str = "") -> tuple[bool, list[str]]:
    """(pasa, motivos). Sin motivos = se puede mostrar."""
    if not prosa.strip():
        return False, ["no hay prosa que mostrar (sin elementos que sugerir, o salida cortada)"]
    motivos: list[str] = []
    permitidos = {c.record_id for c in casos}
    por_caso = _documentados(casos)
    en_casos = set().union(*por_caso.values()) if por_caso else set()
    de_consulta = {t for ts in terminos(consulta).values() for t in ts} | obvios(consulta)
    etiquetas = _etiquetas(casos)
    n = len(casos)

    for oracion in oraciones(prosa):
        citas = {m.group(1) for m in RX_CITA.finditer(oracion)}
        inventadas = citas - permitidos
        if inventadas:
            motivos.append(f"cita a casos que no están entre los recuperados: {', '.join(sorted(inventadas))}")
        nombrados = {t for ts in terminos(RX_CITA.sub(" ", oracion)).values() for t in ts}

        sin_respaldo = nombrados - en_casos - de_consulta
        for t in sorted(sin_respaldo):
            motivos.append(f"«{t}» no consta en ningún caso recuperado")

        propios = (nombrados & en_casos) - de_consulta
        validas = citas & permitidos
        if propios and not validas:
            motivos.append(f"nombra {', '.join(sorted(propios))} sin citar el caso")
        elif propios:
            for t in sorted(propios):
                if not any(t in por_caso[rid] for rid in validas):
                    motivos.append(f"«{t}» se cita a casos que no lo documentan")

        norm = normalizar(oracion)
        posibles = {sum(t in ts for ts in por_caso.values()) for t in nombrados}
        # las etiquetas se buscan sin los términos del léxico: "tórax" de
        # "tubo de tórax" no es la región, y validaba "todos los casos"
        sin_terminos = norm
        for _, _, patron in hallazgos._LEXICO:
            sin_terminos = patron.sub(" ", sin_terminos)
        posibles |= {k for et, k in etiquetas.items() if et and et in sin_terminos}
        for m in _RX_CONTEO.finditer(norm):
            cuanto, total = _numero(m.group(1)), _numero(m.group(2))
            if total is not None and total != n:
                motivos.append(f"«{m.group(0)}»: se recuperaron {n} casos")
            elif cuanto not in posibles:
                motivos.append(f"«{m.group(0)}» no coincide con los casos")
        for m in _RX_CONTEO_CASOS.finditer(norm):
            if _numero(m.group(1)) not in posibles:
                motivos.append(f"«{m.group(0)}» no coincide con los casos")
        if _RX_TODOS.search(norm) and n not in posibles:
            motivos.append("dice «todos» y no es así en los casos")
        if _RX_MAYORIA.search(norm) and not any(k > n / 2 for k in posibles):
            motivos.append("dice «la mayoría» y no es así en los casos")
        orden = _RX_ORDEN.search(norm)
        if orden:
            motivos.append(f"da una orden («{orden.group(0)}») en vez de orientar")

    ajenas = palabras_sin_raiz(prosa, casos, consulta)
    if ajenas:
        motivos.append(f"palabras que no aparecen en los casos: {', '.join(ajenas)}")
    return not motivos, motivos
