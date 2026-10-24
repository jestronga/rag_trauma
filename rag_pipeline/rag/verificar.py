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

Con los libros de trauma (PLAN_INDEXAR_LIBROS.md, fase 5), una oración que
cita [guía: ID] se revisa contra los fragmentos que cita, no contra los casos:
  7. citas      toda [guía: ID] está entre las secciones recuperadas
  8. separación no cita casos y guías a la vez: cada afirmación tiene una sola
                fuente, y así no se atribuye a un caso lo que dice un manual.
                Si nombra un manual (ATLS, HUV), cita un pasaje de ese manual
  9. cifras     cada número está escrito en el fragmento citado
 10. términos y vocabulario (reglas 2 y 6) contra el fragmento, solo si es
                del manual del HUV (en español). Del ATLS (en inglés) solo se
                revisan la cita, las cifras y el tono (decisión 2 del plan: no se
                traduce, y la app muestra el fragmento original)
 11. con fragmentos en el prompt, una oración sin citas no puede dar cifras
Las reglas 1 a 6 siguen igual para las demás oraciones.
"""
from __future__ import annotations

import inspect
import re
from collections import Counter
from dataclasses import dataclass, field

from . import build_prompt, config, hallazgos
from .hallazgos import normalizar, obvios, terminos
from .retrieve import FragmentoGuia, Resultado

# `[caso: 12345]` es el formato pedido; los modelos pequeños a veces escriben
# `[12345]`. Si no se reconoce, una cita inventada pasaría sin marcarse.
RX_CITA = re.compile(r"\[(?:caso[:\s]+)?([0-9]{1,8})\]", re.IGNORECASE)
# `[guía: HUV-11-003]` (la sección). Se acepta también el ID del fragmento
# (`HUV-11-003-02`), que se lee como su sección. El libro puede llevar
# dígitos: `ATLS10-03-005` (PLAN_INDEXAR_ATLS10.md).
RX_CITA_GUIA = re.compile(r"\[gu[ií]as?[:\s]+([a-z][a-z0-9]{1,6}-\d{2}-\d{3})(?:-\d{2})?\]",
                          re.IGNORECASE)
_RX_CIFRA = re.compile(r"\d+(?:[.,]\d+)*")

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
aplicar momento debido podria podrian rapidamente mantuvo
""".split()
# Medido (28/09, 25 prosas): la penúltima línea eran falsos rechazos. "proximal" y
# "estabilidad" NO van aquí: eran detalles que el modelo agregó sin que
# estuvieran en los casos ("región proximal", "evaluar la estabilidad").
# Medido (17/10, 41 prosas con los manuales): la última línea. NO van
# "profundas", "inestabilidad", "riesgo", "aliviar" ni "aereas": agregaban un
# detalle clínico o una razón que no estaba en la fuente.


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


def _formas(cifra: str) -> set[str]:
    """Formas equivalentes de un número: 0,5 = 0.5; 1,000 = 1.000 = 1000."""
    punto = cifra.replace(",", ".")
    formas = {punto}
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", punto):
        formas.add(punto.replace(".", ""))
    return formas


def cifras(texto: str) -> set[str]:
    return {f for m in _RX_CIFRA.finditer(texto) for f in _formas(m.group(0))}


def _sin_citas(oracion: str) -> str:
    return RX_CITA.sub(" ", RX_CITA_GUIA.sub(" ", oracion))


def _verificar_oracion_guia(oracion: str, citadas: set[str], textos: dict[str, str],
                            idioma: dict[str, str], consulta: str, de_consulta: set[str]) -> list[str]:
    """Reglas 7 a 10, para una oración que cita manuales."""
    motivos = []
    inventadas = citadas - textos.keys()
    if inventadas:
        motivos.append(f"cita a guías que no están entre los fragmentos recuperados: "
                       f"{', '.join(sorted(inventadas))}")
    limpia = _sin_citas(oracion)
    norm = normalizar(limpia)
    orden = _RX_ORDEN.search(norm)
    if orden:
        motivos.append(f"da una orden («{orden.group(0)}») en vez de orientar")
    validas = citadas & textos.keys()
    if not validas:
        return motivos
    # "El ATLS recomienda… [guía: HUV-04-013]": el manual nombrado es el citado
    for fuente_nombrada in ("ATLS", "HUV"):
        if re.search(rf"\b{fuente_nombrada.lower()}\b", norm) and \
                not any(s.startswith(f"{fuente_nombrada}-") for s in validas):
            motivos.append(f"atribuye al {fuente_nombrada} un pasaje de otro manual")
    fuente = " ".join(textos[s] for s in validas)
    en_fuente = cifras(fuente)
    for m in _RX_CIFRA.finditer(limpia):
        if not _formas(m.group(0)) & en_fuente:
            motivos.append(f"«{m.group(0)}» no está en la guía citada")
    if any(idioma[s] != "es" for s in validas):
        return motivos                     # ATLS: solo cita, cifras y tono (decisión 2)
    nombrados = {t for ts in terminos(limpia).values() for t in ts}
    en_guia = {t for ts in terminos(fuente).values() for t in ts}
    for t in sorted(nombrados - en_guia - de_consulta):
        motivos.append(f"«{t}» no consta en la guía citada")
    base = normalizar(fuente + " " + consulta) + " " + _VOCABULARIO_CODIGO
    ajenas = sorted({w for w in re.findall(r"[a-z]{6,}", normalizar(limpia)) if w[:5] not in base})
    if ajenas:
        motivos.append(f"palabras que no aparecen en la guía citada: {', '.join(ajenas)}")
    return motivos


@dataclass
class Verificacion:
    """La prosa partida por fuente, con los motivos de rechazo de cada parte.

    Casos: las oraciones sin [guía: ID] (también las que no citan nada).
    Manuales: las que citan [guía: ID]. Cada parte se puede mostrar si sus
    oraciones pasan, aunque la otra no (generate.prosa_por_fuente)."""
    oraciones_casos: list[str] = field(default_factory=list)
    oraciones_guias: list[str] = field(default_factory=list)
    motivos_casos: list[str] = field(default_factory=list)
    motivos_guias: list[str] = field(default_factory=list)
    vacia: bool = False

    @property
    def motivos(self) -> list[str]:
        if self.vacia:
            return ["no hay prosa que mostrar (sin elementos que sugerir, o salida cortada)"]
        return self.motivos_casos + self.motivos_guias

    @property
    def pasa(self) -> bool:
        return not self.motivos


def verificar_prosa(prosa: str, casos: list[Resultado], consulta: str = "",
                    guias: list[FragmentoGuia] | None = None) -> tuple[bool, list[str]]:
    """(pasa, motivos). Sin motivos = se puede mostrar."""
    v = verificar_por_fuente(prosa, casos, consulta, guias)
    return v.pasa, v.motivos


def verificar_por_fuente(prosa: str, casos: list[Resultado], consulta: str = "",
                         guias: list[FragmentoGuia] | None = None) -> Verificacion:
    if not prosa.strip():
        return Verificacion(vacia=True)
    v = Verificacion()
    motivos = v.motivos_casos          # las reglas 1 a 6 escriben aquí
    permitidos = {c.record_id for c in casos}
    por_caso = _documentados(casos)
    en_casos = set().union(*por_caso.values()) if por_caso else set()
    de_consulta = {t for ts in terminos(consulta).values() for t in ts} | obvios(consulta)
    etiquetas = _etiquetas(casos)
    n = len(casos)
    # sección -> texto con sus títulos (el modelo los ve en la cabecera)
    grupos = build_prompt.agrupar_guias(guias or [])
    textos_guia = {s: "\n".join(f.texto for f in fs) for s, fs in grupos.items()}
    idioma = {s: fs[0].idioma for s, fs in grupos.items()}
    de_casos = v.oraciones_casos       # las oraciones que se revisan contra los casos

    for oracion in oraciones(prosa):
        citas_guia = {m.group(1).upper() for m in RX_CITA_GUIA.finditer(oracion)}
        if citas_guia:
            v.oraciones_guias.append(oracion)
            if RX_CITA.search(RX_CITA_GUIA.sub(" ", oracion)):
                v.motivos_guias.append("mezcla citas de casos y de manuales en una oración")
            v.motivos_guias.extend(_verificar_oracion_guia(oracion, citas_guia, textos_guia, idioma,
                                                           consulta, de_consulta))
            continue
        de_casos.append(oracion)
        if guias and not RX_CITA.search(oracion) and _RX_CIFRA.search(oracion):
            motivos.append("da cifras sin citar de dónde salen")
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

    ajenas = palabras_sin_raiz(" ".join(de_casos), casos, consulta)
    if ajenas:
        motivos.append(f"palabras que no aparecen en los casos: {', '.join(ajenas)}")
    return v
