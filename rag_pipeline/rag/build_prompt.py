"""
Construcción de los prompts y de las secciones deterministas de la respuesta.

Separado de generate.py a propósito: el prompt es el artefacto que más se
itera en un sistema clínico, y conviene poder leerlo y cambiarlo sin tocar la
lógica de recuperación ni la de abstención.

El asistente hace dos cosas, y las dos salen SOLO de los casos recuperados:

  1. Muestra lo que se hizo en los casos similares.
  2. Sugiere qué valorar, pero solo lo que los casos respaldan: cada
     sugerencia dice en cuántos casos se apoya y los cita.

Reparto de trabajo, pensado para un modelo pequeño (3B en CPU) que, probado,
inventa detalles cuando resume casos o sugiere en libre:

  · La respuesta que lee el médico la arma el CÓDIGO: conclusión
    (`conclusion_calculada`), tabla de casos (`tabla_casos`), contexto de
    cada sugerencia (`contexto_elemento`), hallazgos que diferencian los
    casos (`para_afinar`) y límites. Todo sale del léxico de hallazgos.py y
    de los metadatos; el texto completo de cada caso queda en la app. El 3B,
    resumiendo, atribuía una laparoscopia al caso equivocado.
  · Las sugerencias las redacta el MODELO, pero solo a partir de la lista de
    elementos documentados (hallazgos.py); después generate.py reescribe el
    conteo y las citas de cada viñeta desde esa lista y descarta las que no
    corresponden a ningún elemento.
  · Los conteos de mecanismo, región y desenlace se calculan aquí y se le
    entregan hechos: un 3B cuenta mal.
  · La plantilla va al final del mensaje del usuario, con los títulos
    literales. Descrita en prosa en el prompt de sistema, el modelo copiaba
    las instrucciones dentro de la respuesta.
"""
from __future__ import annotations

import re
from collections import Counter

from . import config
from .hallazgos import (
    PROCEDIMIENTOS_QUIRURGICOS, Elemento, extraer, formatear, normalizar, obvios, terminos,
)
from .retrieve import Resultado

# Títulos del protocolo con el modelo: la plantilla se los pide y generate.py
# los busca en su salida. No son los que ve el médico.
TITULO_CASOS = "### Casos similares"
TITULO_SUGERENCIAS = "### Sugerencias basadas en los casos"
TITULO_LIMITES = "### Lo que el registro no permite afirmar"
SIN_SUGERENCIAS = "Sin sugerencias con respaldo en estos casos."

# Lo que ve el médico. Es un asistente al diagnóstico: lo PRIMERO son las
# sugerencias, cortas y accionables. Después, la base de cada una (síntesis,
# tabla, citas), que el médico tiene que poder revisar por su cuenta.
ROTULO_SUGERENCIAS = "**Sugerencias según casos similares**"
ROTULO_SINTESIS = "**En síntesis.**"
ROTULO_AFINAR = "**Para afinar la búsqueda**"

# Cómo se muestra el desenlace. "cirugía" es un procedimiento, no un desenlace:
# el clasificador de anonimización lo usa para "hospitalizado y operado".
DESENLACE_VISIBLE = {
    "alta": "alta", "hospitalizacion": "hospitalizado", "cirugia": "operado", "uci": "UCI",
    "remision": "remitido", "fallecimiento": "falleció", "no_determinado": "sin clasificar",
}
# Gravedad para elegir qué escalada contar en la síntesis.
_GRAVEDAD = {"fallecimiento": 3, "uci": 2, "cirugia": 1}

_REGLAS_COMUNES = f"""\
FUENTE ÚNICA
Trabajas SOLO con los casos que se te entregan, del registro de trauma \
({config.COBERTURA_TEMPORAL}) de un hospital de Colombia. No uses conocimiento médico \
general, guías ni protocolos externos. Si algo no aparece en los casos, no existe para ti.

PROHIBIDO INVENTAR
- No menciones síntomas, medicamentos, dosis, estudios, procedimientos ni diagnósticos \
que no estén escritos en los casos.
- No completes datos que falten. Si algo no consta, dilo.
- Los datos personales aparecen como marcadores ([FECHA], [EDAD:30-39], [INSTITUCION], \
[MUNICIPIO], [MEDICO]). Son información no disponible: no intentes deducirlos.
- Si un caso viene marcado como TRUNCADO, no infieras cómo continuaba.

CITAS
Cada afirmación sobre un caso lleva su identificador en el formato [caso: 12345]. \
Nunca cites un identificador que no esté en la lista entregada.

ESTILO
Español de Colombia, registro clínico, directo."""

SISTEMA = f"""Eres un asistente de apoyo al diagnóstico en trauma para personal de salud. \
Ante una presentación clínica, comparas con casos similares del registro y sugieres qué \
valorar según lo que se hizo en ellos. No reemplazas el juicio clínico de quien consulta.

{_REGLAS_COMUNES}

CÓMO SUGERIR
- Las sugerencias salen SOLO de la lista "ELEMENTOS DOCUMENTADOS EN LOS CASOS" que \
viene en el mensaje. Nada que no esté en esa lista puede ser sugerencia.
- Usa el nombre del elemento tal como aparece en la lista, con su conteo y sus citas.
- Hallazgos: "Valorar ...". Estudios: "Considerar ...". Manejo: "Considerar ..., según \
el cuadro clínico".
- Prioriza los elementos más frecuentes y los que tienen que ver con la consulta.
- Redacta como orientación, nunca como orden ("se debe", "hay que", "el tratamiento \
indicado es")."""

SISTEMA_RESUMEN = f"""Eres un asistente de consulta sobre un registro de casos de trauma. \
Resumes un caso concreto tal como consta en el registro, sin agregar nada.

{_REGLAS_COMUNES}"""

_PLANTILLA_SINTESIS = f"""\
{TITULO_CASOS}
(1 o 2 frases: en qué se parecen los casos a la consulta y en qué difieren su manejo o \
su desenlace. Lo general, solo con los DATOS CALCULADOS; lo de un caso concreto, con su \
cita. No escribas "todos" si el conteo no es N de N. No especules sobre el paciente de \
la consulta.)

"""

PLANTILLA = f"""\
{TITULO_SUGERENCIAS}
- Valorar (hallazgo de la lista) [caso: ID]
- Considerar (estudio o manejo de la lista), según el cuadro clínico [caso: ID]
(entre 2 y 4 viñetas, tomadas SOLO de la lista de elementos documentados. Si la lista \
está vacía, escribe una sola línea: "{SIN_SUGERENCIAS}")"""

_PLANTILLA_LIMITES = f"""

{TITULO_LIMITES}
(1 o 2 frases sobre qué información CLÍNICA falta en los casos para responder mejor la \
consulta. Los datos personales anonimizados no cuentan.)"""


def _etiqueta(mapa: dict, clave: str) -> str:
    return mapa.get(clave, clave)


def resumen_calculado(casos: list[Resultado]) -> str:
    """Conteos exactos desde los metadatos, para que el modelo no tenga que contar.

    `no_determinado` se presenta como "sin clasificar": significa que el
    clasificador automático no lo reconoció, no que el caso no tenga desenlace.
    """
    n = len(casos)
    mecanismos = Counter(c.metadata.get("mecanismo_lesion") or "no_determinado" for c in casos)
    regiones = Counter(r for c in casos for r in (c.metadata.get("region_anatomica") or []))
    desenlaces = Counter(c.metadata.get("desenlace") or "no_determinado" for c in casos)
    pediatricos = sum(1 for c in casos if c.metadata.get("es_pediatrico"))

    def fmt(conteo: Counter, mapa: dict) -> str:
        partes = []
        for clave, k in conteo.most_common():
            nombre = "sin clasificar" if clave == "no_determinado" else _etiqueta(mapa, clave)
            partes.append(f"{nombre} {k} de {n}")
        return ", ".join(partes) or "sin datos"

    return "\n".join([
        f"- Casos: {n}" + (f" ({pediatricos} pediátricos)" if pediatricos else ""),
        f"- Mecanismo: {fmt(mecanismos, config.ETIQUETAS_MECANISMO)}",
        f"- Regiones: {fmt(regiones, config.ETIQUETAS_REGION)}",
        f"- Desenlace (clasificación automática): {fmt(desenlaces, config.ETIQUETAS_DESENLACE)}",
    ])


def linea_base(casos: list[Resultado]) -> str:
    """Conteos exactos en una línea, bajo la síntesis del modelo, para que quien
    lee tenga los números reales aunque la síntesis generalice de más."""
    n = len(casos)
    partes = [f"{n} casos"]
    for clave, mapa, nombre in (("mecanismo_lesion", config.ETIQUETAS_MECANISMO, "mecanismo"),
                                ("desenlace", config.ETIQUETAS_DESENLACE, "desenlace")):
        conteo = Counter(c.metadata.get(clave) or "no_determinado" for c in casos)
        valores = ", ".join(
            f"{'sin clasificar' if k == 'no_determinado' else _etiqueta(mapa, k).lower()} {v}"
            for k, v in conteo.most_common())
        partes.append(f"{nombre}: {valores}")
    return f"*Base: {' · '.join(partes)}*"


def _bloque_caso(caso: Resultado, indice: int) -> str:
    meta = caso.metadata
    etiquetas = []
    if meta.get("mecanismo_lesion"):
        etiquetas.append(f"mecanismo: {_etiqueta(config.ETIQUETAS_MECANISMO, meta['mecanismo_lesion'])}")
    if meta.get("region_anatomica"):
        regiones = ", ".join(_etiqueta(config.ETIQUETAS_REGION, r) for r in meta["region_anatomica"])
        etiquetas.append(f"región: {regiones}")
    if meta.get("desenlace") and meta["desenlace"] != "no_determinado":
        etiquetas.append(f"desenlace: {_etiqueta(config.ETIQUETAS_DESENLACE, meta['desenlace'])}")
    if meta.get("es_pediatrico"):
        etiquetas.append("paciente pediátrico")

    # Sin número de orden: con "CASO 1" el modelo escribía "los casos 1, 2 y 5"
    # en lugar de citar identificadores.
    cabecera = f"--- [caso: {caso.record_id}] ---"
    if etiquetas:
        cabecera += f"\n({' | '.join(etiquetas)})"

    # El corpus apto_rag ya no trae truncados; el aviso queda para cuando se
    # indexa el corpus de ablación (--incluir-incompletos).
    aviso = ""
    if meta.get("truncado"):
        campos = meta.get("campos_truncados") or []
        donde = ", ".join(c.lower() for c in campos) if campos else "alguno de sus campos"
        aviso = (f"\n[ATENCIÓN: este registro está TRUNCADO en {donde}. "
                 f"El desenlace puede no constar. No infieras cómo terminó.]")

    return f"{cabecera}\n{caso.text}{aviso}"


def construir_prompt(consulta: str, casos: list[Resultado]) -> str:
    bloques = "\n\n".join(_bloque_caso(c, i) for i, c in enumerate(casos, start=1))
    return (
        f"CONSULTA:\n{consulta}\n\n"
        f"CASOS SIMILARES DEL REGISTRO ({len(casos)}):\n\n{bloques}\n\n"
        f"DATOS CALCULADOS SOBRE ESTOS CASOS (úsalos tal cual para los conteos):\n"
        f"{resumen_calculado(casos)}\n\n"
        f"ELEMENTOS DOCUMENTADOS EN LOS CASOS (única fuente de las sugerencias):\n"
        f"{formatear(extraer(casos), len(casos))}\n\n"
        f"{_instruccion_salida()}"
    )


def _instruccion_salida() -> str:
    """Con SINTESIS_LLM apagado el modelo solo elige sugerencias; síntesis y
    límites los arma el código (sintesis_calculada, limites_calculados)."""
    if config.SINTESIS_LLM:
        return (f"Responde con EXACTAMENTE estas tres secciones y estos títulos. Reemplaza "
                f"lo que está entre paréntesis por el contenido; no copies los paréntesis. "
                f"Máximo 200 palabras.\n\n"
                f"{_PLANTILLA_SINTESIS}{PLANTILLA}{_PLANTILLA_LIMITES}")
    return (f"Responde SOLO con esta sección y este título. Reemplaza lo que está entre "
            f"paréntesis por el contenido; no copies los paréntesis.\n\n{PLANTILLA}")


# ---------------------------------------------------------------------------
# Respuesta para el médico: determinista, desde los casos
# ---------------------------------------------------------------------------
def desenlace_visible(caso: Resultado) -> str:
    return DESENLACE_VISIBLE.get(caso.metadata.get("desenlace") or "no_determinado", "sin clasificar")


def _citas(ids: list[str]) -> str:
    return " ".join(f"[caso: {rid}]" for rid in ids)


def _similitud(caso: Resultado) -> str:
    # 0 = el caso lo trajo solo BM25 (por palabras), sin coseno denso
    return f"{caso.score_norm:.2f}".replace(".", ",") if caso.score_norm > 0 else "—"


def _conteo(pares: Counter) -> str:
    """"alta 2, operado 1": de más a menos frecuente, "sin clasificar" al final."""
    orden = sorted(pares.items(), key=lambda kv: (kv[0] == "sin clasificar", -kv[1]))
    return ", ".join(f"{k} {v}" for k, v in orden)


def _por_caso(casos: list[Resultado]) -> dict[str, dict[str, list[str]]]:
    return {c.record_id: terminos(c.text) for c in casos}


def _manejo_visible(ts: dict[str, list[str]]) -> list[str]:
    """Manejo de un caso sin "cirugía" cuando ya consta el procedimiento concreto."""
    if any(p in ts["Manejo"] for p in PROCEDIMIENTOS_QUIRURGICOS):
        return [m for m in ts["Manejo"] if m != "cirugía"]
    return ts["Manejo"]


def _operado(caso: Resultado, ts: dict[str, list[str]]) -> bool:
    return (caso.metadata.get("desenlace") == "cirugia") or "cirugía" in ts["Manejo"]


def conclusion_calculada(casos: list[Resultado], consulta: str = "") -> str:
    """Conclusión primero: qué casos son, qué se hizo más y la escalada más grave.

    Todo sale de conteos y del léxico; cada caso concreto va citado. La última
    línea dice sobre cuánto se apoya: n y rango de similitud.
    """
    n = len(casos)
    por_caso = _por_caso(casos)
    ya_dicho = obvios(consulta)

    mecanismos = Counter(_etiqueta(config.ETIQUETAS_MECANISMO, c.metadata.get("mecanismo_lesion") or "no_determinado").lower()
                         for c in casos)
    if len(mecanismos) == 1:
        quienes = f"{n} casos similares, todos por {next(iter(mecanismos))}"
    else:
        quienes = f"{n} casos similares ({_conteo(mecanismos)})"
    regiones = Counter(r for c in casos for r in (c.metadata.get("region_anatomica") or []))
    if regiones:
        region, k = regiones.most_common(1)[0]
        quienes += f", {_etiqueta(config.ETIQUETAS_REGION, region).lower()} en {k}"
    frases = [quienes + "."]

    # La escalada más grave, con los hallazgos que tenían esos casos: es lo que
    # un registro de casos aporta que una guía no.
    graves: dict[int, list[Resultado]] = {}
    for c in casos:
        d = c.metadata.get("desenlace") or "no_determinado"
        nivel = _GRAVEDAD.get(d, 1 if _operado(c, por_caso[c.record_id]) else 0)
        if nivel:
            graves.setdefault(nivel, []).append(c)
    escalada = ""
    if graves:
        nivel = max(graves)
        grupo = graves[nivel]
        uno = len(grupo) == 1
        verbo = {3: "falleció" if uno else "fallecieron",
                 2: "pasó a UCI" if uno else "pasaron a UCI",
                 1: "fue operado o tuvo indicación quirúrgica" if uno
                 else "fueron operados o tuvieron indicación quirúrgica"}[nivel]
        escalada = f"{len(grupo)} de {n} {verbo} {_citas([c.record_id for c in grupo])}"
        con = Counter(h for c in grupo for h in por_caso[c.record_id]["Hallazgos"] if h not in ya_dicho)
        if con:
            escalada += "; " + ("tenía" if uno else "en ellos") + ": " + ", ".join(
                h if uno else f"{h} ({k})" for h, k in con.most_common(3))
        escalada += "."

    # "cirugía" no se repite en "lo más documentado" si ya la cuenta la escalada
    manejo = [e for e in extraer(casos) if e.categoria == "Manejo"
              and not (e.termino == "cirugía" and graves and max(graves) == 1)]
    if manejo:
        top = ", ".join(f"**{e.termino}** ({len(e.casos)} de {n})" for e in manejo[:2])
        frases.append(f"Lo más documentado: {top}.")
    elif not escalada:
        frases.append("No hay manejo reconocible en los registros: ver la tabla y los casos completos.")
    if escalada:
        frases.append(escalada)

    sims = [c.score_norm for c in casos if c.score_norm > 0]
    respaldo = f"Respaldo: {n} casos del registro"
    if sims:
        respaldo += f" · similitud {min(sims):.2f}–{max(sims):.2f}".replace(".", ",")
    if len(sims) < n:
        respaldo += f" · {n - len(sims)} hallado(s) solo por palabras clave"
    salida = f"{ROTULO_SINTESIS} {' '.join(frases)}\n\n*{respaldo}.*"
    if not sims or max(sims) < config.SIMILITUD_BAJA:
        salida += ("\n\n**Similitud baja:** estos casos pueden no corresponder a la consulta. "
                   "Revisar la tabla antes de usarlos.")
    return salida


def tabla_casos(casos: list[Resultado], consulta: str = "") -> str:
    """Un caso por fila, celdas cortas con términos del léxico (no texto libre).

    Los hallazgos que la consulta también menciona van en negrita: es lo que
    hace parecido al caso. El texto completo queda en las tarjetas de la app.
    """
    ya_dicho = set(terminos(consulta)["Hallazgos"]) if consulta else set()
    filas = ["| Caso | Similitud | Lesión | Hallazgos | Manejo | Desenlace |",
             "|---|---|---|---|---|---|"]
    for c in casos:
        ts = terminos(c.text)
        meta = c.metadata
        lesion = _etiqueta(config.ETIQUETAS_MECANISMO, meta.get("mecanismo_lesion") or "no_determinado")
        if meta.get("region_anatomica"):
            lesion += " · " + ", ".join(_etiqueta(config.ETIQUETAS_REGION, r).lower()
                                        for r in meta["region_anatomica"][:2])
        if meta.get("es_pediatrico"):
            lesion += " · pediátrico"
        hallazgos = ", ".join(f"**{h}**" if h in ya_dicho else h for h in ts["Hallazgos"]) or "—"
        hecho = ", ".join(ts["Estudios"] + _manejo_visible(ts)) or "—"
        filas.append(f"| [caso: {c.record_id}] | {_similitud(c)} | {lesion} | {hallazgos} | {hecho} "
                     f"| {desenlace_visible(c)} |")
    return "\n".join(filas)


def contexto_elemento(e: Elemento, casos: list[Resultado], consulta: str = "") -> str:
    """Viñeta de "Para considerar": el elemento, en cuántos casos, y en qué casos.

    Para un manejo o estudio: qué hallazgos tenían esos pacientes y cómo
    terminaron. Para un hallazgo: qué se les hizo y cómo terminaron. Conteo y
    citas salen de la lista, nunca del modelo.
    """
    n = len(casos)
    ids = set(e.casos)
    sub = [c for c in casos if c.record_id in ids]
    por_caso = _por_caso(sub)
    ya_dicho = obvios(consulta)
    desenlaces = Counter(desenlace_visible(c) for c in sub)

    # Línea 1: la acción, corta y en negrita, con su respaldo. Es lo que el
    # médico tiene que poder leer de un vistazo.
    if e.categoria == "Hallazgos":
        cabeza = f"- **Valorar {e.termino}** — presente en {len(ids)} de {n} casos similares"
        otros = Counter(t for c in sub for t in _manejo_visible(por_caso[c.record_id]))
        contexto = f"En esos casos se hizo: {_conteo(otros)}" if otros else ""
    else:
        verbo = {"Estudios": "se solicitó", "Manejo": "se hizo"}[e.categoria]
        if e.termino == "cirugía":
            verbo = "se hizo o se indicó"
        cabeza = f"- **Considerar {e.termino}** — {verbo} en {len(ids)} de {n} casos similares"
        otros = Counter(t for c in sub for t in por_caso[c.record_id]["Hallazgos"] if t not in ya_dicho)
        contexto = f"En esos casos: {_conteo(otros)}" if otros else ""
    # Línea 2: el detalle y las citas, para quien quiera verificar.
    detalle = " · ".join(([contexto] if contexto else []) + [f"desenlace: {_conteo(desenlaces)}"])
    return f"{cabeza}  \n  {detalle[0].upper()}{detalle[1:]} {_citas(e.casos)}"


def para_afinar(casos: list[Resultado], consulta: str = "", maximo: int = 4,
                ya_sugeridos: set[str] | None = None) -> str:
    """Hallazgos que DIFERENCIAN los casos y la consulta no dice.

    Presentes en algunos casos y no en todos: son los que separan un manejo de
    otro. Solo términos del léxico: no se sugiere preguntar nada que no esté
    en los casos. Los que ya van como sugerencia no se repiten.
    """
    n = len(casos)
    omitir = obvios(consulta) | (ya_sugeridos or set())
    difieren = [e for e in extraer(casos) if e.categoria == "Hallazgos"
                and 0 < len(e.casos) < n and e.termino not in omitir][:maximo]
    if not difieren:
        return ""
    lista = ", ".join(f"{e.termino} ({len(e.casos)} de {n})" for e in difieren)
    return (f"{ROTULO_AFINAR}: los casos se diferencian por {lista}. Si su paciente los tiene "
            f"o no, agregarlo a la consulta acota los casos.")


_RX_VITALES = re.compile(r"\b(t\.?a\.?|p\.?a\.?|f\.?c\.?|f\.?r\.?|sat\s?o2|spo2|tension arterial|presion arterial"
                         r"|frecuencia cardiaca|signos vitales)\b")
_RX_GLASGOW = re.compile(r"glasgow|\bgcs\b")


def limites_calculados(casos: list[Resultado]) -> str:
    """Límites de ESTOS casos, en una línea (el aviso general va fijo en la app)."""
    n = len(casos)
    textos = [normalizar(c.text) for c in casos]
    partes = []
    sin_desenlace = sum(1 for c in casos if (c.metadata.get("desenlace") or "no_determinado") == "no_determinado")
    if sin_desenlace:
        partes.append(f"{sin_desenlace} de {n} sin desenlace clasificado")
    partes.append(f"signos vitales anotados en {sum(1 for t in textos if _RX_VITALES.search(t))} de {n}")
    partes.append(f"Glasgow en {sum(1 for t in textos if _RX_GLASGOW.search(t))} de {n}")
    return f"*Límites de estos casos: {' · '.join(partes)}.*"


def resumen_de_caso(caso: Resultado) -> str:
    return (
        f"Resume el siguiente caso en un párrafo: mecanismo, hallazgos al examen, manejo "
        f"y desenlace, tal como constan. Si algo no consta, dilo. Cita el caso una vez "
        f"con [caso: {caso.record_id}].\n\n"
        f"{_bloque_caso(caso, 1)}"
    )
