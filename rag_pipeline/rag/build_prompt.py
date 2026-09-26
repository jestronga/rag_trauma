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

  · "Qué se hizo" lo arma el CÓDIGO (`seccion_que_se_hizo`): hallazgos
    extraídos por reglas y el manejo citado textualmente del registro. Una
    cita textual no se puede inventar. El 3B, resumiendo, atribuía una
    laparoscopia al caso equivocado.
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
from .hallazgos import extraer, formatear, terminos
from .retrieve import Resultado

TITULO_CASOS = "### Casos similares"
TITULO_QUE_SE_HIZO = "### Qué se hizo"
TITULO_SUGERENCIAS = "### Sugerencias basadas en los casos"
TITULO_LIMITES = "### Lo que el registro no permite afirmar"
SIN_SUGERENCIAS = "Sin sugerencias con respaldo en estos casos."

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


def limites_calculados(casos: list[Resultado]) -> str:
    """Límites verdaderos por construcción: cobertura, tamaño y desenlaces sin clasificar."""
    n = len(casos)
    sin_desenlace = sum(1 for c in casos if (c.metadata.get("desenlace") or "no_determinado") == "no_determinado")
    frases = [f"Son {n} casos del registro ({config.COBERTURA_TEMPORAL}): los conteos describen "
              f"estos casos, no frecuencias generales, y los protocolos de manejo pueden haber "
              f"cambiado desde entonces."]
    if sin_desenlace:
        frases.append(f"En {sin_desenlace} de {n} casos el desenlace no quedó clasificado "
                      f"automáticamente; el registro de manejo de cada caso tiene el detalle.")
    return " ".join(frases)


def sintesis_calculada(casos: list[Resultado]) -> str:
    """Síntesis de "Casos similares" armada con conteos: no puede inventar.

    Es la opción por defecto (config.SINTESIS_LLM apagado): el 3B, escribiendo
    la síntesis en libre, atribuía manejos y desenlaces a casos que no los tenían.
    """
    n = len(casos)

    def contar(clave: str, mapa: dict) -> list[tuple[str, int]]:
        conteo = Counter(c.metadata.get(clave) or "no_determinado" for c in casos)
        return [("sin clasificar" if k == "no_determinado" else _etiqueta(mapa, k).lower(), v)
                for k, v in conteo.most_common()]

    mecanismos = contar("mecanismo_lesion", config.ETIQUETAS_MECANISMO)
    if len(mecanismos) == 1:
        frase_mec = f"{n} casos similares, todos por {mecanismos[0][0]}"
    else:
        frase_mec = f"{n} casos similares: " + ", ".join(f"{m} en {k}" for m, k in mecanismos)

    regiones = Counter(r for c in casos for r in (c.metadata.get("region_anatomica") or []))
    frase_reg = ", ".join(f"{_etiqueta(config.ETIQUETAS_REGION, r).lower()} en {k}"
                          for r, k in regiones.most_common(3))

    manejo = [e for e in extraer(casos, max_por_categoria=4) if e.categoria == "Manejo"]
    frase_man = ", ".join(f"{e.termino} en {len(e.casos)}" for e in manejo)

    frase_des = ", ".join(f"{d} en {k}" for d, k in contar("desenlace", config.ETIQUETAS_DESENLACE))

    frases = [frase_mec + (f"; regiones: {frase_reg}" if frase_reg else "") + "."]
    if frase_man:
        frases.append(f"Manejo documentado: {frase_man}.")
    frases.append(f"Desenlace: {frase_des}.")
    return " ".join(frases)


# ---------------------------------------------------------------------------
# "Qué se hizo": determinista, desde el texto del registro
# ---------------------------------------------------------------------------
_RX_SECCION = re.compile(
    r"(Motivo de consulta|Examen físico|Evolución y hospitalización):\s*(.*?)(?=\n\n(?:Motivo de consulta|"
    r"Examen físico|Evolución y hospitalización):|\Z)",
    re.DOTALL,
)


def _secciones(texto: str) -> dict[str, str]:
    return {m.group(1): " ".join(m.group(2).split()) for m in _RX_SECCION.finditer(texto)}


def _recortar(texto: str, maximo: int) -> str:
    if len(texto) <= maximo:
        return texto
    corte = texto[:maximo].rsplit(" ", 1)[0]
    return corte.rstrip(" ,;:.") + " …"


def seccion_que_se_hizo(casos: list[Resultado], max_manejo: int = 260) -> str:
    """Una viñeta por caso con el manejo CITADO TEXTUALMENTE del registro.

    No pasa por el modelo: lo que se muestra es lo que dice el registro. El
    texto completo está en las tarjetas de caso debajo de la respuesta.
    """
    lineas = [TITULO_QUE_SE_HIZO]
    for c in casos:
        meta = c.metadata
        partes = [_etiqueta(config.ETIQUETAS_MECANISMO, meta.get("mecanismo_lesion") or "no_determinado")]
        if meta.get("region_anatomica"):
            partes.append(", ".join(_etiqueta(config.ETIQUETAS_REGION, r) for r in meta["region_anatomica"]))
        desenlace = meta.get("desenlace") or "no_determinado"
        if desenlace != "no_determinado":
            partes.append(f"desenlace: {_etiqueta(config.ETIQUETAS_DESENLACE, desenlace).lower()}")
        if meta.get("es_pediatrico"):
            partes.append("pediátrico")

        lineas.append(f"- **[caso: {c.record_id}]** {' · '.join(partes)}")
        hallazgos = terminos(c.text)["Hallazgos"]
        if hallazgos:
            lineas.append(f"  - Hallazgos: {', '.join(hallazgos)}")
        manejo = _secciones(c.text).get("Evolución y hospitalización")
        if manejo:
            lineas.append(f"  - Registro de manejo: «{_recortar(manejo, max_manejo)}»")
    return "\n".join(lineas)


def resumen_de_caso(caso: Resultado) -> str:
    return (
        f"Resume el siguiente caso en un párrafo: mecanismo, hallazgos al examen, manejo "
        f"y desenlace, tal como constan. Si algo no consta, dilo. Cita el caso una vez "
        f"con [caso: {caso.record_id}].\n\n"
        f"{_bloque_caso(caso, 1)}"
    )
