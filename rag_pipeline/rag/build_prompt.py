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
  · Las sugerencias las ELIGE el modelo de la lista de elementos elegibles
    (hallazgos.elegibles). Por defecto responde JSON restringido por esquema
    a esa lista (INSTRUCCION_ESTRUCTURADA), así que no puede elegir nada sin
    respaldo; generate.ensamblar escribe el texto, el conteo y las citas. Con
    SALIDA_ESTRUCTURADA=0 vuelve a las viñetas de PLANTILLA, de las que
    generate.py descarta las que no corresponden a ningún elemento.
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
from .retrieve import FragmentoGuia, Resultado

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


def construir_prompt(consulta: str, casos: list[Resultado], elementos: list[Elemento] | None = None,
                     estructurada: bool = False, prosa: bool = False) -> str:
    """`elementos`: la lista que el modelo puede elegir. Por defecto, todo lo que
    extrae hallazgos.py; con salida estructurada o prosa, `hallazgos.elegibles`
    (la misma lista que el enum del esquema).

    En prosa no van los DATOS CALCULADOS: son conteos, y la prosa no debe
    tenerlos (el verificador rechaza los que no cuadran)."""
    if elementos is None:
        elementos = extraer(casos)
    bloques = "\n\n".join(_bloque_caso(c, i) for i, c in enumerate(casos, start=1))
    datos = ("" if prosa else
             f"DATOS CALCULADOS SOBRE ESTOS CASOS (úsalos tal cual para los conteos):\n"
             f"{resumen_calculado(casos)}\n\n")
    return (
        f"CONSULTA:\n{consulta}\n\n"
        f"CASOS SIMILARES DEL REGISTRO ({len(casos)}):\n\n{bloques}\n\n"
        f"{datos}"
        f"ELEMENTOS DOCUMENTADOS EN LOS CASOS (única fuente de las sugerencias):\n"
        f"{formatear(elementos, len(casos))}\n\n"
        f"{_instruccion_salida(estructurada, prosa)}"
    )


# Con salida estructurada el modelo solo ELIGE: el texto de cada sugerencia, el
# conteo y las citas los escribe generate.ensamblar. Las reglas de "al menos un
# hallazgo y un manejo" las vuelve a imponer el código; decírselas al modelo
# evita que el código tenga que corregir su elección.
INSTRUCCION_ESTRUCTURADA = """\
Elige qué elementos de la lista sugerir a quien consulta y responde SOLO con JSON:
{"sugerencias": ["elemento", "elemento", "elemento"]}
- Entre 3 y 4 elementos, escritos exactamente como en la lista (sin conteos ni citas).
- En orden de utilidad para la consulta: primero lo que más orienta el diagnóstico.
- Si la lista los tiene, incluye al menos un hallazgo a valorar y al menos un manejo."""


# --- Sugerencia en prosa (SALIDA=prosa) ---------------------------------------
# El modelo redacta la sugerencia como un asistente médico; verificar.py la
# revisa oración por oración y, si no pasa, se muestra la versión determinista.
# Por eso las reglas piden exactamente lo que el verificador comprueba: solo
# elementos de la lista, cita en cada oración que los nombra, sin conteos.
SISTEMA_PROSA = f"""Eres un asistente de apoyo al diagnóstico en trauma para personal de salud. \
Ante una presentación clínica, comparas con casos similares del registro y redactas una \
sugerencia breve basada en lo que se hizo en ellos. No reemplazas el juicio clínico de quien \
consulta.

{_REGLAS_COMUNES}

CÓMO REDACTAR
- Solo puedes nombrar hallazgos, estudios y manejos que estén en la lista "ELEMENTOS \
DOCUMENTADOS EN LOS CASOS". Nada más, aunque lo sepas.
- Cada oración que nombra un hallazgo, un estudio o un manejo lleva al final las citas de los \
casos donde consta, copiadas de la lista.
- No escribas cantidades ("dos de ellos", "3 de 5", "la mayoría", "todos"): quien lee tiene \
los casos a la vista.
- Redacta como orientación ("conviene valorar", "en casos similares se hizo"), nunca como \
orden ("se debe", "hay que")."""

INSTRUCCION_PROSA = """\
Redacta la sugerencia para quien consulta y responde SOLO con JSON:
{"sugerencia": "texto"}
- Entre 3 y 5 oraciones, en este orden: qué hallazgos valorar, qué estudios considerar y qué \
manejo se hizo en los casos similares.
- Empieza por lo que más orienta el diagnóstico en esta consulta.
- Cada oración que nombra un elemento de la lista termina con sus citas [caso: ID].
- Usa "conviene" o "se puede considerar"; nunca "se debe", "debe" ni "hay que".
- No agregues razones, condiciones ni detalles que no estén escritos en los casos \
("si no hay…", "por presentar…").
- Sin títulos, viñetas ni cantidades."""


# --- Con los libros de trauma (PLAN_INDEXAR_LIBROS.md, fase 5) -----------------
# Dos fuentes con el mismo peso y dos tipos de afirmación que nunca se mezclan:
# lo que se hizo en casos similares ([caso: ID]) y lo que recomiendan los
# manuales ([guía: ID], sin conteos). verificar.py revisa cada oración contra
# su fuente. Si solo pasó la guarda de los casos, el prompt es el de arriba,
# sin cambios; estos son para cuando hay fragmentos.
_REGLAS_DOS_FUENTES = f"""\
DOS FUENTES
Trabajas SOLO con lo que se te entrega: casos del registro de trauma \
({config.COBERTURA_TEMPORAL}) de un hospital de Colombia y fragmentos de dos manuales, el ATLS \
(en inglés) y el manual de manejo inicial del HUV. No uses conocimiento médico general ni otras \
guías. Si algo no aparece en los casos ni en los fragmentos, no existe para ti.

PROHIBIDO INVENTAR
- No menciones síntomas, medicamentos, dosis, estudios, procedimientos ni diagnósticos \
que no estén escritos en los casos o en los fragmentos.
- No completes datos que falten. Si algo no consta, dilo.
- Los datos personales aparecen como marcadores ([FECHA], [EDAD:30-39], [INSTITUCION], \
[MUNICIPIO], [MEDICO]). Son información no disponible: no intentes deducirlos.

CITAS
- Lo que viene de un caso lleva su identificador: [caso: 12345].
- Lo que viene de un manual lleva el identificador de la cabecera de su fragmento, en la forma \
[guía: LIBRO-00-000].
- Una oración cita casos o cita manuales, nunca las dos cosas.
- Nunca cites un identificador que no esté entre los entregados.

ESTILO
Español de Colombia, registro clínico, directo. Lo que dice el ATLS se escribe en español."""

SISTEMA_PROSA_GUIAS = f"""Eres un asistente de apoyo al diagnóstico en trauma para personal de \
salud. Ante una presentación clínica, redactas una sugerencia breve con dos miradas: lo que se \
hizo en casos similares del registro y lo que recomiendan los manuales. No reemplazas el juicio \
clínico de quien consulta.

{_REGLAS_DOS_FUENTES}

CÓMO REDACTAR
- De los casos solo puedes nombrar hallazgos, estudios y manejos que estén en la lista \
"ELEMENTOS DOCUMENTADOS EN LOS CASOS".
- De los manuales solo puedes decir lo que está escrito en los fragmentos, sin agregar nada \
aunque lo sepas.
- No escribas cantidades de casos ("dos de ellos", "la mayoría", "todos"): quien lee tiene los \
casos a la vista.
- Redacta como orientación ("conviene valorar", "en casos similares se hizo", "el ATLS \
recomienda"), nunca como orden ("se debe", "hay que")."""

INSTRUCCION_PROSA_GUIAS = """\
Redacta la sugerencia para quien consulta y responde SOLO con JSON:
{"sugerencia": "texto"}
- Primero, de 2 a 4 oraciones sobre los casos similares, en este orden: qué hallazgos valorar, \
qué estudios considerar y qué manejo se hizo. Cada oración que nombra un elemento de la lista \
termina con sus citas [caso: ID].
- Después, de 1 a 3 oraciones sobre lo que recomiendan los manuales para esta consulta. Cada una \
empieza por "El ATLS recomienda" o "El manual del HUV recomienda", según el manual del fragmento \
que cita, y termina con su cita [guía: ID].
- Cada oración habla de una sola fuente. No compares los casos con los manuales ni juzgues lo que \
se hizo: quien lee tiene las dos miradas a la vista.
- Cifras solo si están escritas en el fragmento que citas.
- Usa "conviene", "se puede considerar" o "recomienda"; nunca "se debe", "debe" ni "hay que".
- No agregues razones, condiciones ni detalles que no estén escritos ("si no hay…", "por \
presentar…").
- Sin títulos ni viñetas."""

INSTRUCCION_PROSA_SOLO_GUIAS = """\
En el registro no hay casos suficientemente similares: la sugerencia sale solo de los manuales.
Redacta la sugerencia para quien consulta y responde SOLO con JSON:
{"sugerencia": "texto"}
- De 2 a 4 oraciones sobre lo que recomiendan los manuales para esta consulta. Cada una empieza \
por "El ATLS recomienda" o "El manual del HUV recomienda", según el manual del fragmento que cita, \
y termina con su cita [guía: ID].
- No hables de casos.
- Cifras solo si están escritas en el fragmento que citas.
- Usa "conviene", "se puede considerar" o "recomienda"; nunca "se debe", "debe" ni "hay que".
- No agregues razones, condiciones ni detalles que no estén escritos.
- Sin títulos ni viñetas."""

# Con pasajes del ATLS 10.ª ed. (PLAN_INDEXAR_ATLS10.md, fase 5): tres
# manuales, y la oración nombra la edición (decisión 3), porque el verificador
# rechaza "el ATLS recomienda" con una cita del ATLS 10. Sin pasajes del ATLS 10,
# los textos de arriba sin cambios.
_DOS_MANUALES = "fragmentos de dos manuales, el ATLS (en inglés) y el manual de manejo inicial del HUV"
_TRES_MANUALES = ("fragmentos de manuales: la 11.ª edición del ATLS (en inglés), la 10.ª edición del "
                  "ATLS (2018, la edición anterior, en una traducción no oficial al español) y el manual "
                  "de manejo inicial del HUV")
_COMO_EMPIEZA = '"El ATLS recomienda" o "El manual del HUV recomienda"'
_COMO_EMPIEZA_TRES = ('"La 11.ª edición del ATLS recomienda", "La 10.ª edición del ATLS recomienda" o '
                      '"El manual del HUV recomienda"')
_NO_COMPARES = "No compares los casos con los manuales"
_NO_COMPARES_TRES = "No compares los casos con los manuales ni una edición del ATLS con la otra"
# La traducción de la 10.ª edición es imperativa y el modelo copiaba sus
# órdenes (medido el 23/10: 6 rechazos por "orden" en 41 consultas, 2 sin ella)
_TONO = '- Usa "conviene", "se puede considerar" o "recomienda"; nunca "se debe", "debe" ni "hay que".'
_TONO_TRES = (_TONO + ' Aunque el fragmento lo diga como orden ("debe", "se requiere"), escríbelo como '
              'recomendación.')


def _con_tres_manuales(texto: str) -> str:
    for viejo, nuevo in ((_DOS_MANUALES, _TRES_MANUALES), (_COMO_EMPIEZA, _COMO_EMPIEZA_TRES),
                         (_NO_COMPARES, _NO_COMPARES_TRES), (_TONO, _TONO_TRES)):
        texto = texto.replace(viejo, nuevo)
    return texto


SISTEMA_PROSA_GUIAS_ATLS10 = _con_tres_manuales(SISTEMA_PROSA_GUIAS)
INSTRUCCION_PROSA_GUIAS_ATLS10 = _con_tres_manuales(INSTRUCCION_PROSA_GUIAS)
INSTRUCCION_PROSA_SOLO_GUIAS_ATLS10 = _con_tres_manuales(INSTRUCCION_PROSA_SOLO_GUIAS)


def con_atls10(fragmentos: list[FragmentoGuia]) -> bool:
    return any(f.fuente == "ATLS10" for f in fragmentos)


def sistema_prosa_guias(fragmentos: list[FragmentoGuia]) -> str:
    return SISTEMA_PROSA_GUIAS_ATLS10 if con_atls10(fragmentos) else SISTEMA_PROSA_GUIAS


# Los apartados de la respuesta (generate.armar_secciones): uno por fuente,
# cada uno con su información y, en la app, su desplegable de texto completo.
ROTULO_CASOS = "### Casos similares del registro"
ROTULO_MANUALES = "### Manuales de trauma (ATLS y HUV)"
# Decisión 3: si no coinciden, no se elige un ganador. Lo dice el código, no el
# modelo: medido el 17/10 (21 consultas), la oración de comparación que
# escribía el 4B nombraba elementos sin cita y juzgaba la atención recibida
# (ejemplos en PLAN_INDEXAR_LIBROS.md, fase 5; no van aquí porque todo lo
# escrito en este archivo cuenta como vocabulario permitido en verificar.py).
AVISO_DOS_FUENTES = ("*Lo que se hizo en casos del registro ({cobertura}) y lo que recomiendan "
                     "los manuales pueden no coincidir: el asistente muestra las dos miradas y no "
                     "elige entre ellas.*").format(cobertura=config.COBERTURA_TEMPORAL)
AVISO_SOLO_CASOS =("*Los manuales (ATLS y manual del HUV) no tratan esta situación: la sugerencia "
                    "sale solo de los casos del registro.*")
AVISO_SOLO_GUIAS = ("*No hay casos suficientemente similares en el registro: la sugerencia sale "
                    "solo de los manuales.*")
# Decisiones 1 y 4 de PLAN_INDEXAR_ATLS10.md: lo dice el código, no el modelo,
# y no elige entre ediciones (como no elige entre casos y manuales).
AVISO_ATLS10 = ("*Los pasajes de la 10.ª edición del ATLS (2018) son de la edición anterior, en una "
                "traducción no oficial al español.*")
AVISO_EDICIONES = ("*Los pasajes del ATLS son de dos ediciones: la 11.ª (en inglés) y la 10.ª (2018), "
                   "anterior y en una traducción no oficial al español. Pueden no coincidir: el "
                   "asistente no elige entre ellas.*")


def aviso_ediciones(fragmentos: list[FragmentoGuia]) -> str:
    """El aviso del apartado de los manuales si hay pasajes del ATLS 10 (vacío si no)."""
    fuentes = {f.fuente for f in fragmentos}
    if "ATLS10" not in fuentes:
        return ""
    return AVISO_EDICIONES if "ATLS" in fuentes else AVISO_ATLS10


def agrupar_guias(fragmentos: list[FragmentoGuia]) -> dict[str, list[FragmentoGuia]]:
    """seccion_id -> sus fragmentos, en orden de lectura; las secciones, en el
    orden de su mejor fragmento. La cita es por sección ([guía: HUV-11-003]):
    dos ventanas de una misma sección son un solo pasaje para quien lee, y un
    identificador corto se copia mejor."""
    grupos: dict[str, list[FragmentoGuia]] = {}
    for f in fragmentos:
        grupos.setdefault(f.seccion_id, []).append(f)
    return {s: sorted(fs, key=lambda f: f.id) for s, fs in grupos.items()}


def _sin_ruta(f: FragmentoGuia) -> str:
    """El texto del fragmento sin la ruta de títulos del inicio: va una vez, en la cabecera."""
    primera, _, resto = f.texto_contexto.partition("\n")
    return resto if primera.startswith(f"{f.fuente} › ") else f.texto_contexto


def _unir(textos: list[str], maximo: int = 80) -> str:
    """Une ventanas consecutivas quitando el solape (fijo-256 repite ~32 tokens)."""
    salida: list[str] = []
    for texto in textos:
        nuevas = texto.split()
        solape = next((m for m in range(min(maximo, len(salida), len(nuevas)), 0, -1)
                       if salida[-m:] == nuevas[:m]), 0)
        salida += nuevas[solape:]
    return " ".join(salida)


def texto_guia(fragmentos: list[FragmentoGuia]) -> str:
    """El pasaje de una sección: sus fragmentos de texto unidos y sus tablas aparte."""
    texto = _unir([_sin_ruta(f) for f in fragmentos if f.tipo != "tabla"])
    tablas = [_sin_ruta(f) for f in fragmentos if f.tipo == "tabla"]
    return "\n\n".join([t for t in [texto] if t] + tablas)


def cabecera_guia(fragmentos: list[FragmentoGuia]) -> str:
    """Libro, capítulo, títulos y páginas impresas: lo que permite ir al libro."""
    f = fragmentos[0]
    libro = config.NOMBRES_LIBROS.get(f.fuente, f.fuente)
    # los apéndices del ATLS 10 van numerados del 14 al 20 por dentro: "cap.
    # 14 Apéndice A" confundiría al buscarlo en el libro
    capitulo = (f"cap. {f.capitulo_num} {f.capitulo}" if f.capitulo_num
                and not f.capitulo.startswith("Apéndice") else f.capitulo)
    ini = min(x.pagina_ini for x in fragmentos)
    fin = max(x.pagina_fin for x in fragmentos)
    paginas = f"p. {ini}" if ini == fin else f"pp. {ini}–{fin}"
    return f"{libro} · {' › '.join([capitulo, *f.titulos])} · {paginas}"


def bloque_guias(fragmentos: list[FragmentoGuia]) -> str:
    bloques = []
    for sid, fs in agrupar_guias(fragmentos).items():
        idioma = " (en inglés)" if fs[0].idioma == "en" else ""
        bloques.append(f"--- [guía: {sid}] {cabecera_guia(fs)}{idioma} ---\n{texto_guia(fs)}")
    return "\n\n".join(bloques)


def construir_prompt_guias(consulta: str, casos: list[Resultado], elementos: list[Elemento],
                           guias: list[FragmentoGuia]) -> str:
    """Prompt de la prosa con los manuales. Sin casos (no pasaron su guarda),
    solo los fragmentos y la instrucción de una sola fuente."""
    partes = [f"CONSULTA:\n{consulta}"]
    if casos:
        bloques = "\n\n".join(_bloque_caso(c, i) for i, c in enumerate(casos, start=1))
        partes += [f"CASOS SIMILARES DEL REGISTRO ({len(casos)}):\n\n{bloques}",
                   f"ELEMENTOS DOCUMENTADOS EN LOS CASOS (única fuente de lo que digas sobre los "
                   f"casos):\n{formatear(elementos, len(casos))}"]
    if con_atls10(guias):
        instruccion = INSTRUCCION_PROSA_GUIAS_ATLS10 if casos else INSTRUCCION_PROSA_SOLO_GUIAS_ATLS10
    else:
        instruccion = INSTRUCCION_PROSA_GUIAS if casos else INSTRUCCION_PROSA_SOLO_GUIAS
    partes += [f"FRAGMENTOS DE LOS MANUALES ({len(agrupar_guias(guias))}):\n\n{bloque_guias(guias)}",
               instruccion]
    return "\n\n".join(partes)


def aviso_pasajes(fragmentos: list[FragmentoGuia]) -> str:
    """Lo que dice el apartado de los manuales cuando no hay prosa sobre ellos:
    dónde ver los pasajes. Libro, capítulo, página y texto de cada uno están en
    el desplegable de la app (la lista en la respuesta se quitó: pedido del
    usuario, 18/10)."""
    n = len(agrupar_guias(fragmentos))
    if n == 1:
        return ("*Hay un pasaje de los manuales relacionado con la consulta: su texto está en "
                "«Fragmentos de los manuales».*")
    return (f"*Hay {n} pasajes de los manuales relacionados con la consulta: el texto de cada "
            f"uno está en «Fragmentos de los manuales».*")


def _instruccion_salida(estructurada: bool = False, prosa: bool = False) -> str:
    """Con SINTESIS_LLM apagado el modelo solo elige sugerencias; síntesis y
    límites los arma el código (sintesis_calculada, limites_calculados)."""
    if prosa:
        return INSTRUCCION_PROSA
    if estructurada:
        return INSTRUCCION_ESTRUCTURADA
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

    return f"{ROTULO_SINTESIS} {' '.join(frases)}\n\n{linea_respaldo(casos)}"


def linea_respaldo(casos: list[Resultado], revisar: str = "la tabla") -> str:
    """Sobre cuánto se apoya la respuesta: n y rango de similitud, y el aviso de
    similitud baja si el mejor caso no llega a SIMILITUD_BAJA. `revisar` dice
    dónde mirar los casos: la tabla, o el apartado de casos de la app."""
    n = len(casos)
    sims = [c.score_norm for c in casos if c.score_norm > 0]
    respaldo = f"Respaldo: {n} casos del registro"
    if sims:
        respaldo += f" · similitud {min(sims):.2f}–{max(sims):.2f}".replace(".", ",")
    if len(sims) < n:
        respaldo += f" · {n - len(sims)} hallado(s) solo por palabras clave"
    salida = f"*{respaldo}.*"
    if not sims or max(sims) < config.SIMILITUD_BAJA:
        salida += ("\n\n**Similitud baja:** estos casos pueden no corresponder a la consulta. "
                   f"Revisar {revisar} antes de usarlos.")
    return salida


# La línea "*Respaldo: …*" de linea_respaldo, para quitarla de la pantalla
# (app.py, pedido del usuario para la demo, 19/10). El aviso de similitud baja
# va aparte y se sigue mostrando.
RX_LINEA_RESPALDO = re.compile(r"^\*Respaldo: [^\n]*\*$\n*", re.M)


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
