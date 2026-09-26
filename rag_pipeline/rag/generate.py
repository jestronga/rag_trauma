"""
Capa de generación anclada, sobre Ollama.

Tres guardas, en orden de importancia:

1. ABSTENCIÓN. Si el coseno del mejor caso no llega al umbral, no se llama al
   modelo. Se responde que no hay casos suficientemente similares. Un RAG que
   siempre responde es un RAG que alucina cuando no sabe.

2. CITAS OBLIGATORIAS. Cada afirmación lleva el identificador del caso que la
   sustenta, y después se VERIFICA que los identificadores citados existan
   entre los recuperados. Una cita inventada se detecta y se marca.

3. SUGERENCIAS CON RESPALDO. El asistente sugiere qué valorar, pero cada
   sugerencia dice en cuántos casos se apoya y los cita; los conteos se le
   entregan calculados (ver build_prompt.py). Sin respaldo, no sugiere.

El corpus solo trae casos completos (apto_rag): sin truncados ni campos
vacíos, que son los que el generador tiende a completar inventando.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import config
from .build_prompt import (
    SIN_SUGERENCIAS, SISTEMA, SISTEMA_RESUMEN, TITULO_CASOS, TITULO_LIMITES,
    TITULO_SUGERENCIAS, construir_prompt, limites_calculados, linea_base, resumen_de_caso,
    seccion_que_se_hizo, sintesis_calculada,
)
from .hallazgos import Elemento, elemento_mencionado, extraer, terminos
from .retrieve import Filtros, Resultado, buscar, confianza

# `[caso: 12345]` es el formato pedido; los modelos pequeños a veces escriben
# `[12345]`. Si no se reconoce, una cita inventada pasaría sin marcarse.
RX_CITA = re.compile(r"\[(?:caso[:\s]+)?([0-9]{1,8})\]", re.IGNORECASE)


@dataclass
class Respuesta:
    texto: str
    casos: list[Resultado] = field(default_factory=list)
    confianza: float = 0.0
    abstuvo: bool = False
    citas_validas: list[str] = field(default_factory=list)
    citas_invalidas: list[str] = field(default_factory=list)
    consulta_reescrita: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Cliente Ollama (urllib: sin dependencias extra)
# ---------------------------------------------------------------------------
def _ollama(mensajes: list[dict], modelo: str | None = None, temperatura: float | None = None,
            max_tokens: int | None = None, timeout: int | None = None) -> str:
    cuerpo = json.dumps({
        "model": modelo or config.MODELO_LLM,
        "messages": mensajes,
        "stream": False,
        "options": {
            "temperature": config.TEMPERATURA if temperatura is None else temperatura,
            "num_predict": max_tokens or config.MAX_TOKENS,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/chat", data=cuerpo,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout or config.TIMEOUT_LLM) as r:
        return json.loads(r.read())["message"]["content"].strip()


def ollama_disponible() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as r:
            modelos = [m["name"] for m in json.loads(r.read()).get("models", [])]
        if not modelos:
            return False, "Ollama responde pero no tiene modelos. Corre: ollama pull " + config.MODELO_LLM
        if config.MODELO_LLM not in modelos:
            return False, (f"Ollama no tiene «{config.MODELO_LLM}». Disponibles: {', '.join(modelos)}.\n"
                           f"Corre: ollama pull {config.MODELO_LLM}")
        return True, f"Ollama ok · {config.MODELO_LLM}"
    except (urllib.error.URLError, OSError) as e:
        return False, f"No hay conexión con Ollama en {config.OLLAMA_URL} ({e}). ¿Está corriendo `ollama serve`?"


# ---------------------------------------------------------------------------
# Reescritura de consulta con historial
# ---------------------------------------------------------------------------
_PROMPT_REESCRITURA = """Reescribe la última pregunta del usuario como una consulta \
autónoma para buscar casos clínicos, resolviendo las referencias al historial.

Reglas:
- Devuelve SOLO la consulta reescrita, sin explicaciones ni comillas.
- Conserva la terminología clínica tal como la escribió el usuario.
- Si la pregunta ya es autónoma, devuélvela sin cambios.

Historial:
{historial}

Última pregunta: {pregunta}

Consulta reescrita:"""


def reescribir(pregunta: str, historial: list[dict]) -> str:
    """En un chat la segunda pregunta es «¿y en pediátricos?».

    Sin resolver esa referencia contra el historial, la recuperación se cae
    justo cuando alguien empieza a jugar con la demo.
    """
    if not historial:
        return pregunta
    previos = historial[-config.VENTANA_HISTORIAL:]
    texto = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in previos)
    try:
        salida = _ollama(
            [{"role": "user",
              "content": _PROMPT_REESCRITURA.format(historial=texto, pregunta=pregunta)}],
            temperatura=0.0, max_tokens=120, timeout=60,
        )
        salida = salida.strip().strip('"').split("\n")[0].strip()
        # si el modelo se va por las ramas, es más seguro la pregunta original
        return salida if 3 <= len(salida) <= 400 else pregunta
    except Exception:
        return pregunta


# ---------------------------------------------------------------------------
# Respuesta
# ---------------------------------------------------------------------------
MENSAJE_ABSTENCION = (
    "No encontré casos en el registro que se parezcan lo suficiente a esa consulta, "
    "así que prefiero no responder a partir de material poco relacionado.\n\n"
    "El registro cubre **atención de trauma {cobertura}**: heridas por arma de fuego y "
    "cortopunzantes, accidentes de tránsito, caídas, quemaduras y trauma contundente. "
    "No contiene patología médica no traumática ni conocimiento clínico general.\n\n"
    "Si la consulta sí es de trauma, prueba describiendo el caso con más detalle "
    "clínico: mecanismo, región anatómica y hallazgos al examen físico."
)


def responder(
    pregunta: str,
    historial: list[dict] | None = None,
    ruta: str | None = None,
    k: int | None = None,
    filtros: Filtros | None = None,
    umbral: float | None = None,
    reescribir_consulta: bool = True,
) -> Respuesta:
    historial = historial or []
    umbral = config.UMBRAL_ABSTENCION if umbral is None else umbral

    consulta = pregunta
    reescrita = None
    if reescribir_consulta and historial:
        reescrita = reescribir(pregunta, historial)
        if reescrita != pregunta:
            consulta = reescrita

    conf = confianza(consulta, filtros)
    if conf < umbral:
        return Respuesta(
            texto=MENSAJE_ABSTENCION.format(cobertura=config.COBERTURA_TEMPORAL),
            casos=[], confianza=conf, abstuvo=True, consulta_reescrita=reescrita,
        )

    casos = buscar(consulta, k=k or config.TOP_K, ruta=ruta, filtros=filtros)
    if len(casos) < config.MIN_CASOS:
        return Respuesta(
            texto=MENSAJE_ABSTENCION.format(cobertura=config.COBERTURA_TEMPORAL),
            casos=casos, confianza=conf, abstuvo=True, consulta_reescrita=reescrita,
        )

    mensajes = [{"role": "system", "content": SISTEMA},
                {"role": "user", "content": construir_prompt(consulta, casos)}]
    try:
        texto = _ollama(mensajes)
    except Exception as e:
        return Respuesta(texto="", casos=casos, confianza=conf,
                         consulta_reescrita=reescrita,
                         error=f"No se pudo generar la respuesta: {e}")

    texto = ensamblar(texto, casos, consulta)
    validas, invalidas = verificar_citas(texto, casos)
    return Respuesta(texto=texto, casos=casos, confianza=conf,
                     citas_validas=validas, citas_invalidas=invalidas,
                     consulta_reescrita=reescrita)


MIN_SUGERENCIAS = 3
MAX_SUGERENCIAS = 4
# Al completar sugerencias, primero lo que se HIZO: es lo que el usuario pide ver.
_PRIORIDAD = {"Manejo": 0, "Estudios": 1, "Hallazgos": 2}


def _seccion(texto: str, titulo: str) -> str | None:
    """Cuerpo de una sección `### ...` de la salida del modelo, o None si no está."""
    ini = texto.find(titulo)
    if ini < 0:
        return None
    cuerpo = ini + len(titulo)
    fin = texto.find("\n### ", cuerpo)
    return texto[cuerpo:len(texto) if fin < 0 else fin].strip()


def _vineta(e: Elemento, n: int) -> str:
    citas = " ".join(f"[caso: {rid}]" for rid in e.casos)
    if e.categoria == "Hallazgos":
        return f"- Valorar {e.termino} — documentado en {len(e.casos)} de {n} casos {citas}"
    return (f"- Considerar {e.termino}, según el cuadro clínico — se realizó en "
            f"{len(e.casos)} de {n} casos {citas}")


def elegir_sugerencias(texto_modelo: str, casos: list[Resultado], consulta: str = "") -> list[Elemento]:
    """Qué elementos sugerir. Del modelo se toma SOLO la elección, no el texto.

    Una viñeta del modelo que no corresponde a ningún elemento extraído de los
    casos se descarta: es una sugerencia sin respaldo. Si el modelo elige menos
    de MIN_SUGERENCIAS, se completa con los elementos más frecuentes de la
    lista, empezando por el manejo. El conteo y las citas salen siempre de la
    lista: el 3B escribía "4 de 5" citando casos que no documentaban el elemento.

    Dos reglas más: un hallazgo que la consulta ya menciona no se sugiere
    ("valorar hipoventilación" a quien preguntó por hipoventilación), y si
    hay manejo documentado, al menos una sugerencia es de manejo: es lo que
    se hizo en los casos.
    """
    ya_dicho = terminos(consulta)["Hallazgos"]
    elementos = [e for e in extraer(casos)
                 if not (e.categoria == "Hallazgos" and e.termino in ya_dicho)]
    cuerpo = _seccion(texto_modelo, TITULO_SUGERENCIAS) or ""
    elegidos: list[Elemento] = []
    for linea in cuerpo.split("\n"):
        if linea.lstrip().startswith(("-", "*", "•")):
            e = elemento_mencionado(linea, elementos)
            if e is not None and e not in elegidos:
                elegidos.append(e)
    resto = sorted((e for e in elementos if e not in elegidos),
                   key=lambda e: (_PRIORIDAD[e.categoria], -len(e.casos)))
    elegidos.extend(resto[:max(0, MIN_SUGERENCIAS - len(elegidos))])
    elegidos = elegidos[:MAX_SUGERENCIAS]
    manejo = [e for e in resto if e.categoria == "Manejo" and e not in elegidos]
    if manejo and not any(e.categoria == "Manejo" for e in elegidos):
        elegidos = elegidos[:MAX_SUGERENCIAS - 1] + [manejo[0]]
    return elegidos


def ensamblar(texto_modelo: str, casos: list[Resultado], consulta: str = "") -> str:
    """Respuesta final, armada por el código desde los casos:

      · Síntesis y límites: calculados con conteos (build_prompt.py). Solo con
        config.SINTESIS_LLM se usan los que escribió el modelo; probado con el
        3B, la síntesis libre atribuía manejos y desenlaces a casos que no los
        tenían.
      · "Qué se hizo": manejo citado textualmente. Si el modelo escribió uno
        propio, se descarta: es donde más inventaba.
      · Sugerencias: elegidas por el modelo, redactadas y citadas por el código.
    """
    n = len(casos)
    if config.SINTESIS_LLM:
        sintesis = _seccion(texto_modelo, TITULO_CASOS) or ""
        sintesis = "\n".join(s for s in (sintesis, linea_base(casos)) if s)
        limites = _seccion(texto_modelo, TITULO_LIMITES)
    else:
        sintesis = sintesis_calculada(casos)
        limites = limites_calculados(casos)
    sugerencias = elegir_sugerencias(texto_modelo, casos, consulta)

    partes = [TITULO_CASOS, sintesis]
    partes.append("\n" + seccion_que_se_hizo(casos))
    partes.append("\n" + TITULO_SUGERENCIAS)
    partes.extend(_vineta(e, n) for e in sugerencias)
    if not sugerencias:
        partes.append(SIN_SUGERENCIAS)
    if limites:
        partes += ["\n" + TITULO_LIMITES, limites]
    return "\n".join(partes)


def verificar_citas(texto: str, casos: list[Resultado]) -> tuple[list[str], list[str]]:
    """Comprueba que cada identificador citado exista entre los recuperados.

    Es barato y atrapa el fallo más vergonzoso en una demo: una cita inventada.
    """
    permitidos = {c.record_id for c in casos}
    citados = {m.group(1) for m in RX_CITA.finditer(texto)}
    return sorted(citados & permitidos), sorted(citados - permitidos)


def resumir_caso(record_id: str) -> Respuesta:
    """Resumen de un caso concreto: trazabilidad, con el dato crudo a la vista."""
    from .retrieve import por_id

    caso = por_id(record_id)
    if caso is None:
        return Respuesta(texto=f"No existe ningún caso con el identificador **{record_id}** "
                               f"en el corpus indexado.", abstuvo=True)
    mensajes = [{"role": "system", "content": SISTEMA_RESUMEN},
                {"role": "user", "content": resumen_de_caso(caso)}]
    try:
        return Respuesta(texto=_ollama(mensajes), casos=[caso], confianza=1.0,
                         citas_validas=[caso.record_id])
    except Exception as e:
        return Respuesta(texto="", casos=[caso], error=f"No se pudo generar el resumen: {e}")
