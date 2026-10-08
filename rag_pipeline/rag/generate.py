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
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import config
from .build_prompt import (
    ROTULO_SINTESIS, ROTULO_SUGERENCIA, ROTULO_SUGERENCIAS, SIN_SUGERENCIAS, SISTEMA, SISTEMA_PROSA,
    SISTEMA_RESUMEN, TITULO_CASOS, TITULO_LIMITES, TITULO_SUGERENCIAS, conclusion_calculada,
    construir_prompt, contexto_elemento, limites_calculados, linea_base, linea_respaldo, para_afinar,
    resumen_de_caso, tabla_casos,
)
from .hallazgos import PROCEDIMIENTOS_QUIRURGICOS, Elemento, elegibles, elemento_mencionado, extraer
from .retrieve import Filtros, Resultado, buscar, confianza
from .verificar import RX_CITA, suavizar, verificar_prosa


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
    relevancia: float | None = None      # prob. máxima del reranker (solo ruta rerank)
    # SALIDA=prosa: True si se mostró la prosa del modelo; False si no pasó la
    # verificación y se mostró la versión determinista (con `motivos`).
    prosa_verificada: bool | None = None
    motivos: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cliente Ollama (urllib: sin dependencias extra)
# ---------------------------------------------------------------------------
# Medido con qwen3 sobre prompts reales: 2,9-3,3 caracteres por token.
# Se estima por lo bajo para que la ventana nunca quede corta.
_CARACTERES_POR_TOKEN = 2.7


def _opciones_contexto(mensajes: list[dict], max_tokens: int) -> dict:
    """num_ctx (y num_gpu) para esta llamada.

    Con la ventana normal se fuerzan las capas en GPU. Si el prompt no cabe,
    se agranda la ventana al múltiplo de 1024 siguiente y se deja que Ollama
    reparta las capas: forzarlas con más contexto puede agotar la VRAM.
    """
    necesario = int(sum(len(m["content"]) for m in mensajes) / _CARACTERES_POR_TOKEN) + max_tokens
    if necesario <= config.NUM_CTX:
        opciones = {"num_ctx": config.NUM_CTX}
        if config.NUM_GPU >= 0:
            opciones["num_gpu"] = config.NUM_GPU
        return opciones
    return {"num_ctx": -(-necesario // 1024) * 1024}


def _ollama(mensajes: list[dict], modelo: str | None = None, temperatura: float | None = None,
            max_tokens: int | None = None, timeout: int | None = None,
            formato: dict | None = None, metricas: dict | None = None) -> str:
    """Una llamada a /api/chat. `formato` = JSON Schema que restringe la salida.
    Si se pasa `metricas` (un dict), se llena con los tiempos y tokens de Ollama."""
    max_tokens = max_tokens or config.MAX_TOKENS
    peticion = {
        "model": modelo or config.MODELO_LLM,
        "messages": mensajes,
        "stream": False,
        # Los modelos con razonamiento (qwen3, qwen3.5) piensan antes de
        # responder y se comen los tokens y el tiempo; aquí no aporta nada.
        "think": False,
        "keep_alive": config.KEEP_ALIVE,
        "options": {
            "temperature": config.TEMPERATURA if temperatura is None else temperatura,
            "num_predict": max_tokens,
            **_opciones_contexto(mensajes, max_tokens),
        },
    }
    if formato is not None:
        peticion["format"] = formato
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/chat", data=json.dumps(peticion).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout or config.TIMEOUT_LLM) as r:
        datos = json.loads(r.read())
    if metricas is not None:
        # Ollama da las duraciones en nanosegundos
        metricas.update({
            "tokens_prompt": datos.get("prompt_eval_count", 0),
            "tokens_salida": datos.get("eval_count", 0),
            "s_prompt": datos.get("prompt_eval_duration", 0) / 1e9,
            "s_salida": datos.get("eval_duration", 0) / 1e9,
            "s_carga": datos.get("load_duration", 0) / 1e9,
            "s_total": datos.get("total_duration", 0) / 1e9,
        })
    return datos["message"]["content"].strip()


def porcentaje_gpu(modelo: str | None = None) -> int | None:
    """% del modelo cargado en VRAM según /api/ps (la columna PROCESSOR de
    `ollama ps`). None si no está cargado o Ollama no responde."""
    modelo = modelo or config.MODELO_LLM
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/ps", timeout=5) as r:
            for m in json.loads(r.read()).get("models", []):
                if m.get("name") == modelo or m.get("model") == modelo:
                    return round(100 * m.get("size_vram", 0) / max(m.get("size", 1), 1))
    except OSError:
        pass
    return None


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
    umbral_rerank: float | None = None,
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
    relevancia = max((c.relevancia for c in casos if c.relevancia is not None), default=None)
    casos = filtrar_relevantes(casos, umbral_rerank)
    if len(casos) < config.MIN_CASOS:
        return Respuesta(
            texto=MENSAJE_ABSTENCION.format(cobertura=config.COBERTURA_TEMPORAL),
            casos=casos, confianza=conf, abstuvo=True, consulta_reescrita=reescrita,
            relevancia=relevancia,
        )

    prosa_verificada, motivos = None, []
    try:
        if config.SALIDA == "prosa" and not config.SINTESIS_LLM:
            prosa = generar_prosa(consulta, casos)
            prosa_verificada, motivos = verificar_prosa(prosa, casos, consulta)
            # Si no pasa, la versión determinista con la elección de las reglas
            # del código: una segunda llamada al modelo duplicaría la espera.
            # Con PROSA_SIEMPRE se muestra aunque no pase; sin prosa no hay qué mostrar.
            mostrar = prosa and (prosa_verificada or config.PROSA_SIEMPRE)
            texto = ensamblar_prosa(prosa, casos) if mostrar else ensamblar("", casos, consulta)
        else:
            texto = ensamblar(generar_eleccion(consulta, casos), casos, consulta)
    except Exception as e:
        return Respuesta(texto="", casos=casos, confianza=conf,
                         consulta_reescrita=reescrita,
                         error=f"No se pudo generar la respuesta: {e}")

    validas, invalidas = verificar_citas(texto, casos)
    return Respuesta(texto=texto, casos=casos, confianza=conf,
                     citas_validas=validas, citas_invalidas=invalidas,
                     consulta_reescrita=reescrita, relevancia=relevancia,
                     prosa_verificada=prosa_verificada, motivos=motivos)


def generar_prosa(consulta: str, casos: list[Resultado], metricas: dict | None = None) -> str:
    """La sugerencia en prosa que escribe el modelo, sin verificar.

    Salida estructurada `{"sugerencia": "..."}`: evita preámbulos, títulos y
    razonamientos alrededor. Sin elementos elegibles no hay nada que el modelo
    pueda nombrar con respaldo: no se le llama y la respuesta es determinista.
    """
    elementos = elegibles(casos, consulta)
    if not elementos:
        return ""
    mensajes = [{"role": "system", "content": SISTEMA_PROSA},
                {"role": "user", "content": construir_prompt(consulta, casos, elementos, prosa=True)}]
    formato = {"type": "object", "properties": {"sugerencia": {"type": "string"}},
               "required": ["sugerencia"]}
    salida = _ollama(mensajes, formato=formato, max_tokens=config.MAX_TOKENS_PROSA, metricas=metricas)
    try:
        datos = json.loads(salida)
        prosa = str(datos.get("sugerencia", "")).strip() if isinstance(datos, dict) else ""
        prosa = suavizar(prosa)
        return prosa[:1].upper() + prosa[1:]   # el 4B suele empezar con "conviene…"
    except json.JSONDecodeError:
        # cortado por MAX_TOKENS_PROSA, o un modelo que ignora el formato
        return ""


def ensamblar_prosa(prosa: str, casos: list[Resultado]) -> str:
    """Prosa verificada + sobre cuánto se apoya. El detalle de cada caso está en
    el apartado de casos de la app."""
    return f"{ROTULO_SUGERENCIA}\n\n{prosa}\n\n{linea_respaldo(casos, 'los casos consultados')}"


def generar_eleccion(consulta: str, casos: list[Resultado], estructurada: bool | None = None,
                     metricas: dict | None = None) -> str:
    """Salida cruda del modelo: la elección de sugerencias (y, con SINTESIS_LLM,
    la síntesis y los límites).

    Estructurada (por defecto): el modelo ve solo los elementos elegibles y
    responde `{"sugerencias": [...]}` restringido por JSON Schema a esa lista.
    Si no hay nada elegible, no se llama al modelo: no hay qué elegir.
    En texto (SALIDA_ESTRUCTURADA=0, o SINTESIS_LLM): viñetas bajo
    TITULO_SUGERENCIAS, como antes del 27/09/2026.
    """
    if estructurada is None:
        estructurada = config.SALIDA_ESTRUCTURADA and not config.SINTESIS_LLM
    if estructurada:
        elementos = elegibles(casos, consulta)
        if not elementos:
            return ""
        prompt = construir_prompt(consulta, casos, elementos, estructurada=True)
        formato = esquema_sugerencias(elementos)
    else:
        prompt, formato = construir_prompt(consulta, casos), None
    mensajes = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": prompt}]
    return _ollama(mensajes, formato=formato, metricas=metricas)


def filtrar_relevantes(casos: list[Resultado], umbral: float | None = None) -> list[Resultado]:
    """Con la ruta rerank, solo pasan los casos que el cross-encoder juzga relevantes.

    Es la segunda guarda de abstención: el coseno filtra lo que no se parece
    a nada del corpus ("capital de Colombia"); el reranker, lo que se parece de
    palabra pero no de caso ("cetoacidosis diabética en urgencias"). Las demás
    rutas no traen relevancia y pasan intactas.
    """
    umbral = config.UMBRAL_RERANK if umbral is None else umbral
    return [c for c in casos if c.relevancia is None or c.relevancia >= umbral]


MIN_SUGERENCIAS = 3
MAX_SUGERENCIAS = 4
# Al completar sugerencias, primero lo que se HIZO: es lo que el usuario pide ver.
_PRIORIDAD = {"Manejo": 0, "Estudios": 1, "Hallazgos": 2}
# Al MOSTRARLAS, en el orden en que se razona un diagnóstico: qué valorar,
# qué estudiar, qué se hizo.
_ORDEN_DIAGNOSTICO = {"Hallazgos": 0, "Estudios": 1, "Manejo": 2}


def _seccion(texto: str, titulo: str) -> str | None:
    """Cuerpo de una sección `### ...` de la salida del modelo, o None si no está."""
    ini = texto.find(titulo)
    if ini < 0:
        return None
    cuerpo = ini + len(titulo)
    fin = texto.find("\n### ", cuerpo)
    return texto[cuerpo:len(texto) if fin < 0 else fin].strip()


def esquema_sugerencias(elementos: list[Elemento]) -> dict:
    """JSON Schema para el campo `format` de Ollama: la salida SOLO puede ser
    una lista de términos de `elementos`. Ollama lo compila a una gramática, así
    que el modelo no puede escribir un término fuera de la lista ni texto libre.
    Los repetidos (la gramática no impone unicidad) se quitan al leer."""
    terminos = [e.termino for e in elementos]
    return {
        "type": "object",
        "properties": {"sugerencias": {
            "type": "array",
            "items": {"type": "string", "enum": terminos},
            "minItems": min(MIN_SUGERENCIAS, len(terminos)),
            # con menos términos que MAX_SUGERENCIAS, el modelo rellenaba
            # repitiendo (medido: "laparotomía" 4 veces con 1 elegible)
            "maxItems": min(MAX_SUGERENCIAS, len(terminos)),
        }},
        "required": ["sugerencias"],
    }


def eleccion_modelo(texto_modelo: str, elementos: list[Elemento]) -> tuple[list[Elemento], int]:
    """(elementos que eligió el modelo y están en la lista, cuántos propuso).

    Lee la salida estructurada (`{"sugerencias": [...]}`) y, si no lo es, las
    viñetas bajo TITULO_SUGERENCIAS (modo texto y SINTESIS_LLM). La diferencia
    entre propuestos y elegidos son las sugerencias descartadas por no tener
    respaldo: lo que mide scripts/comparar_modelos.py.
    """
    try:
        datos = json.loads(texto_modelo)
        propuestas = datos.get("sugerencias") if isinstance(datos, dict) else None
    except (json.JSONDecodeError, TypeError):
        propuestas = None
    if isinstance(propuestas, list):
        por_termino = {e.termino: e for e in elementos}
        # exacto primero (el enum lo garantiza); si un modelo sin gramática
        # escribió algo parecido, la coincidencia del modo texto
        candidatos = [por_termino.get(str(p)) or elemento_mencionado(str(p), elementos)
                      for p in propuestas]
    else:
        cuerpo = _seccion(texto_modelo, TITULO_SUGERENCIAS) or ""
        vinetas = [l for l in cuerpo.split("\n") if l.lstrip().startswith(("-", "*", "•"))]
        propuestas = vinetas
        candidatos = [elemento_mencionado(l, elementos) for l in vinetas]
    elegidos: list[Elemento] = []
    for e in candidatos:
        if e is not None and e not in elegidos:
            elegidos.append(e)
    return elegidos, len(propuestas)


def elegir_sugerencias(texto_modelo: str, casos: list[Resultado], consulta: str = "") -> list[Elemento]:
    """Qué elementos sugerir. Del modelo se toma SOLO la elección, no el texto.

    Una viñeta del modelo que no corresponde a ningún elemento extraído de los
    casos se descarta: es una sugerencia sin respaldo. Si el modelo elige menos
    de MIN_SUGERENCIAS, se completa con los elementos más frecuentes de la
    lista, empezando por el manejo. El conteo y las citas salen siempre de la
    lista: el 3B escribía "4 de 5" citando casos que no documentaban el elemento.

    Reglas más:
      · un hallazgo que la consulta ya menciona o implica no se sugiere
        ("valorar hipoventilación" a quien preguntó por hipoventilación,
        "herida penetrante" a quien preguntó por una herida por arma);
      · si hay manejo documentado, al menos una sugerencia es de manejo: es lo
        que se hizo en los casos;
      · si hay hallazgos documentados, al menos una sugerencia es "valorar…":
        es un asistente al DIAGNÓSTICO. Se prefiere el hallazgo de los casos
        que escalaron (operados, UCI, fallecidos): el modelo tendía a elegir
        solo manejo y dejaba fuera la irritación peritoneal de los operados.
    """
    todos = extraer(casos)
    elementos = elegibles(casos, consulta)
    elegidos, _ = eleccion_modelo(texto_modelo, elementos)
    resto = sorted((e for e in elementos if e not in elegidos),
                   key=lambda e: (_PRIORIDAD[e.categoria], -len(e.casos)))
    elegidos.extend(resto[:max(0, MIN_SUGERENCIAS - len(elegidos))])
    elegidos = elegidos[:MAX_SUGERENCIAS]
    manejo = [e for e in resto if e.categoria == "Manejo" and e not in elegidos]
    if manejo and not any(e.categoria == "Manejo" for e in elegidos):
        elegidos = elegidos[:MAX_SUGERENCIAS - 1] + [manejo[0]]

    hallazgos = [e for e in elementos if e.categoria == "Hallazgos" and e not in elegidos]
    if hallazgos and not any(e.categoria == "Hallazgos" for e in elegidos):
        graves = _casos_escalados(casos, todos)
        mejor = max(hallazgos, key=lambda e: (len(set(e.casos) & graves), len(e.casos)))
        if len(elegidos) < MAX_SUGERENCIAS:
            elegidos.append(mejor)
        else:
            # reemplaza el último que no deje la respuesta sin manejo
            n_manejo = sum(e.categoria == "Manejo" for e in elegidos)
            for i in range(len(elegidos) - 1, -1, -1):
                if elegidos[i].categoria != "Manejo" or n_manejo > 1:
                    elegidos[i] = mejor
                    break
    return elegidos


_ESCALADA = {"cirugía", "UCI", "intubación / ventilación mecánica", *PROCEDIMIENTOS_QUIRURGICOS}


def _casos_escalados(casos: list[Resultado], elementos: list[Elemento]) -> set[str]:
    """Casos que fueron a cirugía o UCI, o fallecieron: sus hallazgos son los que más orientan."""
    ids = {c.record_id for c in casos if c.metadata.get("desenlace") in ("cirugia", "uci", "fallecimiento")}
    return ids | {rid for e in elementos if e.termino in _ESCALADA for rid in e.casos}


def ensamblar(texto_modelo: str, casos: list[Resultado], consulta: str = "") -> str:
    """Respuesta final para el médico, armada por el código desde los casos.

    Es un asistente al diagnóstico: lo primero es la asistencia, fácil de usar.
      1. Sugerencias: elegidas por el modelo; conteo, contexto y citas del
         código. En orden diagnóstico: qué valorar, qué estudiar, qué se hizo.
      2. En síntesis: qué casos son, qué se hizo más, la escalada más grave y
         sobre cuánto se apoya. Solo con config.SINTESIS_LLM la escribe el
         modelo; probado con el 3B, atribuía manejos a casos que no los tenían.
      3. Tabla: un caso por fila con hallazgos, manejo y desenlace.
      4. Para afinar: hallazgos que diferencian los casos y no van ya sugeridos.
      5. Límites de estos casos, en una línea.
    """
    if config.SINTESIS_LLM:
        sintesis = _seccion(texto_modelo, TITULO_CASOS) or ""
        sintesis = f"{ROTULO_SINTESIS} " + "\n\n".join(s for s in (sintesis, linea_base(casos)) if s)
        limites = _seccion(texto_modelo, TITULO_LIMITES)
    else:
        sintesis = conclusion_calculada(casos, consulta)
        limites = limites_calculados(casos)
    # orden diagnóstico; dentro de cada categoría, el orden que eligió el modelo
    sugerencias = sorted(elegir_sugerencias(texto_modelo, casos, consulta),
                         key=lambda e: _ORDEN_DIAGNOSTICO[e.categoria])

    partes = [ROTULO_SUGERENCIAS,
              "\n".join(contexto_elemento(e, casos, consulta) for e in sugerencias)
              if sugerencias else SIN_SUGERENCIAS,
              sintesis, tabla_casos(casos, consulta)]
    afinar = para_afinar(casos, consulta, ya_sugeridos={e.termino for e in sugerencias})
    if afinar:
        partes.append(afinar)
    if limites:
        partes.append(limites)
    return "\n\n".join(partes)


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
        return Respuesta(texto=_ollama(mensajes, max_tokens=config.MAX_TOKENS_RESUMEN),
                         casos=[caso], confianza=1.0,
                         citas_validas=[caso.record_id])
    except Exception as e:
        return Respuesta(texto="", casos=[caso], error=f"No se pudo generar el resumen: {e}")
