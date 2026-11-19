"""
La consulta con que se buscan los manuales (PLAN_CHATBOT_CLINICO.md, fase 1).

El corpus de casos ya trae las abreviaturas expandidas (lo hizo la
anonimización); los manuales no: "HPAF en cráneo" no encuentra "herida por arma
de fuego". Aquí se expanden de forma aumentativa, como en el corpus: "HPAF en
cráneo" → "HPAF (herida por arma de fuego) en cráneo". Y, aparte, el nombre del
mecanismo que la consulta dice con otras palabras ("sujeto baleado"). Solo para
buscar en los manuales (retrieve.candidatos_guias; ABREV_LIBROS,
MECANISMO_LIBROS): la búsqueda de casos no cambia, y así siguen valiendo
eval_rutas.csv y las calibraciones de los casos. La guarda por señal de trauma
mira la consulta original.

El diccionario es el de la anonimización (anonimizacion/config/abreviaturas.yaml,
ABREVIATURAS_YAML), leído como dato: no se importa código de esa carpeta.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from . import config
from .hallazgos import normalizar
from .senal import senal_trauma

# Ambiguas: no se expanden. tx y qx ya dieron errores en el corpus
# (pendiente 6 de CONTEXTO_SESION.md: "Tx (tratamiento) renal" era trauma,
# "Qx (quirúrgico) de II grado" era quemadura); "ha" es casi siempre el verbo
# ("se ha caído"), no hipertensión arterial.
NO_EXPANDIR = {"tx", "qx", "ha"}

# Las que los manuales ya usan (10 veces o más en guias/secciones.jsonl el
# 24/10/2026): expandirlas solo mete ruido en la búsqueda; con "FAST (ecografia
# FAST abdominal)", sin-114 perdió su sección. Dos significan otra cosa en los
# manuales: Fr es el calibre French (tubo de 28 Fr) y AP, anteroposterior.
# tests/test_consulta.py las recuenta si está el archivo.
MINIMO_EN_MANUALES = 10
LAS_USAN_LOS_MANUALES = {"fast", "iv", "gcs", "tc", "rx", "tac", "fr", "ap", "ecg", "tce",
                         "spo2", "uci", "fc"}


# El nombre de cada mecanismo de rag/senal.py, para agregarlo a la búsqueda
# cuando la consulta lo dice con otras palabras: "sujeto baleado" → "sujeto
# baleado (herida por arma de fuego)" (MECANISMO_LIBROS). Una consulta sin señal
# de trauma no cambia: no pesa en las fuera de dominio.
NOMBRES_MECANISMO = {
    "arma_de_fuego": "herida por arma de fuego",
    "cortopunzante": "herida por arma cortopunzante",
    "accidente_transito": "accidente de tránsito",
    "caida": "caída",
    "contundente": "trauma contundente",
    "quemadura": "quemadura",
    "mordedura": "mordedura",
    "cuerpo_extrano": "cuerpo extraño",
    "intoxicacion": "intoxicación",
    "autoinfligido": "lesión autoinfligida",
}


@lru_cache(maxsize=1)
def diccionario() -> dict[str, str]:
    """abreviatura (sin tildes, en minúsculas) -> expansión. Vacío sin el archivo."""
    ruta = Path(config.ABREVIATURAS_YAML)
    if not ruta.exists():
        return {}
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    excluidas = set(datos.get("excluidas") or {}) | NO_EXPANDIR | LAS_USAN_LOS_MANUALES
    return {normalizar(str(k)): str(v) for k, v in (datos.get("abreviaturas") or {}).items()
            if normalizar(str(k)) not in excluidas}


def expandir_abreviaturas(consulta: str) -> str:
    """La consulta con cada abreviatura expandida entre paréntesis, la primera
    vez que aparece. Sin abreviaturas, la misma consulta."""
    dic = diccionario()
    salida, desde, vistas = [], 0, set()
    for m in re.finditer(r"\w+", consulta):
        clave = normalizar(m.group(0))
        if clave in dic and clave not in vistas:
            vistas.add(clave)
            salida.append(f"{consulta[desde:m.end()]} ({dic[clave]})")
            desde = m.end()
    return "".join(salida) + consulta[desde:]


def agregar_mecanismo(consulta: str) -> str:
    """La consulta con el nombre de los mecanismos que nombra con otras
    palabras, entre paréntesis al final. Los que ya están escritos, no."""
    texto = normalizar(consulta)
    nombres = [NOMBRES_MECANISMO[c.split(":", 1)[1]] for c in senal_trauma(consulta)
               if c.startswith("mecanismo:")]
    nuevos = [n for n in nombres if normalizar(n) not in texto]
    return f"{consulta} ({', '.join(nuevos)})" if nuevos else consulta


# --- la consulta en inglés, para el ATLS 11 (TRADUCIR_ATLS11, 18/11/2026) ------
# El reranker (bge-reranker-v2-m3) es multilingüe, pero una consulta en español
# contra un pasaje en inglés puntúa muy por debajo de la misma contra un pasaje
# en español: con tres manuales, el ATLS 11 casi nunca pasaba la guarda. La
# traduce el generador; el código solo se queda con la primera línea y la
# descarta si no parece una traducción (el modelo a veces responde la pregunta:
# "Closed thoracostomy technique step-by-step: 1. Position…").
SISTEMA_TRADUCCION = ("You translate Spanish clinical trauma queries into concise medical English, "
                      "as an ATLS course manual would phrase them. Expand abbreviations. "
                      "Answer with the translation only.")
_traducciones: dict[str, str | None] = {}


def limpiar_traduccion(original: str, crudo: str) -> str | None:
    """La primera línea de lo que devolvió el modelo, o None si no sirve: vacía,
    sin letras, mucho más larga que la consulta o igual a ella."""
    lineas = [l.strip() for l in crudo.splitlines() if l.strip()]
    if not lineas:
        return None
    t = lineas[0].strip(" \"'«»*").rstrip(":").strip()
    if not re.search(r"[A-Za-z]", t) or len(t) > 3 * len(original) + 40:
        return None
    return None if normalizar(t) == normalizar(original) else t


def _pedir_traduccion(consulta: str) -> str:
    from .generate import _ollama               # generate → retrieve → consulta
    return _ollama([{"role": "system", "content": SISTEMA_TRADUCCION},
                    {"role": "user", "content": consulta}],
                   temperatura=0, max_tokens=config.MAX_TOKENS_TRADUCCION,
                   timeout=config.TIMEOUT_TRADUCCION)


def en_ingles(consulta: str) -> str | None:
    """La consulta de los manuales traducida al inglés, para buscar el ATLS 11
    (retrieve.candidatos_guias). None si TRADUCIR_ATLS11 está apagado, si
    Ollama no responde (no se guarda: se reintenta en la consulta siguiente) o
    si la traducción no sirve; entonces el ATLS 11 se busca en español."""
    if not config.TRADUCIR_ATLS11 or not consulta.strip():
        return None
    if consulta not in _traducciones:
        try:
            crudo = _pedir_traduccion(consulta)
        except (OSError, ValueError, KeyError):  # URLError y TimeoutError son OSError
            return None
        if len(_traducciones) >= 1000:
            _traducciones.clear()
        _traducciones[consulta] = limpiar_traduccion(consulta, crudo)
    return _traducciones[consulta]


def para_libros(consulta: str, abreviaturas: bool | None = None, mecanismo: bool | None = None) -> str:
    """La consulta con que se buscan los manuales: con las abreviaturas
    expandidas (ABREV_LIBROS) y el nombre del mecanismo (MECANISMO_LIBROS)."""
    abreviaturas = config.ABREV_LIBROS if abreviaturas is None else abreviaturas
    mecanismo = config.MECANISMO_LIBROS if mecanismo is None else mecanismo
    if abreviaturas:
        consulta = expandir_abreviaturas(consulta)
    if mecanismo:
        consulta = agregar_mecanismo(consulta)
    return consulta
