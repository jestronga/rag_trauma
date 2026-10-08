"""
Fragmentación de las secciones de los libros (fase 2 de PLAN_INDEXAR_LIBROS.md).

Varias configuraciones, para elegir con scripts/comparar_fragmentacion.py la
que mejor recupera. Reglas comunes a todas:

  · un fragmento nunca cruza secciones (hereda su `seccion_id`, que es la
    unidad con la que se evalúa: así las configuraciones son comparables);
  · las tablas y recuadros van en fragmentos propios; si no caben, se parten
    por filas y cada parte repite el pie;
  · ningún fragmento pasa de LIMITE tokens de bge-m3: el encoder trunca a 512
    sin avisar, y lo que queda fuera no se recupera nunca;
  · salvo `ruta=False`, cada fragmento empieza con la ruta de títulos
    ("ATLS › Disability… › TRAUMATIC BRAIN INJURY"), que le dice al encoder y
    al modelo de qué trata un fragmento que solo dice "Treatment is…".

El contador de tokens se inyecta (`contar`), para que los tests no carguen el
tokenizador; `contador_bge()` da el real.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Callable

import numpy as np

from . import config

LIMITE = 510            # 512 de bge-m3 menos [CLS] y [SEP]

CONFIGS: dict[str, dict] = {
    # la sección entera; si no cabe, en partes parejas por párrafo
    "seccion": {"estrategia": "seccion", "maximo": LIMITE},
    # ventanas de tokens que ignoran los párrafos
    "fijo-256": {"estrategia": "fijo", "maximo": 256, "solape": 32},
    "fijo-512": {"estrategia": "fijo", "maximo": LIMITE, "solape": 64},
    # párrafos agrupados hasta ~350 tokens, solapando el último párrafo
    "parrafos-350": {"estrategia": "parrafos", "maximo": 350, "solape": 1},
    "parrafos-350-sin-ruta": {"estrategia": "parrafos", "maximo": 350, "solape": 1, "ruta": False},
    # small-to-big: se indexan trozos de ~128 tokens; al prompt va su padre
    # (la sección, o su parte de ≤ 512 en la configuración "seccion")
    "oraciones-128": {"estrategia": "oraciones", "maximo": 128},
    # corta entre oraciones donde cae la similitud de bge-m3
    "semantico": {"estrategia": "semantico", "maximo": 450, "percentil": 90, "minimo": 50},
}

Contador = Callable[[str], int]
Embebedor = Callable[[list[str]], np.ndarray]


def contador_bge() -> Contador:
    """Cuenta tokens con el tokenizador de bge-m3, sin los especiales."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.MODELO_EMBEDDING)

    @lru_cache(maxsize=500_000)
    def contar(texto: str) -> int:
        return len(tok(texto, add_special_tokens=False)["input_ids"])

    return contar


# ---------------------------------------------------------------------------
# Oraciones y palabras
# ---------------------------------------------------------------------------
# Corte tras . ! ? seguido de mayúscula, número, viñeta o signo de apertura.
_CORTE = re.compile(r"(?<=[.!?])\s+(?=[\"“(¿¡•●]?[A-ZÁÉÍÓÚÑ0-9•●])")
# Abreviaturas tras las que no se corta ("et al. Smith", "Fig. 3", "p. ej. La").
_ABREV = re.compile(
    r"(\bet al|\bfig|\bfigs|\bp\. ej|\be\.g|\bi\.e|\bdr|\bdra|\bvs|\baprox|\bapprox|\bno|\bcf|\bsr|\bsra)\.$",
    re.I)


def oraciones(texto: str) -> list[str]:
    """Parte en oraciones. Cada línea (elemento de lista) es al menos una."""
    salida: list[str] = []
    for linea in texto.split("\n"):
        linea = linea.strip()
        if not linea:
            continue
        propias: list[str] = []
        for parte in _CORTE.split(linea):
            if propias and _ABREV.search(propias[-1]):
                propias[-1] += " " + parte
            else:
                propias.append(parte)
        salida += propias
    return salida


def _palabras(texto: str) -> list[str]:
    """Palabras con su espacio: unirlas reconstruye el texto exacto."""
    return re.findall(r"\S+\s*", texto)


def _por_palabras(texto: str, presupuesto: int, contar: Contador, solape: int = 0) -> list[str]:
    """Ventanas de palabras de hasta `presupuesto` tokens, con `solape` tokens
    repetidos entre ventanas consecutivas."""
    palabras = _palabras(texto)
    tams = [contar(p.strip()) for p in palabras]
    ventanas, inicio = [], 0
    while inicio < len(palabras):
        fin, n = inicio, 0
        while fin < len(palabras) and (n + tams[fin] <= presupuesto or fin == inicio):
            n += tams[fin]
            fin += 1
        ventanas.append("".join(palabras[inicio:fin]).strip())
        if fin >= len(palabras):
            break
        siguiente, arrastre = fin, 0
        while solape and siguiente - 1 > inicio and arrastre + tams[siguiente - 1] <= solape:
            siguiente -= 1
            arrastre += tams[siguiente]
        inicio = siguiente
    return ventanas


def _cabe(unidades: list[str], presupuesto: int, contar: Contador) -> list[str]:
    """Garantiza que cada unidad quepa: parte por oraciones y, si una oración
    sola no cabe, por palabras."""
    salida = []
    for u in unidades:
        if contar(u) <= presupuesto:
            salida.append(u)
            continue
        for o in oraciones(u):
            salida += [o] if contar(o) <= presupuesto else _por_palabras(o, presupuesto, contar)
    return salida


def _empacar(unidades: list[str], presupuesto: int, contar: Contador, solape: int = 0,
             sep: str = "\n\n") -> list[str]:
    """Agrupa unidades consecutivas sin pasar de `presupuesto` tokens. Con
    `solape=n`, cada grupo empieza con las últimas n unidades del anterior
    (si caben). Las unidades deben caber solas (ver `_cabe`)."""
    grupos: list[list[str]] = []
    actual: list[str] = []
    i = 0
    while i < len(unidades):
        candidato = sep.join(actual + [unidades[i]])
        if actual and contar(candidato) > presupuesto:
            grupos.append(actual)
            arrastre = actual[-solape:] if solape and len(actual) > solape else []
            actual = arrastre if contar(sep.join(arrastre + [unidades[i]])) <= presupuesto else []
            continue
        actual.append(unidades[i])
        i += 1
    if actual:
        grupos.append(actual)
    return [sep.join(g) for g in grupos]


# ---------------------------------------------------------------------------
# Piezas comunes
# ---------------------------------------------------------------------------
def ruta(seccion: dict) -> str:
    return " › ".join([seccion["fuente"], seccion["capitulo"], *seccion["titulos"]])


def _tabla(anexo: str, presupuesto: int, contar: Contador) -> list[str]:
    """Una tabla entera si cabe; si no, por filas, y cada parte empieza con el
    título de la tabla (la primera oración del pie, como mucho un cuarto del
    presupuesto). El resto del pie va como una fila más: los pies del ATLS
    llegan a 90 tokens ("Figure 1-2: National Guideline for the Field
    Triage…") y repetidos enteros no dejaban lugar a las filas."""
    if contar(anexo) <= presupuesto:
        return [anexo]
    pie, *filas = anexo.split("\n")
    titulo, *resto_pie = oraciones(pie) or [pie]
    if contar(titulo) > presupuesto // 4:
        titulo = _por_palabras(titulo, presupuesto // 4, contar)[0] + " …"
        resto_pie = [pie]
    # margen de 3 tokens para los saltos de línea entre título y filas
    resto = presupuesto - contar(titulo) - 3
    unidades = _cabe([" ".join(resto_pie)] * bool(resto_pie) + filas, resto, contar)
    return [titulo + "\n" + parte for parte in _empacar(unidades, resto, contar, sep="\n")]


def _partes_seccion(parrafos: list[str], presupuesto: int, contar: Contador) -> list[str]:
    """La sección entera si cabe; si no, en partes parejas por párrafo (con un
    tope proporcional, para no dejar una cola de dos líneas)."""
    texto = "\n\n".join(parrafos)
    if not parrafos:
        return []
    total = contar(texto)
    if total <= presupuesto:
        return [texto]
    partes = math.ceil(total / presupuesto)
    tope = min(presupuesto, math.ceil(total / partes * 1.15))
    return _empacar(_cabe(parrafos, tope, contar), tope, contar)


# ---------------------------------------------------------------------------
# Fragmentación
# ---------------------------------------------------------------------------
def fragmentar(secciones: list[dict], nombre: str, contar: Contador,
               embeber: Embebedor | None = None) -> list[dict]:
    """Fragmentos de todas las secciones con la configuración `nombre`.

    Cada fragmento: id, seccion_id, tipo (texto | tabla), texto (lo que se
    embebe), texto_contexto (lo que iría al prompt: igual a texto salvo en
    small-to-big), n_tokens y los metadatos de la sección.
    """
    cfg = CONFIGS[nombre]
    umbral = None
    if cfg["estrategia"] == "semantico":
        if embeber is None:
            raise ValueError("la configuración semántica necesita `embeber`")
        umbral, cortes = _cortes_semanticos(secciones, embeber, cfg["percentil"])
    salida = []
    for s in secciones:
        cabecera = ruta(s) + "\n" if cfg.get("ruta", True) else ""
        presupuesto = cfg["maximo"] - contar(cabecera)
        piezas: list[tuple[str, str, str]] = []          # (tipo, texto, contexto)
        est = cfg["estrategia"]
        if est == "seccion":
            piezas += [("texto", p, p) for p in _partes_seccion(s["parrafos"], presupuesto, contar)]
        elif est == "fijo":
            cuerpo = "\n\n".join(s["parrafos"])
            if cuerpo:
                piezas += [("texto", p, p) for p in _por_palabras(cuerpo, presupuesto, contar, cfg["solape"])]
        elif est == "parrafos":
            unidades = _cabe(s["parrafos"], presupuesto, contar)
            piezas += [("texto", p, p) for p in _empacar(unidades, presupuesto, contar, cfg["solape"])]
        elif est == "oraciones":
            padre_max = LIMITE - contar(cabecera)
            for padre in _partes_seccion(s["parrafos"], padre_max, contar):
                hijos = _empacar(_cabe(oraciones(padre), presupuesto, contar), presupuesto, contar, sep=" ")
                piezas += [("texto", h, padre) for h in hijos]
        elif est == "semantico":
            piezas += [("texto", p, p) for p in segmentar(cortes[s["id"]], umbral, presupuesto,
                                                            cfg["minimo"], contar)]
        for anexo in s["anexos"]:
            if est == "oraciones":
                # small-to-big: se indexan trozos de filas (con el pie) y al
                # prompt va la tabla entera, o su parte de ≤ 512
                for padre in _tabla(anexo, LIMITE - contar(cabecera), contar):
                    piezas += [("tabla", t, padre) for t in _tabla(padre, presupuesto, contar)]
            else:
                piezas += [("tabla", t, t) for t in _tabla(anexo, presupuesto, contar)]

        for k, (tipo, texto, contexto) in enumerate(piezas, 1):
            completo = cabecera + texto
            salida.append({
                "id": f"{s['id']}-{k:02d}",
                "seccion_id": s["id"],
                "fuente": s["fuente"],
                "idioma": s["idioma"],
                "capitulo_num": s["capitulo_num"],
                "clinico": s["clinico"],
                "pagina_ini": s["pagina_ini"],
                "pagina_fin": s["pagina_fin"],
                "tipo": tipo,
                "texto": completo,
                "texto_contexto": cabecera + contexto,
                "n_tokens": contar(completo),
            })
    return salida


def _cortes_semanticos(secciones: list[dict], embeber: Embebedor,
                       percentil: float) -> tuple[float, dict[str, list[tuple[str, float]]]]:
    """Por sección, sus oraciones con la distancia (1 - coseno) a la anterior.

    El umbral sale del percentil de TODAS las distancias del libro, no de cada
    sección: con un percentil por sección, toda sección se cortaría aunque
    hable de un solo tema.
    """
    por_seccion = {s["id"]: oraciones("\n".join(s["parrafos"])) for s in secciones}
    todas = [o for os_ in por_seccion.values() for o in os_]
    vectores = embeber(todas) if todas else np.zeros((0, 1))
    salida, distancias, i = {}, [], 0
    for sid, os_ in por_seccion.items():
        v = vectores[i:i + len(os_)]
        i += len(os_)
        d = [0.0] + [float(1 - v[j] @ v[j - 1]) for j in range(1, len(os_))]
        distancias += d[1:]
        salida[sid] = list(zip(os_, d))
    umbral = float(np.percentile(distancias, percentil)) if distancias else 1.0
    return umbral, salida


def segmentar(pares: list[tuple[str, float]], umbral: float, presupuesto: int, minimo: int,
              contar: Contador) -> list[str]:
    """Corta donde la distancia a la oración anterior supera `umbral`; respeta
    el presupuesto y une al anterior los trozos de menos de `minimo` tokens
    (un tema de una oración suelta no sirve como fragmento)."""
    temas: list[list[str]] = []
    for o, d in pares:
        if not temas or d > umbral:
            temas.append([o])
        else:
            temas[-1].append(o)
    trozos: list[str] = []
    for t in temas:
        trozos += _empacar(_cabe(t, presupuesto, contar), presupuesto, contar, sep=" ")
    unidos: list[str] = []
    for t in trozos:
        if unidos and contar(t) < minimo and contar(unidos[-1] + " " + t) <= presupuesto:
            unidos[-1] += " " + t
        else:
            unidos.append(t)
    return unidos
