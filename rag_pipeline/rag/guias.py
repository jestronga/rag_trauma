"""
Extracción del texto de los libros de trauma (fase 1 de PLAN_INDEXAR_LIBROS.md).

Los dos libros tienen capa de texto, pero no se puede leer de corrido:

  · van a dos columnas con tramos a todo el ancho, y PyMuPDF no entrega los
    bloques en orden de lectura;
  · llevan encabezados, números de página, tablas, flujogramas y la
    bibliografía de cada capítulo;
  · ninguno trae índice incrustado.

Por eso cada bloque recibe un ROL según su fuente dominante (reglas por libro
en guias/perfiles.yaml): títulos, cuerpo, pie de figura o tabla, o
"material" (el texto suelto de figuras y tablas). El material solo se
conserva pegado a su pie ("Table 7-4: Glasgow Coma Scale…"); sin pie (los
flujogramas, los objetivos del capítulo) se descarta. Los pies de foto y de
dibujo del HUV se descartan con su material.

Lo que el PDF no trae como texto legible (flujogramas, que sin las flechas no
se pueden leer, y tablas en imagen) sale de una transcripción a mano
(cargar_transcripciones).

Salida: una SECCIÓN por título, con la ruta de títulos, el capítulo y las
páginas impresas, para que el médico encuentre el pasaje en el libro.

Las funciones de texto y de maquetación no dependen de PyMuPDF: los tests las
prueban con bloques sintéticos, sin los PDF (que no están en git).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from . import config

# Caracteres de uso privado (viñetas de Wingdings), el ETX que deja PyMuPDF
# junto a ellas, y espacios especiales.
_BASURA = re.compile(r"[-\x00-\x08\x0b-\x1f]")
_ESPACIOS = re.compile(r"[ \t   ]+")
# Una línea que empieza un elemento de lista va en línea nueva.
_LISTA = re.compile(r"^(\d{1,2}[.)]\s|[•●▪◦–]\s?)")
_FIN_ORACION = re.compile(r"[.:;!?)\]\"”»]$")
_FUENTES_SIMBOLO = ("wingdings", "symbol", "zapf")


def cargar_perfiles(ruta: Path | None = None) -> dict:
    with Path(ruta or config.GUIAS_PERFILES).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def equivalentes_atls11(capitulo_num: int, perfiles: dict | None = None) -> list[tuple[int, str]]:
    """Los capítulos del ATLS 11 (número, título) que tratan lo mismo que un
    capítulo del ATLS 10 (`equivalente_11` en su perfil; propuesta de Claude
    que valida el médico, PLAN_INDEXAR_ATLS10.md, fase 2). Vacío si no hay."""
    perfiles = perfiles or cargar_perfiles()
    titulos = {fila[0]: fila[1] for fila in perfiles["atls"]["capitulos"]}
    tabla = perfiles.get("atls10", {}).get("equivalente_11", {})
    return [(n, titulos[n]) for n in tabla.get(capitulo_num, [])]


# Ancla por defecto de cada tipo de transcripción: el título "Flujograma" que
# abre la sección del diagrama, o el pie "Tabla." de la tabla en imagen.
ANCLAS = {"flujograma": r"^Flujograma\b", "tabla": r"^Tabla\b"}


def cargar_transcripciones(ruta: Path | None = None) -> dict[str, dict[int, dict]]:
    """{clave del libro: {página del PDF: transcripción}}; vacío si no está el
    archivo (fuera de git, junto a los PDF).

    Cada transcripción: `tipo` (flujograma | tabla), `texto` (la primera línea
    es el título; una fila o un paso por línea, como las tablas, para que
    rag/fragmentar.py la parta por filas), `ancla` opcional y `notas` para la
    revisión médica (no se indexan)."""
    ruta = Path(ruta or config.GUIAS_TRANSCRIPCIONES)
    if not ruta.exists():
        return {}
    with ruta.open(encoding="utf-8") as f:
        datos = yaml.safe_load(f) or {}
    salida: dict[str, dict[int, dict]] = {}
    for libro, entradas in datos.items():
        por_pagina = salida.setdefault(libro, {})
        for e in entradas:
            if e["tipo"] not in ANCLAS:
                raise ValueError(f"{libro} p. {e['pagina']}: tipo desconocido {e['tipo']!r}")
            if e["pagina"] in por_pagina:
                raise ValueError(f"{libro} p. {e['pagina']}: dos transcripciones en la misma página")
            por_pagina[e["pagina"]] = {**e, "ancla": e.get("ancla") or ANCLAS[e["tipo"]],
                                       "texto": e["texto"].strip()}
    return salida


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
def limpiar(linea: str) -> str:
    return _BASURA.sub("", linea)


def unir_lineas(lineas: list[str], guion_corta_palabra: bool = True) -> str:
    """Une las líneas de un bloque en un párrafo.

    El guion blando (U+00AD) siempre corta palabra. El guion normal al final de
    línea, entre letra y minúscula, corta palabra en español
    ("desorgani-/zados"); en los demás casos es parte de la palabra y se une
    sin espacio ("intra-/abdominal", "24-/hour", "IMIST-/AMBO").
    """
    out = ""
    for bruta in lineas:
        linea = _ESPACIOS.sub(" ", limpiar(bruta)).strip()
        if not linea:
            continue
        if not out:
            out = linea
        elif out.endswith("­"):
            out = out[:-1] + linea
        elif re.search(r"\w-$", out):
            corta = guion_corta_palabra and out[-2].isalpha() and linea[:1].islower()
            out = (out[:-1] if corta else out) + linea
        elif _LISTA.match(linea):
            out += "\n" + linea
        else:
            out += " " + linea
    return out.replace("­", "").strip()


def unir_celdas(lineas: list[str]) -> str:
    """Texto de una tabla o figura: cada línea es una celda, salvo las que
    continúan la anterior (empiezan en minúscula o la anterior termina en coma)."""
    out = ""
    for bruta in lineas:
        linea = _ESPACIOS.sub(" ", limpiar(bruta)).strip()
        if not linea:
            continue
        if not out:
            out = linea
        elif out.endswith("­"):
            out = out[:-1] + linea
        elif linea[:1].islower() or linea[:1] == "(" or out.endswith((",", "-", "/")):
            out += ("" if out.endswith("-") else " ") + linea
        else:
            out += " | " + linea
    return out.replace("­", "").strip()


def corregir(linea: str, reemplazos: list[tuple[re.Pattern, str]]) -> str:
    """Arreglos del PDF, no del libro (`reemplazos` del perfil). En el ATLS 10,
    traducido a máquina: la viñeta "▸" traducida como "norte" ("( norte FIGURA
    1-1 )") y la letra capitular separada del resto ("W gallina que tratan…",
    de "When"). Los errores de traducción del contenido no se tocan."""
    for patron, reemplazo in reemplazos:
        linea = patron.sub(reemplazo, linea)
    return linea


_RX_CORTE_ORACION = re.compile(r"(?<=[.;:?!])\s+")


def sin_repeticiones(parrafo: str, minimo: int = 40, ventana: int = 3) -> str:
    """Quita la oración que repite una de las `ventana` anteriores del mismo
    párrafo. En el ATLS 10 la traducción deja frases repetidas en otra
    posición de la página, con otra caja ("…fugas de LCR. ( antibióticos de
    amplio espectro… fugas de LCR. (" ×3, p. 179 del PDF). Las cortas y las de
    otro párrafo no se tocan: dos pacientes del escenario de triage tienen la
    misma frase."""
    clave = lambda o: re.sub(r"^[(\s]+|[(\s]+$", "", o)
    salida: list[str] = []
    for oracion in _RX_CORTE_ORACION.split(parrafo):
        if len(clave(oracion)) >= minimo and clave(oracion) in {clave(o) for o in salida[-ventana:]}:
            continue
        salida.append(oracion)
    return " ".join(salida)


def continua(anterior: str, siguiente: str) -> bool:
    """¿El párrafo siguiente es la continuación del anterior? (corte de
    columna o de página a mitad de oración)."""
    return bool(anterior) and not _FIN_ORACION.search(anterior) and siguiente[:1].islower()


# ---------------------------------------------------------------------------
# Bloques y maquetación
# ---------------------------------------------------------------------------
@dataclass
class Bloque:
    x0: float
    y0: float
    x1: float
    y1: float
    lineas: list[str]
    fuente: str = ""
    tam: float = 0.0
    color: int = 0
    rol: str = ""
    anexo: str = ""            # texto de la tabla o figura, en los pies
    # La caja de cada renglón, si se conoce (para armar tablas por columnas)
    cajas: list[tuple] = field(default_factory=list)

    def texto(self, guion_corta_palabra: bool = True) -> str:
        return unir_lineas(self.lineas, guion_corta_palabra)

    def solapa_x(self, otro: "Bloque") -> bool:
        return min(self.x1, otro.x1) - max(self.x0, otro.x0) > 0


def bloques_de_pagina(pagina, deduplicar: bool = False) -> list[Bloque]:
    """Bloques de texto de una página de PyMuPDF, con su fuente dominante.

    `deduplicar`: descarta la línea que repite el texto de otra de la página en
    la misma caja (a menos de 1 punto: las copias difieren en milésimas). En el
    ATLS 10 (traducción automática) el 17,9 % de los caracteres viene
    superpuesto 2 a 4 veces ("…los segundo pico se produce en" ×3, p. 32 del
    PDF); sin esto entraría repetido."""
    bloques = []
    vistas: dict[str, list[tuple]] = {}
    for b in pagina.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        pesos: dict[tuple, int] = {}
        lineas, cajas = [], []
        for l in b["lines"]:
            texto = "".join(s["text"] for s in l["spans"])
            caja = tuple(l.get("bbox", b["bbox"]))
            if deduplicar:
                clave = _ESPACIOS.sub(" ", texto).strip()
                if any(max(abs(a - o) for a, o in zip(caja, otra)) < 1 for otra in vistas.get(clave, [])):
                    continue
                vistas.setdefault(clave, []).append(caja)
            lineas.append(texto)
            cajas.append(caja)
            for s in l["spans"]:
                n = len(s["text"].strip())
                if n and not s["font"].lower().startswith(_FUENTES_SIMBOLO):
                    clave = (s["font"], round(s["size"], 1), s["color"])
                    pesos[clave] = pesos.get(clave, 0) + n
        if not pesos:
            continue
        fuente, tam, color = max(pesos, key=pesos.get)
        bloques.append(Bloque(*b["bbox"], lineas=lineas, fuente=fuente, tam=tam, color=color,
                              cajas=cajas))
    return bloques


def _calza(regla: dict, b: Bloque, texto: str) -> bool:
    if not re.search(regla["fuente"], b.fuente):
        return False
    if "tam" in regla and not regla["tam"][0] <= b.tam <= regla["tam"][1]:
        return False
    if "color" in regla and b.color != int(regla["color"], 16):
        return False
    if "no_color" in regla and b.color == int(regla["no_color"], 16):
        return False
    if "empieza" in regla and not re.match(regla["empieza"], texto):
        return False
    if regla.get("mayusculas"):
        letras = [c for c in texto if c.isalpha()]
        if len(letras) < 3 or any(c.islower() for c in letras):
            return False
    return True


def clasificar(b: Bloque, perfil: dict) -> str:
    if b.y1 <= perfil["margen_sup"] or b.y0 >= perfil["margen_inf"]:
        return "margen"
    texto = b.texto()
    # Por el texto y no por la fuente: en el ATLS el título "REFERENCES" viene
    # en OpenSans en unos capítulos y en Arial en otros (pp. 81, 206).
    if re.match(perfil["referencias"], texto, re.I):
        return "referencias"
    # Sin `pie`, el perfil reconoce los pies por su color en `roles` (ATLS 10:
    # "TABLA 3-1 esboza…" también abre párrafos del cuerpo)
    if perfil.get("pie") and re.match(perfil["pie"], texto):
        return "pie"
    if perfil.get("figura") and re.match(perfil["figura"], texto):
        return "figura"
    for regla in perfil["roles"]:
        if _calza(regla, b, texto):
            return regla["rol"]
    return "material"


def orden_lectura(bloques: list[Bloque], ancho: float) -> list[Bloque]:
    """Orden de lectura en una página a dos columnas con tramos a todo el ancho.

    Los bloques que cruzan el centro de la página parten la página en franjas;
    dentro de cada franja se lee la columna izquierda y después la derecha.
    """
    mitad = ancho / 2

    def cruza(b: Bloque) -> bool:
        return b.x0 < mitad - 15 and b.x1 > mitad + 15

    anchos = sorted((b for b in bloques if cruza(b)), key=lambda b: b.y0)
    pendientes = [b for b in bloques if not cruza(b)]
    salida: list[Bloque] = []
    for i, limite in enumerate([b.y0 for b in anchos] + [float("inf")]):
        franja = [b for b in pendientes if (b.y0 + b.y1) / 2 < limite]
        pendientes = [b for b in pendientes if (b.y0 + b.y1) / 2 >= limite]
        salida += sorted(franja, key=lambda b: (b.x0 >= mitad - 15, b.y0))
        if i < len(anchos):
            salida.append(anchos[i])
    return salida


def _material(bloques: list[Bloque], pies: list[Bloque], posicion: str) -> list[Bloque]:
    """El material de la página, más los "títulos" que caen dentro de una
    figura o tabla: en la Figure 2-1 del ATLS (p. 46) el rótulo "TQEH ICU
    INTUBATION CHECKLIST" tiene la fuente de un título de sección."""
    material = [b for b in bloques if b.rol == "material"]
    cuerpo = [b for b in bloques if b.rol == "cuerpo"]
    for t in (b for b in bloques if b.rol.startswith("titulo")):
        for p in pies:
            if not p.solapa_x(t):
                continue
            figura = [m for m in material if m.solapa_x(p) and len(unir_celdas(m.lineas)) >= 4]
            # Entre el pie y el título no puede haber texto corrido: las
            # figuras no lo tienen, y si lo hay el título es de una sección.
            if posicion == "arriba":
                arriba, abajo = p.y1, t.y0
                dentro = p.y1 <= t.y0 and any(m.y0 >= t.y1 for m in figura)
            else:
                arriba, abajo = t.y1, p.y0
                dentro = t.y1 <= p.y0 and any(m.y1 <= t.y0 for m in figura)
            dentro = dentro and not any(
                c.solapa_x(t) and arriba <= (c.y0 + c.y1) / 2 <= abajo for c in cuerpo)
            if dentro:
                t.rol = "material"
                material.append(t)
                break
    return material


def continuar_figuras(bloques: list[Bloque], salto: float = 14) -> None:
    """Marca como "figura" la continuación de un pie de foto partido en varios
    bloques: pegada debajo (a menos de `salto` puntos), en la misma columna y
    la misma fuente. Medido el 19/10 en el HUV: 20 pies siguen en un segundo
    bloque ("…con compresión y" / "hemostáticos y trauma hepático…", p. 97)."""
    pendientes = [b for b in bloques if b.rol == "figura"]
    while pendientes:
        f = pendientes.pop()
        for m in bloques:
            if (m.rol == "material" and m.fuente == f.fuente and abs(m.tam - f.tam) < 0.5
                    and m.solapa_x(f) and -6 <= m.y0 - f.y1 <= salto):
                m.rol = "figura"
                pendientes.append(m)


def unir_pies_partidos(bloques: list[Bloque], salto: float = 6) -> list[Bloque]:
    """Un pie de dos renglones que viene en dos bloques (ATLS 10: "Tabla 8-1
    deformidades luxación de la articulación" / "comunes", p. 210 del PDF): el
    de abajo, del mismo color, pegado y en la misma columna, se suma al pie.
    Sin esto, el segundo renglón se leería como un título."""
    for p in [b for b in bloques if b.rol == "pie"]:
        for b in sorted(bloques, key=lambda b: b.y0):
            if (b is not p and b.rol not in ("pie", "unido") and b.color == p.color
                    and b.solapa_x(p) and 0 <= b.y0 - p.y1 <= salto):
                p.lineas += b.lineas
                p.cajas += b.cajas
                p.y1 = max(p.y1, b.y1)
                b.rol = "unido"
    return [b for b in bloques if b.rol != "unido"]


def agrupar_pies(bloques: list[Bloque], posicion: str = "arriba",
                 por_columnas: bool = False) -> tuple[list[Bloque], int]:
    """Pega a cada pie el material de su tabla o figura.

    Un bloque de material va con el pie más cercano que comparta columna y
    quede encima de él (`posicion="arriba"`, ATLS) o debajo (`"abajo"`, HUV).
    Devuelve los bloques sin el material y cuántos caracteres de material
    quedaron sin pie (se descartan). Los pies de foto o dibujo (rol "figura")
    no recogen material: medido el 19/10, el pie de una foto se llevaba listas
    sin pie que estaban encima ("Indicaciones para realizar TAC corporal
    Total", HUV p. 59). Con `por_columnas`, las tablas se arman con
    `_filas_por_columnas`.
    """
    continuar_figuras(bloques)
    pies = [b for b in bloques if b.rol == "pie"]
    asignado: dict[int, list[Bloque]] = {id(p): [] for p in pies}
    sueltos: list[Bloque] = []
    for m in _material(bloques, pies, posicion):
        if posicion == "arriba":
            cands = [p for p in pies if p.solapa_x(m) and m.y0 >= p.y0 - 2]
            distancia = lambda p: m.y0 - p.y1
        else:
            cands = [p for p in pies if p.solapa_x(m) and m.y1 <= p.y1 + 2]
            distancia = lambda p: p.y0 - m.y1
        if cands:
            asignado[id(min(cands, key=distancia))].append(m)
        else:
            sueltos.append(m)
    if por_columnas:
        # ATLS 10: el pie es más angosto que su tabla. Lo que cae dentro del
        # alto y el ancho de las celdas ya asignadas también es de la tabla
        # (los encabezados "Clase III" y "Clase IV" de la tabla 3-1, p. 103)
        for p in pies:
            celdas = asignado[id(p)]
            if not celdas:
                continue
            x0, x1 = min(c.x0 for c in celdas), max(c.x1 for c in celdas)
            y1 = max(c.y1 for c in celdas)
            dentro = [m for m in sueltos if m.y0 >= p.y0 - 2 and m.y1 <= y1 + 2
                      and m.x0 >= x0 - 2 and m.x1 <= x1 + 2]
            celdas += dentro
            sueltos = [m for m in sueltos if m not in dentro]
    filas = _filas_por_columnas if por_columnas else _filas
    for p in pies:
        p.anexo = filas(asignado[id(p)])
    cajas, descartados = recuadros(sueltos, por_columnas=por_columnas)
    return [b for b in bloques if b.rol != "material"] + cajas, descartados


def _filas(partes: list[Bloque]) -> str:
    """Texto de una tabla o figura: los bloques a la misma altura son una fila."""
    filas: list[str] = []
    y_fila = None
    for m in sorted(partes, key=lambda b: (round(b.y0), b.x0)):
        celda = unir_celdas(m.lineas)
        if len(celda) < 4:                          # rótulos "A.", "SI", "NO"
            continue
        if y_fila is not None and abs(m.y0 - y_fila) < 3:
            filas[-1] += " | " + celda
        else:
            filas.append(celda)
            y_fila = m.y0
    return "\n".join(filas)


def _filas_por_columnas(bloques: list[Bloque], salto: float = 8, columna: float = 30) -> str:
    """Texto de una tabla armada renglón por renglón (ATLS 10). Ahí un bloque
    puede ser un renglón de una celda (con `_filas`, los renglones de las
    celdas de dos o más líneas se intercalan entre columnas: "Diagnóstico de
    choque se puede | • Use toda la información… / la frecuencia cardíaca, la
    presión | perder cuando…") o una fila entera con sus celdas.

    Las columnas salen de dónde empiezan los renglones (más de `columna` puntos
    entre sí; la viñeta y su texto quedan juntos). Cada fila empieza donde
    empieza una celda de la primera columna (renglones a menos de `salto`
    puntos son la misma celda), y lo de las otras columnas va a la fila que
    empezó a su altura o encima."""
    partes = [Bloque(*caja, lineas=[linea]) for b in bloques
              for linea, caja in zip(b.lineas, b.cajas if len(b.cajas) == len(b.lineas)
                                     else [(b.x0, b.y0, b.x1, b.y1)] * len(b.lineas))
              if linea.strip()]
    if not partes:
        return ""
    xs = sorted(b.x0 for b in partes)
    cortes = [x for a, x in zip(xs, xs[1:]) if x - a > columna]
    col = lambda b: sum(b.x0 >= c for c in cortes)
    inicios: list[float] = []
    fin_celda = None
    for b in sorted((b for b in partes if col(b) == 0), key=lambda b: b.y0):
        if fin_celda is None or b.y0 - fin_celda > salto:
            inicios.append(b.y0)
        fin_celda = b.y1
    celdas: dict[tuple[int, int], list[Bloque]] = {}
    for b in partes:
        fila = max([i for i, y in enumerate(inicios) if y <= b.y0 + 4], default=0)
        celdas.setdefault((fila, col(b)), []).append(b)
    filas = []
    for fila in range(len(inicios) or 1):
        textos = []
        for c in range(len(cortes) + 1):
            bs = sorted(celdas.get((fila, c), []), key=lambda b: (b.y0, b.x0))
            texto = unir_lineas([l for b in bs for l in b.lineas]).replace("\n", " ")
            if texto:
                textos.append(texto)
        if textos:
            filas.append(" | ".join(textos))
    return "\n".join(filas)


# Los flujogramas se descartan: sin las flechas, "¿Penetra? | NO | SI |
# Cirugía" no dice qué rama lleva a qué, y el modelo podría leer una conducta
# que el libro no da. Se reconocen por sus ramas de decisión. El largo de los
# fragmentos no sirve para separarlos: las listas valiosas también los tienen
# cortos (laboratorios del ATLS p. 139, signos vasculares del HUV p. 259).
_RAMA = re.compile(r"^(s[ií]|no|yes|\(\s*[+-]\s*\)|[+-])$|^¿", re.I)
RECUADRO_MIN_TOTAL = 60


def es_flujograma(celdas: list[str]) -> bool:
    return any(_RAMA.search(c.strip()) for c in celdas)


def recuadros(sueltos: list[Bloque], margen: float = 30,
              por_columnas: bool = False) -> tuple[list[Bloque], int]:
    """Agrupa el material sin pie en recuadros (bloques a menos de `margen`
    puntos entre sí) y conserva los que no son flujogramas.

    Devuelve los recuadros (rol "recuadro", texto en `anexo`) y cuántos
    caracteres se descartaron.
    """
    grupos: list[list[Bloque]] = []
    for b in sorted(sueltos, key=lambda b: b.y0):
        cerca = [g for g in grupos if any(
            b.x0 - margen < o.x1 and o.x0 < b.x1 + margen
            and b.y0 - margen < o.y1 and o.y0 < b.y1 + margen for o in g)]
        nuevo = [b] + [o for g in cerca for o in g]
        grupos = [g for g in grupos if g not in cerca] + [nuevo]
    cajas, descartados = [], 0
    for g in grupos:
        celdas = [c for b in g for c in unir_celdas(b.lineas).split(" | ")]
        largos = [len(c) for c in celdas]
        if sum(largos) >= RECUADRO_MIN_TOTAL and not es_flujograma(celdas):
            caja = Bloque(min(b.x0 for b in g), min(b.y0 for b in g),
                          max(b.x1 for b in g), max(b.y1 for b in g), lineas=[], rol="recuadro")
            caja.anexo = _filas_por_columnas(g) if por_columnas else _filas(g)
            cajas.append(caja)
        else:
            descartados += sum(largos)
    return cajas, descartados


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------
@dataclass
class Capitulo:
    numero: int
    titulo: str
    pdf_ini: int
    pdf_fin: int
    clinico: bool


def capitulos(perfil: dict) -> list[Capitulo]:
    """Un capítulo termina donde empieza el siguiente, salvo que su fila traiga
    la última página (ATLS 10: el índice alfabético va entre el apéndice G y
    las secciones del final)."""
    filas = perfil["capitulos"]
    no_clinicos = set(perfil.get("no_clinicos", []))
    salida = []
    for i, (num, titulo, ini, *fin) in enumerate(filas):
        ultima = fin[0] if fin else filas[i + 1][2] - 1 if i + 1 < len(filas) else perfil["fin"]
        salida.append(Capitulo(num, titulo, ini, ultima, num not in no_clinicos))
    return salida


def pagina_impresa(perfil: dict, pdf: int) -> int:
    """La página impresa, para citar. `desfase_pagina` si es fijo; si cambia a
    lo largo del libro, `desfases`: [[página del PDF desde la que rige,
    desfase], …], en orden (ATLS 10: de 53 a 58, por páginas repetidas o en
    blanco)."""
    tramos = perfil.get("desfases")
    if not tramos:
        return pdf - perfil.get("desfase_pagina", 0)
    desfase = tramos[0][1]
    for desde, d in tramos:
        if pdf >= desde:
            desfase = d
    return pdf - desfase


@dataclass
class Seccion:
    fuente: str
    capitulo: Capitulo
    titulos: list[str]
    parrafos: list[str] = field(default_factory=list)
    # Tablas y figuras van al final de la sección: en el orden de lectura caen
    # a mitad de una oración ("…one side of the [Figure 7-13] spinal cord").
    anexos: list[str] = field(default_factory=list)
    paginas_pdf: list[int] = field(default_factory=list)

    def agregar(self, texto: str, pagina_pdf: int, es_cuerpo: bool = True) -> None:
        if not texto:
            return
        if not es_cuerpo:
            self.anexos.append(texto)
        elif self.parrafos and continua(self.parrafos[-1], texto):
            self.parrafos[-1] += " " + texto
        else:
            self.parrafos.append(texto)
        if pagina_pdf not in self.paginas_pdf:
            self.paginas_pdf.append(pagina_pdf)

    @property
    def vacia(self) -> bool:
        return not (self.parrafos or self.anexos)

    @property
    def texto(self) -> str:
        return "\n\n".join(self.parrafos + self.anexos)


class Extractor:
    """Recorre un libro capítulo a capítulo y emite una sección por título."""

    def __init__(self, perfil: dict, transcripciones: dict[int, dict] | None = None):
        self.perfil = perfil
        self.guion = perfil.get("guion_corta_palabra", True)
        # {página del PDF: transcripción} de este libro (cargar_transcripciones)
        self.transcripciones = transcripciones or {}
        self.reemplazos = [(re.compile(a), b) for a, b in perfil.get("reemplazos", [])]
        self.stats = {"material_sin_pie": 0, "descartado": 0, "referencias": 0,
                      "figuras": 0, "reemplazado": 0, "transcritas": 0}

    def secciones(self, doc) -> Iterator[dict]:
        saltar = set(self.perfil.get("saltar", []))
        self.colocadas: set[int] = set()
        for cap in capitulos(self.perfil):
            portada = {cap.pdf_ini} if self.perfil.get("saltar_portada") else set()
            yield from self._capitulo(doc, cap, saltar | portada)
        fuera = sorted(set(self.transcripciones) - self.colocadas)
        if fuera:
            raise ValueError(f"{self.perfil['fuente']}: transcripciones en páginas que no se "
                             f"extraen (saltadas o fuera de los capítulos): {fuera}")

    def _capitulo(self, doc, cap: Capitulo, saltar: set[int]) -> Iterator[dict]:
        p = self.perfil
        titulos: list[tuple[int, str]] = []
        actual = Seccion(p["fuente"], cap, [])
        n = 0
        en_referencias = False
        for pdf in range(cap.pdf_ini, cap.pdf_fin + 1):
            if pdf in saltar:
                continue
            pagina = doc[pdf - 1]
            bloques = bloques_de_pagina(pagina, p.get("deduplicar_lineas", False))
            if self.reemplazos:
                for b in bloques:
                    b.lineas = [corregir(l, self.reemplazos) for l in b.lineas]
            # ATLS 10: en la primera página de cada apéndice los objetivos van
            # arriba del texto; se descarta lo que termina antes de esa altura
            arriba = p.get("recortar_arriba", {}).get(pdf)
            if arriba:
                self.stats["descartado"] += sum(len(b.texto()) for b in bloques if b.y1 <= arriba)
                bloques = [b for b in bloques if b.y1 > arriba]
            for b in bloques:
                b.rol = clasificar(b, p)
            if p.get("unir_pies"):
                bloques = unir_pies_partidos(bloques)
            bloques, sin_pie = agrupar_pies(bloques, p.get("pie_posicion", "arriba"),
                                            p.get("tablas_por_columnas", False))
            self.stats["material_sin_pie"] += sin_pie
            trans = self.transcripciones.get(pdf)
            colocada = reemplazando = previa_leyenda = ultimo_titulo = False
            for b in orden_lectura(bloques, pagina.rect.width):
                texto = b.texto(self.guion)
                es_titulo = b.rol == "referencias" or b.rol.startswith("titulo")
                if reemplazando and not es_titulo:
                    # las cajas del diagrama (recuadros, rótulos leídos como
                    # cuerpo): su contenido ya va en la transcripción
                    self.stats["reemplazado"] += len(texto) + len(b.anexo)
                    continue
                reemplazando = False
                # la leyenda de una figura sigue en el bloque de abajo si este
                # no abre otra ("FIGURA 1-4 …" en rojo, el resto en gris)
                pegar_leyenda = (previa_leyenda and b.rol == "leyenda"
                                 and not re.match(p["leyenda"], texto))
                previa_leyenda = b.rol == "leyenda"
                previo_titulo, ultimo_titulo = ultimo_titulo, es_titulo
                ancla = bool(trans) and not colocada and bool(re.match(trans["ancla"], texto))
                if es_titulo:
                    if not actual.vacia:
                        n += 1
                        yield self._registro(actual, n)
                    if b.rol == "referencias":
                        en_referencias = True
                    else:
                        nivel = int(b.rol[-1])
                        if (p.get("unir_titulos") and previo_titulo and actual.vacia
                                and titulos and titulos[-1][0] == nivel):
                            # ATLS 10: un título de dos renglones viene en dos
                            # bloques ("15% a 30% … Hemorragia" / "Clase II")
                            titulos[-1] = (nivel, titulos[-1][1] + " " + texto)
                        else:
                            titulos = [t for t in titulos if t[0] < nivel] + [(nivel, texto)]
                    actual = Seccion(p["fuente"], cap, [t for _, t in titulos])
                    if ancla and trans["tipo"] == "flujograma" and b.rol != "referencias":
                        # el diagrama entero, en la sección que abre su título
                        actual.agregar(trans["texto"], pdf, es_cuerpo=False)
                        colocada = reemplazando = True
                elif en_referencias:
                    self.stats["referencias"] += len(texto)
                elif b.rol == "cuerpo":
                    actual.agregar(texto, pdf)
                elif b.rol == "pie":
                    if ancla and trans["tipo"] == "tabla":
                        # la transcripción trae su título: reemplaza pie y material
                        self.stats["reemplazado"] += len(b.anexo)
                        tabla = trans["texto"]
                        colocada = True
                    else:
                        tabla = texto + ("\n" + b.anexo if b.anexo else "")
                    actual.agregar(tabla, pdf, es_cuerpo=False)
                elif b.rol == "recuadro":
                    actual.agregar("Recuadro:\n" + b.anexo, pdf, es_cuerpo=False)
                elif b.rol == "leyenda":
                    # ATLS 10: la leyenda de la figura dice algo (la técnica,
                    # lo que muestra); los rótulos de dentro se descartan
                    if pegar_leyenda:
                        actual.anexos[-1] += " " + texto
                    else:
                        actual.agregar(texto, pdf, es_cuerpo=False)
                elif b.rol == "figura":
                    self.stats["figuras"] += len(texto) + len(b.anexo)
                elif b.rol == "descartar":
                    self.stats["descartado"] += len(texto)
            if trans and not colocada:
                # mejor fallar que perder el diagrama sin aviso (p. ej., si
                # cambian las reglas y el título deja de ser título)
                raise ValueError(f"{p['fuente']} p. PDF {pdf}: no se encontró el ancla "
                                 f"{trans['ancla']!r} de la transcripción ({trans['tipo']})")
            if trans:
                self.colocadas.add(pdf)
                self.stats["transcritas"] += 1
        if not actual.vacia:
            n += 1
            yield self._registro(actual, n)

    def _registro(self, s: Seccion, n: int) -> dict:
        cap = s.capitulo
        if self.perfil.get("quitar_repetidas"):
            s.parrafos = [sin_repeticiones(x) for x in s.parrafos]
        return {
            "id": f"{s.fuente}-{cap.numero:02d}-{n:03d}",
            "fuente": s.fuente,
            "libro": self.perfil["libro"],
            "idioma": self.perfil["idioma"],
            "capitulo_num": cap.numero,
            "capitulo": cap.titulo,
            "clinico": cap.clinico,
            "titulos": s.titulos,
            "pagina_ini": pagina_impresa(self.perfil, min(s.paginas_pdf)),
            "pagina_fin": pagina_impresa(self.perfil, max(s.paginas_pdf)),
            "pagina_pdf_ini": min(s.paginas_pdf),
            "texto": s.texto,
            # Por separado para la fragmentación (rag/fragmentar.py): las
            # tablas y recuadros no se parten como texto corrido.
            "parrafos": s.parrafos,
            "anexos": s.anexos,
        }
