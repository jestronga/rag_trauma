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
flujogramas, los objetivos del capítulo) se descarta.

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

    def texto(self, guion_corta_palabra: bool = True) -> str:
        return unir_lineas(self.lineas, guion_corta_palabra)

    def solapa_x(self, otro: "Bloque") -> bool:
        return min(self.x1, otro.x1) - max(self.x0, otro.x0) > 0


def bloques_de_pagina(pagina) -> list[Bloque]:
    """Bloques de texto de una página de PyMuPDF, con su fuente dominante."""
    bloques = []
    for b in pagina.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        pesos: dict[tuple, int] = {}
        lineas = []
        for l in b["lines"]:
            lineas.append("".join(s["text"] for s in l["spans"]))
            for s in l["spans"]:
                n = len(s["text"].strip())
                if n and not s["font"].lower().startswith(_FUENTES_SIMBOLO):
                    clave = (s["font"], round(s["size"], 1), s["color"])
                    pesos[clave] = pesos.get(clave, 0) + n
        if not pesos:
            continue
        fuente, tam, color = max(pesos, key=pesos.get)
        bloques.append(Bloque(*b["bbox"], lineas=lineas, fuente=fuente, tam=tam, color=color))
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
    if re.match(perfil["pie"], texto):
        return "pie"
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


def agrupar_pies(bloques: list[Bloque], posicion: str = "arriba") -> tuple[list[Bloque], int]:
    """Pega a cada pie el material de su tabla o figura.

    Un bloque de material va con el pie más cercano que comparta columna y
    quede encima de él (`posicion="arriba"`, ATLS) o debajo (`"abajo"`, HUV).
    Devuelve los bloques sin el material y cuántos caracteres de material
    quedaron sin pie (se descartan).
    """
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
    for p in pies:
        p.anexo = _filas(asignado[id(p)])
    cajas, descartados = recuadros(sueltos)
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


# Los flujogramas se descartan: sin las flechas, "¿Penetra? | NO | SI |
# Cirugía" no dice qué rama lleva a qué, y el modelo podría leer una conducta
# que el libro no da. Se reconocen por sus ramas de decisión. El largo de los
# fragmentos no sirve para separarlos: las listas valiosas también los tienen
# cortos (laboratorios del ATLS p. 139, signos vasculares del HUV p. 259).
_RAMA = re.compile(r"^(s[ií]|no|yes|\(\s*[+-]\s*\)|[+-])$|^¿", re.I)
RECUADRO_MIN_TOTAL = 60


def es_flujograma(celdas: list[str]) -> bool:
    return any(_RAMA.search(c.strip()) for c in celdas)


def recuadros(sueltos: list[Bloque], margen: float = 30) -> tuple[list[Bloque], int]:
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
            caja.anexo = _filas(g)
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
    filas = perfil["capitulos"]
    no_clinicos = set(perfil.get("no_clinicos", []))
    salida = []
    for i, (num, titulo, ini) in enumerate(filas):
        fin = filas[i + 1][2] - 1 if i + 1 < len(filas) else perfil["fin"]
        salida.append(Capitulo(num, titulo, ini, fin, num not in no_clinicos))
    return salida


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

    def __init__(self, perfil: dict):
        self.perfil = perfil
        self.guion = perfil.get("guion_corta_palabra", True)
        self.stats = {"material_sin_pie": 0, "descartado": 0, "referencias": 0}

    def secciones(self, doc) -> Iterator[dict]:
        saltar = set(self.perfil.get("saltar", []))
        for cap in capitulos(self.perfil):
            portada = {cap.pdf_ini} if self.perfil.get("saltar_portada") else set()
            yield from self._capitulo(doc, cap, saltar | portada)

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
            bloques = bloques_de_pagina(pagina)
            for b in bloques:
                b.rol = clasificar(b, p)
            bloques, sin_pie = agrupar_pies(bloques, p.get("pie_posicion", "arriba"))
            self.stats["material_sin_pie"] += sin_pie
            for b in orden_lectura(bloques, pagina.rect.width):
                texto = b.texto(self.guion)
                if b.rol == "referencias" or b.rol.startswith("titulo"):
                    if not actual.vacia:
                        n += 1
                        yield self._registro(actual, n)
                    if b.rol == "referencias":
                        en_referencias = True
                    else:
                        nivel = int(b.rol[-1])
                        titulos = [t for t in titulos if t[0] < nivel] + [(nivel, texto)]
                    actual = Seccion(p["fuente"], cap, [t for _, t in titulos])
                elif en_referencias:
                    self.stats["referencias"] += len(texto)
                elif b.rol == "cuerpo":
                    actual.agregar(texto, pdf)
                elif b.rol == "pie":
                    tabla = texto + ("\n" + b.anexo if b.anexo else "")
                    actual.agregar(tabla, pdf, es_cuerpo=False)
                elif b.rol == "recuadro":
                    actual.agregar("Recuadro:\n" + b.anexo, pdf, es_cuerpo=False)
                elif b.rol == "descartar":
                    self.stats["descartado"] += len(texto)
        if not actual.vacia:
            n += 1
            yield self._registro(actual, n)

    def _registro(self, s: Seccion, n: int) -> dict:
        desfase = self.perfil.get("desfase_pagina", 0)
        cap = s.capitulo
        return {
            "id": f"{s.fuente}-{cap.numero:02d}-{n:03d}",
            "fuente": s.fuente,
            "libro": self.perfil["libro"],
            "idioma": self.perfil["idioma"],
            "capitulo_num": cap.numero,
            "capitulo": cap.titulo,
            "clinico": cap.clinico,
            "titulos": s.titulos,
            "pagina_ini": min(s.paginas_pdf) - desfase,
            "pagina_fin": max(s.paginas_pdf) - desfase,
            "pagina_pdf_ini": min(s.paginas_pdf),
            "texto": s.texto,
            # Por separado para la fragmentación (rag/fragmentar.py): las
            # tablas y recuadros no se parten como texto corrido.
            "parrafos": s.parrafos,
            "anexos": s.anexos,
        }
