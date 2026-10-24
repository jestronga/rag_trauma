"""
Señal de trauma en la consulta (PLAN_INDEXAR_ATLS10.md, decisión 5).

Las consultas fuera de dominio que pasan las guardas (23/10/2026: sangrado
vaginal en el embarazo, hipoglucemia, ACV por los libros; apendicitis, ACV e
insuficiencia cardiaca por los casos) comparten algo: no nombran un mecanismo
de lesión, una lesión traumática ni un procedimiento de trauma. Este módulo
dice si la consulta nombra alguno. La idea a medir: sin señal, las guardas
piden más. No se abstienen sin señal, porque hay preguntas legítimas sin esas
palabras ("cuándo activar el protocolo de transfusión masiva", "cómo se puntúa
la escala de coma de Glasgow").

Las listas se escribieron ANTES de medir, para no ajustarlas a las consultas
de evaluación:
  · mecanismos: los de anonimizacion/src/enrich.py (MECANISMOS), copiados (no
    se importa código de esa carpeta), sin "choque", que en clínica es shock;
  · lesiones y procedimientos: listas generales de trauma. No llevan regiones
    ni el léxico de los casos ("edema pulmonar", "hemorragia digestiva" y
    "cefalea" también son de enfermedades).
Se comparan sobre el texto sin tildes y en minúsculas (hallazgos.normalizar).
"""
from __future__ import annotations

import re

from .hallazgos import normalizar

MECANISMOS = {
    "arma_de_fuego": [r"hpaf", r"\bhaf\b", r"arma\s+de\s+fuego", r"proyectil", r"disparar", r"disparo",
                      r"me\s+pegaron\s+un\s+tiro", r"balea", r"pafm?\b", r"impacto\s+de\s+bala"],
    "cortopunzante": [r"hpacp", r"\bhacp\b", r"\bhpcp\b", r"cortopunzante", r"apunal", r"punal",
                      r"cuchill", r"machete", r"navaja", r"vidrio", r"me\s+cort",
                      r"herida\s+por\s+arma\s+blanca"],
    "accidente_transito": [r"accidente\s+de\s+transito", r"\bavt\b", r"colision", r"volcamiento",
                           r"atropell", r"moto\s+vs", r"vehiculo", r"me\s+accident"],
    "caida": [r"\bcaida\b", r"me\s+cai", r"se\s+cayo", r"propia\s+altura", r"cayo\s+de",
              r"precipitacion", r"desde\s+su\s+altura"],
    # con límite de palabra (en enrich.py no lo tienen): "rina" estaba en
    # "warfarina" y "urinaria", "acido" en "cetoacidosis" (medido el 23/10)
    "contundente": [r"contundente", r"\bhpco\b", r"me\s+golpearon", r"le\s+golpearon",
                    r"me\s+pegaron(?!\s+un\s+tiro)", r"\bpuno\b", r"\bpatada", r"\bpalo\b", r"\bpiedra",
                    r"\brina\b", r"\bpelea", r"agresion\s+fisica"],
    "quemadura": [r"quemadur", r"\bscq\b", r"escaldadura", r"\bacido\b", r"electrocucion"],
    "mordedura": [r"mordedur", r"mordio", r"mordieron", r"canino", r"serpiente", r"ofidic"],
    "cuerpo_extrano": [r"cuerpo\s+extrano", r"\bce\s+en\b"],
    "intoxicacion": [r"intoxica", r"envenenamiento", r"sobredosis", r"ingesta\s+de"],
    "autoinfligido": [r"autolitic", r"autoinfligid", r"intento\s+suicid"],
}
LESIONES = [r"trauma", r"traumat", r"politrauma", r"\btce\b", r"\btec\b", r"lesionad", r"herid",
            r"penetrante", r"fractur", r"\bfx\b", r"luxac", r"esguince", r"contusi", r"laceraci",
            r"avulsi", r"amputa", r"quemad", r"aplastamiento", r"hemotorax", r"neumotorax",
            r"taponamiento", r"hematoma"]
PROCEDIMIENTOS = [r"torniquete", r"toracostom", r"tubo\s+(de|a|en)\s+torax", r"faja\s+pelvica",
                  r"\be-?fast\b", r"cricotiroid", r"ventana\s+pericardica", r"\batls\b"]

_PATRONES = ([(f"mecanismo:{k}", re.compile(p)) for k, ps in MECANISMOS.items() for p in ps]
             + [("lesion", re.compile(p)) for p in LESIONES]
             + [("procedimiento", re.compile(p)) for p in PROCEDIMIENTOS])


def senal_trauma(consulta: str) -> list[str]:
    """Las clases de señal que nombra la consulta ("mecanismo:caida",
    "lesion", "procedimiento"), sin repetir; vacía si no nombra ninguna."""
    texto = normalizar(consulta)
    vistas: list[str] = []
    for clase, patron in _PATRONES:
        if clase not in vistas and patron.search(texto):
            vistas.append(clase)
    return vistas
