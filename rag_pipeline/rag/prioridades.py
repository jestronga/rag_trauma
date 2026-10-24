"""
El orden de la revisión primaria del ATLS, xABCDE (PLAN_CHATBOT_CLINICO.md,
fase 4): qué letra le toca a cada término del léxico y a cada pasaje de los
manuales. Lo decide el código; el modelo solo redacta dentro de cada letra.

  x  hemorragia exanguinante      C  circulación
  A  vía aérea                    D  déficit neurológico
  B  respiración                  E  exposición
  y "después de la revisión primaria" para lo demás (fractura, sutura,
  antibiótico…)

La letra de un pasaje sale de su capítulo cuando el capítulo es de una letra
(ATLS 11, capítulos 3 a 8, uno por letra; HUV y ATLS 10, los de vía aérea,
shock, hipotermia…) y, si no, de los términos del léxico que nombra, solo en
los manuales en español. Se calcula al buscar, con capitulo_num: no hay que
reindexar. Las tablas las tiene que validar el médico (decisión 4 del plan);
los casos dudosos están en DUDOSOS.

Los rótulos van aquí y no en build_prompt.py: todo lo escrito allí cuenta como
vocabulario permitido para el verificador.
"""
from __future__ import annotations

from collections import Counter

from .hallazgos import LEXICO, terminos

DESPUES = "después"
LETRAS = ["x", "A", "B", "C", "D", "E", DESPUES]
NOMBRES = {"x": "Hemorragia exanguinante", "A": "Vía aérea", "B": "Respiración", "C": "Circulación",
           "D": "Déficit neurológico", "E": "Exposición", DESPUES: "Después de la revisión primaria"}


def rotulo(letra: str) -> str:
    """"**B · Respiración.**"; "**Después de la revisión primaria.**"."""
    return f"**{NOMBRES[letra]}.**" if letra == DESPUES else f"**{letra} · {NOMBRES[letra]}.**"


# El plan: intubación → A; hipoventilación, neumotórax, hemotórax y tubo de
# tórax → B; hipotensión o choque, hemoperitoneo, FAST y transfusión → C;
# Glasgow, pérdida de conciencia, anisocoria y déficit neurológico → D. Lo que no
# es de la revisión primaria va en "después".
LETRA_TERMINO = {
    "sangrado activo": "x",
    "intubación / ventilación mecánica": "A",
    "hipoventilación": "B", "hemotórax": "B", "neumotórax": "B", "tubo de tórax": "B",
    "hipotensión o choque": "C", "hemoperitoneo / líquido libre": "C", "ecografía / FAST": "C",
    "transfusión": "C", "líquidos endovenosos": "C", "compromiso vascular": "C", "toracotomía": "C",
    "déficit neurológico": "D", "alteración del Glasgow": "D", "pérdida de conciencia": "D",
    "anisocoria": "D", "convulsión": "D",
    **{t: DESPUES for t in (
        "herida penetrante", "fractura", "lesión tendinosa o nerviosa", "irritación peritoneal",
        "amnesia", "emesis", "cefalea", "otorragia / otoliquia", "radiografía", "TAC",
        "angiografía / doppler", "laparotomía / laparoscopia", "cirugía", "sutura",
        "lavado / desbridamiento", "curación", "escarectomía / injerto", "osteosíntesis / reducción",
        "inmovilización / férula", "antibiótico", "analgesia", "profilaxis antitetánica", "UCI",
        "remisión", "observación")},
}
# Para el médico (fase 7 de los libros): la letra que se eligió y la otra posible.
DUDOSOS = {
    "sangrado activo": "x o C", "compromiso vascular": "C o x", "toracotomía": "C o B",
    "convulsión": "D o después", "irritación peritoneal": "después o C",
    "laparotomía / laparoscopia": "después o C", "angiografía / doppler": "después o C",
}

# Capítulos que son de una letra. ATLS 11: los de la revisión primaria, uno por
# letra. HUV: vía aérea (5), shock (6), hipotermia (3); el vascular periférico
# ("primero controlar el sangrado", 20), el ultrasonido (21) y la toracostomía
# (22) se agregaron al mirar los títulos (el plan nombraba 5, 6 y 3). ATLS 10:
# vía aérea y manejo ventilatorio (2), choque (3), hipotermia (apéndice B, 15).
LETRA_CAPITULO = {
    "ATLS": {3: "x", 4: "A", 5: "B", 6: "C", 7: "D", 8: "E"},
    "HUV": {3: "E", 5: "A", 6: "C", 20: "x", 21: "C", 22: "B"},
    "ATLS10": {2: "A", 3: "C", 15: "E"},
}
EN_ESPANOL = {"HUV", "ATLS10"}


def letra_termino(termino: str) -> str:
    return LETRA_TERMINO.get(termino, DESPUES)


# En un pasaje, "hemorragia" (que el léxico lee como "sangrado activo") es casi
# siempre la interna: la pélvica, la abdominal. Con la x, 74 secciones del ATLS
# 10 quedaban en "hemorragia exanguinante" (medido el 24/10). La x de un pasaje
# sale del capítulo; en los casos, "sangrado activo" es la herida que sangra.
_LETRA_EN_PASAJES = {"sangrado activo": "C"}


def letra_por_terminos(texto: str) -> str:
    """La letra de la revisión primaria que más nombra un pasaje (empate: la
    primera en el orden xABCDE); "después" si no nombra ninguna."""
    cuenta = Counter(_LETRA_EN_PASAJES.get(t, letra_termino(t)) for ts in terminos(texto).values() for t in ts)
    primarias = [(cuenta[letra], -LETRAS.index(letra)) for letra in LETRAS[:-1] if cuenta[letra]]
    return LETRAS[-max(primarias)[1]] if primarias else DESPUES


def letra_pasaje(fuente: str, capitulo_num: int | None, texto: str) -> str:
    """La letra de un pasaje (una sección de un manual): la del capítulo si es
    de una letra; si no, en español, la de sus términos; si no, "después"."""
    letra = LETRA_CAPITULO.get(fuente, {}).get(capitulo_num)
    if letra:
        return letra
    return letra_por_terminos(texto) if fuente in EN_ESPANOL else DESPUES


def ordenar(letras) -> list[str]:
    return sorted(set(letras), key=LETRAS.index)


def sin_letra_lexico() -> list[str]:
    """Términos del léxico sin letra asignada (debe estar vacía: un test)."""
    return [t for _, t, _ in LEXICO if t not in LETRA_TERMINO]
