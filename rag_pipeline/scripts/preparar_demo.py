"""
Chequeo antes de una demostración: correrlo en el PC de la demo, con el
chatbot CERRADO (Qdrant local admite un solo proceso), 10 minutos antes.

    python scripts/preparar_demo.py

Qué revisa:
  1. GPU y configuración elegida: VRAM, ruta (en 4 GB debe ser rrf: el
     reranker no cabe junto al generador) y umbral de abstención.
  2. Ollama: que tenga el modelo, lo carga y mide cuánto quedó en la GPU
     (debe ser 100 %; si no, algo más ocupa la VRAM: cerrar el navegador
     con aceleración por hardware, juegos, etc.).
  3. Los botones de la demo (campo `boton` de eval/consultas.yaml y de
     eval/consultas_guias.yaml) por el sistema completo, como en la app:
     tiempo, si se abstuvo (solo el de "fuera de alcance" debe hacerlo), qué
     fuentes respondieron (casos, manuales), qué quedó en prosa verificada y
     que no haya citas inválidas.
  4. La conversación de la demo (PLAN_CHATBOT_CLINICO.md, fase 5), como en la
     app: "HPAF en abdomen" → el botón "Hipotenso o en choque" → "¿qué dice el
     ATLS del FAST?". Que la ficha siga al paciente, que salga la pregunta y su
     respuesta se sume, que se diga qué cambió y que la respuesta vaya por
     letras (xABCDE). Con TARJETAS, que los dos primeros turnos traigan el
     protocolo y el del FAST no (es una pregunta concreta).
  5. El peatón de la respuesta ideal (PLAN_RESPUESTA_IDEAL.md): "hombre
     arrollado por vehiculo automotor" → su tarjeta y la pregunta de si las
     ruedas pasaron por encima → el botón "Le pasaron por encima" → la tarjeta
     con lo del aplastamiento.

Al final deja el modelo cargado (KEEP_ALIVE) para que la primera pregunta de
la demo no espere la carga.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from rag import config                                                      # noqa: E402
from rag.build_prompt import SISTEMA_PROSA, SISTEMA_PROSA_GUIAS, SISTEMA_PROSA_GUIAS_ATLS10  # noqa: E402
from rag.consulta import diccionario                                         # noqa: E402
from rag.generate import (_ollama, ollama_disponible, porcentaje_gpu, responder,  # noqa: E402
                          usar_guias)
from rag.retrieve import (dispositivo_reranker, libros_indexados, ruta_guias_por_defecto,  # noqa: E402
                          ruta_por_defecto, umbral_guias, vram_gb)
from rag.tarjetas import cargar as cargar_tarjetas                           # noqa: E402

OK, MAL, AVISO = "  [ok]", "  [!!]", "  [ ?]"


def main() -> int:
    problemas = 0
    print("1 · GPU y configuración")
    vram = vram_gb()
    ruta = ruta_por_defecto()
    print(f"{OK if vram else AVISO} VRAM {vram:.1f} GB" + ("" if vram else " (torch no ve una GPU NVIDIA)"))
    print(f"{OK} ruta {ruta} · reranker en {dispositivo_reranker()} · "
          f"umbral de abstención {config.UMBRAL_ABSTENCION} · salida {config.SALIDA} · "
          f"prosa siempre {'sí' if config.PROSA_SIEMPRE else 'no'} · keep_alive {config.KEEP_ALIVE}")
    if 0 < vram < config.VRAM_MIN_RERANK and ruta == "rerank":
        print(f"{MAL} RUTA=rerank con {vram:.1f} GB: el reranker no cabe junto al generador. "
              f"Quitar la variable RUTA.")
        problemas += 1
    con_guias = usar_guias()
    if con_guias:
        ruta_libros = ruta_guias_por_defecto()
        print(f"{OK} manuales: ruta {ruta_libros} · umbral {umbral_guias(ruta_libros)} · "
              f"presupuesto {config.PRESUPUESTO_GUIAS} tokens{' con cuota por libro' if config.CUOTA_GUIAS else ''}"
              f" · NUM_CTX {config.NUM_CTX}")
        if config.SENAL_TRAUMA:
            print(f"{OK} guarda por señal de trauma: sin señal, libros {config.UMBRAL_GUIAS_SIN_SENAL} "
                  f"y casos {config.UMBRAL_RERANK_SIN_SENAL}")
        # cómo se sigue la conversación (PLAN_CHATBOT_CLINICO.md, fase 2)
        print(f"{OK} conversación: " + ("ficha del paciente armada por el código, sin reescribir"
                                        if config.CONVERSACION == "ficha" else
                                        "el modelo reescribe cada seguimiento (CONVERSACION=llm)"))
        # las preguntas (fase 3): sus respuestas se suman a la ficha
        if config.PREGUNTAS and config.CONVERSACION != "ficha":
            print(f"{AVISO} PREGUNTAS=1 necesita CONVERSACION=ficha: no se van a mostrar")
        elif config.PREGUNTAS:
            print(f"{OK} preguntas: hasta 2 bajo cada respuesta, en el orden xABCDE")
        # la respuesta por letras (fase 4)
        print(f"{OK} respuesta: " + ("por letras de la revisión primaria (xABCDE) dentro de cada apartado, "
                                     "con un paciente" if config.ORDEN_ATLS else "prosa de siempre (ORDEN_ATLS=0)"))
        # las tarjetas por mecanismo (PLAN_RESPUESTA_IDEAL.md)
        if config.TARJETAS:
            tarjetas = cargar_tarjetas()
            print(f"{OK if tarjetas else MAL} tarjetas por mecanismo: {len(tarjetas)} "
                  f"({', '.join(t.titulo for t in tarjetas.values()) or f'no encuentro {config.TARJETAS_YAML}'})")
            problemas += not tarjetas
            if not config.PATRONES_REGISTRO.exists():
                print(f"{AVISO} sin {config.PATRONES_REGISTRO.name}: los casos van sin la línea del registro. "
                      f"Correr: python scripts\\patrones_registro.py")
        else:
            print(f"{AVISO} TARJETAS=0: los manuales van en prosa del modelo, sin el protocolo del mecanismo")
        # la consulta de los manuales (PLAN_CHATBOT_CLINICO.md, fase 1)
        if config.ABREV_LIBROS and not diccionario():
            print(f"{AVISO} ABREV_LIBROS=1 pero no encuentro {config.ABREVIATURAS_YAML}: "
                  f"los manuales se buscan sin expandir las abreviaturas")
        else:
            print(f"{OK} búsqueda en los manuales: abreviaturas "
                  f"{'expandidas' if config.ABREV_LIBROS else 'sin expandir'} · nombre del mecanismo "
                  f"{'sí' if config.MECANISMO_LIBROS else 'no'}")
        # el ATLS 10 (PLAN_INDEXAR_ATLS10.md): indexado desde la fase 3, se busca con ATLS10=1
        indexado = "ATLS10" in libros_indexados()
        if config.ATLS10 and not indexado:
            print(f"{MAL} ATLS10=1 pero el ATLS 10 no está indexado: extraer y correr "
                  f"python -m rag.index --fuente guias")
            problemas += 1
        elif config.ATLS10:
            print(f"{OK} ATLS 10.ª ed. encendido (edición anterior, traducción no oficial)")
        else:
            print(f"{AVISO} ATLS10=0: el ATLS 10.ª ed. {'está indexado pero ' if indexado else ''}"
                  f"no se busca")
    elif config.GUIAS:
        print(f"{MAL} GUIAS=1 pero los manuales no están indexados: la respuesta saldrá solo de "
              f"los casos. Correr: python -m rag.index --fuente guias")
        problemas += 1
    else:
        print(f"{AVISO} GUIAS=0: solo casos")

    print("\n2 · Ollama")
    ok, msg = ollama_disponible()
    print(f"{OK if ok else MAL} {msg}")
    if not ok:
        return 1
    t0 = time.perf_counter()
    sistema = ((SISTEMA_PROSA_GUIAS_ATLS10 if config.ATLS10 else SISTEMA_PROSA_GUIAS)
               if con_guias else SISTEMA_PROSA)
    _ollama([{"role": "system", "content": sistema}, {"role": "user", "content": "ok"}],
            max_tokens=1, timeout=300)
    gpu = porcentaje_gpu()
    print(f"{OK} modelo cargado en {time.perf_counter() - t0:.0f} s")
    if vram and gpu is not None:
        bien = gpu >= 100
        print(f"{OK if bien else MAL} {gpu} % en GPU"
              + ("" if bien else " → algo más ocupa la VRAM (navegador, juegos). Cerrarlo y reintentar; "
                                 "si sigue, $env:NUM_GPU = \"-1\" (más lento)."))
        problemas += not bien

    print("\n3 · Botones de la demo (pipeline completo)")
    consultas = yaml.safe_load((RAIZ / "eval" / "consultas.yaml").read_text(encoding="utf-8"))["consultas"]
    if con_guias:
        # las de los manuales (proc-*): solo las que no están ya en consultas.yaml
        ids = {c["id"] for c in consultas}
        consultas += [c for c in yaml.safe_load((RAIZ / "eval" / "consultas_guias.yaml").read_text(
            encoding="utf-8"))["consultas"] if c["id"] not in ids]
    for c in sorted((c for c in consultas if c.get("boton")), key=lambda c: c["boton"]):
        if c.get("atls10") and not (config.ATLS10 and con_guias):
            print(f"{AVISO} {c['boton']:28} se salta: solo con el ATLS 10 encendido (ATLS10=1)")
            continue
        espera = bool(c.get("espera_abstencion")) or c.get("categoria") == "fuera_de_dominio"
        t0 = time.perf_counter()
        r = responder(c["texto"], reescribir_consulta=False)
        s = time.perf_counter() - t0
        fallos = []
        if r.error:
            fallos.append(r.error)
        if r.abstuvo != espera:
            fallos.append(f"{'no ' if espera else ''}se abstuvo (coseno {r.confianza:.3f})")
        if r.citas_invalidas:
            fallos.append(f"citas inválidas {r.citas_invalidas}")
        if c.get("atls10") and not any(f.fuente == "ATLS10" for f in r.guias):
            fallos.append("no trajo pasajes del ATLS 10")
        if r.abstuvo:
            detalle = f"se abstiene (coseno {r.confianza:.3f})"
        else:
            fuentes = " + ".join(n for n, hay in ((f"protocolo ({r.tarjeta})", r.tarjeta), ("casos", r.casos),
                                                  ("manuales", r.guias and not r.tarjeta)) if hay)
            en_prosa = " + ".join({"casos": "casos", "guias": "manuales"}[n] for n in r.fuentes_en_prosa)
            if r.prosa_verificada:
                detalle = f"{fuentes} · prosa verificada"
            elif r.orden_atls and r.letras_armadas:
                # por letras (fase 4) con PROSA_SIEMPRE_ATLS=0: la que no pasa va armada desde su fuente
                detalle = f"{fuentes} · por letras; armadas desde la fuente: {', '.join(r.letras_armadas)}"
            elif r.orden_atls:
                detalle = f"{fuentes} · por letras SIN verificar ({r.motivos[0] if r.motivos else ''})"
            elif config.PROSA_SIEMPRE and en_prosa:
                # se muestra igual: así lo decidió el usuario para la demo
                detalle = f"{fuentes} · prosa SIN verificar ({r.motivos[0] if r.motivos else ''})"
            elif en_prosa:
                detalle = f"{fuentes} · en prosa: {en_prosa} ({r.motivos[0] if r.motivos else ''})"
            else:
                detalle = f"{fuentes} · versión determinista ({r.motivos[0] if r.motivos else 'sin prosa'})"
        print(f"{MAL if fallos else OK} {c['boton']:28} {s:5.1f} s  {detalle}")
        for f in fallos:
            print(f"        {f}")
        problemas += bool(fallos)

    print("\n4 · Conversación de la demo (ficha, preguntas, letras)")
    if config.CONVERSACION != "ficha":
        print(f"{AVISO} CONVERSACION={config.CONVERSACION}: la conversación de la demo necesita la ficha")
    else:
        problemas += secuencia_guion()
        problemas += conversacion_demo(con_guias)
        print("\n5 · El peatón de la respuesta ideal (tarjeta y su pregunta)")
        if not (config.TARJETAS and con_guias and config.PREGUNTAS):
            print(f"{AVISO} se salta: necesita TARJETAS, los manuales y PREGUNTAS")
        else:
            problemas += peaton_demo()

    print(f"\n{'Todo listo' if not problemas else f'{problemas} problema(s): ver arriba'}. "
          f"Abrir el chatbot desde rag_pipeline\\: streamlit run app.py")
    return 1 if problemas else 0


# El guion de SETUP.md §6, escrito en un solo chat: el tipo que debe dar la
# ficha a cada mensaje (solo reglas, sin el modelo). "Paciente nuevo" = el botón
# de la barra lateral. Si una regla de rag/ficha.py cambia, esto lo avisa.
GUION = [
    ("herida por arma de fuego en tórax con hipoventilación", "nuevo"),       # 2
    ("paciente apuñalado en el pecho, herida penetrante", "nuevo"),           # 4
    ("HPAF en cráneo con orificio de entrada parietal", "nuevo"),             # 5
    ("HPAF en abdomen", "nuevo"),                                             # 6
    ("hipotenso o en choque", "seguimiento"),                                 # 6, el botón
    ("¿qué dice el ATLS del FAST?", "seguimiento"),                           # 6
    ("mujer de 75 años con fractura de cadera", "nuevo"),                     # 6, la corrección
    ("hombre arrollado por vehiculo automotor", "nuevo"),                     # 6b
    ("las ruedas le pasaron por encima", "seguimiento"),                      # 6b, el botón
    ("herida por arma de fuego con desenlace fatal en sala de urgencias", "nuevo"),   # 7
    ("Paciente nuevo", None),                                                 # 7a, el botón
    ("cuándo activar el protocolo de transfusión masiva", "nuevo"),           # 7a
    ("TCE leve con pérdida de conciencia transitoria", "nuevo"),              # 7b
    ("manejo de cetoacidosis diabética en urgencias", "nuevo"),               # 8
]
# las respuestas de los botones (rag/preguntas.py): la app las manda como seguimiento
BOTONES = {"hipotenso o en choque", "las ruedas le pasaron por encima"}


def secuencia_guion() -> int:
    """El guion en un solo chat: ninguna consulta se suma al paciente anterior
    salvo los seguimientos del paso 6."""
    from rag.ficha import actualizar
    ficha, fallos = None, []
    for texto, esperado in GUION:
        if esperado is None:
            ficha = None
            continue
        forzar = "seguimiento" if texto in BOTONES else None                   # el clic
        tipo, ficha = actualizar(ficha, texto, forzar)
        if tipo != esperado:
            fallos.append(f"«{texto}»: {tipo}, se esperaba {esperado}")
    print(f"{MAL if fallos else OK} el guion en un solo chat: {len(GUION) - 1 - len(fallos)} de "
          f"{len(GUION) - 1} mensajes con el tipo esperado")
    for f in fallos:
        print(f"        {f}")
    return int(bool(fallos))


# La conversación del guion (SETUP.md §6, paso 6): tres turnos como en la app.
TURNO_1 = "HPAF en abdomen"
RESPUESTA_C = "Hipotenso o en choque"          # el botón de la pregunta C (rag/preguntas.py)
TURNO_3 = "¿qué dice el ATLS del FAST?"


def _turno(nombre: str, s: float, r, fallos: list[str], detalle: str) -> int:
    if r.error:
        fallos.insert(0, r.error)
    print(f"{MAL if fallos else OK} {nombre:28} {s:5.1f} s  {detalle}")
    for f in fallos:
        print(f"        {f}")
    return int(bool(fallos))


def conversacion_demo(con_guias: bool) -> int:
    """Los tres turnos de la demo por generate.responder, con la ficha y la
    respuesta anterior como los pasa la app. Devuelve cuántos turnos fallan."""
    problemas = 0
    t0 = time.perf_counter()
    r1 = responder(TURNO_1)
    s = time.perf_counter() - t0
    fallos = []
    if r1.abstuvo or r1.tipo != "nuevo" or r1.ficha is None:
        fallos.append(f"esperaba un paciente nuevo con respuesta (tipo {r1.tipo}, abstuvo {r1.abstuvo})")
    elif r1.ficha.mecanismo != "arma_de_fuego" or "abdomen" not in r1.ficha.regiones:
        fallos.append(f"la ficha no tiene arma de fuego y abdomen: {r1.ficha.linea()}")
    c = next((p for p in r1.preguntas if p.clave == "C"), None)
    opcion = next((o for o in c.opciones if o.etiqueta == RESPUESTA_C), None) if c else None
    if config.PREGUNTAS and opcion is None:
        fallos.append(f"no sale la pregunta C con «{RESPUESTA_C}»: {[p.clave for p in r1.preguntas]}")
    if config.ORDEN_ATLS and not r1.orden_atls:
        fallos.append("la respuesta no va por letras (ORDEN_ATLS)")
    con_tarjeta = config.TARJETAS and con_guias
    if con_tarjeta and r1.tarjeta != "arma_de_fuego":
        fallos.append(f"no trae el protocolo de arma de fuego (tarjeta {r1.tarjeta})")
    problemas += _turno("a · " + TURNO_1, s, r1, fallos,
                        f"{r1.ficha.linea() if r1.ficha else ''} · preguntas {[p.clave for p in r1.preguntas]}"
                        + (f" · protocolo {r1.tarjeta}" if r1.tarjeta else ""))
    if fallos or not config.PREGUNTAS:
        if not config.PREGUNTAS:
            print(f"{AVISO} PREGUNTAS=0: el turno b (el botón) se salta")
        return problemas

    t0 = time.perf_counter()
    r2 = responder(opcion.mensaje, ficha=r1.ficha, anterior=r1, tipo="seguimiento")   # el clic
    s = time.perf_counter() - t0
    fallos = []
    if r2.abstuvo or r2.tipo != "seguimiento":
        fallos.append(f"esperaba un seguimiento con respuesta (tipo {r2.tipo}, abstuvo {r2.abstuvo})")
    elif r2.ficha.estado != "inestable" or "hipotensión o choque" not in r2.ficha.afirmados:
        fallos.append(f"la respuesta no se sumó a la ficha: {r2.ficha.linea()}")
    if not r2.cambio:
        fallos.append("no dice qué cambió respecto de la respuesta anterior")
    if any(p.clave == "C" for p in r2.preguntas):
        fallos.append("vuelve a preguntar la C")
    if con_tarjeta and not (r2.tarjeta and "*Está inestable:*" in r2.secciones[0].texto):
        fallos.append("el protocolo no se ajustó al paciente inestable")
    problemas += _turno(f"b · botón «{RESPUESTA_C}»", s, r2, fallos, r2.cambio or "")

    t0 = time.perf_counter()
    r3 = responder(TURNO_3, ficha=r2.ficha, anterior=r2)
    s = time.perf_counter() - t0
    fallos = []
    busqueda = r3.consulta_reescrita or ""
    if r3.abstuvo or r3.tipo != "seguimiento":
        fallos.append(f"esperaba un seguimiento con respuesta (tipo {r3.tipo}, abstuvo {r3.abstuvo})")
    elif "HPAF" not in busqueda or "FAST" not in busqueda:
        fallos.append(f"la búsqueda perdió al paciente o la pregunta: «{busqueda}»")
    if con_guias and not r3.guias:
        fallos.append("los manuales no responden la pregunta del FAST")
    if r3.ficha and "ecografía / FAST" in r3.ficha.afirmados:
        fallos.append("tomó la pregunta por el FAST como un hecho")
    if r3.tarjeta:
        fallos.append("una pregunta concreta (el FAST) no lleva el protocolo, sino la prosa de los manuales")
    libros = sorted({f.fuente for f in r3.guias})
    problemas += _turno("c · " + TURNO_3, s, r3, fallos, f"busca «{busqueda}» · manuales {libros}")
    return problemas


# El peatón de la respuesta ideal (SETUP.md §6, paso 6b; PLAN_RESPUESTA_IDEAL.md)
PEATON = "hombre arrollado por vehiculo automotor"
RESPUESTA_RUEDAS = "Le pasaron por encima"      # el botón de la pregunta del peatón


def peaton_demo() -> int:
    """La tarjeta del peatón y su pregunta, como en la app. Devuelve cuántos turnos fallan."""
    problemas = 0
    t0 = time.perf_counter()
    r1 = responder(PEATON)
    s = time.perf_counter() - t0
    fallos = []
    if r1.abstuvo or r1.tarjeta != "peaton":
        fallos.append(f"no trae el protocolo del peatón (tarjeta {r1.tarjeta}, abstuvo {r1.abstuvo})")
    pregunta = next((p for p in r1.preguntas if p.clave == "aplastamiento"), None)
    opcion = next((o for o in pregunta.opciones if o.etiqueta == RESPUESTA_RUEDAS), None) if pregunta else None
    if opcion is None:
        fallos.append(f"no pregunta si las ruedas pasaron por encima: {[p.clave for p in r1.preguntas]}")
    if r1.citas_invalidas:
        fallos.append(f"citas inválidas {r1.citas_invalidas}")
    protocolo = r1.secciones[0].texto if r1.secciones else ""
    letras = sum(f"**{l} · " in protocolo for l in "xABCDE")
    problemas += _turno("a · " + PEATON, s, r1, fallos,
                        f"protocolo con {letras} letras y {len({f.seccion_id for f in r1.pasajes_tarjeta})} "
                        f"pasajes · preguntas {[p.clave for p in r1.preguntas]}")
    if fallos:
        return problemas
    t0 = time.perf_counter()
    r2 = responder(opcion.mensaje, ficha=r1.ficha, anterior=r1, tipo="seguimiento")    # el clic
    s = time.perf_counter() - t0
    fallos = []
    if r2.tarjeta != "peaton" or "*Las ruedas le pasaron por encima:*" not in r2.secciones[0].texto:
        fallos.append("el protocolo no muestra lo del aplastamiento")
    if any(p.clave == "aplastamiento" for p in r2.preguntas):
        fallos.append("vuelve a preguntar si las ruedas pasaron por encima")
    problemas += _turno(f"b · botón «{RESPUESTA_RUEDAS}»", s, r2, fallos, r2.ficha.linea() if r2.ficha else "")
    return problemas


if __name__ == "__main__":
    sys.exit(main())
