"""
Chatbot de consulta sobre el registro de trauma.

    streamlit run app.py

Requiere los índices construidos (`python -m rag.index`; los manuales, con
`--fuente guias`) y Ollama corriendo.
"""
from __future__ import annotations

import html
import re
import time
import warnings

import streamlit as st

warnings.filterwarnings("ignore")

from rag import config                                    # noqa: E402
from rag.build_prompt import agrupar_guias, cabecera_guia, desenlace_visible, texto_guia  # noqa: E402
from rag.generate import ollama_disponible, responder, resumir_caso, usar_guias   # noqa: E402
from rag.hallazgos import resaltar_html                        # noqa: E402
from rag.retrieve import RUTAS, Filtros, hay_guias, ruta_por_defecto  # noqa: E402

st.set_page_config(page_title="Asistente Clínico de Trauma",
                   page_icon="🩺", layout="wide")

RX_ID = re.compile(r"^\s*(?:caso|id|registro)?\s*#?\s*(\d{3,8})\s*$", re.IGNORECASE)

CSS = """
<style>
  .cabecera { display:flex; gap:16px; align-items:center; padding:18px 22px;
              border-left:6px solid #C8102E; border-radius:10px; background:#FCEEEF;
              margin-bottom:18px; }
  .cabecera h1 { color:#C8102E; margin:0; padding:0; font-size:2rem; }
  .cab-icono { flex:none; width:54px; height:54px; border-radius:50%; background:#C8102E;
               color:#FFFFFF; font-size:1.9rem; display:flex; align-items:center;
               justify-content:center; }
  .cab-lema { font-size:1.1rem; opacity:.8; margin:2px 0 0; }
  .bienvenida { display:flex; flex-wrap:wrap; gap:12px; margin:6px 0 10px; }
  .bv-item { flex:1 1 220px; padding:12px 14px; border-radius:8px; font-size:0.98rem;
             border:1px solid rgba(200,16,46,.22); line-height:1.45; }
  .bv-item b { color:#C8102E; }
  .stChatMessage { font-size: 1.05rem; line-height: 1.6; }
  .stChatInput textarea { font-size: 1.05rem; }
  .caso-box { border:1px solid rgba(128,128,128,.28); border-radius:8px;
              padding:10px 12px; margin-bottom:10px; font-size:0.95rem; line-height:1.55; }
  .caso-id  { font-weight:600; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .etq { display:inline-block; padding:1px 7px; border-radius:10px; font-size:0.82rem;
         border:1px solid rgba(128,128,128,.35); margin:2px 3px 2px 0; }
  .caso-box mark { background:rgba(255,200,0,.35); color:inherit; padding:0 2px; border-radius:3px; }
  .guia-box { border-left:4px solid #1F6FB2; }
  .nota { font-size:0.95rem; padding:8px 12px; margin-bottom:10px; border-radius:6px;
          border-left:4px solid #B26B00; background:rgba(178,107,0,.08); line-height:1.45; }
  .tiempo { font-size:0.8rem; opacity:.45; text-align:right; margin-top:2px;
            font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .aviso { font-size:0.9rem; opacity:.75; line-height:1.45; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def preparar_modelos() -> bool:
    """Carga todo antes de la primera pregunta: bge-m3, el índice, el reranker
    si se usa, y el generador en Ollama con el prompt de sistema en su caché.

    Sin esto, la primera pregunta de una demo tardaba 25-35 s más que las
    demás. Corre una vez por proceso de Streamlit. Si falla, lanza: Streamlit
    no cachea excepciones y vuelve a intentarlo (p. ej. tras abrir Ollama).
    """
    from rag.build_prompt import SISTEMA_PROSA, SISTEMA_PROSA_GUIAS
    from rag.generate import _ollama
    from rag.retrieve import buscar, buscar_guias
    buscar("herida en mano", k=1, ruta=ruta_por_defecto())
    sistema = SISTEMA_PROSA
    if usar_guias():
        buscar_guias("herida en mano")
        sistema = SISTEMA_PROSA_GUIAS
    # carga el modelo con la ventana NUM_CTX: si después cambiara, Ollama lo recargaría
    _ollama([{"role": "system", "content": sistema},
             {"role": "user", "content": "ok"}], max_tokens=1, timeout=180)
    return True


def manuales_indexados() -> bool:
    try:
        return hay_guias()
    except Exception:        # Qdrant abierto por otro proceso: lo dice la precarga
        return False


error_preparacion = None
with st.spinner("Cargando modelos (solo la primera vez, ~30 s)…"):
    try:
        preparar_modelos()
    except Exception as e:           # Ollama apagado, índice ausente: lo dice la barra lateral
        error_preparacion = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuración")

    ok_ollama, msg_ollama = ollama_disponible()
    (st.success if ok_ollama else st.error)(msg_ollama)
    if error_preparacion:
        st.warning(f"No se pudieron precargar los modelos: {error_preparacion}")

    ruta = st.selectbox("Ruta de recuperación", RUTAS,
                        index=RUTAS.index(ruta_por_defecto()) if ruta_por_defecto() in RUTAS else 2,
                        help="rrf fusiona BM25 y densa por posición. rerank reordena "
                             "los candidatos de rrf con un cross-encoder y descarta "
                             "los que no juzga relevantes.")
    k = st.slider("Casos recuperados", 3, 10, config.TOP_K)
    umbral = st.slider("Umbral de abstención", 0.0, 0.9, config.UMBRAL_ABSTENCION, 0.05,
                       help="Coseno mínimo del mejor caso para que el sistema responda. "
                            "Más alto = más conservador.")
    umbral_rerank = None
    if ruta == "rerank":
        umbral_rerank = st.slider("Relevancia mínima por caso", 0.0, 0.95, config.UMBRAL_RERANK, 0.05,
                                  help="Probabilidad del reranker para que un caso llegue a la "
                                       f"respuesta. Con menos de {config.MIN_CASOS} casos, "
                                       "el sistema se abstiene.")

    st.markdown("### Fuentes")
    hay_manuales = manuales_indexados()
    con_guias = st.toggle(
        "Manuales de trauma (ATLS y HUV)", value=config.GUIAS and hay_manuales,
        disabled=not hay_manuales,
        help="Segunda fuente, con el mismo peso que los casos: lo que recomiendan el ATLS "
             "11.ª ed. y el manual de manejo inicial del HUV. Apagado, la respuesta sale solo "
             "de los casos del registro.")
    if not hay_manuales:
        st.caption("Los manuales no están indexados: `python -m rag.index --fuente guias`.")

    st.markdown("### Filtros (casos)")
    mecanismo = st.selectbox("Mecanismo", ["(todos)"] + list(config.ETIQUETAS_MECANISMO),
                             format_func=lambda v: config.ETIQUETAS_MECANISMO.get(v, v))
    region = st.selectbox("Región anatómica", ["(todos)"] + list(config.ETIQUETAS_REGION),
                          format_func=lambda v: config.ETIQUETAS_REGION.get(v, v))
    desenlace = st.selectbox("Desenlace", ["(todos)"] + list(config.ETIQUETAS_DESENLACE),
                             format_func=lambda v: config.ETIQUETAS_DESENLACE.get(v, v))
    # El corpus ya no trae registros truncados (apto_rag): no hay filtro que ofrecer.
    filtros = Filtros(
        mecanismo=None if mecanismo == "(todos)" else mecanismo,
        region=None if region == "(todos)" else region,
        desenlace=None if desenlace == "(todos)" else desenlace,
    )

    st.divider()
    manuales = (" · <b>Manuales</b>: ATLS 11.ª ed. (en inglés) y el manual de manejo inicial "
                "del HUV, lo que recomiendan. Lo hecho en los casos y lo que recomiendan los "
                "manuales puede no coincidir: el sistema muestra las dos miradas y no elige."
                if con_guias else "")
    st.markdown(
        f"<div class='aviso'><b>Alcance</b><br>"
        f"<b>Casos</b>: atención de trauma del registro, {config.COBERTURA_TEMPORAL}, lo que se "
        f"hizo en casos históricos{manuales}<br><br>"
        f"Sugiere solo lo que esas fuentes respaldan, con la cita de cada afirmación. Los "
        f"protocolos cambian con el tiempo. No reemplaza el juicio clínico.<br><br>"
        f"Los datos personales están reemplazados por marcadores "
        f"(<code>[FECHA]</code>, <code>[EDAD:30-39]</code>).</div>",
        unsafe_allow_html=True,
    )
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

st.markdown(
    f"<div class='cabecera'>"
    f"<div class='cab-icono'>✚</div>"
    f"<div><h1>Asistente Clínico de Trauma</h1>"
    f"<div class='cab-lema'>Apoyo al diagnóstico basado en casos similares del registro de trauma"
    f"{' y en los manuales ATLS y HUV' if con_guias else ''}</div>"
    f"</div></div>",
    unsafe_allow_html=True,
)

if not st.session_state.mensajes:
    revise = ("Qué valorar y qué se hizo en los casos similares, y qué recomiendan los manuales, "
              "con la cita de cada afirmación." if con_guias else
              "Qué valorar y qué se hizo en los casos similares, con la cita de cada uno.")
    verifique = ("El texto completo de cada caso y de cada pasaje está en «Casos consultados» y "
                 "«Fragmentos de los manuales»." if con_guias else
                 "El texto completo de cada caso está en «Casos consultados».")
    st.markdown(
        "<div class='bienvenida'>"
        "<div class='bv-item'><b>1 · Describa el caso</b><br>Mecanismo, región y hallazgos: "
        "«herida por arma cortopunzante en abdomen con irritación peritoneal».</div>"
        f"<div class='bv-item'><b>2 · Revise la sugerencia</b><br>{revise}</div>"
        f"<div class='bv-item'><b>3 · Verifique el origen</b><br>{verifique} "
        "También acepta un número de caso.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def aviso_respuesta(respuesta) -> str:
    """El aviso fijo bajo cada respuesta: qué es cada fuente y sus límites."""
    partes = []
    if respuesta.casos:
        partes.append(f"Casos históricos del registro ({config.COBERTURA_TEMPORAL}): los conteos "
                      f"describen estos casos, no frecuencias generales, y los protocolos pueden "
                      f"haber cambiado.")
    if respuesta.guias:
        partes.append("Pasajes del ATLS 11.ª ed. y del manual del HUV: lo del ATLS (en inglés) va "
                      "resumido en español; el texto original está en «Fragmentos de los manuales».")
    partes.append("Apoyo a la decisión; no reemplaza el juicio clínico.")
    return " ".join(partes)


def tarjeta_caso(caso, consulta: str = "") -> str:
    m = caso.metadata
    etiquetas = []
    if m.get("mecanismo_lesion"):
        etiquetas.append(f"<span class='etq'>{config.ETIQUETAS_MECANISMO.get(m['mecanismo_lesion'], m['mecanismo_lesion'])}</span>")
    for r in (m.get("region_anatomica") or [])[:3]:
        etiquetas.append(f"<span class='etq'>{config.ETIQUETAS_REGION.get(r, r)}</span>")
    etiquetas.append(f"<span class='etq'>{desenlace_visible(caso)}</span>")
    texto = resaltar_html(caso.text, consulta).replace("\n", "<br>")
    return (f"<div class='caso-box'><span class='caso-id'>caso {caso.record_id}</span> "
            f"· similitud {caso.score_norm:.2f}<br>{''.join(etiquetas)}<br><br>{texto}</div>")


def tarjeta_guia(seccion: str, fragmentos, consulta: str = "") -> str:
    """Un pasaje de un manual: libro, capítulo, sección y páginas impresas, y su
    texto tal cual (el ATLS en inglés)."""
    f = fragmentos[0]
    escala = "relevancia" if f.relevancia is not None else "similitud"
    puntaje = max(x.puntaje for x in fragmentos)
    idioma = "<span class='etq'>en inglés</span>" if f.idioma == "en" else ""
    texto = resaltar_html(texto_guia(fragmentos), consulta).replace("\n", "<br>")
    return (f"<div class='caso-box guia-box'><span class='caso-id'>guía {seccion}</span> "
            f"· {escala} {puntaje:.2f}<br><span class='etq'>{html.escape(cabecera_guia(fragmentos))}"
            f"</span>{idioma}<br><br>{texto}</div>")


def nota_verificacion(respuesta) -> None:
    """Qué parte de la respuesta armó el código porque la redacción del modelo
    no pasó la verificación (o no la hubo). Va arriba de la respuesta: como
    caption gris al pie nadie la veía (pendiente 13 de CONTEXTO_SESION.md)."""
    if respuesta is None or respuesta.prosa_verificada is not False or config.PROSA_SIEMPRE:
        return
    nombres = {"casos": "los casos", "guias": "los manuales"}
    presentes = [n for n, hay in (("casos", respuesta.casos), ("guias", respuesta.guias)) if hay]
    armadas = [nombres[n] for n in presentes if n not in respuesta.fuentes_en_prosa]
    if not armadas:
        return
    motivo = respuesta.motivos[0] if respuesta.motivos else "sin detalle"
    extra = f" y {len(respuesta.motivos) - 1} más" if len(respuesta.motivos) > 1 else ""
    if motivo.startswith("no hay prosa"):
        porque = "el modelo no redactó una sugerencia con respaldo"
    else:
        porque = f"la redacción del modelo no pasó la verificación: {motivo}{extra}"
    st.markdown(f"<div class='nota'>Lo de {' y '.join(armadas)} va armado desde las fuentes, sin "
                f"redactar ({html.escape(porque)}).</div>", unsafe_allow_html=True)


def pie_tiempo(segundos: float | None) -> None:
    """Tiempo hasta la respuesta, discreto, para medir modelos desde la interfaz."""
    if segundos is None:
        return
    st.markdown(f"<div class='tiempo'>{segundos:.1f} s · {config.MODELO_LLM}</div>",
                unsafe_allow_html=True)


def desplegable_casos(respuesta, consulta: str) -> None:
    """El texto completo de cada caso: la base de cada afirmación sobre los
    casos, que el médico tiene que poder revisar por su cuenta."""
    if not respuesta.casos:
        return
    buscada = respuesta.consulta_reescrita or consulta
    etiqueta = f"Casos consultados ({len(respuesta.casos)}) · texto completo"
    if respuesta.consulta_reescrita:
        etiqueta += "  ·  consulta reescrita"
    with st.expander(etiqueta):
        if respuesta.consulta_reescrita:
            st.caption(f"Búsqueda realizada: «{respuesta.consulta_reescrita}»")
        relevancia = (f" · relevancia del reranker: {respuesta.relevancia:.2f}"
                      if respuesta.relevancia is not None else "")
        st.caption(f"Confianza (coseno del mejor caso): {respuesta.confianza:.3f}{relevancia} · "
                   f"resaltado: lo que coincide con la consulta")
        for c in respuesta.casos:
            st.markdown(tarjeta_caso(c, buscada), unsafe_allow_html=True)


def desplegable_guias(respuesta, consulta: str) -> None:
    """El texto completo de cada pasaje de los manuales."""
    if not respuesta.guias:
        return
    buscada = respuesta.consulta_reescrita or consulta
    grupos = agrupar_guias(respuesta.guias)
    with st.expander(f"Fragmentos de los manuales ({len(grupos)}) · texto completo"):
        st.caption("Pasajes que pasaron la guarda de los manuales, del más al menos relevante. "
                   "Páginas impresas del libro; el ATLS, en su idioma original.")
        for seccion, fragmentos in grupos.items():
            st.markdown(tarjeta_guia(seccion, fragmentos, buscada), unsafe_allow_html=True)


def cuerpo_respuesta(respuesta, consulta: str, de_a_poco: bool = False) -> None:
    """La respuesta por apartados: casos (prosa, tabla de todos los casos y su
    desplegable) y manuales (prosa, lista de todos los pasajes y su
    desplegable). Sin apartados (abstención, resumen de un caso), el texto y
    los desplegables al final."""
    def escribir(texto: str) -> None:
        if de_a_poco:
            st.write_stream(escribir_de_a_poco(texto, segundos_max=1.0))
        else:
            st.markdown(texto)

    if not respuesta.secciones:
        escribir(respuesta.texto)
        desplegable_casos(respuesta, consulta)
        desplegable_guias(respuesta, consulta)
        return
    for seccion in respuesta.secciones:
        escribir(seccion.texto)
        if seccion.fuente == "casos":
            desplegable_casos(respuesta, consulta)
        elif seccion.fuente == "guias":
            desplegable_guias(respuesta, consulta)


def detalle_respuesta(respuesta, segundos: float | None) -> None:
    """Lo que va al final: tiempo, aviso fijo y alertas."""
    pie_tiempo(segundos)
    if respuesta.error:
        return
    if (respuesta.casos or respuesta.guias) and not respuesta.abstuvo:
        st.caption(aviso_respuesta(respuesta))
    if respuesta.citas_invalidas:
        st.warning("El modelo citó identificadores que no están entre los casos ni los pasajes "
                   f"recuperados: {', '.join(respuesta.citas_invalidas)}. "
                   "Esa parte de la respuesta no está respaldada.")


def escribir_de_a_poco(texto: str, segundos_max: float = 2.0):
    """El texto ya está completo: se muestra letra por letra solo por estética.
    Los trozos crecen con el largo para que nunca tarde más de `segundos_max`."""
    paso = max(1, len(texto) // int(segundos_max / 0.008))
    for i in range(0, len(texto), paso):
        yield texto[i:i + paso]
        time.sleep(0.008)


def pintar(rol: str, contenido: str, respuesta=None, segundos: float | None = None,
           consulta: str = "") -> None:
    with st.chat_message(rol):
        if respuesta is None or respuesta.error:
            st.markdown(contenido)
        else:
            nota_verificacion(respuesta)
            cuerpo_respuesta(respuesta, consulta)
        if respuesta is not None:
            detalle_respuesta(respuesta, segundos)


for m in st.session_state.mensajes:
    pintar(m["role"], m["content"], m.get("respuesta"), m.get("segundos"), m.get("consulta", ""))


# ---------------------------------------------------------------------------
# Turno
# ---------------------------------------------------------------------------
escrita = st.chat_input("Ej.: varón joven con herida por arma cortopunzante en cuello, estable")
pregunta = escrita

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    pintar("user", pregunta)

    historial = [{"role": m["role"], "content": m["content"]}
                 for m in st.session_state.mensajes[:-1]]

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        with st.spinner("Pensando…"):
            m = RX_ID.match(pregunta)
            if m:
                r = resumir_caso(m.group(1))
            else:
                r = responder(pregunta, historial=historial, ruta=ruta, k=k,
                              filtros=filtros, umbral=umbral, umbral_rerank=umbral_rerank,
                              guias=con_guias)
        segundos = time.perf_counter() - t0

        if r.error:
            st.error(r.error)
            texto = f"_{r.error}_"
        else:
            texto = r.texto
            nota_verificacion(r)
            cuerpo_respuesta(r, pregunta, de_a_poco=True)
        detalle_respuesta(r, segundos)

    st.session_state.mensajes.append(
        {"role": "assistant", "content": texto, "respuesta": r, "segundos": segundos,
         "consulta": pregunta}
    )
