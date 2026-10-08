"""
Chatbot de consulta sobre el registro de trauma.

    streamlit run app.py

Requiere los índices construidos (`python -m rag.index`) y Ollama corriendo.
"""
from __future__ import annotations

import re
import time
import warnings

import streamlit as st

warnings.filterwarnings("ignore")

from rag import config                                    # noqa: E402
from rag.build_prompt import desenlace_visible                 # noqa: E402
from rag.generate import ollama_disponible, responder, resumir_caso   # noqa: E402
from rag.hallazgos import resaltar_html                        # noqa: E402
from rag.retrieve import RUTAS, Filtros, ruta_por_defecto  # noqa: E402

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
    from rag.build_prompt import SISTEMA_PROSA
    from rag.generate import _ollama
    from rag.retrieve import buscar
    buscar("herida en mano", k=1, ruta=ruta_por_defecto())
    _ollama([{"role": "system", "content": SISTEMA_PROSA},
             {"role": "user", "content": "ok"}], max_tokens=1, timeout=180)
    return True


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

    st.markdown("### Filtros")
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
    st.markdown(
        f"<div class='aviso'><b>Alcance del registro</b><br>"
        f"Atención de trauma, {config.COBERTURA_TEMPORAL}. Los protocolos de manejo "
        f"cambian con el tiempo: el sistema muestra lo que se hizo en casos "
        f"históricos y sugiere solo lo que esos casos respaldan. No reemplaza el "
        f"juicio clínico.<br><br>"
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
    f"<div class='cab-lema'>Apoyo al diagnóstico basado en casos similares del registro de trauma</div>"
    f"</div></div>",
    unsafe_allow_html=True,
)

if not st.session_state.mensajes:
    st.markdown(
        "<div class='bienvenida'>"
        "<div class='bv-item'><b>1 · Describa el caso</b><br>Mecanismo, región y hallazgos: "
        "«herida por arma cortopunzante en abdomen con irritación peritoneal».</div>"
        "<div class='bv-item'><b>2 · Revise la sugerencia</b><br>Qué valorar y qué se hizo "
        "en los casos similares, con la cita de cada uno.</div>"
        "<div class='bv-item'><b>3 · Verifique el origen</b><br>El texto completo de cada caso "
        "está en «Casos consultados». También acepta un número de caso.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


AVISO_RESPUESTA = (f"Casos históricos del registro ({config.COBERTURA_TEMPORAL}): los conteos "
                   f"describen estos casos, no frecuencias generales, y los protocolos pueden "
                   f"haber cambiado. Apoyo a la decisión; no reemplaza el juicio clínico.")


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


def pie_tiempo(segundos: float | None) -> None:
    """Tiempo hasta la respuesta, discreto, para medir modelos desde la interfaz."""
    if segundos is None:
        return
    st.markdown(f"<div class='tiempo'>{segundos:.1f} s · {config.MODELO_LLM}</div>",
                unsafe_allow_html=True)


def detalle_respuesta(respuesta, segundos: float | None, consulta: str) -> None:
    """Lo que va bajo el texto: tiempo, aviso fijo, alertas y los casos completos."""
    pie_tiempo(segundos)
    if respuesta.error:
        return
    if respuesta.casos and not respuesta.abstuvo:
        st.caption(AVISO_RESPUESTA)
    if respuesta.prosa_verificada is False and not config.PROSA_SIEMPRE:
        # Se muestra la versión armada desde los casos; se dice por qué, para
        # que no parezca un cambio de formato sin razón.
        motivo = respuesta.motivos[0] if respuesta.motivos else "sin detalle"
        extra = f" (y {len(respuesta.motivos) - 1} más)" if len(respuesta.motivos) > 1 else ""
        st.caption(f"La redacción del modelo no pasó la verificación ({motivo}{extra}): "
                   f"se muestran las sugerencias armadas desde los casos.")
    if respuesta.citas_invalidas:
        st.warning("El modelo citó identificadores que no están entre los casos "
                   f"recuperados: {', '.join(respuesta.citas_invalidas)}. "
                   "Esa parte de la respuesta no está respaldada.")
    if not respuesta.casos:
        return
    # El texto completo de cada caso es la base de cada afirmación: el médico
    # tiene que poder revisarla por su cuenta.
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
            st.markdown(tarjeta_caso(c, respuesta.consulta_reescrita or consulta),
                        unsafe_allow_html=True)


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
        st.markdown(contenido)
        if respuesta is not None:
            detalle_respuesta(respuesta, segundos, consulta)


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
                              filtros=filtros, umbral=umbral, umbral_rerank=umbral_rerank)
        segundos = time.perf_counter() - t0

        if r.error:
            st.error(r.error)
            texto = f"_{r.error}_"
        else:
            texto = r.texto
            st.write_stream(escribir_de_a_poco(texto))
        detalle_respuesta(r, segundos, pregunta)

    st.session_state.mensajes.append(
        {"role": "assistant", "content": texto, "respuesta": r, "segundos": segundos,
         "consulta": pregunta}
    )
