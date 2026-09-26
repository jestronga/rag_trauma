"""
Chatbot de consulta sobre el registro de trauma.

    streamlit run app.py

Requiere los índices construidos (`python -m rag.index`) y Ollama corriendo.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import streamlit as st
import yaml

warnings.filterwarnings("ignore")

from rag import config                                    # noqa: E402
from rag.generate import ollama_disponible, responder, resumir_caso   # noqa: E402
from rag.retrieve import RUTAS, Filtros                   # noqa: E402

CONSULTAS_YAML = Path(__file__).resolve().parent / "eval" / "consultas.yaml"

st.set_page_config(page_title="Registro de trauma — consulta de casos",
                   page_icon="🩺", layout="wide")

RX_ID = re.compile(r"^\s*(?:caso|id|registro)?\s*#?\s*(\d{3,8})\s*$", re.IGNORECASE)

CSS = """
<style>
  .stChatMessage { font-size: 0.95rem; }
  .caso-box { border:1px solid rgba(128,128,128,.28); border-radius:8px;
              padding:10px 12px; margin-bottom:10px; font-size:0.82rem; }
  .caso-id  { font-weight:600; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .etq { display:inline-block; padding:1px 7px; border-radius:10px; font-size:0.72rem;
         border:1px solid rgba(128,128,128,.35); margin:2px 3px 2px 0; }
  .trunc { border-color:#c9822f; color:#c9822f; }
  .aviso { font-size:0.78rem; opacity:.75; line-height:1.45; }
  .rotulo-demo { font-size:0.74rem; text-transform:uppercase; letter-spacing:.06em;
                 opacity:.6; margin:6px 0 4px; }
  div[data-testid="column"] .stButton > button { font-size:0.8rem; padding:6px 4px;
                                                 white-space:normal; height:100%; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def consultas_demo(_mtime: float) -> tuple[list[dict], str | None]:
    """Las consultas con `boton` en eval/consultas.yaml.

    Se leen del mismo archivo que usa el arnés de evaluación para que no haya
    dos listas de consultas que mantener por separado: si cambias una consulta
    ahí, cambia también el botón.

    `_mtime` no se usa dentro: existe para que la caché se invalide sola cuando
    editas el YAML, sin reiniciar Streamlit.

    Devuelve (consultas, motivo_del_fallo). El motivo se pinta en la interfaz:
    si los botones no aparecen, hay que poder ver POR QUÉ en vez de mirar un
    hueco en blanco.
    """
    if not CONSULTAS_YAML.exists():
        return [], f"No encuentro {CONSULTAS_YAML}"
    try:
        datos = yaml.safe_load(CONSULTAS_YAML.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return [], f"{CONSULTAS_YAML.name} no es YAML válido: {e}"

    todas = datos.get("consultas") or []
    conb = [c for c in todas if c.get("boton")]
    if not todas:
        return [], f"{CONSULTAS_YAML.name} no tiene la clave «consultas»"
    if not conb:
        return [], (f"{CONSULTAS_YAML.name} tiene {len(todas)} consultas pero ninguna "
                    f"con el campo «boton». ¿Es una versión anterior del archivo?")
    return conb, None


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuración")

    ok_ollama, msg_ollama = ollama_disponible()
    (st.success if ok_ollama else st.error)(msg_ollama)

    ruta = st.selectbox("Ruta de recuperación", RUTAS,
                        index=RUTAS.index(config.RUTA_DEFECTO) if config.RUTA_DEFECTO in RUTAS else 2,
                        help="rrf fusiona BM25 y densa por posición. rerank añade "
                             "un cross-encoder (más VRAM).")
    k = st.slider("Casos recuperados", 3, 10, config.TOP_K)
    umbral = st.slider("Umbral de abstención", 0.0, 0.9, config.UMBRAL_ABSTENCION, 0.05,
                       help="Coseno mínimo del mejor caso para que el sistema responda. "
                            "Más alto = más conservador.")

    st.markdown("### Filtros")
    mecanismo = st.selectbox("Mecanismo", ["(todos)"] + list(config.ETIQUETAS_MECANISMO),
                             format_func=lambda v: config.ETIQUETAS_MECANISMO.get(v, v))
    region = st.selectbox("Región anatómica", ["(todos)"] + list(config.ETIQUETAS_REGION),
                          format_func=lambda v: config.ETIQUETAS_REGION.get(v, v))
    desenlace = st.selectbox("Desenlace", ["(todos)"] + list(config.ETIQUETAS_DESENLACE),
                             format_func=lambda v: config.ETIQUETAS_DESENLACE.get(v, v))
    solo_completos = st.checkbox("Excluir registros truncados", value=False,
                                 help="El 22 % del corpus está cortado a 500 caracteres "
                                      "por el export original.")

    filtros = Filtros(
        mecanismo=None if mecanismo == "(todos)" else mecanismo,
        region=None if region == "(todos)" else region,
        desenlace=None if desenlace == "(todos)" else desenlace,
        excluir_truncados=solo_completos,
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

st.title("Consulta de casos de trauma")
st.caption("Describe una presentación clínica: el sistema recupera casos análogos del "
           "registro, muestra su manejo y desenlace, y sugiere qué valorar según esos "
           "casos. También acepta un identificador de caso.")

# --- Consultas de demostración ---------------------------------------------
# Un clic = una consulta. El orden reproduce el guion de SETUP.md: caso fácil,
# brecha de vocabulario, abreviatura, desenlace y truncamiento, y abstención.
# El tooltip de cada botón dice qué demuestra: sirve de chuleta en vivo.
demo, motivo = consultas_demo(
    CONSULTAS_YAML.stat().st_mtime if CONSULTAS_YAML.exists() else 0.0
)
pregunta_boton = None
if demo:
    st.markdown("<div class='rotulo-demo'>Consultas de demostración</div>",
                unsafe_allow_html=True)
    for col, c in zip(st.columns(len(demo)), demo):
        if col.button(c["boton"], key=f"demo-{c['id']}", use_container_width=True,
                      help=f"«{c['texto']}»\n\n{c.get('muestra', '')}"):
            pregunta_boton = c["texto"]
elif motivo:
    st.warning(f"Botones de demostración no disponibles — {motivo}")


def tarjeta_caso(caso) -> str:
    m = caso.metadata
    etiquetas = []
    if m.get("mecanismo_lesion"):
        etiquetas.append(f"<span class='etq'>{config.ETIQUETAS_MECANISMO.get(m['mecanismo_lesion'], m['mecanismo_lesion'])}</span>")
    for r in (m.get("region_anatomica") or [])[:3]:
        etiquetas.append(f"<span class='etq'>{config.ETIQUETAS_REGION.get(r, r)}</span>")
    if m.get("desenlace") and m["desenlace"] != "no_determinado":
        etiquetas.append(f"<span class='etq'>{config.ETIQUETAS_DESENLACE.get(m['desenlace'], m['desenlace'])}</span>")
    if m.get("truncado"):
        etiquetas.append("<span class='etq trunc'>truncado</span>")
    texto = caso.text.replace("\n", "<br>")
    return (f"<div class='caso-box'><span class='caso-id'>caso {caso.record_id}</span> "
            f"· similitud {caso.score_norm:.2f}<br>{''.join(etiquetas)}<br><br>{texto}</div>")


def pintar(rol: str, contenido: str, respuesta=None) -> None:
    with st.chat_message(rol):
        st.markdown(contenido)
        if respuesta is None:
            return
        if respuesta.citas_invalidas:
            st.warning("El modelo citó identificadores que no están entre los casos "
                       f"recuperados: {', '.join(respuesta.citas_invalidas)}. "
                       "Esa parte de la respuesta no está respaldada.")
        if respuesta.casos:
            etiqueta = f"Casos consultados ({len(respuesta.casos)})"
            if respuesta.consulta_reescrita:
                etiqueta += "  ·  consulta reescrita"
            with st.expander(etiqueta):
                if respuesta.consulta_reescrita:
                    st.caption(f"Búsqueda realizada: «{respuesta.consulta_reescrita}»")
                st.caption(f"Confianza (coseno del mejor caso): {respuesta.confianza:.3f}")
                for c in respuesta.casos:
                    st.markdown(tarjeta_caso(c), unsafe_allow_html=True)


for m in st.session_state.mensajes:
    pintar(m["role"], m["content"], m.get("respuesta"))


# ---------------------------------------------------------------------------
# Turno
# ---------------------------------------------------------------------------
escrita = st.chat_input("Ej.: varón joven con herida por arma cortopunzante en cuello, estable")
pregunta = escrita or pregunta_boton

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    pintar("user", pregunta)

    historial = [{"role": m["role"], "content": m["content"]}
                 for m in st.session_state.mensajes[:-1]]

    with st.chat_message("assistant"):
        with st.spinner("Buscando casos…"):
            m = RX_ID.match(pregunta)
            if m:
                r = resumir_caso(m.group(1))
            else:
                # Las consultas de botón ya son autónomas: reescribirlas contra
                # el historial solo puede estropearlas a mitad de una demo.
                r = responder(pregunta, historial=historial, ruta=ruta, k=k,
                              filtros=filtros, umbral=umbral,
                              reescribir_consulta=escrita is not None)

        if r.error:
            st.error(r.error)
            texto = f"_{r.error}_"
        else:
            texto = r.texto
            st.markdown(texto)
            if r.citas_invalidas:
                st.warning("El modelo citó identificadores que no están entre los casos "
                           f"recuperados: {', '.join(r.citas_invalidas)}. "
                           "Esa parte de la respuesta no está respaldada.")
            if r.casos:
                etiqueta = f"Casos consultados ({len(r.casos)})"
                if r.consulta_reescrita:
                    etiqueta += "  ·  consulta reescrita"
                with st.expander(etiqueta):
                    if r.consulta_reescrita:
                        st.caption(f"Búsqueda realizada: «{r.consulta_reescrita}»")
                    st.caption(f"Confianza (coseno del mejor caso): {r.confianza:.3f}")
                    for c in r.casos:
                        st.markdown(tarjeta_caso(c), unsafe_allow_html=True)

    st.session_state.mensajes.append(
        {"role": "assistant", "content": texto, "respuesta": r}
    )
