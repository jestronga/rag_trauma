"""
Las cuatro rutas de recuperación, tras una interfaz común.

    bm25    léxica pura
    densa   bge-m3 sobre Qdrant
    rrf     fusión Reciprocal Rank Fusion de las dos anteriores
    rerank  rrf + cross-encoder sobre los candidatos

RRF fusiona por POSICIÓN, no por puntaje. Es deliberado: los puntajes de BM25
y del coseno no son comparables ni normalizables de forma estable entre
consultas, y cualquier combinación lineal exige calibrar un peso que se
desajusta con cada consulta nueva. RRF no necesita calibración.

    RRF(d) = Σ_rutas 1 / (k + posición(d))     con k = 60

Todas las rutas devuelven `Resultado`, con `score_norm` en [0, 1] para que la
abstención tenga un criterio comparable: el coseno del mejor caso.

Los libros de trauma (colección COLECCION_GUIAS) se buscan aparte, con
`buscar_guias`, y devuelven `FragmentoGuia`: nunca se mezclan con los casos.
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import config
from .corpus import tokenizar

# Qdrant en modo local avisa a partir de 20.000 puntos. Medido sobre este
# corpus (22.176 docs): ~85 ms por consulta densa, ~100 ms con RRF. Suficiente
# para un chatbot. Si el corpus crece mucho, se pasa a Qdrant en Docker
# cambiando solo el constructor de _qdrant().
warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
warnings.filterwarnings("ignore", message=".*Local mode is not recommended.*")

RUTAS = ("bm25", "densa", "rrf", "rerank")


@dataclass
class Resultado:
    record_id: str
    text: str
    metadata: dict
    score: float
    score_norm: float = 0.0          # coseno denso: comparable entre consultas y rutas
    posiciones: dict = field(default_factory=dict)   # ruta -> posición, para depurar RRF
    # Probabilidad del cross-encoder (solo ruta rerank). Va aparte del coseno:
    # la interfaz y SIMILITUD_BAJA leen score_norm en escala de coseno.
    relevancia: float | None = None

    @property
    def truncado(self) -> bool:
        return bool(self.metadata.get("truncado"))


@dataclass
class Filtros:
    mecanismo: str | None = None
    region: str | None = None
    desenlace: str | None = None
    pediatrico: bool | None = None
    excluir_truncados: bool = False

    def vacio(self) -> bool:
        return not any([self.mecanismo, self.region, self.desenlace,
                        self.pediatrico is not None, self.excluir_truncados])

    def acepta(self, meta: dict) -> bool:
        if self.mecanismo and meta.get("mecanismo_lesion") != self.mecanismo:
            return False
        if self.region and self.region not in (meta.get("region_anatomica") or []):
            return False
        if self.desenlace and meta.get("desenlace") != self.desenlace:
            return False
        if self.pediatrico is not None and bool(meta.get("es_pediatrico")) != self.pediatrico:
            return False
        if self.excluir_truncados and meta.get("truncado"):
            return False
        return True


# ---------------------------------------------------------------------------
# Carga perezosa de los índices (se cachean para que Streamlit no recargue)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _bm25():
    if not config.BM25_PATH.exists():
        raise SystemExit(f"Falta el índice BM25 ({config.BM25_PATH}). Corre: python -m rag.index")
    with config.BM25_PATH.open("rb") as f:
        d = pickle.load(f)
    return d["bm25"], d["ids"]


@lru_cache(maxsize=1)
def _qdrant():
    import atexit
    from qdrant_client import QdrantClient
    if not config.QDRANT_PATH.exists():
        raise SystemExit(f"Falta el índice denso ({config.QDRANT_PATH}). Corre: python -m rag.index")
    cliente = QdrantClient(path=str(config.QDRANT_PATH))
    # sin esto, el __del__ del cliente explota durante el apagado del intérprete
    atexit.register(lambda: _cerrar(cliente))
    return cliente


def _cerrar(cliente) -> None:
    try:
        cliente.close()
    except Exception:
        pass


@lru_cache(maxsize=1)
def _encoder():
    from .encoder import cargar_encoder
    return cargar_encoder(config.DISPOSITIVO_CONSULTA, precision=config.PRECISION_CONSULTA)


@lru_cache(maxsize=1)
def vram_gb() -> float:
    """VRAM total de la GPU NVIDIA, o 0 si no hay (o torch no la ve)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except ImportError:
        pass
    return 0.0


def _reranker_cabe() -> bool:
    """El reranker en GPU (~1,3 GB) solo cabe junto al generador con 7 GB o más.
    En la GTX 1650 (4 GB), qwen3:4b-instruct ya ocupa ~3,0: juntos, Ollama da HTTP 500."""
    return vram_gb() >= config.VRAM_MIN_RERANK


def ruta_por_defecto() -> str:
    """RUTA=auto: rerank si el reranker cabe en la GPU; si no, rrf (en CPU tarda ~30 s)."""
    if config.RUTA_DEFECTO != "auto":
        return config.RUTA_DEFECTO
    return "rerank" if _reranker_cabe() else "rrf"


def dispositivo_reranker() -> str:
    if config.DISPOSITIVO_RERANKER != "auto":
        return config.DISPOSITIVO_RERANKER
    return "cuda" if _reranker_cabe() else "cpu"


@lru_cache(maxsize=1)
def _reranker():
    import torch
    from sentence_transformers import CrossEncoder
    dev = dispositivo_reranker()
    kwargs = {"dtype": torch.float16} if dev.startswith("cuda") else {}
    return CrossEncoder(config.MODELO_RERANKER, device=dev, max_length=512, model_kwargs=kwargs)


def _probabilidades(pares: list[tuple[str, str]]) -> np.ndarray:
    """Probabilidad de relevancia de cada par (consulta, caso).

    sentence-transformers >= 4 ya aplica la sigmoide a los modelos de una
    etiqueta; aplicarla otra vez comprimía todo a [0,5, 0,73]. Se piden los
    logits crudos (Identity) y la sigmoide se aplica una sola vez aquí.
    """
    import torch
    logits = _reranker().predict(pares, batch_size=16, show_progress_bar=False,
                                 activation_fn=torch.nn.Identity())
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))


@lru_cache(maxsize=1)
def _payloads() -> dict:
    """record_id -> payload. BM25 solo devuelve posiciones; el texto vive aquí."""
    cliente = _qdrant()
    salida, offset = {}, None
    while True:
        puntos, offset = cliente.scroll(
            config.COLECCION, limit=2048, offset=offset, with_payload=True, with_vectors=False
        )
        for p in puntos:
            salida[str(p.payload["record_id"])] = p.payload
        if offset is None:
            break
    return salida


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
def buscar_bm25(consulta: str, k: int, filtros: Filtros | None = None) -> list[Resultado]:
    bm25, ids = _bm25()
    pays = _payloads()
    puntajes = bm25.get_scores(tokenizar(consulta, expandir=True))

    orden = np.argsort(puntajes)[::-1]
    salida: list[Resultado] = []
    for i in orden:
        if puntajes[i] <= 0:
            break
        rid = ids[i]
        pay = pays.get(rid)
        if pay is None:
            continue
        if filtros and not filtros.acepta(pay):
            continue
        # score_norm se deja en 0: BM25 no tiene escala absoluta. Normalizarlo
        # contra el máximo de la MISMA consulta haría que el top-1 valga
        # siempre 1.0 y la abstención nunca se dispararía. La confianza se
        # calcula aparte, con el coseno denso -> confianza().
        salida.append(
            Resultado(rid, pay.get("text", ""), _meta(pay), float(puntajes[i]), 0.0)
        )
        if len(salida) >= k:
            break
    return salida


def _vector_consulta(consulta: str) -> np.ndarray:
    return _encoder().encode([config.PREFIJO_CONSULTA + consulta],
                             normalize_embeddings=True, convert_to_numpy=True)[0]


def buscar_densa(consulta: str, k: int, filtros: Filtros | None = None) -> list[Resultado]:
    from qdrant_client import models

    vector = _vector_consulta(consulta)

    condiciones = []
    if filtros and not filtros.vacio():
        if filtros.mecanismo:
            condiciones.append(models.FieldCondition(
                key="mecanismo_lesion", match=models.MatchValue(value=filtros.mecanismo)))
        if filtros.region:
            condiciones.append(models.FieldCondition(
                key="region_anatomica", match=models.MatchValue(value=filtros.region)))
        if filtros.desenlace:
            condiciones.append(models.FieldCondition(
                key="desenlace", match=models.MatchValue(value=filtros.desenlace)))
        if filtros.pediatrico is not None:
            condiciones.append(models.FieldCondition(
                key="es_pediatrico", match=models.MatchValue(value=filtros.pediatrico)))
        if filtros.excluir_truncados:
            condiciones.append(models.FieldCondition(
                key="truncado", match=models.MatchValue(value=False)))

    puntos = _qdrant().query_points(
        config.COLECCION,
        query=vector.tolist(),
        limit=k,
        with_payload=True,
        query_filter=models.Filter(must=condiciones) if condiciones else None,
    ).points

    # Vectores normalizados + distancia coseno -> score ya está en [-1, 1]
    return [
        Resultado(str(p.payload["record_id"]), p.payload.get("text", ""), _meta(p.payload),
                  float(p.score), max(0.0, float(p.score)))
        for p in puntos
    ]


def buscar_rrf(consulta: str, k: int, filtros: Filtros | None = None,
               profundidad: int | None = None) -> list[Resultado]:
    prof = profundidad or config.TOP_K_CANDIDATOS
    listas = {"bm25": buscar_bm25(consulta, prof, filtros),
              "densa": buscar_densa(consulta, prof, filtros)}

    acumulado: dict[str, float] = {}
    posiciones: dict[str, dict] = {}
    objetos: dict[str, Resultado] = {}
    for ruta, lista in listas.items():
        for pos, r in enumerate(lista, start=1):
            acumulado[r.record_id] = acumulado.get(r.record_id, 0.0) + 1.0 / (config.RRF_K + pos)
            posiciones.setdefault(r.record_id, {})[ruta] = pos
            objetos.setdefault(r.record_id, r)

    ordenados = sorted(acumulado.items(), key=lambda kv: kv[1], reverse=True)[:k]
    # La suma RRF no es interpretable como confianza (depende de cuántas rutas
    # vieron el documento). Se hereda el coseno denso cuando existe, y 0 si el
    # documento solo lo trajo BM25: es el valor conservador.
    cos = {r.record_id: r.score_norm for r in listas["densa"]}
    return [
        Resultado(rid, objetos[rid].text, objetos[rid].metadata, puntaje,
                  cos.get(rid, 0.0), posiciones.get(rid, {}))
        for rid, puntaje in ordenados
    ]


def buscar_rerank(consulta: str, k: int, filtros: Filtros | None = None,
                  profundidad: int | None = None) -> list[Resultado]:
    candidatos = buscar_rrf(consulta, profundidad or config.PROFUNDIDAD_RERANK, filtros)
    if not candidatos:
        return []
    probas = _probabilidades([(consulta, c.text) for c in candidatos])
    orden = np.argsort(probas)[::-1][:k]
    return [
        Resultado(candidatos[i].record_id, candidatos[i].text, candidatos[i].metadata,
                  float(probas[i]), candidatos[i].score_norm, candidatos[i].posiciones,
                  relevancia=float(probas[i]))
        for i in orden
    ]


_DESPACHO = {
    "bm25": buscar_bm25,
    "densa": buscar_densa,
    "rrf": buscar_rrf,
    "rerank": buscar_rerank,
}


def buscar(consulta: str, k: int | None = None, ruta: str | None = None,
           filtros: Filtros | None = None) -> list[Resultado]:
    ruta = (ruta or ruta_por_defecto()).lower()
    if ruta not in _DESPACHO:
        raise ValueError(f"Ruta desconocida: {ruta!r}. Opciones: {', '.join(RUTAS)}")
    return _DESPACHO[ruta](consulta, k or config.TOP_K, filtros)


def confianza(consulta: str, filtros: Filtros | None = None) -> float:
    """¿Hay ALGO en el corpus parecido a esta consulta? Coseno del mejor caso.

    Es la única señal con escala absoluta y comparable entre consultas, así que
    es la que decide la abstención — independientemente de la ruta que se use
    para ordenar los resultados. Cuesta una búsqueda densa (~85 ms).
    """
    mejor = buscar_densa(consulta, 1, filtros)
    return mejor[0].score_norm if mejor else 0.0


def por_id(record_id: str) -> Resultado | None:
    """Recuperar un caso concreto por su identificador."""
    pay = _payloads().get(str(record_id).strip())
    if pay is None:
        return None
    return Resultado(str(pay["record_id"]), pay.get("text", ""), _meta(pay), 1.0, 1.0)


_CAMPOS_META = ("mecanismo_lesion", "region_anatomica", "desenlace", "es_pediatrico",
                "truncado", "campos_truncados", "doc_largo", "n_chars", "abreviaturas")


def _meta(payload: dict) -> dict:
    return {c: payload.get(c) for c in _CAMPOS_META if c in payload}


# ---------------------------------------------------------------------------
# Libros de trauma (PLAN_INDEXAR_LIBROS.md, fase 4)
# ---------------------------------------------------------------------------
RUTAS_GUIAS = ("densa", "rerank")


@dataclass
class FragmentoGuia:
    """Un fragmento de los libros. Es un tipo aparte de `Resultado` a
    propósito: un fragmento en la lista de casos rompería los conteos ("2 de
    5") de hallazgos.py, la tabla y verificar.py."""
    id: str                     # ATLS-07-019-02: sección + número de fragmento
    seccion_id: str
    fuente: str                 # ATLS | HUV
    libro: str
    capitulo: str
    titulos: list[str]
    pagina_ini: int             # páginas impresas: las que se citan
    pagina_fin: int
    tipo: str                   # texto | tabla
    idioma: str                 # en (ATLS) | es (HUV)
    texto: str                  # lo que se embebe: ruta de títulos + texto
    texto_contexto: str         # lo que va al prompt
    n_tokens_contexto: int
    coseno: float
    relevancia: float | None = None     # probabilidad del reranker (ruta rerank)

    @property
    def puntaje(self) -> float:
        """Lo que decide la guarda: la probabilidad si hubo reranker, si no el coseno."""
        return self.coseno if self.relevancia is None else self.relevancia


def ruta_guias_por_defecto() -> str:
    """RUTA_GUIAS=auto: rerank si el reranker cabe en la GPU; si no, densa."""
    if config.RUTA_GUIAS != "auto":
        return config.RUTA_GUIAS
    return "rerank" if _reranker_cabe() else "densa"


def umbral_guias(ruta: str) -> float:
    return config.UMBRAL_GUIAS if ruta == "rerank" else config.UMBRAL_GUIAS_COSENO


@lru_cache(maxsize=1)
def hay_guias() -> bool:
    """¿Se indexaron los libros? (python -m rag.index --fuente guias)"""
    return config.QDRANT_PATH.exists() and _qdrant().collection_exists(config.COLECCION_GUIAS)


def _fragmento(punto) -> FragmentoGuia:
    p = punto.payload
    return FragmentoGuia(
        id=str(p["record_id"]), seccion_id=p["seccion_id"], fuente=p["fuente"],
        libro=p.get("libro", ""), capitulo=p.get("capitulo", ""), titulos=list(p.get("titulos", [])),
        pagina_ini=p.get("pagina_ini"), pagina_fin=p.get("pagina_fin"), tipo=p.get("tipo", "texto"),
        idioma=p.get("idioma", ""), texto=p.get("text", ""),
        texto_contexto=p.get("texto_contexto", p.get("text", "")),
        n_tokens_contexto=int(p.get("n_tokens_contexto", p.get("n_tokens", 0))),
        coseno=float(punto.score),
    )


def candidatos_guias(consulta: str, ruta: str | None = None,
                     profundidad: int | None = None) -> list[FragmentoGuia]:
    """Los `profundidad` fragmentos más cercanos por la densa, de mejor a peor;
    con la ruta rerank, reordenados por el cross-encoder. Sin guarda ni
    presupuesto: eso lo hace `seleccionar_guias` (y así calibrar_guias.py
    prueba varios umbrales sin repetir la búsqueda)."""
    ruta = (ruta or ruta_guias_por_defecto()).lower()
    if ruta not in RUTAS_GUIAS:
        raise ValueError(f"Ruta de libros desconocida: {ruta!r}. Opciones: {', '.join(RUTAS_GUIAS)}")
    if not hay_guias():
        raise SystemExit(f"Falta la colección de los libros ({config.COLECCION_GUIAS!r}). "
                         "Corre: python -m rag.index --fuente guias")
    puntos = _qdrant().query_points(
        config.COLECCION_GUIAS,
        query=_vector_consulta(consulta).tolist(),
        limit=profundidad or config.PROFUNDIDAD_RERANK,
        with_payload=True,
    ).points
    frags = [_fragmento(p) for p in puntos]
    if ruta == "rerank" and frags:
        for f, p in zip(frags, _probabilidades([(consulta, f.texto) for f in frags])):
            f.relevancia = float(p)
        frags.sort(key=lambda f: f.relevancia, reverse=True)
    return frags


def seleccionar_guias(candidatos: list[FragmentoGuia], umbral: float,
                      presupuesto: int) -> list[FragmentoGuia]:
    """Guarda y presupuesto. Si el mejor fragmento no llega a `umbral`, los
    libros se abstienen (lista vacía). Si llega, entran en orden mientras
    quepan en `presupuesto` tokens, sin repetir contexto (small-to-big trae el
    mismo padre para varios hijos). Se corta en el primero que no cabe, como
    R@ppto en scripts/comparar_fragmentacion.py: así lo medido es lo que llega
    al prompt."""
    if not candidatos or max(f.puntaje for f in candidatos) < umbral:
        return []
    salida, tokens, vistos = [], 0, set()
    for f in candidatos:
        if f.texto_contexto in vistos:
            continue
        if tokens + f.n_tokens_contexto > presupuesto:
            break
        vistos.add(f.texto_contexto)
        tokens += f.n_tokens_contexto
        salida.append(f)
    return salida


def buscar_guias(consulta: str, ruta: str | None = None) -> list[FragmentoGuia]:
    """Fragmentos de los libros para el prompt; vacío si los libros no tratan
    la consulta (la guarda de esta fuente no pasa)."""
    ruta = (ruta or ruta_guias_por_defecto()).lower()
    return seleccionar_guias(candidatos_guias(consulta, ruta), umbral_guias(ruta),
                             config.PRESUPUESTO_GUIAS)
