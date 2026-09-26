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
    score_norm: float = 0.0          # comparable entre consultas: coseno o prob. del reranker
    posiciones: dict = field(default_factory=dict)   # ruta -> posición, para depurar RRF

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
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.MODELO_RERANKER, device=config.DISPOSITIVO_RERANKER, max_length=512)


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


def buscar_densa(consulta: str, k: int, filtros: Filtros | None = None) -> list[Resultado]:
    from qdrant_client import models

    vector = _encoder().encode([config.PREFIJO_CONSULTA + consulta],
                               normalize_embeddings=True, convert_to_numpy=True)[0]

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
    candidatos = buscar_rrf(consulta, profundidad or config.TOP_K_CANDIDATOS, filtros)
    if not candidatos:
        return []
    pares = [(consulta, c.text) for c in candidatos]
    puntajes = _reranker().predict(pares, batch_size=16, show_progress_bar=False)
    # El cross-encoder devuelve logits: sigmoide -> [0, 1] interpretable
    probas = 1.0 / (1.0 + np.exp(-np.asarray(puntajes, dtype=float)))
    orden = np.argsort(probas)[::-1][:k]
    return [
        Resultado(candidatos[i].record_id, candidatos[i].text, candidatos[i].metadata,
                  float(probas[i]), float(probas[i]), candidatos[i].posiciones)
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
    ruta = (ruta or config.RUTA_DEFECTO).lower()
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
