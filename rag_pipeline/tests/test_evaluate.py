"""Ideal común de la evaluación: el nDCG tiene que comparar rutas entre sí."""
from eval.evaluate import ideal_comun, ndcg_at, recall_at
from rag.retrieve import Resultado

CONSULTA = {"mecanismo": "arma_de_fuego", "regiones": ["torax"]}
BUENO = {"mecanismo_lesion": "arma_de_fuego", "region_anatomica": ["torax"]}
MALO = {"mecanismo_lesion": "caida", "region_anatomica": ["cara"]}


def _res(rid, meta):
    return Resultado(rid, "", meta, 0.0)


def test_ruta_que_trae_menos_relevantes_no_saca_1():
    # antes el ideal salía de la propia ruta: 1 relevante en la posición 1 valía 1,0
    resultados = {
        "pobre": [_res("a", BUENO), _res("x", MALO)],
        "rica": [_res("a", BUENO), _res("b", BUENO), _res("c", BUENO)],
    }
    pool = ideal_comun(resultados, CONSULTA)
    ideal = list(pool.values())
    pobre = ndcg_at([pool[r.record_id] for r in resultados["pobre"]], 10, ideal)
    rica = ndcg_at([pool[r.record_id] for r in resultados["rica"]], 10, ideal)
    assert rica == 1.0
    assert pobre < 0.5


def test_caso_repetido_entre_rutas_cuenta_una_vez():
    resultados = {"bm25": [_res("a", BUENO)], "densa": [_res("a", BUENO), _res("x", MALO)]}
    assert ideal_comun(resultados, CONSULTA) == {"a": 2, "x": 0}


def test_recall_sobre_el_pool():
    assert recall_at(["a", "x"], 10, {"a", "b"}) == 0.5
    assert recall_at(["a"], 10, set()) == 0.0
