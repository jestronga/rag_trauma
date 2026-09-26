"""Carga del corpus y tokenización para el índice disperso."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Documento:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def truncado(self) -> bool:
        return bool(self.metadata.get("truncado"))


def cargar(ruta: Path | str | None = None) -> list[Documento]:
    ruta = Path(ruta or config.CORPUS_JSONL)
    if not ruta.exists():
        raise SystemExit(
            f"No encuentro el corpus en {ruta}.\n"
            "Genéralo con:  python ../anonimizacion/run_pipeline.py"
        )
    docs = []
    with ruta.open(encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                d = json.loads(linea)
                docs.append(Documento(d["id"], d["text"], d.get("metadata", {})))
    return docs


# ---------------------------------------------------------------------------
# Tokenización para BM25
#
# Aquí SÍ se quitan tildes y se baja a minúsculas, al revés que en el texto que
# va al modelo denso. BM25 hace match exacto de términos, así que 'REGIÓN' y
# 'region' tienen que colapsar o el corpus se parte en dos vocabularios: solo
# 10.497 de 27.157 registros originales usaban tildes.
# ---------------------------------------------------------------------------
_ACENTO = re.compile(r"[̀-ͯ]")
_TOKEN = re.compile(r"[a-zñ0-9]+")

VACIAS = {
    "a", "al", "algo", "ante", "antes", "aqui", "asi", "aun", "cada", "como",
    "con", "contra", "cual", "cuando", "de", "del", "desde", "donde", "dos",
    "el", "ella", "ellos", "en", "entre", "era", "es", "esa", "ese", "eso",
    "esta", "estan", "este", "esto", "estos", "fue", "fueron", "ha", "hace",
    "han", "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas", "me",
    "mi", "mientras", "muy", "ni", "no", "nos", "o", "otra", "otro", "para",
    "pero", "por", "porque", "que", "quien", "se", "segun", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "tal", "tambien", "te", "tiene", "todo",
    "todos", "tras", "un", "una", "uno", "unos", "y", "ya",
}

# Sinónimos de consulta: registro médico -> como aparece en el corpus.
# El campo `Motivo de consulta` está en voz del paciente ("ME APUÑALARON",
# "ME DISPARARON"), mientras que quien usa el sistema escribe en registro
# clínico. Sin este puente, BM25 falla en consultas perfectamente razonables.
SINONIMOS_CONSULTA = {
    "apunalado": ["apunalaron", "hpacp", "cortopunzante"],
    "punalada": ["apunalaron", "hpacp", "cortopunzante"],
    "baleado": ["dispararon", "hpaf", "fuego"],
    "disparo": ["dispararon", "hpaf", "fuego"],
    "balazo": ["dispararon", "hpaf", "fuego"],
    "atropellado": ["atropello", "transito"],
    "golpeado": ["golpearon", "contundente"],
    "agredido": ["agresion", "rina", "pelea"],
    "precipitacion": ["caida", "cayo", "altura"],
    "quemado": ["quemadura", "quemo"],
}


def normalizar(texto: str) -> str:
    return _ACENTO.sub("", unicodedata.normalize("NFD", texto)).lower()


def tokenizar(texto: str, expandir: bool = False) -> list[str]:
    toks = [t for t in _TOKEN.findall(normalizar(texto)) if t not in VACIAS and len(t) > 1]
    if expandir:
        extra = []
        for t in toks:
            extra.extend(SINONIMOS_CONSULTA.get(t, ()))
        toks = toks + extra
    return toks
