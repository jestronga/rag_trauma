"""Los tests corren sin Ollama: la traducción de la consulta para el ATLS 11
(TRADUCIR_ATLS11, rag/consulta.py) se apaga en todos. Los que la prueban la
encienden y reemplazan la llamada al modelo."""
from __future__ import annotations

import pytest

from rag import config, consulta


@pytest.fixture(autouse=True)
def sin_traduccion(monkeypatch):
    monkeypatch.setattr(config, "TRADUCIR_ATLS11", False)
    monkeypatch.setattr(consulta, "_traducciones", {})
