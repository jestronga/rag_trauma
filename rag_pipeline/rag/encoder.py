"""
Codificador denso, con tres implementaciones y un orden de preferencia claro.

  1. `transformers` directo (POR DEFECTO). AutoTokenizer + AutoModel, pooling
     CLS y normalización L2 — que es exactamente lo que hace bge-m3 para su
     vector denso. Sin capas intermedias: no depende de la API de
     sentence-transformers, que cambia entre versiones mayores.

  2. `sentence-transformers`  (ENCODER=st). Cómodo, pero la versión 6.x
     reestructuró la carga de modelos y ahora intenta instanciar un
     `AutoProcessor` para todo modelo. bge-m3 es un XLM-RoBERTa sin processor,
     así que falla con:

         ValueError: Unrecognized processing class in BAAI/bge-m3.
         Can't instantiate a processor, a tokenizer, an image processor...

     Si usas esta ruta, fija `sentence-transformers<6`.

  3. `ENCODER=falso`. Sustituto hash determinista. No mide calidad: sirve para
     validar la tubería (índices, rutas, métricas, interfaz) sin descargar
     2,3 GB de pesos.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import config
from .corpus import tokenizar


def dispositivo(preferido: str = "auto") -> str:
    if preferido != "auto":
        return preferido
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# 1. transformers directo — la ruta por defecto
# ---------------------------------------------------------------------------
class EncoderHF:
    """bge-m3 sobre `transformers`, con pooling CLS y normalización L2.

    bge-m3 produce su vector denso tomando el token CLS de la última capa y
    normalizándolo. Implementarlo aquí en diez líneas evita depender de que
    una librería intermedia adivine bien la arquitectura del modelo.
    """

    def __init__(self, nombre: str, dev: str, max_length: int = 512, precision: str = "fp32"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dev = dev
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(nombre)

        # fp16 en GPU: la mitad de VRAM y más rápido, sin pérdida apreciable
        # para recuperación. En CPU, fp32 por defecto (fp16 en CPU va más lento,
        # no más); `precision="bf16"` reduce la RAM a la mitad (~1,1 GB), a
        # costa de ~250 ms por consulta. Ver config.PRECISION_CONSULTA.
        if dev.startswith("cuda"):
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.model = AutoModel.from_pretrained(nombre, dtype=dtype).to(dev).eval()
        self.dim = int(self.model.config.hidden_size)

    def encode(self, textos, batch_size: int = 32, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, convert_to_numpy: bool = True, **_):
        if isinstance(textos, str):
            textos = [textos]
        textos = list(textos)

        iterador = range(0, len(textos), batch_size)
        if show_progress_bar and len(textos) > batch_size:
            try:
                from tqdm.auto import tqdm
                iterador = tqdm(iterador, desc="embeddings", unit="lote")
            except ImportError:
                pass

        salida = []
        with self.torch.no_grad():
            for inicio in iterador:
                lote = textos[inicio:inicio + batch_size]
                entradas = self.tokenizer(
                    lote, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.dev)
                estados = self.model(**entradas).last_hidden_state
                vec = estados[:, 0]                       # pooling CLS
                if normalize_embeddings:
                    vec = self.torch.nn.functional.normalize(vec, p=2, dim=1)
                salida.append(vec.float().cpu().numpy())
        return np.vstack(salida) if salida else np.zeros((0, self.dim), dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


# ---------------------------------------------------------------------------
# 3. Sustituto determinista para pruebas
# ---------------------------------------------------------------------------
class EncoderHash:
    """Hashing trick + normalización L2.

    Captura solapamiento léxico y nada más: se comporta como un BM25 pobre, no
    como un modelo denso. Solo para comprobar que los datos fluyen.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, texto: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenizar(texto):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, textos, batch_size=32, normalize_embeddings=True,
               show_progress_bar=False, convert_to_numpy=True, **_):
        if isinstance(textos, str):
            textos = [textos]
        return np.vstack([self._vector(t) for t in textos])

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


# ---------------------------------------------------------------------------
def _cargar_st(nombre: str, dev: str):
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(nombre, device=dev)
    modelo.max_seq_length = 512
    return modelo


def cargar_encoder(dispositivo_preferido: str | None = None, precision: str = "fp32"):
    tipo = os.environ.get("ENCODER", "hf").lower()
    dev = dispositivo(dispositivo_preferido or config.DISPOSITIVO_INDEXACION)

    if tipo in {"falso", "hash", "fake"}:
        print("    [ENCODER=falso] sustituto hash: solo valida la tubería, no mide calidad")
        return EncoderHash()

    if tipo in {"st", "sentence-transformers", "sbert"}:
        try:
            return _cargar_st(config.MODELO_EMBEDDING, dev)
        except Exception as e:
            print(f"    [aviso] sentence-transformers falló ({type(e).__name__}). "
                  f"Usando transformers directo.")
            if "processing class" in str(e):
                print("    (versión 6.x incompatible con bge-m3: "
                      "pip install 'sentence-transformers<6')")
            return EncoderHF(config.MODELO_EMBEDDING, dev, precision=precision)

    return EncoderHF(config.MODELO_EMBEDDING, dev, precision=precision)
