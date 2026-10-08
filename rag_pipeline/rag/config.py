"""
Configuración central del prototipo RAG.

Presupuesto de VRAM para una GPU de 8 GB — es la restricción que manda:

    bge-m3          ~2,3 GB   |  reranker-v2-m3  ~2,3 GB
    Qwen2.5 7B Q4   ~4,7 GB   |  contexto 4k     ~0,5 GB

Las tres piezas juntas no caben. El reparto por defecto:

    · bge-m3 en GPU SOLO durante la indexación (una vez, minutos)
    · en consulta, bge-m3 en CPU: embeber una consulta corta cuesta ~150 ms
    · Ollama se queda con la GPU para el generador
    · el reranker está APAGADO por defecto; se enciende con RERANKER=1
      y en 8 GB conviene correrlo en CPU (sobre 50 candidatos, ~1 s)

Todo se puede sobrescribir por variable de entorno.
"""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
# El corpus lo produce el pipeline de anonimización, que vive en una carpeta hermana.
RAIZ_ANONIMIZACION = RAIZ.parent / "anonimizacion"


def _env(nombre: str, defecto: str) -> str:
    return os.environ.get(nombre, defecto)


def _env_bool(nombre: str, defecto: bool = False) -> bool:
    v = os.environ.get(nombre)
    return defecto if v is None else v.strip().lower() in {"1", "true", "si", "sí", "yes"}


def _env_int(nombre: str, defecto: int) -> int:
    try:
        return int(os.environ.get(nombre, defecto))
    except ValueError:
        return defecto


def _env_float(nombre: str, defecto: float) -> float:
    try:
        return float(os.environ.get(nombre, defecto))
    except ValueError:
        return defecto


# --- rutas -----------------------------------------------------------------
CORPUS_JSONL = Path(_env("CORPUS_JSONL", str(RAIZ_ANONIMIZACION / "outputs" / "corpus_indexable.jsonl")))
INDICE_DIR = Path(_env("INDICE_DIR", str(RAIZ / "indices")))
QDRANT_PATH = INDICE_DIR / "qdrant"          # modo local embebido: sin Docker
BM25_PATH = INDICE_DIR / "bm25.pkl"
META_PATH = INDICE_DIR / "meta.json"
COLECCION = _env("COLECCION", "trauma")
# Libros de trauma (PLAN_INDEXAR_LIBROS.md): los PDF viven junto a trauma.csv,
# fuera de git; el texto extraído también queda fuera (derechos de autor).
LIBROS_DIR = Path(_env("LIBROS_DIR", str(RAIZ_ANONIMIZACION / "data")))
GUIAS_DIR = RAIZ / "guias"
GUIAS_PERFILES = GUIAS_DIR / "perfiles.yaml"
GUIAS_SECCIONES = Path(_env("GUIAS_SECCIONES", str(GUIAS_DIR / "secciones.jsonl")))

# --- modelos ---------------------------------------------------------------
MODELO_EMBEDDING = _env("MODELO_EMBEDDING", "BAAI/bge-m3")
DIM_EMBEDDING = _env_int("DIM_EMBEDDING", 1024)
MODELO_RERANKER = _env("MODELO_RERANKER", "BAAI/bge-reranker-v2-m3")
# Qwen3-4B-Instruct-2507: cabe entero en 4 GB de VRAM (ver NUM_GPU). OJO: la
# etiqueta `qwen3:4b` es la variante Thinking, que razona siempre aunque se
# pida "think": false; con 900 tokens no alcanzaba a escribir las sugerencias
# y tardaba ~80 s. Sin GPU NVIDIA y poca RAM: MODELO_LLM=qwen2.5:3b.
MODELO_LLM = _env("MODELO_LLM", "qwen3:4b-instruct")
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")

# bge-m3 es un modelo de recuperación asimétrico: la consulta y el documento
# se codifican igual, pero el modelo espera textos, no palabras sueltas.
PREFIJO_CONSULTA = _env("PREFIJO_CONSULTA", "")

DISPOSITIVO_INDEXACION = _env("DISPOSITIVO_INDEXACION", "auto")   # auto | cuda | cpu
DISPOSITIVO_CONSULTA = _env("DISPOSITIVO_CONSULTA", "cpu")
# bge-m3 en CPU: fp32 ocupa ~2,3 GB de RAM; bf16, ~1,1 GB. Medido sobre este
# índice (construido en fp32): coseno 0,9999 entre ambos vectores de consulta
# y el mismo top-5. Esa RAM es la que necesita Ollama para cargar el generador.
PRECISION_CONSULTA = _env("PRECISION_CONSULTA", "bf16")   # bf16 | fp32
LOTE_INDEXACION = _env_int("LOTE_INDEXACION", 32)

# --- recuperación ----------------------------------------------------------
# auto = rerank si torch ve una GPU NVIDIA, rrf si no (retrieve.ruta_por_defecto).
# El reranker en CPU no es viable: medido en un i5-12450H, 31 s por consulta.
RUTA_DEFECTO = _env("RUTA", "auto")           # auto | bm25 | densa | rrf | rerank
TOP_K = _env_int("TOP_K", 5)                   # casos que se le pasan al generador
TOP_K_CANDIDATOS = _env_int("TOP_K_CANDIDATOS", 50)   # profundidad antes de fusionar
RRF_K = _env_int("RRF_K", 60)                  # constante estándar de Reciprocal Rank Fusion
# auto = cuda si hay GPU NVIDIA; en CUDA se carga en fp16 (~1,1 GB de pesos).
# Presupuesto en 8 GB: qwen3:4b-instruct (~3,0 GB con NUM_CTX=3072) + reranker
# (~1,3 GB) + Windows (~0,5 GB) caben. Con qwen3:8b (~6 GB) NO: poner
# DISPOSITIVO_RERANKER=cpu y RUTA=rrf, o Ollama se queda sin memoria (HTTP 500).
DISPOSITIVO_RERANKER = _env("DISPOSITIVO_RERANKER", "auto")   # auto | cuda | cpu
# RUTA=auto usa rerank solo con esta VRAM o más (GTX 1650 de 4 GB → rrf).
VRAM_MIN_RERANK = _env_float("VRAM_MIN_RERANK", 7.0)
# Candidatos de RRF que se reordenan. Cada par cuesta un pase del cross-encoder
# (los casos miden 118 tokens de mediana, 371 como máximo): es la palanca de la
# latencia.
PROFUNDIDAD_RERANK = _env_int("PROFUNDIDAD_RERANK", 30)
# Probabilidad mínima del reranker para que un caso llegue al generador. Si
# pasan menos de MIN_CASOS, el sistema se abstiene: el cross-encoder lee la
# consulta y el caso juntos y separa lo que el coseno de bge-m3 no separa
# ("cetoacidosis diabética" sacaba 0,50 de coseno). Calibrado con
# scripts/calibrar_umbral.py (27/09/2026, 25 consultas de trauma y 10 fuera de
# dominio, profundidad 30), sobre la probabilidad del 2.º caso:
#   0,1 → 0 abstenciones falsas, 2 respuestas falsas
#   0,2 → 0 abstenciones falsas, 1 respuesta falsa (apendicitis aguda, 0,42)
#   0,5 → 3 abstenciones falsas, 0 respuestas falsas
# La consulta de trauma más baja: "agresión con objeto romo en región craneal",
# 0,22. Con coseno solo, 8 de 10 fuera de dominio pasaban el umbral de 0,45.
UMBRAL_RERANK = _env_float("UMBRAL_RERANK", 0.2)

# --- abstención ------------------------------------------------------------
# Si el mejor caso no supera este coseno, el sistema dice que no encontró nada
# suficientemente parecido en vez de responder. Calibrado a ojo sobre bge-m3;
# `scripts/calibrar_umbral.py` lo ajusta con las consultas de evaluación.
# 0,52 desde el 28/09/2026: sin reranker (GTX 1650), el coseno es la única
# guarda. Medido: trauma más bajo "TCE" 0,541 y "FX de radio distal" 0,551;
# bloquea cetoacidosis 0,500, infarto 0,497, asma 0,493, apendicitis 0,489,
# diabetes 0,464. NO separa "dolor torácico" (0,59) ni "dolor de cabeza"
# (0,62): eso solo lo hace el reranker (UMBRAL_RERANK).
UMBRAL_ABSTENCION = _env_float("UMBRAL_ABSTENCION", 0.52)
MIN_CASOS = _env_int("MIN_CASOS", 2)
# Por encima del umbral de abstención pero con el mejor caso por debajo de esto,
# la respuesta avisa que los casos pueden no corresponder. Medido (26/09/2026):
# consultas de trauma 0,58-0,72; "cetoacidosis diabética" (fuera de dominio) 0,47.
SIMILITUD_BAJA = _env_float("SIMILITUD_BAJA", 0.55)

# --- generación ------------------------------------------------------------
TEMPERATURA = _env_float("TEMPERATURA", 0.1)   # baja: es un contexto clínico
# Quién escribe la síntesis de "Casos similares". Apagado: la arma el código
# con conteos y manejo extraído (no puede inventar). Probado con el 3B, la
# síntesis libre atribuía manejos y desenlaces a casos que no los tenían.
# Encenderlo solo con un modelo mayor y revisando las respuestas.
SINTESIS_LLM = _env_bool("SINTESIS_LLM", False)
# El modelo responde `{"sugerencias": [...]}` restringido por JSON Schema (campo
# `format` de Ollama) a los términos de la lista de elementos: no puede elegir
# algo sin respaldo. Apagado, vuelve a las viñetas de texto. No aplica con
# SINTESIS_LLM, que necesita texto libre.
SALIDA_ESTRUCTURADA = _env_bool("SALIDA_ESTRUCTURADA", True)
# Qué ve el médico. prosa: el modelo redacta la sugerencia como un asistente
# médico, con citas, y rag/verificar.py la revisa oración por oración; si no
# pasa, se muestra la respuesta determinista. sugerencias: la respuesta
# determinista siempre (el modelo solo elige de la lista).
SALIDA = _env("SALIDA", "prosa")                 # prosa | sugerencias
# Demo 28/09: muestra la prosa aunque no pase la verificación (se sigue
# verificando y queda en Respuesta.motivos). Va contra "nada inventado":
# volver a PROSA_SIEMPRE=0 después de la demo.
PROSA_SIEMPRE = _env_bool("PROSA_SIEMPRE", True)
# La prosa son 3-5 oraciones con citas: ~150-250 tokens.
MAX_TOKENS_PROSA = _env_int("MAX_TOKENS_PROSA", 400)
# Sin SINTESIS_LLM el modelo solo escribe 2-4 viñetas de sugerencias (~100
# tokens medidos). Con SINTESIS_LLM escribe cuatro secciones: ~600-800.
MAX_TOKENS = _env_int("MAX_TOKENS", 900 if SINTESIS_LLM else 250)
MAX_TOKENS_RESUMEN = _env_int("MAX_TOKENS_RESUMEN", 700)
# En CPU el 3B genera ~5 tokens/s: una respuesta completa tarda 2-3 minutos.
TIMEOUT_LLM = _env_int("TIMEOUT_LLM", 400)
# Ventana de contexto de Ollama. Con TOP_K=5 el prompt mide 1.400-2.050
# tokens; 3.072 los cubre con la salida y deja VRAM para meter el modelo entero
# en la GPU. Si un prompt no cabe (p. ej. 10 casos), generate.py agranda la
# ventana solo para esa llamada: sin eso Ollama recorta el comienzo del
# prompt, que es donde están las reglas.
NUM_CTX = _env_int("NUM_CTX", 3072)
# Capas en GPU (-1 = que decida Ollama). Ollama calcula de más y en la GTX 1650
# dejaba 10 de 37 capas en CPU: 33 % CPU, 15 tok/s. Con todas: 100 % GPU,
# 42 tok/s, ~3,0 GB de VRAM. Si da HTTP 500 por memoria (p. ej. el navegador
# ocupa VRAM), NUM_GPU=-1. Solo se aplica con la ventana NUM_CTX.
NUM_GPU = _env_int("NUM_GPU", 99)
VENTANA_HISTORIAL = _env_int("VENTANA_HISTORIAL", 4)   # turnos para reescribir la consulta
# Cuánto deja Ollama el modelo cargado tras cada llamada. Su valor por defecto
# son 5 min: en una demo, la pregunta después de una pausa recargaba el modelo
# (25-35 s en la GTX 1650). Se envía en cada llamada: no hace falta
# OLLAMA_KEEP_ALIVE ni reiniciar Ollama. "-1" = para siempre.
KEEP_ALIVE = _env("KEEP_ALIVE", "2h")

# --- corpus ----------------------------------------------------------------
# El registro llega hasta 2018. Aparece en el prompt y en la interfaz: es una
# limitación real y es mejor declararla que esperar a que alguien la note.
COBERTURA_TEMPORAL = _env("COBERTURA_TEMPORAL", "hasta 2018")

ETIQUETAS_MECANISMO = {
    "arma_de_fuego": "Arma de fuego",
    "cortopunzante": "Arma cortopunzante",
    "accidente_transito": "Accidente de tránsito",
    "caida": "Caída",
    "contundente": "Trauma contundente",
    "quemadura": "Quemadura",
    "mordedura": "Mordedura",
    "cuerpo_extrano": "Cuerpo extraño",
    "intoxicacion": "Intoxicación",
    "autoinfligido": "Autoinfligido",
    "no_determinado": "No determinado",
}
ETIQUETAS_REGION = {
    "craneo_encefalo": "Cráneo / encéfalo",
    "cara": "Cara",
    "ojo": "Ojo",
    "cuello": "Cuello",
    "torax": "Tórax",
    "abdomen": "Abdomen",
    "pelvis": "Pelvis",
    "columna": "Columna",
    "miembro_superior": "Miembro superior",
    "miembro_inferior": "Miembro inferior",
}
ETIQUETAS_DESENLACE = {
    "fallecimiento": "Fallecimiento",
    "cirugia": "Cirugía",
    "uci": "UCI",
    "hospitalizacion": "Hospitalización",
    "remision": "Remisión",
    "alta": "Alta",
    "no_determinado": "No determinado",
}
