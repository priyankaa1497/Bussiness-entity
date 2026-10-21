"""ber2 — Business Entity Resolution v2 (high-recall blocking + two-stage LightGBM matcher)."""
import os as _os

# Keep HuggingFace libraries on the PyTorch path only (a broken TensorFlow install must not
# take down the optional embedding channel).
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_TORCH", "1")
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "2.3.0-final-dual"
