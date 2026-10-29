"""ActionCodec2Sparse: ActionCodec2 physics with a zero-free spatio-temporal Set-BPE token layer."""

from .fitting import SparseProfileTrainer, sparse_profile_config
from .grammar import SparseGrammar
from .layout import SparseLayout
from .processor import ActionCodec2SparseProcessor
from .routing import ActionCodec2SparseRouter, RoutedSparseGrammar
from .training import fit_sparse_vocabulary
from .vocabulary import SparseVocabulary

__all__ = [
    "ActionCodec2SparseProcessor",
    "ActionCodec2SparseRouter",
    "RoutedSparseGrammar",
    "SparseGrammar",
    "SparseLayout",
    "SparseProfileTrainer",
    "SparseVocabulary",
    "fit_sparse_vocabulary",
    "sparse_profile_config",
]
