"""Pattern generators for different chess narratives."""

from .base import PatternGenerator
from .markov import MarkovChainPattern

__all__ = ['PatternGenerator', 'MarkovChainPattern']
