# src/engine/context/__init__.py
"""
Dynamic Context-Aware keying for DCASS.

This is the layer the project name refers to: per-epoch permutations of the
cluster -> byte mapping derived from a shared context source (time bucket +
optional public signal + optional out-of-band secret). See key_manager.py
for scope and security properties.
"""

from .key_manager import (
    ContextEpoch,
    ContextKeyManager,
    apply_symbol_permutation,
    invert_symbol_permutation,
)

__all__ = [
    "ContextEpoch",
    "ContextKeyManager",
    "apply_symbol_permutation",
    "invert_symbol_permutation",
]
