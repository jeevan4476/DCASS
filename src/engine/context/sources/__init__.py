# src/engine/context/sources/__init__.py
"""
External context signal fetchers.

Each source returns a short, slowly-varying, PUBLIC string that both Alice
and Bob can independently obtain. Fetch failures must never break the
channel - ContextKeyManager degrades to the remaining sources.
"""

__all__ = []
