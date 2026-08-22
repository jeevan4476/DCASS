# src/diagnostics/__init__.py
"""
Runtime diagnostics for DCASS ("dcass doctor").

Validates the entire runtime in one shot: dependencies, index/metadata
agreement, codebook binding (count check + content fingerprints), cluster
population histogram, empty clusters, checkpoints, disk footprint.

Produces the Phase-0 ground-truth numbers and defines "system ready".
"""

from .doctor import DoctorReport, CheckResult, run_doctor

__all__ = ["DoctorReport", "CheckResult", "run_doctor"]
