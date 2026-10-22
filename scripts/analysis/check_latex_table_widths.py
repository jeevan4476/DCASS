#!/usr/bin/env python3
r"""
Measure each generated LaTeX table's rendered width against the text block.

Compiles every table on its own and reports the overfull hbox width, if
any. A table that overflows the text block cannot be dropped into a
two-column paper as-is; this tells the integration guide which ones need a
full-width float or a width reduction.

The probe document is staged at the project root, because `\input` paths
resolve relative to the source file's directory.

This module is importable: `measure_tables()` returns the measurements so
other tools can act on them.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = PROJECT_ROOT / "storage" / "benchmarks" / "tables"

# Point and text width of the check document, in points. geometry with
# 1in margins on US Letter gives a 6.5in text block.
TEXT_WIDTH_PT = 6.5 * 72.27

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\begin{document}
\input{%s}
\end{document}
"""

OVERFULL = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\)")
STAGED_NAME = "_dcass_width_probe.tex"


def find_tectonic() -> str | None:
    found = shutil.which("tectonic")
    if found:
        return found
    candidate = Path("/tmp/opencode/tectonic")
    return str(candidate) if candidate.exists() else None


def measure_tables(tectonic: str | None = None) -> list[dict]:
    """Compile each table alone and report its overfull width.

    Returns one record per table with the overfull width in points, whether
    it fits, and any LaTeX error line.
    """
    tectonic = tectonic or find_tectonic()
    if tectonic is None:
        raise RuntimeError("no LaTeX engine found")

    build_dir = Path(tempfile.mkdtemp(prefix="dcass_texwidth_"))
    staged = PROJECT_ROOT / STAGED_NAME
    rows: list[dict] = []
    try:
        for table in sorted(TABLES_DIR.glob("*.tex")):
            rel = str(table.relative_to(PROJECT_ROOT))
            staged.write_text(PREAMBLE % rel, encoding="utf-8")
            subprocess.run(
                [
                    tectonic,
                    "--outdir",
                    str(build_dir),
                    "--keep-logs",
                    str(staged),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            log = build_dir / f"{Path(STAGED_NAME).stem}.log"
            worst = 0.0
            error = None
            if log.exists():
                text = log.read_text(encoding="utf-8", errors="replace")
                error = next((ln for ln in text.splitlines() if ln.startswith("!")), None)
                for m in OVERFULL.finditer(text):
                    worst = max(worst, float(m.group(1)))
            rows.append(
                {
                    "table": table.name,
                    "overfull_pt": round(worst, 1),
                    "fits": worst <= 0.0,
                    "error": error,
                }
            )
            for suffix in (".aux", ".log", ".xdv", ".pdf"):
                (build_dir / f"{Path(STAGED_NAME).stem}{suffix}").unlink(missing_ok=True)
        return rows
    finally:
        staged.unlink(missing_ok=True)
        shutil.rmtree(build_dir, ignore_errors=True)


def main() -> int:
    tectonic = find_tectonic()
    if tectonic is None:
        print("ERROR: no LaTeX engine found; cannot measure.")
        print("Install tectonic, or place a binary at /tmp/opencode/tectonic.")
        print("The check is skipped rather than reported as passing.")
        return 3

    rows = measure_tables(tectonic)
    print(f"{'table':36s} {'overfull (pt)':>14s} {'status':>10s}")
    print("-" * 64)
    for r in rows:
        status = "OK" if r["fits"] else f"+{r['overfull_pt']:.0f}pt"
        if r["error"]:
            status = "ERROR"
        print(f"{r['table']:36s} {r['overfull_pt']:14.1f} {status:>10s}")
    bad = [r for r in rows if not r["fits"] or r["error"]]
    print()
    print(f"{len(rows) - len(bad)}/{len(rows)} tables fit the {TEXT_WIDTH_PT:.0f}pt text block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
