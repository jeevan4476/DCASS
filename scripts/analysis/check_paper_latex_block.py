#!/usr/bin/env python3
r"""
Extract the LaTeX block from the paper writeup and compile it.

The paper writeup carries the exact `\input` and `\includegraphics` calls a
submission would use. This script pulls that block out of the document,
wraps it in a minimal two-column article, and compiles it, so a broken
include path or a missing package is caught before submission rather than
after.

The block is extracted verbatim; nothing is rewritten.

Usage:
    python scripts/analysis/check_paper_latex_block.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER = PROJECT_ROOT / "docs" / "benchmarking" / "BENCHMARK_PAPER_WRITEUP.md"

PREAMBLE = r"""\documentclass[11pt,twocolumn]{article}
\usepackage[margin=1in]{geometry}
"""

BLOCK_RE = re.compile(r"```latex\n(.*?)```", re.S)


def find_tectonic() -> str | None:
    found = shutil.which("tectonic")
    if found:
        return found
    candidate = Path("/tmp/opencode/tectonic")
    return str(candidate) if candidate.exists() else None


def main() -> int:
    tectonic = find_tectonic()
    if tectonic is None:
        print("ERROR: no LaTeX engine found; cannot check the paper block.")
        print("Install tectonic, or place a binary at /tmp/opencode/tectonic.")
        return 3

    if not PAPER.exists():
        print(f"ERROR: {PAPER} not found.")
        return 2

    text = PAPER.read_text(encoding="utf-8")
    blocks = BLOCK_RE.findall(text)
    if not blocks:
        print("ERROR: no LaTeX block found in the paper writeup.")
        return 2

    # The integration block is the one that contains \input of the tables.
    block = next(
        (b for b in blocks if "storage/benchmarks/tables/" in b),
        blocks[0],
    )

    # The preamble lives in its own block; merge it in so the packages the
    # block depends on are actually loaded. Any \usepackage inside the body
    # block is moved up, because LaTeX rejects it after \begin{document}.
    preamble_block = next(
        (b for b in blocks if b.lstrip().startswith("% main.tex preamble")),
        "",
    )
    preamble_lines = [
        ln for ln in preamble_block.splitlines() if ln.startswith("\\usepackage")
    ]
    body_lines = [
        ln
        for ln in block.splitlines()
        if not ln.lstrip().startswith("\\usepackage")
    ]
    body = "\n".join(body_lines)

    doc = (
        PREAMBLE
        + "\n".join(preamble_lines)
        + "\n\\begin{document}\n"
        + body
        + "\n\\end{document}\n"
    )

    build_dir = Path(tempfile.mkdtemp(prefix="dcass_papertex_"))
    staged = PROJECT_ROOT / "_dcass_paper_check.tex"
    try:
        staged.write_text(doc, encoding="utf-8")
        proc = subprocess.run(
            [tectonic, "--outdir", str(build_dir), "--keep-logs", str(staged)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        pdf = build_dir / "_dcass_paper_check.pdf"
        ok = proc.returncode == 0 and pdf.exists()
        if ok:
            print(f"PAPER BLOCK COMPILES: {pdf.stat().st_size} byte PDF")
        else:
            print("PAPER BLOCK FAILED TO COMPILE")
            print(proc.stdout[-4000:])
            print(proc.stderr[-2000:])
        log = build_dir / "_dcass_paper_check.log"
        if log.exists():
            errors = [
                ln
                for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.startswith("!")
            ]
            if errors:
                print(f"--- {len(errors)} LaTeX error line(s) ---")
                for ln in errors[:20]:
                    print(ln)
            overfull = [
                ln
                for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.startswith("Overfull")
            ]
            if overfull:
                print(f"--- {len(overfull)} overfull hbox warning(s) ---")
                for ln in overfull:
                    print(ln)
        return 0 if ok else 1
    finally:
        staged.unlink(missing_ok=True)
        shutil.rmtree(build_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
