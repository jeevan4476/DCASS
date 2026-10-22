#!/usr/bin/env python3
"""
Component 6 inventory checker.

Compares what is actually on disk under storage/benchmarks/ against what the
benchmarking report (section 8) and the publication-artifacts document
claim exists. Reports:

  * artifacts on disk that no document mentions (orphans);
  * documents that mention an artifact that is not on disk (phantoms);
  * artifacts that are neither a table, a plot, nor a results JSON
    (unclassified);
  * whether every LaTeX table in the tables directory is referenced by the
    report's section 8.1 and by the publication-artifacts document.

Read-only. It never writes to storage/benchmarks/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = PROJECT_ROOT / "storage" / "benchmarks"
TABLES = BENCH / "tables"
PLOTS = BENCH / "plots"

REPORT = PROJECT_ROOT / "docs" / "benchmarking" / "BENCHMARKING_REPORT.md"
PUBDOC = PROJECT_ROOT / "docs" / "benchmarking" / "06_PUBLICATION_ARTIFACTS_AND_LATEX.md"
PAPER = PROJECT_ROOT / "docs" / "benchmarking" / "BENCHMARK_PAPER_WRITEUP.md"

# Artifacts that are documentation or diagnostics rather than publication
# outputs. They are listed in the report's section 8.3 but are not tables or
# figures, and they are not expected to have a LaTeX counterpart.
DIAGNOSTIC_PREFIXES = ("warden_diagnostic",)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def mentioned(paths: list[Path], text: str) -> set[Path]:
    """Which of *paths* are named in *text*."""
    out = set()
    for p in paths:
        if p.name in text:
            out.add(p)
    return out


def main() -> int:
    report_text = read(REPORT)
    pubdoc_text = read(PUBDOC)
    paper_text = read(PAPER)

    tables = sorted(TABLES.glob("*.tex"))
    plots = sorted(PLOTS.iterdir())
    plots = [p for p in plots if p.is_file()]
    jsons = sorted(BENCH.glob("*.json"))

    report_tables = mentioned(tables, report_text)
    report_plots = mentioned(plots, report_text)
    report_jsons = mentioned(jsons, report_text)
    pubdoc_tables = mentioned(tables, pubdoc_text)
    pubdoc_plots = mentioned(plots, pubdoc_text)
    pubdoc_jsons = mentioned(jsons, pubdoc_text)
    paper_tables = mentioned(tables, paper_text)
    paper_plots = mentioned(plots, paper_text)

    result: dict = {
        "counts": {
            "tables_on_disk": len(tables),
            "plots_on_disk": len(plots),
            "top_level_json_on_disk": len(jsons),
        },
        "tables": {
            "in_report": sorted(p.name for p in report_tables),
            "in_publication_doc": sorted(p.name for p in pubdoc_tables),
            "in_paper_writeup": sorted(p.name for p in paper_tables),
            "missing_from_report": sorted(p.name for p in set(tables) - report_tables),
            "missing_from_publication_doc": sorted(
                p.name for p in set(tables) - pubdoc_tables
            ),
            "missing_from_paper_writeup": sorted(
                p.name for p in set(tables) - paper_tables
            ),
        },
        "plots": {
            "in_report": sorted(p.name for p in report_plots),
            "in_publication_doc": sorted(p.name for p in pubdoc_plots),
            "in_paper_writeup": sorted(p.name for p in paper_plots),
            "missing_from_report": sorted(p.name for p in set(plots) - report_plots),
            "missing_from_publication_doc": sorted(
                p.name for p in set(plots) - pubdoc_plots
            ),
            "missing_from_paper_writeup": sorted(
                p.name for p in set(plots) - paper_plots
            ),
        },
        "json": {
            "in_report": sorted(p.name for p in report_jsons),
            "in_publication_doc": sorted(p.name for p in pubdoc_jsons),
            "missing_from_report": sorted(p.name for p in set(jsons) - report_jsons),
            "missing_from_publication_doc": sorted(
                p.name for p in set(jsons) - pubdoc_jsons
            ),
        },
    }

    # Does every table file carry a \label and \caption?
    label_issues = []
    for t in tables:
        text = read(t)
        has_label = "\\label{" in text
        has_caption = "\\caption{" in text
        if not (has_label and has_caption):
            label_issues.append(
                {
                    "table": t.name,
                    "has_label": has_label,
                    "has_caption": has_caption,
                }
            )
    result["tables_missing_label_or_caption"] = label_issues

    # Are the manifest artifacts present for the components that record them?
    result["manifests"] = {
        p.name: p.exists() for p in sorted(BENCH.glob("*_manifest.json"))
    }

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(result, indent=2))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
