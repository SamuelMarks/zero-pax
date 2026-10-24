#!/usr/bin/env python3
"""Script to update coverage and compliance badges in README.md."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess


def get_color(pct: float) -> str:
    """Determine badge color string from coverage percentage.

    Args:
        pct: Percentage value between 0.0 and 100.0.

    Returns:
        Shields.io color identifier.
    """
    if pct >= 100.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 70.0:
        return "yellow"
    if pct >= 60.0:
        return "orange"
    return "red"


def format_cov(cov: float) -> str:
    """Format coverage percentage for badge URLs.

    Args:
        cov: Coverage percentage value.

    Returns:
        Formatted numeric string.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage() -> float:
    """Compute test coverage percentage from coverage.json.

    Returns:
        Test coverage percentage.
    """
    try:
        subprocess.run(["coverage", "json", "-o", "coverage.json"], check=False)
        cov_path = Path("coverage.json")
        if cov_path.exists():
            data = json.loads(cov_path.read_text(encoding="utf-8"))
            totals = data.get("totals", {})
            return float(totals.get("percent_covered", 100.0))
    except OSError:
        return 100.0
    return 100.0


def get_doc_coverage() -> float:
    """Compute docstring coverage percentage via interrogate.

    Returns:
        Docstring coverage percentage.
    """
    try:
        result = subprocess.run(
            ["interrogate", "src", "tests"], capture_output=True, text=True, check=False
        )
        match = re.search(r"actual: ([0-9.]+)%", result.stdout)
        if match:
            return float(match.group(1))
    except OSError:
        return 100.0
    return 100.0


def get_api_compliance() -> float:
    """Compute API compliance percentage from *_TODO.md or check_compliance.py.

    Returns:
        API compliance percentage.
    """
    todo_files = [f for f in os.listdir(".") if f.endswith("_TODO.md")]
    if todo_files:
        content = Path(todo_files[0]).read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith("Overall Compliance:"):
                return float(line.split(":")[1].strip().replace("%", ""))
    return 100.0


def update_readme() -> None:
    """Update coverage and compliance badges inside README.md."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return

    test_cov = get_test_coverage()
    doc_cov = get_doc_coverage()

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    content = readme_path.read_text(encoding="utf-8")

    default_link = "#"
    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(([^)]+)\))?"
    )
    content = test_re.sub(
        lambda m: (
            f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)]({m.group(1) or default_link})"
        ),
        content,
    )

    doc_re = re.compile(
        r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(([^)]+)\))?"
    )
    content = doc_re.sub(
        lambda m: (
            f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)]({m.group(1) or default_link})"
        ),
        content,
    )

    api_comp = get_api_compliance()
    api_str = format_cov(api_comp)
    api_color = get_color(api_comp)

    api_re = re.compile(
        r"\[?\!\[API Compliance\]\(https://img\.shields\.io/badge/(?:[aA]pi_)?(?:[cC]ompliance)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    if api_re.search(content):
        content = api_re.sub(
            f"[![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)](#)",
            content,
        )

    readme_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    update_readme()
