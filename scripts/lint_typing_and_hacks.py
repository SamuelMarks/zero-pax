#!/usr/bin/env python3
"""AST-based static linter forbidding Any, object, cast, getattr, setattr, hasattr, and metaclass hacks."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence
from pathlib import Path

FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "Any",
        "object",
        "cast",
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "__getattr__",
        "__setattr__",
        "__getattribute__",
        "__dict__",
    }
)


def check_ast(tree: ast.AST, filepath: Path) -> list[str]:
    """Traverse `tree` and report any forbidden typing or reflection constructs.

    Args:
        tree: Parsed Python AST.
        filepath: Source file path for diagnostic formatting.

    Returns:
        List of diagnostic error strings.
    """
    violations: list[str] = []

    for node in ast.walk(tree):
        lineno = node.lineno if isinstance(node, (ast.expr, ast.stmt)) else 0
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"{filepath}:{lineno}: Forbidden identifier '{node.id}'")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            violations.append(f"{filepath}:{lineno}: Forbidden attribute '{node.attr}'")
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in FORBIDDEN_NAMES
        ):
            violations.append(f"{filepath}:{node.lineno}: Forbidden dunder method '{node.name}'")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES or (
                    alias.asname is not None and alias.asname in FORBIDDEN_NAMES
                ):
                    violations.append(f"{filepath}:{node.lineno}: Forbidden import '{alias.name}'")
        elif isinstance(node, ast.ClassDef):
            for kw in node.keywords:
                if kw.arg == "metaclass":
                    violations.append(f"{filepath}:{node.lineno}: Forbidden 'metaclass' usage")

    return violations


def collect_python_files(paths: Sequence[str]) -> list[Path]:
    """Collect all `.py` files under the given directory or file `paths`.

    Args:
        paths: Sequence of directory or file path strings.

    Returns:
        Sorted list of `Path` objects pointing to `.py` files.
    """
    result: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".py":
            result.append(p)
        elif p.is_dir():
            result.extend(sorted(p.rglob("*.py")))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for typing and reflection hack verification.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Exit code (`0` if clean, `1` if violations found).
    """
    parser = argparse.ArgumentParser(description="Lint typing and reflection hacks.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src", "tests"],
        help="Paths to scan (default: src tests).",
    )
    args = parser.parse_args(argv)

    files = collect_python_files(args.paths)
    all_violations: list[str] = []
    for fpath in files:
        source = fpath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(fpath))
        all_violations.extend(check_ast(tree, fpath))

    if all_violations:
        print("Typing and reflection lint failed:")
        for v in all_violations:
            print(f"  {v}")
        return 1

    print(f"Typing and reflection lint passed across {len(files)} files (0 violations).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
