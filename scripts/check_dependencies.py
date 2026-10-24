#!/usr/bin/env python3
"""Script to verify that no disallowed third-party dependencies are used in non-test code."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence
from pathlib import Path

from stdlib_list import stdlib_list

ALLOWED_THIRD_PARTY: frozenset[str] = frozenset(
    {
        "pydantic",
        "cdd_python",
        "cdd",
        "ml_switcheroo_ir",
        "ml_switcheroo_compiler",
        "ml_switcheroo",
        "ml_ecosystem_snapshots",
        "zero_jax",
        "zero_pax",
    }
)

STDLIB: frozenset[str] = frozenset(stdlib_list("3.9"))


def is_allowed_module(module_name: str) -> bool:
    """Check whether a module import is allowed in runtime code.

    Args:
        module_name: Dotted module name being imported.

    Returns:
        True if the top-level module is part of the Python standard library or allowlist.
    """
    top_level = module_name.split(".")[0]
    return top_level in STDLIB or top_level in ALLOWED_THIRD_PARTY


def check_file(file_path: Path) -> list[str]:
    """Check a single Python file for disallowed third-party imports.

    Args:
        file_path: Path to the Python source file to inspect.

    Returns:
        A list of human-readable violation strings found in the file.
    """
    errors: list[str] = []
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return [f"{file_path}: Error parsing file: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_allowed_module(alias.name):
                    errors.append(f"{file_path}:{node.lineno}: Disallowed import: {alias.name}")
        elif (
            isinstance(node, ast.ImportFrom) and node.module and not is_allowed_module(node.module)
        ):
            errors.append(f"{file_path}:{node.lineno}: Disallowed import: {node.module}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the disallowed dependency check across source files.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Exit code 0 if no disallowed dependencies are found, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Check for disallowed dependencies.")
    parser.add_argument(
        "filenames",
        nargs="*",
        type=Path,
        help="Files to check (defaults to all Python files under src/).",
    )
    args = parser.parse_args(argv)

    raw_filenames: list[Path] = list(args.filenames)
    files_to_check: list[Path] = (
        raw_filenames if raw_filenames else sorted(Path("src").rglob("*.py"))
    )

    all_errors: list[str] = []
    for file_path in files_to_check:
        if "test" in file_path.parts:
            continue
        all_errors.extend(check_file(file_path))

    if all_errors:
        print("Disallowed dependencies found in non-test code:")
        for error in all_errors:
            print(error)
        return 1

    print("Dependency check passed: 0 disallowed third-party imports found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
