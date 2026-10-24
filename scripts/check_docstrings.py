#!/usr/bin/env python3
"""AST-based docstring verifier enforcing 100% file, module, class, function, and argument documentation."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Sequence
from pathlib import Path


def _extract_documented_args(docstring: str) -> set[str]:
    """Extract parameter names documented under the `Args:` section of a Google-style docstring.

    Args:
        docstring: Raw docstring text.

    Returns:
        Set of parameter names found in the `Args:` block.
    """
    documented: set[str] = set()
    in_args = False
    args_indent: int | None = None

    for line in docstring.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            args_indent = indent
            continue
        if in_args:
            if args_indent is not None and indent <= args_indent and stripped.endswith(":"):
                in_args = False
                continue
            match = re.match(r"^\*{0,2}([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:", stripped)
            if match:
                documented.add(match.group(1))
    return documented


def check_file_docstrings(file_path: Path) -> list[str]:
    """Check a single Python file for 100% module, class, function, and argument docstrings.

    Args:
        file_path: Path to the Python file to inspect.

    Returns:
        List of violation messages.
    """
    errors: list[str] = []
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))

    mod_doc = ast.get_docstring(tree)
    if not mod_doc or not mod_doc.strip():
        errors.append(f"{file_path}:1: Missing module docstring")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node)
            if not cls_doc or not cls_doc.strip():
                errors.append(
                    f"{file_path}:{node.lineno}: Class '{node.name}' is missing a docstring"
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_doc = ast.get_docstring(node)
            if not fn_doc or not fn_doc.strip():
                errors.append(
                    f"{file_path}:{node.lineno}: Function/method '{node.name}' is missing a docstring"
                )
                continue

            param_names: list[str] = []
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.arg not in ("self", "cls"):
                    param_names.append(arg.arg)
            if node.args.vararg is not None:
                param_names.append(node.args.vararg.arg)
            if node.args.kwarg is not None:
                param_names.append(node.args.kwarg.arg)

            if param_names:
                doc_args = _extract_documented_args(fn_doc)
                for p_name in param_names:
                    if p_name not in doc_args:
                        errors.append(
                            f"{file_path}:{node.lineno}: Function/method '{node.name}' is missing docstring for argument '{p_name}'"
                        )

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run docstring completeness verification across src/, scripts/, and tests/.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Exit code 0 if all docstrings are complete, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Verify 100% docstring coverage across modules, classes, functions, and arguments."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Directories or files to check (defaults to src, scripts, tests).",
    )
    args = parser.parse_args(argv)

    raw_paths: list[Path] = list(args.paths)
    target_paths: list[Path] = (
        raw_paths if raw_paths else [Path("src"), Path("scripts"), Path("tests")]
    )

    files: list[Path] = []
    for p in target_paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))

    all_errors: list[str] = []
    for fp in files:
        all_errors.extend(check_file_docstrings(fp))

    if all_errors:
        print(f"Docstring check failed with {len(all_errors)} violations:")
        for err in all_errors:
            print(f"  {err}")
        return 1

    print(
        f"Docstring check passed: 100% file, module, class, function, and argument coverage across {len(files)} files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
