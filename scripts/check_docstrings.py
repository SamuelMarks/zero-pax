#!/usr/bin/env python3
"""AST-based docstring verifier enforcing 100% file, module, class, attribute, function, argument, return, and raises documentation."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Sequence
from pathlib import Path


def _extract_section_entries(docstring: str, section_headers: tuple[str, ...]) -> set[str]:
    """Extract identifier names documented under a Google-style docstring section.

    Args:
        docstring: Raw docstring text.
        section_headers: Tuple of section header strings (e.g., `('Args:', 'Arguments:')`).

    Returns:
        Set of documented identifier names found in the section.
    """
    documented: set[str] = set()
    in_section = False
    section_indent: int | None = None

    for line in docstring.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped in section_headers:
            in_section = True
            section_indent = indent
            continue
        if in_section:
            if section_indent is not None and indent <= section_indent and stripped.endswith(":"):
                in_section = False
                continue
            match = re.match(r"^\*{0,2}([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:", stripped)
            if match:
                documented.add(match.group(1))
    return documented


def _extract_documented_args(docstring: str) -> set[str]:
    """Extract parameter names documented under the `Args:` section of a Google-style docstring.

    Args:
        docstring: Raw docstring text.

    Returns:
        Set of parameter names found in the `Args:` block.
    """
    return _extract_section_entries(docstring, ("Args:", "Arguments:", "Parameters:"))


def _extract_documented_attributes(docstring: str) -> set[str]:
    """Extract attribute names documented under the `Attributes:` section of a class docstring.

    Args:
        docstring: Raw class docstring text.

    Returns:
        Set of attribute names found in the `Attributes:` block.
    """
    return _extract_section_entries(docstring, ("Attributes:",))


def _has_section(docstring: str, section_headers: tuple[str, ...]) -> bool:
    """Check whether `docstring` contains any of the specified Google-style `section_headers`.

    Args:
        docstring: Raw docstring text.
        section_headers: Section header lines to search for (e.g., `('Returns:', 'Yields:')`).

    Returns:
        True if at least one matching section header is present.
    """
    return any(line.strip() in section_headers for line in docstring.splitlines())


def _is_none_return(returns_node: ast.expr | None) -> bool:
    """Determine whether a function return annotation represents `None`.

    Args:
        returns_node: AST expression for the return annotation, or `None`.

    Returns:
        True if the return annotation is absent or `None`.
    """
    return (
        returns_node is None
        or (isinstance(returns_node, ast.Constant) and returns_node.value is None)
        or (isinstance(returns_node, ast.Name) and returns_node.id == "None")
    )


def _function_directly_raises(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether `fn_node` contains an explicit `raise` statement (excluding nested defs).

    Args:
        fn_node: Function or method AST node.

    Returns:
        True if the function body directly raises an exception.
    """
    queue: list[ast.AST] = list(fn_node.body)
    while queue:
        current = queue.pop(0)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(current, ast.Raise):
            return True
        queue.extend(ast.iter_child_nodes(current))
    return False


def check_file_docstrings(file_path: Path) -> list[str]:
    """Check a single Python file for 100% module, class, attribute, function, arg, return, and raises docstrings.

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
                continue

            public_attrs: list[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    attr_name = stmt.target.id
                    if not attr_name.startswith("_") and attr_name != "model_config":
                        public_attrs.append(attr_name)

            if public_attrs:
                doc_attrs = _extract_documented_attributes(cls_doc)
                for attr_name in public_attrs:
                    if attr_name not in doc_attrs:
                        errors.append(
                            f"{file_path}:{node.lineno}: Class '{node.name}' is missing docstring for attribute '{attr_name}'"
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

            if (
                node.name != "__init__"
                and not _is_none_return(node.returns)
                and not _has_section(fn_doc, ("Returns:", "Yields:"))
            ):
                errors.append(
                    f"{file_path}:{node.lineno}: Function/method '{node.name}' is missing 'Returns:' or 'Yields:' docstring section"
                )

            if _function_directly_raises(node) and not _has_section(fn_doc, ("Raises:",)):
                errors.append(
                    f"{file_path}:{node.lineno}: Function/method '{node.name}' is missing 'Raises:' docstring section"
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
        description="Verify 100% docstring coverage across modules, classes, attributes, functions, arguments, returns, and raises."
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
        f"Docstring check passed: 100% file, module, class, attribute, function, argument, return, and raises coverage across {len(files)} files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
