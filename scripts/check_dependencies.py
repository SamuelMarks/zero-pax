#!/usr/bin/env python3
"""Script to verify that no disallowed third-party dependencies are used across src, scripts, tests, or requirements."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path

ALLOWED_THIRD_PARTY: frozenset[str] = frozenset(
    {
        "pydantic",
        "cdd_python",
        "cdd",
        "ml_switcheroo_ir",
        "ml_switcheroo_compiler",
        "ml_ecosystem_snapshots",
        "zero_pax",
        "zero_jax",
        "zero_flax",
        "zero_optax",
        "zero_chex",
        "zero_orbax",
        "zero_grain",
        "zero_keras",
        "zero_mlx",
        "zero_pytorch",
        "zero_tensorflow",
        "zero_zoo",
    }
)

ALLOWED_DEV_PACKAGES: frozenset[str] = frozenset(
    {
        "pytest",
        "pytest_cov",
        "mypy",
        "interrogate",
        "ruff",
        "build",
    }
)

FORBIDDEN_SIBLING_FRAMEWORKS: frozenset[str] = frozenset(
    {
        "grain",
        "chex",
        "orbax",
        "optax",
    }
)


def _discover_stdlib_modules() -> frozenset[str]:
    """Build the set of Python standard library top-level module names.

    Returns:
        Frozenset of standard library module names.
    """
    modules: set[str] = set(sys.builtin_module_names)
    stdlib_dir = sysconfig.get_path("stdlib") or ""
    stdlib_path = Path(stdlib_dir)
    for entry in stdlib_path.iterdir():
        if entry.suffix == ".py":
            modules.add(entry.stem)
        elif entry.is_dir() and entry.name not in ("site-packages", "__pycache__"):
            modules.add(entry.name)
    lib_dynload = stdlib_path / "lib-dynload"
    for entry in lib_dynload.iterdir():
        modules.add(entry.name.split(".")[0])
    modules.add("__future__")
    return frozenset(modules)


STDLIB: frozenset[str] = _discover_stdlib_modules()


def is_allowed_module(module_name: str, is_test: bool = False) -> bool:
    """Check whether a module import is allowed in runtime or test code.

    Args:
        module_name: Dotted module name being imported.
        is_test: Whether the import occurs inside `tests/`.

    Returns:
        True if the top-level module is part of the Python standard library or allowlist.
    """
    top_level = module_name.split(".")[0].replace("-", "_")
    if top_level in FORBIDDEN_SIBLING_FRAMEWORKS:
        return False
    if top_level in STDLIB or top_level in ALLOWED_THIRD_PARTY:
        return True
    return is_test and top_level in ("pytest", "tests", "scripts", "check_docstrings")


def check_file(file_path: Path) -> list[str]:
    """Check a single Python file for disallowed third-party imports (including dynamic imports).

    Args:
        file_path: Path to the Python source file to inspect.

    Returns:
        A list of human-readable violation strings found in the file.
    """
    errors: list[str] = []
    is_test = (
        "test" in file_path.parts
        or "tests" in file_path.parts
        or "scripts" in file_path.parts
        or file_path.name.startswith("test_")
    )
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return [f"{file_path}: Error parsing file: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_allowed_module(alias.name, is_test=is_test):
                    errors.append(f"{file_path}:{node.lineno}: Disallowed import: {alias.name}")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and not is_allowed_module(node.module, is_test=is_test)
        ):
            errors.append(f"{file_path}:{node.lineno}: Disallowed import: {node.module}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and not is_allowed_module(node.args[0].value, is_test=is_test)
        ):
            errors.append(
                f"{file_path}:{node.lineno}: Disallowed dynamic import: {node.args[0].value}"
            )
    return errors


def check_requirements_file(req_path: Path, is_test: bool = False) -> list[str]:
    """Validate that all dependencies in `req_path` belong to the allowed dependency set.

    Args:
        req_path: Path to `requirements.txt` or `requirements-test.txt`.
        is_test: Whether `req_path` is a test/dev requirements file.

    Returns:
        List of violation messages for any disallowed dependency specifications.
    """
    if not req_path.is_file():
        return []
    errors: list[str] = []
    for idx, raw_line in enumerate(req_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"^([a-zA-Z0-9_\-]+)", line)
        pkg_raw = match.group(1) if match else line
        pkg_name = pkg_raw.replace("-", "_")
        allowed = is_allowed_module(pkg_name) or (is_test and pkg_name in ALLOWED_DEV_PACKAGES)
        if not allowed:
            errors.append(f"{req_path}:{idx}: Disallowed dependency: {pkg_raw}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the disallowed dependency check across `src/`, `scripts/`, `tests/`, and requirements files.

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
        help="Files or directories to check (defaults to src/, scripts/, and tests/).",
    )
    args = parser.parse_args(argv)

    raw_filenames: list[Path] = list(args.filenames)
    target_paths: list[Path] = (
        raw_filenames if raw_filenames else [Path("src"), Path("scripts"), Path("tests")]
    )

    files_to_check: list[Path] = []
    for p in target_paths:
        if p.is_file() and p.suffix == ".py":
            files_to_check.append(p)
        elif p.is_dir():
            files_to_check.extend(sorted(p.rglob("*.py")))

    all_errors: list[str] = []
    for file_path in files_to_check:
        all_errors.extend(check_file(file_path))

    if not raw_filenames:
        all_errors.extend(check_requirements_file(Path("requirements.txt"), is_test=False))
        all_errors.extend(check_requirements_file(Path("requirements-test.txt"), is_test=True))

    if all_errors:
        print("Disallowed dependencies found:")
        for error in all_errors:
            print(error)
        return 1

    print("Dependency check passed: 0 disallowed third-party imports found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
