#!/usr/bin/env python3
"""Script to verify that no disallowed third-party dependencies are used in non-test code."""

import ast
import sys
from pathlib import Path
from stdlib_list import stdlib_list

ALLOWED_THIRD_PARTY = {
    "numpy",
    "pydantic",
    "cdd_python",  # Usually imported as cdd or something else? We'll check the top level module
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "ml_switcheroo",
    "zero_jax",
    "zero_pax",  # The project itself
}

# Get a set of standard library module names for Python 3.9
STDLIB = set(stdlib_list("3.9"))


def is_allowed_module(module_name: str) -> bool:
    """Checks if a module is allowed."""
    top_level = module_name.split(".")[0]
    return top_level in STDLIB or top_level in ALLOWED_THIRD_PARTY


def check_file(file_path: Path) -> list[str]:
    """Checks a single file for disallowed imports."""
    errors = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except Exception as e:
        return [f"{file_path}: Error parsing file: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if not is_allowed_module(name.name):
                    errors.append(
                        f"{file_path}:{node.lineno}: Disallowed import: {name.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module and not is_allowed_module(node.module):
                errors.append(
                    f"{file_path}:{node.lineno}: Disallowed import: {node.module}"
                )
    return errors


def main() -> int:
    """Main execution block."""
    import argparse

    parser = argparse.ArgumentParser(description="Check for disallowed dependencies.")
    parser.add_argument("filenames", nargs="*", type=Path, help="Files to check.")
    args = parser.parse_args()

    all_errors = []

    # If no filenames provided, default to all non-test python files in src
    files_to_check = args.filenames
    if not files_to_check:
        files_to_check = Path("src").rglob("*.py")

    for file_path in files_to_check:
        # Ignore tests if they somehow get passed
        if "test" in str(file_path):
            continue

        errors = check_file(file_path)
        all_errors.extend(errors)

    if all_errors:
        print("Disallowed dependencies found in non-test code:")
        for error in all_errors:
            print(error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
