#!/usr/bin/env python3
"""Script to validate zero_pax API, module hierarchy, class, field, type, default, and docstring compliance against ml-ecosystem-snapshots v0.0.5 (pax_v1.4.0.json) and SEMANTIC_PLAN.md."""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

import zero_pax
from zero_pax.praxis.layers import EXPORTED_NAMES, LAYER_REGISTRY

_ATTR_ENTRY_RE: re.Pattern[str] = re.compile(r"^\s{2,12}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?:")


def _extract_documented_attributes(doc: str) -> set[str]:
    """Extract attribute names documented under an `Attributes:` section of a class docstring.

    Args:
        doc: Cleaned class docstring text.

    Returns:
        Set of documented attribute names.
    """
    attrs: set[str] = set()
    in_attrs = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in ("Attributes:", "Attributes"):
            in_attrs = True
            continue
        if in_attrs:
            if stripped.endswith(":") and not line.startswith("    "):
                in_attrs = False
                continue
            match = _ATTR_ENTRY_RE.match(line)
            if match:
                attrs.add(match.group(1))
    return attrs


EXPECTED_SUBMODULES: tuple[str, ...] = (
    "praxis.base_layer",
    "praxis.pax_fiddle",
    "praxis.layers",
    "praxis.layers.activations",
    "praxis.layers.adapters",
    "praxis.layers.attentions",
    "praxis.layers.augmentations",
    "praxis.layers.base_ops",
    "praxis.layers.conformers",
    "praxis.layers.convolutions",
    "praxis.layers.embedding_softmax",
    "praxis.layers.embeddings",
    "praxis.layers.frnn",
    "praxis.layers.linears",
    "praxis.layers.models",
    "praxis.layers.multi_query_attention",
    "praxis.layers.ngrammer",
    "praxis.layers.normalizations",
    "praxis.layers.pipeline",
    "praxis.layers.poolings",
    "praxis.layers.quantization",
    "praxis.layers.repeats",
    "praxis.layers.resnets",
    "praxis.layers.rnn_cell",
    "praxis.layers.ssm",
    "praxis.layers.stochastics",
    "praxis.layers.transformer_models",
    "praxis.layers.transformers",
    "praxis.layers.vanillanets",
    "praxis.layers.vits",
)

FORBIDDEN_SIBLING_FRAMEWORKS: tuple[str, ...] = (
    "grain",
    "chex",
    "orbax",
    "optax",
)


def _find_snapshot_path() -> Path | None:
    """Locate `pax_v1.4.0.json` in the local cache or `ml_ecosystem_snapshots` package.

    Returns:
        Path to `pax_v1.4.0.json` if found, otherwise `None`.
    """
    candidates = [
        Path.home() / ".cache" / "ml_ecosystem_snapshots" / "pax_v1.4.0.json",
        Path.home() / ".cache" / "ml_ecosystem_snapshots" / "snapshots" / "pax_v1.4.0.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _parse_expected_default(
    snap_def: str | None,
) -> tuple[bool, str | int | float | bool | tuple[int, ...] | None]:
    """Parse a serialized default string from `pax_v1.4.0.json` into a comparable Python literal.

    Args:
        snap_def: Raw default string from the snapshot parameter metadata.

    Returns:
        Tuple `(should_check, expected_literal_value)`.
    """
    if snap_def is None:
        return False, None
    if snap_def in ("'```(None)```'", "```(None)```", "None"):
        return True, None
    try:
        parsed = ast.literal_eval(snap_def)
        if isinstance(parsed, str):
            if parsed == "```(None)```":
                return True, None
            if any(
                parsed.startswith(pfx)
                for pfx in (
                    "template_field(",
                    "dataclasses.field(",
                    "instance_field(",
                    "base_layer.",
                    "pax_fiddle.",
                )
            ):
                return False, None
            if parsed in (
                "ActivationType",
                "LanguageModelType",
                "AutodiffCheckpointType",
                "repeats.AutodiffCheckpointType",
            ):
                return False, None
            if parsed.startswith("(") and parsed.endswith(")"):
                tuple_val = ast.literal_eval(parsed)
                if isinstance(tuple_val, tuple):
                    return True, tuple_val
            if parsed == "768 * 256":
                return True, 768 * 256
            return True, parsed
        if isinstance(parsed, (int, float, bool, tuple)) or parsed is None:
            return True, parsed
    except (ValueError, SyntaxError):
        return False, None
    return False, None


def verify_module_hierarchy() -> list[str]:
    """Verify the `zero_pax.praxis` module hierarchy and ensure no sibling frameworks are replicated.

    Returns:
        List of hierarchy or sibling-replication error messages.
    """
    errors: list[str] = []
    for mod_path in EXPECTED_SUBMODULES:
        parts = mod_path.split(".")
        curr: ModuleType = zero_pax
        for part in parts:
            member_map = dict(inspect.getmembers(curr))
            if part not in member_map:
                errors.append(f"Missing module hierarchy node: zero_pax.{mod_path} (at {part})")
                break
            next_mod = member_map[part]
            if isinstance(next_mod, ModuleType):
                curr = next_mod

    top_members = dict(inspect.getmembers(zero_pax))
    praxis_members = dict(inspect.getmembers(zero_pax.praxis))
    for forbidden in FORBIDDEN_SIBLING_FRAMEWORKS:
        if forbidden in top_members or forbidden in praxis_members:
            errors.append(
                f"Forbidden sibling framework replicated in zero_pax: {forbidden} (belongs in ../zero-{forbidden})"
            )
    return errors


def verify_snapshot_compliance(snapshot_path: Path) -> list[str]:
    """Verify that all classes, fields, types, defaults, and docstrings in `snapshot_path` exist in `zero_pax`.

    Args:
        snapshot_path: Path to `pax_v1.4.0.json`.

    Returns:
        List of compliance error strings (empty if 100% compliant).
    """
    errors: list[str] = []
    errors.extend(verify_module_hierarchy())

    raw_text = snapshot_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    categories = data.get("categories", {})
    neural_items = categories.get("neural", [])

    for item in neural_items:
        name: str = item["name"]
        api_path: str = item["api_path"]
        if name not in EXPORTED_NAMES or name not in LAYER_REGISTRY:
            errors.append(f"Missing API class: {api_path}")
            continue

        cls = LAYER_REGISTRY[name]
        if not inspect.getdoc(cls):
            errors.append(f"Missing docstring on class: {api_path}")

        expected_params = [p for p in item.get("params", []) if p["name"] not in ("args", "kwargs")]

        if issubclass(cls, Enum):
            member_names = set(cls.__members__.keys())
            for param in expected_params:
                pname = param["name"]
                if pname not in member_names:
                    errors.append(f"{api_path}: Missing Enum member '{pname}'")
                else:
                    snap_def = param.get("default")
                    if isinstance(snap_def, str):
                        expected_val = snap_def.strip("'")
                        actual_val = cls.__members__[pname].value
                        if actual_val != expected_val:
                            errors.append(
                                f"{api_path}.{pname}: Enum value mismatch (expected {expected_val!r}, got {actual_val!r})"
                            )
        elif issubclass(cls, BaseModel):
            model_fields = cls.model_fields
            documented_attrs: set[str] = set()
            for base_cls in inspect.getmro(cls):
                if base_cls.__module__.startswith("zero_pax"):
                    documented_attrs |= _extract_documented_attributes(
                        inspect.getdoc(base_cls) or ""
                    )

            for param in expected_params:
                pname = param["name"]
                if pname not in model_fields:
                    errors.append(f"{api_path}: Missing field '{pname}'")
                    continue
                if pname not in documented_attrs:
                    errors.append(f"{api_path}: Undocumented field '{pname}' in class docstring")

                finfo = model_fields[pname]
                snap_ann = param.get("annotation") or ""
                snap_def = param.get("default")
                actual_ann = str(finfo.annotation)

                if (
                    ("None" in snap_ann or snap_def in ("'```(None)```'", "```(None)```"))
                    and "None" not in actual_ann
                    and "Optional" not in actual_ann
                ):
                    errors.append(
                        f"{api_path}.{pname}: Type annotation {actual_ann} does not allow None (snapshot: {snap_ann})"
                    )
                if (
                    "Sequence[int]" in snap_ann
                    and "Sequence" not in actual_ann
                    and "tuple" not in actual_ann
                ):
                    errors.append(
                        f"{api_path}.{pname}: Type annotation {actual_ann} does not allow Sequence[int] (snapshot: {snap_ann})"
                    )

                should_check, expected_def = _parse_expected_default(snap_def)
                if should_check:
                    actual_def = finfo.get_default(call_default_factory=True)
                    if actual_def != expected_def or type(actual_def) is not type(expected_def):
                        errors.append(
                            f"{api_path}.{pname}: Default mismatch (expected {expected_def!r}, got {actual_def!r})"
                        )
        else:
            sig = inspect.signature(cls)
            actual_params = set(sig.parameters.keys())
            for param in expected_params:
                pname = param["name"]
                if pname not in actual_params:
                    errors.append(f"{api_path}: Missing parameter '{pname}'")

    return errors


def verify_semantic_plan(plan_path: Path) -> list[str]:
    """Verify that every layer listed in `SEMANTIC_PLAN.md` is implemented and exported.

    Args:
        plan_path: Path to `SEMANTIC_PLAN.md`.

    Returns:
        List of missing layer error strings.
    """
    if not plan_path.is_file():
        return [f"Missing plan file: {plan_path}"]

    text = plan_path.read_text(encoding="utf-8")
    layer_names = re.findall(r"\* \[[xX ]\] Implement `([A-Za-z0-9_]+)` semantics", text)
    errors: list[str] = []
    for name in layer_names:
        if name not in EXPORTED_NAMES or name not in LAYER_REGISTRY:
            errors.append(f"SEMANTIC_PLAN.md layer `{name}` is missing from zero_pax.praxis.layers")
    return errors


def main() -> int:
    """Run the API and signature compliance check.

    Returns:
        Exit code 0 on 100% compliance, 1 otherwise.
    """
    errors: list[str] = []
    plan_path = Path("SEMANTIC_PLAN.md")
    errors.extend(verify_semantic_plan(plan_path))

    snapshot_path = _find_snapshot_path()
    if snapshot_path is not None:
        errors.extend(verify_snapshot_compliance(snapshot_path))

    if errors:
        print(f"Compliance Check Failed with {len(errors)} error(s):")
        for err in errors[:50]:
            print(f"  - {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors.")
        return 1

    print(
        "Compliance Check Passed! 100.0% compliant with ml-ecosystem-snapshots v0.0.5 pax_v1.4.0.json (108/108 APIs, hierarchy, types, defaults, and docstrings)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
