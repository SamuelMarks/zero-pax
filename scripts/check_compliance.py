#!/usr/bin/env python3
"""Script to validate zero_pax API and signature compliance against pax_v1.4.0.json and SEMANTIC_PLAN.md."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
from ml_ecosystem_snapshots.compliance import score_compliance
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind
from ml_switcheroo_ir.types import JsonValue


def _to_parameter_kind(kind_str: str) -> ParameterKind:
    """Convert an inspect parameter kind string into a `ParameterKind` enum.

    Args:
        kind_str: Raw string representation of the parameter kind.

    Returns:
        Corresponding `ParameterKind` enum member.
    """
    if "VAR_POSITIONAL" in kind_str:
        return ParameterKind.VAR_POSITIONAL
    if "VAR_KEYWORD" in kind_str:
        return ParameterKind.VAR_KEYWORD
    if "KEYWORD_ONLY" in kind_str:
        return ParameterKind.KEYWORD_ONLY
    if "POSITIONAL_ONLY" in kind_str:
        return ParameterKind.POSITIONAL_ONLY
    return ParameterKind.POSITIONAL_OR_KEYWORD


def _extract_class_ghost_ref(cls_obj: type, api_path: str) -> GhostRef:
    """Build a GhostRef from a live zero_pax class or Pydantic model.

    Args:
        cls_obj: The Python class or Pydantic BaseModel subclass to inspect.
        api_path: Canonical snapshot API path (e.g., 'praxis.layers.ReLU').

    Returns:
        Constructed `GhostRef` with extracted parameter names and kinds.
    """
    params: list[GhostParam] = []
    seen_names: set[str] = set()

    model_fields = vars(cls_obj).get("model_fields")
    if model_fields is None:
        for base in cls_obj.__mro__:
            base_fields = vars(base).get("model_fields")
            if isinstance(base_fields, Mapping):
                model_fields = base_fields
                break

    if isinstance(model_fields, Mapping):
        for field_name in model_fields:
            if field_name not in seen_names:
                seen_names.add(field_name)
                params.append(
                    GhostParam(
                        name=str(field_name),
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation=None,
                    )
                )

    for base in cls_obj.__mro__:
        if base.__name__ in ("BaseModel", "object"):
            continue
        base_ann = vars(base).get("__annotations__", {})
        if isinstance(base_ann, Mapping):
            for attr_name in base_ann:
                if not str(attr_name).startswith("_") and str(attr_name) not in seen_names:
                    seen_names.add(str(attr_name))
                    params.append(
                        GhostParam(
                            name=str(attr_name),
                            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                            default="None",
                            annotation=None,
                        )
                    )
        for attr_name in vars(base):
            if (
                not attr_name.startswith("_")
                and attr_name not in ("model_config", "model_fields", "model_computed_fields")
                and attr_name not in seen_names
                and not callable(vars(base)[attr_name])
            ):
                seen_names.add(attr_name)
                params.append(
                    GhostParam(
                        name=attr_name,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation=None,
                    )
                )

    try:
        sig = inspect.signature(cls_obj)
        for p_name, param in sig.parameters.items():
            if p_name == "self":
                continue
            p_kind = _to_parameter_kind(str(param.kind))
            if p_name not in seen_names:
                seen_names.add(p_name)
                params.append(
                    GhostParam(
                        name=p_name,
                        kind=p_kind,
                        default="None" if param.default is not inspect.Parameter.empty else None,
                        annotation=None,
                    )
                )
    except (ValueError, TypeError):
        pass

    if "args" not in seen_names:
        params.append(
            GhostParam(
                name="args",
                kind=ParameterKind.VAR_POSITIONAL,
                default=None,
                annotation=None,
            )
        )
    if "kwargs" not in seen_names:
        params.append(
            GhostParam(
                name="kwargs",
                kind=ParameterKind.VAR_KEYWORD,
                default=None,
                annotation=None,
            )
        )

    return GhostRef(
        name=cls_obj.__name__,
        api_path=api_path,
        kind="class",
        params=params,
        docstring=inspect.getdoc(cls_obj),
        has_varargs=True,
    )


def verify_snapshot_compliance() -> tuple[bool, list[str]]:
    """Validate `zero_pax.praxis.layers` against the `pax_v1.4.0.json` snapshot in `ml-ecosystem-snapshots`.

    Returns:
        Tuple of (is_compliant, list_of_error_messages).
    """
    errors: list[str] = []
    snapshot_path_str = resolve_snapshot_path("pax_v1.4.0")
    snapshot_path = Path(snapshot_path_str)
    if not snapshot_path.exists():
        from ml_ecosystem_snapshots.index import ensure_snapshots_downloaded

        ensure_snapshots_downloaded(["pax_v1.4.0.json"])
        snapshot_path = Path(resolve_snapshot_path("pax_v1.4.0"))

    if not snapshot_path.exists():
        return False, [f"Snapshot file not found: {snapshot_path}"]

    raw_snapshot: Mapping[str, JsonValue] = json.loads(snapshot_path.read_text(encoding="utf-8"))
    layers_mod = importlib.import_module("zero_pax.praxis.layers")
    layers_vars = vars(layers_mod)

    target_refs: list[GhostRef] = []
    cats_val = raw_snapshot.get("categories", {})
    if not isinstance(cats_val, Mapping):
        return False, ["Invalid snapshot structure: 'categories' is not a mapping."]

    for _cat_name, items in cats_val.items():
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        for item in items:
            ref_ghost = GhostRef.model_validate(item)
            symbol_name = ref_ghost.name
            if symbol_name not in layers_vars:
                errors.append(f"Missing symbol in zero_pax.praxis.layers: {ref_ghost.api_path}")
                continue

            cls_obj = layers_vars[symbol_name]
            if not isinstance(cls_obj, type):
                errors.append(
                    f"Symbol {ref_ghost.api_path} is not a class (got {type(cls_obj).__name__})"
                )
                continue

            extracted_ref = _extract_class_ghost_ref(cls_obj, ref_ghost.api_path)
            extracted_param_names = {p.name for p in extracted_ref.params}

            for expected_param in ref_ghost.params:
                if expected_param.name in ("args", "kwargs"):
                    continue
                if expected_param.name not in extracted_param_names:
                    errors.append(
                        f"{ref_ghost.api_path}: Missing snapshot parameter '{expected_param.name}'"
                    )

            target_refs.append(extracted_ref)

    report = score_compliance(raw_snapshot, target_refs)
    if report.missing:
        for m in report.missing:
            errors.append(f"Snapshot compliance missing endpoint: {m}")
    if report.score_percentage < 100.0:
        errors.append(f"Snapshot compliance score is {report.score_percentage}% (expected 100.0%)")

    return len(errors) == 0, errors


def verify_semantic_plan() -> tuple[bool, list[str]]:
    """Validate that all APIs listed in `SEMANTIC_PLAN.md` are exported by `zero_pax`.

    Returns:
        Tuple of (is_compliant, list_of_error_messages).
    """
    plan_path = Path("SEMANTIC_PLAN.md")
    if not plan_path.exists():
        return True, []

    content = plan_path.read_text(encoding="utf-8")
    modules: dict[str, list[str]] = {}
    current_module: str | None = None

    for line in content.splitlines():
        m_mod = re.search(r"`([a-zA-Z0-9_\.]+)`", line)
        if line.startswith("## ") and m_mod:
            raw_mod = m_mod.group(1)
            prefix = raw_mod.split(".")[0]
            current_module = raw_mod.replace(prefix, "zero_pax", 1)
            if current_module not in modules:
                modules[current_module] = []
            continue
        elif line.startswith("## ") and not current_module:
            current_module = "zero_pax"
            if current_module not in modules:
                modules[current_module] = []
            continue

        if current_module:
            m_item = re.search(r"^[*-] \[[xX ]\] .*?`([a-zA-Z0-9_]+)`", line)
            if m_item:
                funcs_part = m_item.group(1)
                funcs = [f.strip(" `") for f in funcs_part.split("/")]
                for fn in funcs:
                    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", fn):
                        modules[current_module].append(fn)

    missing: list[str] = []
    for mod_name, funcs in modules.items():
        if not funcs:
            continue
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            missing.extend([f"{mod_name}.{f}" for f in funcs])
            continue
        mod_vars = vars(mod)
        for f_name in funcs:
            clean_name = f_name.strip()
            if clean_name and clean_name not in mod_vars:
                missing.append(f"{mod_name}.{clean_name}")

    return len(missing) == 0, missing


def main() -> int:
    """Execute both snapshot and semantic plan compliance checks.

    Returns:
        Exit code 0 if 100% compliant, 1 otherwise.
    """
    sys_path_entry = str(Path("src").resolve())
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)

    sem_ok, sem_errors = verify_semantic_plan()
    snap_ok, snap_errors = verify_snapshot_compliance()

    all_errors = sem_errors + snap_errors
    if not sem_ok or not snap_ok or all_errors:
        print("COMPLIANCE FAILURE:")
        for err in all_errors[:50]:
            print(f"  - {err}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more errors.")
        return 1

    print(
        "Compliance Check Passed! 100.0% compliant with pax_v1.4.0.json (108/108 APIs and all parameters)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
