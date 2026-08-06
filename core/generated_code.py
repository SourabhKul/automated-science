from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
import re


class GeneratedCodeError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractedCode:
    code: str
    thoughts: str = ""
    normalization_applied: bool = False


ALLOWED_TOP_LEVEL = (ast.FunctionDef, ast.Assign, ast.Expr)
ALLOWED_CALL_ROOTS = {"jnp"}
FORBIDDEN_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "locals",
    "open",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "vars",
}
FORBIDDEN_MODULES = {"os", "sys", "subprocess", "socket", "requests", "pathlib", "shutil"}


def _is_safe_math_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return (
            len(node.names) == 1
            and node.names[0].name == "jax.numpy"
            and node.names[0].asname == "jnp"
        )
    if isinstance(node, ast.ImportFrom):
        return (
            node.module == "jax"
            and len(node.names) == 1
            and node.names[0].name == "numpy"
            and node.names[0].asname == "jnp"
        )
    return False


def _is_docstring_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_metadata_assignment(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign):
        return False
    return any(isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets)


def _top_level_surface_nodes(tree: ast.Module) -> tuple[ast.FunctionDef | None, ast.Assign | None, bool]:
    dynamics_node: ast.FunctionDef | None = None
    metadata_node: ast.Assign | None = None
    canonicalizable = True
    for node in tree.body:
        if _is_safe_math_import(node) or _is_docstring_expr(node):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "dynamics":
            if dynamics_node is not None:
                canonicalizable = False
            dynamics_node = node
            continue
        if _is_metadata_assignment(node):
            if metadata_node is not None:
                canonicalizable = False
            metadata_node = node
            continue
        canonicalizable = False
    return dynamics_node, metadata_node, canonicalizable


def _slice_source(lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "\n".join(lines[start:end]).strip()


def _literal_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    scalar = float(value)
    if not math.isfinite(scalar):
        return None
    return scalar


def _validate_metadata_literal_shape(metadata_value: ast.AST, *, max_args_index: int | None) -> None:
    try:
        literal_value = ast.literal_eval(metadata_value)
    except (ValueError, SyntaxError):
        return

    if not isinstance(literal_value, list) or not literal_value:
        raise GeneratedCodeError("metadata must be a non-empty list")

    for index, entry in enumerate(literal_value):
        if not isinstance(entry, dict):
            raise GeneratedCodeError("metadata entries must be dicts")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise GeneratedCodeError(f"metadata[{index}] is missing a non-empty string name")
        value_range = entry.get("range")
        if not isinstance(value_range, (tuple, list)) or len(value_range) != 2:
            raise GeneratedCodeError(f"metadata[{index}] range must be a length-2 tuple/list")
        low = _literal_number(value_range[0])
        high = _literal_number(value_range[1])
        if low is None or high is None:
            raise GeneratedCodeError(f"metadata[{index}] range bounds must be finite numbers")
        if low >= high:
            raise GeneratedCodeError(f"metadata[{index}] range lower bound must be < upper bound")

    if max_args_index is not None and len(literal_value) <= max_args_index:
        raise GeneratedCodeError(
            f"metadata defines {len(literal_value)} parameters but dynamics indexes args[{max_args_index}]"
        )


def _max_args_index(dynamics_node: ast.FunctionDef) -> int | None:
    max_index: int | None = None
    for node in ast.walk(dynamics_node):
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            continue
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
            if slice_node.value < 0:
                raise GeneratedCodeError("args must be indexed with non-negative integer positions")
            max_index = slice_node.value if max_index is None else max(max_index, slice_node.value)
    return max_index


def extract_code(llm_response: str | None) -> ExtractedCode | None:
    if not llm_response:
        return None
    thoughts_match = re.search(r"<think>(.*?)</think>", llm_response, re.DOTALL)
    thoughts = thoughts_match.group(1).strip() if thoughts_match else ""
    match = re.search(r"```python(.*?)```", llm_response, re.DOTALL)
    if match:
        return ExtractedCode(match.group(1).strip(), thoughts)
    fallback = re.search(r"(def dynamics\(.*)", llm_response, re.DOTALL)
    if fallback:
        return ExtractedCode(fallback.group(1).strip(), thoughts)
    stripped = re.sub(r"^```[a-z]*\n?", "", llm_response.strip()).replace("```", "").strip()
    return ExtractedCode(stripped, thoughts)


def normalize_generated_math_code(code: str) -> tuple[str, bool]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False

    lines = code.splitlines()
    dynamics_node, metadata_node, canonicalizable = _top_level_surface_nodes(tree)
    if canonicalizable and dynamics_node is not None and metadata_node is not None:
        canonical_chunks = [
            _slice_source(lines, dynamics_node),
            _slice_source(lines, metadata_node),
        ]
        canonical_code = "\n\n".join(chunk for chunk in canonical_chunks if chunk).strip()
        return canonical_code, canonical_code != code.strip()

    removable_line_indexes: set[int] = set()
    for node in tree.body:
        if not (_is_safe_math_import(node) or _is_docstring_expr(node)):
            continue
        for index in range(node.lineno - 1, node.end_lineno or node.lineno):
            removable_line_indexes.add(index)

    if not removable_line_indexes:
        return code, False

    normalized_lines = [line for index, line in enumerate(lines) if index not in removable_line_indexes]
    normalized_code = "\n".join(normalized_lines).strip()
    return normalized_code, normalized_code != code.strip()


def proposal_fingerprint(code: str) -> str:
    """Return a stable fingerprint for a normalized generated model proposal."""
    normalized_code, _ = normalize_generated_math_code(code)
    try:
        tree = ast.parse(normalized_code)
        canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    except SyntaxError:
        canonical = "\n".join(line.rstrip() for line in normalized_code.splitlines()).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_generated_math_code(
    code: str,
    *,
    dynamics_signature: tuple[str, ...] = ("t", "y", "args"),
) -> None:
    """Reject generated code with obvious non-math side effects.

    This is a lightweight guard, not a sandbox. It is intended as a first filter
    before executing trusted local experiment candidates.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise GeneratedCodeError(f"syntax error: {exc}") from exc

    dynamics_node: ast.FunctionDef | None = None
    metadata_node: ast.Assign | None = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise GeneratedCodeError("imports are not allowed in generated math code")
        if not isinstance(node, ALLOWED_TOP_LEVEL):
            raise GeneratedCodeError(f"top-level {type(node).__name__} is not allowed")
        if _is_docstring_expr(node):
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name != "dynamics":
                raise GeneratedCodeError("only a dynamics function may be defined")
            if dynamics_node is not None:
                raise GeneratedCodeError("only one dynamics function may be defined")
            signature_text = f"dynamics({', '.join(dynamics_signature)})"
            if node.args.defaults or node.args.kw_defaults:
                raise GeneratedCodeError(f"{signature_text} must not define default arguments")
            positional_args = [
                *node.args.posonlyargs,
                *node.args.args,
            ]
            arg_names = [arg.arg for arg in positional_args]
            if arg_names != list(dynamics_signature):
                raise GeneratedCodeError(f"dynamics must have the exact signature {signature_text}")
            if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                raise GeneratedCodeError(f"{signature_text} must not define variadic or keyword-only arguments")
            dynamics_node = node
            continue
        if isinstance(node, ast.Assign):
            if not _is_metadata_assignment(node):
                raise GeneratedCodeError("only a metadata assignment may appear at top level")
            if metadata_node is not None:
                raise GeneratedCodeError("only one metadata assignment may appear at top level")
            metadata_node = node
            continue
        raise GeneratedCodeError("top-level expressions other than a module docstring are not allowed")

    if dynamics_node is None:
        raise GeneratedCodeError(f"missing dynamics({', '.join(dynamics_signature)})")
    if metadata_node is None:
        raise GeneratedCodeError("missing metadata assignment")

    _validate_metadata_literal_shape(
        metadata_node.value,
        max_args_index=_max_args_index(dynamics_node),
    )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise GeneratedCodeError("imports are not allowed")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id in FORBIDDEN_MODULES:
                raise GeneratedCodeError(f"forbidden name: {node.id}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                raise GeneratedCodeError(f"forbidden call: {func.id}")
            if isinstance(func, ast.Attribute):
                root = func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in FORBIDDEN_MODULES:
                    raise GeneratedCodeError(f"forbidden module call: {root.id}")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "args":
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                raise GeneratedCodeError("args must be indexed positionally, not by parameter name")


def extract_and_validate(llm_response: str | None, *, normalize: bool = False) -> ExtractedCode | None:
    extracted = extract_code(llm_response)
    if extracted is None:
        return None
    code = extracted.code
    normalization_applied = False
    if normalize:
        code, normalization_applied = normalize_generated_math_code(code)
    validate_generated_math_code(code)
    return ExtractedCode(
        code=code,
        thoughts=extracted.thoughts,
        normalization_applied=normalization_applied,
    )


def extract_validated_code_tuple(llm_response: str | None) -> tuple[str | None, str]:
    """Compatibility wrapper for legacy runners that expect ``(code, thoughts)``."""
    try:
        extracted = extract_and_validate(llm_response)
    except GeneratedCodeError as exc:
        print(f"Generated code rejected: {exc}")
        return None, ""
    if extracted is None:
        return None, ""
    if extracted.thoughts:
        print("--- Model Thoughts ---")
        print(f"{extracted.thoughts[:300]}...")
        print("----------------------")
    return extracted.code, extracted.thoughts
