from __future__ import annotations

import ast
import re
from typing import Any

from vibeproteinagent.core.errors import InvalidParamsError

_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def iter_refs(value: Any) -> list[str]:
    """Walk ``value`` and return every ``${...}`` expression body (no braces).

    Used by the workflow DAG validator (Codex round-3 RC2) to discover
    every upstream reference declared by a node's inputs/foreach/when
    *without* having to substitute against a scope. Dicts, lists and
    tuples are recursed; non-string scalars contribute nothing.
    """
    out: list[str] = []

    def _walk(v: Any) -> None:
        if isinstance(v, str):
            for m in _VAR_RE.finditer(v):
                out.append(m.group(1).strip())
            return
        if isinstance(v, dict):
            for k, child in v.items():
                _walk(k)
                _walk(child)
            return
        if isinstance(v, (list, tuple)):
            for item in v:
                _walk(item)

    _walk(value)
    return out


# ---------------------------------------------------------------- resolve


def resolve(value: Any, scope: dict[str, Any]) -> Any:
    """Recursively substitute ``${...}`` references in ``value`` from ``scope``.

    Strings: replaced. If the *entire* string is one reference, the typed value
    is returned (so ``${params.n}`` -> ``int`` not ``"4"``). Otherwise string
    interpolation is used.
    Dicts/lists/tuples: recursed.
    Other types: returned unchanged.
    """
    if isinstance(value, str):
        return _resolve_str(value, scope)
    if isinstance(value, dict):
        return {k: resolve(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, scope) for v in value]
    if isinstance(value, tuple):
        return tuple(resolve(v, scope) for v in value)
    return value


def _resolve_str(s: str, scope: dict[str, Any]) -> Any:
    matches = list(_VAR_RE.finditer(s))
    if not matches:
        return s
    if len(matches) == 1 and matches[0].span() == (0, len(s)):
        return _lookup(matches[0].group(1).strip(), scope)
    # Multi-match or partial: stringify each result.
    out: list[str] = []
    last = 0
    for m in matches:
        out.append(s[last : m.start()])
        out.append(str(_lookup(m.group(1).strip(), scope)))
        last = m.end()
    out.append(s[last:])
    return "".join(out)


def _lookup(expr: str, scope: dict[str, Any]) -> Any:
    """Walk ``expr`` (e.g. ``params.x.y[0]``) through ``scope``."""
    # Split on dots; bracket indexing is handled per-segment.
    parts = _split_path(expr)
    if not parts:
        raise InvalidParamsError(
            f"empty workflow reference '${{{expr}}}'",
            details={"expr": expr},
        )
    cur: Any = scope
    walked: list[str] = []
    for part in parts:
        walked.append(part)
        key, indices = _parse_segment(part)
        if isinstance(cur, dict):
            if key not in cur:
                raise InvalidParamsError(
                    f"workflow reference '${{{expr}}}' has no '{key}' "
                    f"(at '{'.'.join(walked)}')",
                    details={"expr": expr, "missing": key,
                             "available": sorted(cur.keys())},
                )
            cur = cur[key]
        else:
            cur = getattr(cur, key, None)
            if cur is None:
                raise InvalidParamsError(
                    f"workflow reference '${{{expr}}}' cannot resolve '{key}' "
                    f"(at '{'.'.join(walked)}')",
                    details={"expr": expr, "missing": key},
                )
        for idx in indices:
            try:
                cur = cur[idx]
            except (KeyError, IndexError, TypeError) as exc:
                raise InvalidParamsError(
                    f"workflow reference '${{{expr}}}' index {idx!r} failed: {exc}",
                    details={"expr": expr, "index": idx},
                ) from exc
    return cur


def _split_path(expr: str) -> list[str]:
    return [p for p in expr.split(".") if p]


def _parse_segment(seg: str) -> tuple[str, list[Any]]:
    """``foo[0][bar]`` -> ('foo', [0, 'bar'])."""
    indices: list[Any] = []
    name = seg
    while "[" in name:
        i = name.index("[")
        j = name.index("]", i)
        token = name[i + 1 : j]
        if token.startswith("'") and token.endswith("'"):
            indices.append(token[1:-1])
        elif token.startswith('"') and token.endswith('"'):
            indices.append(token[1:-1])
        else:
            try:
                indices.append(int(token))
            except ValueError:
                indices.append(token)
        name = name[:i] + name[j + 1 :]
    return name, indices


# ---------------------------------------------------------------- eval_when

_ALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Index,  # py<3.9 compat; py>=3.9 collapses Index into the slice node
    ast.List,
    ast.Tuple,
    ast.Dict,
    # operator nodes
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,
)

_ALLOWED_BUILTINS: dict[str, Any] = {
    "len": len,
    "all": all,
    "any": any,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "True": True,
    "False": False,
    "None": None,
}


def eval_when(expr: str, scope: dict[str, Any], *, max_depth: int = 8) -> bool:
    """Evaluate a restricted Python boolean expression against ``scope``.

    Allowed: literals, names from ``scope``, comparisons, boolean/arithmetic
    operators, subscripting, attribute access on values *already in scope*,
    and whitelisted builtin calls (``len``, ``all``, ``any``, etc.).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise InvalidParamsError(
            f"when expression {expr!r} is not valid Python: {exc.msg}",
            details={"expr": expr},
        ) from exc
    _validate_ast(tree, max_depth=max_depth, expr=expr)

    env = {**_ALLOWED_BUILTINS, **scope}
    try:
        result = eval(
            compile(tree, "<when>", "eval"),
            {"__builtins__": {}},
            env,
        )
    except (NameError, KeyError, AttributeError, TypeError) as exc:
        raise InvalidParamsError(
            f"when expression {expr!r} failed: {exc}",
            details={"expr": expr, "error": str(exc)},
        ) from exc
    return bool(result)


def _validate_ast(tree: ast.AST, *, max_depth: int, expr: str) -> None:
    def _walk(node: ast.AST, depth: int) -> None:
        if depth > max_depth:
            raise InvalidParamsError(
                f"when expression {expr!r} exceeds max depth {max_depth}",
                details={"expr": expr},
            )
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise InvalidParamsError(
                f"when expression {expr!r} contains disallowed node "
                f"{type(node).__name__}",
                details={"expr": expr, "node": type(node).__name__},
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise InvalidParamsError(
                    f"when expression {expr!r} may only call simple names",
                    details={"expr": expr},
                )
            if node.func.id not in _ALLOWED_BUILTINS:
                raise InvalidParamsError(
                    f"when expression {expr!r} called disallowed function "
                    f"{node.func.id!r}",
                    details={"expr": expr, "function": node.func.id},
                )
        for child in ast.iter_child_nodes(node):
            _walk(child, depth + 1)

    _walk(tree, 0)


__all__ = ["eval_when", "iter_refs", "resolve"]
