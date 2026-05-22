from __future__ import annotations

import pytest

from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows import eval_when, resolve


def test_params_substitution_typed() -> None:
    scope = {"params": {"n": 7, "msg": "hi"}}
    assert resolve("${params.n}", scope) == 7
    assert resolve("${params.msg}", scope) == "hi"


def test_outputs_substitution() -> None:
    scope = {"outputs": {"a": {"payload": "hello"}}}
    assert resolve("${outputs.a.payload}", scope) == "hello"


def test_item_substitution() -> None:
    scope = {"item": {"c": 42}}
    assert resolve("${item.c}", scope) == 42


def test_deep_nesting() -> None:
    scope = {"params": {"x": 1}, "outputs": {"a": {"payload": "p"}}}
    template = {
        "outer": {
            "inner_list": ["${params.x}", "literal", "${outputs.a.payload}-suffix"],
            "deep": {"k": "${params.x}"},
        }
    }
    out = resolve(template, scope)
    assert out["outer"]["inner_list"] == [1, "literal", "p-suffix"]
    assert out["outer"]["deep"]["k"] == 1


def test_missing_key_raises_invalid_params() -> None:
    scope = {"params": {"a": 1}}
    with pytest.raises(InvalidParamsError):
        resolve("${params.b}", scope)


def test_string_interpolation_with_two_refs() -> None:
    scope = {"params": {"a": "X", "b": "Y"}}
    assert resolve("${params.a}-${params.b}", scope) == "X-Y"


# ---------------------------------------------------------------- eval_when


def test_eval_when_simple_compare() -> None:
    scope = {"params": {"k": 3}}
    assert eval_when("params['k'] >= 2", scope) is True
    assert eval_when("params['k'] >= 99", scope) is False


def test_eval_when_len_call() -> None:
    scope = {"outputs": {"a": [1, 2, 3]}}
    assert eval_when("len(outputs['a']) >= 1", scope) is True


def test_eval_when_rejects_import() -> None:
    with pytest.raises(InvalidParamsError):
        eval_when("__import__('os').system('echo hi')", {})


def test_eval_when_rejects_lambda() -> None:
    with pytest.raises(InvalidParamsError):
        eval_when("(lambda: 1)()", {})


def test_eval_when_rejects_attr_on_non_scope() -> None:
    # `os` is not in scope; eval will raise NameError → wrapped as InvalidParamsError.
    with pytest.raises(InvalidParamsError):
        eval_when("os.system('x')", {})


def test_eval_when_rejects_comprehensions() -> None:
    with pytest.raises(InvalidParamsError):
        eval_when("[x for x in [1,2,3]]", {})


def test_eval_when_bool_and_or() -> None:
    assert eval_when("True and not False", {}) is True
    assert eval_when("True or False", {}) is True
