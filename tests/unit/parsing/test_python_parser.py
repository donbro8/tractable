"""Unit tests for PythonParser."""

from __future__ import annotations

import pytest

from tractable.parsing.parsers.python_parser import PythonParser
from tractable.protocols.graph_construction import CodeParser

# ── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_SOURCE: bytes = b'''\
import os
import sys
from pathlib import Path

class MyClass:
    """A sample class."""

    def __init__(self, value: int) -> None:
        self.value = value

    def get_value(self) -> int:
        return self.value

async def async_func(x: int) -> str:
    return str(x)

def plain_func() -> None:
    pass
'''

DECORATED_SOURCE: bytes = b"""\
import functools

def my_decorator(fn):
    return fn

@my_decorator
@functools.wraps
async def decorated_async(x: int) -> int:
    return x * 2

@my_decorator
class DecoratedClass:
    pass
"""

SYNTAX_ERROR_SOURCE: bytes = b"""\
def foo(:
    pass
"""

CALL_SOURCE: bytes = b"""\
def helper() -> None:
    pass

def caller() -> None:
    helper()
    os.path.join("a", "b")
"""


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def parser() -> PythonParser:
    return PythonParser()


def test_satisfies_code_parser_protocol(parser: PythonParser) -> None:
    """AC-1: PythonParser is an instance of the CodeParser Protocol."""
    assert isinstance(parser, CodeParser)


def test_supported_extensions(parser: PythonParser) -> None:
    """AC-2: supported_extensions == frozenset({'.py', '.pyi'})."""
    assert parser.supported_extensions == frozenset({".py", ".pyi"})


@pytest.mark.asyncio
async def test_entity_extraction_counts(parser: PythonParser) -> None:
    """AC-3: 3 functions + 1 class → at least 4 entities with correct kinds."""
    result = await parser.parse_file("mymodule/sample.py", SAMPLE_SOURCE)
    kinds = {e.kind for e in result.entities}
    assert "function" in kinds
    assert "class" in kinds
    assert len(result.entities) >= 4  # module + class + 3 functions (at least)


@pytest.mark.asyncio
async def test_syntax_error_resilience(parser: PythonParser) -> None:
    """AC-4: Syntax error → non-empty ParseResult with module entity, no exception."""
    result = await parser.parse_file("bad.py", SYNTAX_ERROR_SOURCE)
    assert len(result.entities) >= 1
    module_entities = [e for e in result.entities if e.kind == "module"]
    assert len(module_entities) == 1


@pytest.mark.asyncio
async def test_import_produces_relationship_or_unresolved(parser: PythonParser) -> None:
    """AC-5: import os → ParsedRelationship(IMPORTS) or UnresolvedReference."""
    result = await parser.parse_file("sample.py", b"import os\n")
    has_imports_rel = any(r.relationship == "IMPORTS" for r in result.relationships)
    has_unresolved = len(result.unresolved_references) > 0
    assert has_imports_rel or has_unresolved


@pytest.mark.asyncio
async def test_function_extraction(parser: PythonParser) -> None:
    """All three functions in SAMPLE_SOURCE are extracted."""
    result = await parser.parse_file("sample.py", SAMPLE_SOURCE)
    fn_names = {e.name for e in result.entities if e.kind == "function"}
    assert "async_func" in fn_names
    assert "plain_func" in fn_names


@pytest.mark.asyncio
async def test_class_extraction(parser: PythonParser) -> None:
    """Class name and base_classes property are correct."""
    result = await parser.parse_file("sample.py", SAMPLE_SOURCE)
    classes = [e for e in result.entities if e.kind == "class"]
    assert len(classes) >= 1
    cls = next(c for c in classes if c.name == "MyClass")
    assert cls.properties["base_classes"] == []


@pytest.mark.asyncio
async def test_async_function_detection(parser: PythonParser) -> None:
    """is_async property is True for async functions."""
    result = await parser.parse_file("sample.py", SAMPLE_SOURCE)
    async_fns = [
        e for e in result.entities if e.kind == "function" and e.properties.get("is_async")
    ]
    assert any(e.name == "async_func" for e in async_fns)


@pytest.mark.asyncio
async def test_decorated_function_detection(parser: PythonParser) -> None:
    """Decorators are captured in properties."""
    result = await parser.parse_file("decorated.py", DECORATED_SOURCE)
    fn = next(
        (e for e in result.entities if e.kind == "function" and e.name == "decorated_async"),
        None,
    )
    assert fn is not None
    assert len(fn.properties["decorators"]) >= 1


@pytest.mark.asyncio
async def test_import_relationship_extraction(parser: PythonParser) -> None:
    """IMPORTS relationships are created for standard imports."""
    result = await parser.parse_file("sample.py", SAMPLE_SOURCE)
    import_rels = [r for r in result.relationships if r.relationship == "IMPORTS"]
    # os, sys, pathlib → at least 3 import relationships
    assert len(import_rels) >= 3


@pytest.mark.asyncio
async def test_module_entity_always_present(parser: PythonParser) -> None:
    """A module entity is always the first entity returned."""
    result = await parser.parse_file("pkg/mod.py", SAMPLE_SOURCE)
    assert result.entities[0].kind == "module"
    assert result.entities[0].qualified_name == "pkg.mod"


@pytest.mark.asyncio
async def test_method_detection(parser: PythonParser) -> None:
    """Methods inside a class have is_method=True."""
    result = await parser.parse_file("sample.py", SAMPLE_SOURCE)
    methods = [e for e in result.entities if e.kind == "function" and e.properties.get("is_method")]
    method_names = {e.name for e in methods}
    assert "__init__" in method_names or "get_value" in method_names


@pytest.mark.asyncio
async def test_relative_import_produces_unresolved(parser: PythonParser) -> None:
    """Relative imports become UnresolvedReferences."""
    src = b"from . import sibling\nfrom ..utils import helper\n"
    result = await parser.parse_file("pkg/mod.py", src)
    assert len(result.unresolved_references) >= 1


@pytest.mark.asyncio
async def test_parse_result_language(parser: PythonParser) -> None:
    """ParseResult.language is 'python'."""
    result = await parser.parse_file("foo.py", b"x = 1\n")
    assert result.language == "python"


@pytest.mark.asyncio
async def test_line_numbers(parser: PythonParser) -> None:
    """line_start / line_end are 1-based and sensible."""
    result = await parser.parse_file("sample.py", SAMPLE_SOURCE)
    for entity in result.entities:
        assert entity.line_start >= 1
        assert entity.line_end >= entity.line_start
