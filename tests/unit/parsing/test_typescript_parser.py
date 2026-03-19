"""Unit tests for TypeScriptParser."""

from __future__ import annotations

import pytest

from tractable.parsing.parsers.typescript_parser import TypeScriptParser
from tractable.protocols.graph_construction import CodeParser
from tractable.types.enums import EdgeConfidence


# ── Fixtures and sample sources ──────────────────────────────────────────────

SAMPLE_SOURCE: bytes = b"""\
import { readFile } from 'fs';
import { helper } from './utils';
import type { Config } from './config';

export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export async function fetchData(url: string): Promise<string> {
    return '';
}

export class Greeter extends BaseGreeter {
    constructor(private name: string) {
        super();
    }
}
"""

ARROW_SOURCE: bytes = b"""\
import axios from 'axios';
import { format } from './format';

export const multiply = (x: number, y: number): number => x * y;

export const asyncFetch = async (url: string): Promise<string> => {
    return '';
};

const privateHelper = (x: number) => x + 1;
"""

SYNTAX_ERROR_SOURCE: bytes = b"""\
export function broken(: string {
    return 'oops';
}
"""

CLASS_SOURCE: bytes = b"""\
export class Animal {
    name: string = '';
}

export class Dog extends Animal {
    bark(): void {}
}

class PrivateClass {}
"""

PACKAGE_IMPORT_SOURCE: bytes = b"""\
import React from 'react';
import { useState } from 'react';
import lodash from 'lodash';
"""

LOCAL_IMPORT_SOURCE: bytes = b"""\
import { foo } from './utils';
import { bar } from '../lib/helpers';
import type { Baz } from './types';
"""

TSX_SOURCE: bytes = b"""\
import React from 'react';
import { helper } from './helper';

export function Button(): JSX.Element {
    return React.createElement('button', null, 'Click me');
}

export class Form extends React.Component<{}, {}> {
    render() {
        return null;
    }
}
"""

DTS_SOURCE: bytes = b"""\
export declare function greet(name: string): string;
export declare class Greeter {}
"""


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def parser() -> TypeScriptParser:
    return TypeScriptParser()


# AC-1: supported_extensions
def test_supported_extensions(parser: TypeScriptParser) -> None:
    assert parser.supported_extensions == frozenset({".ts", ".tsx", ".js", ".jsx"})


# Protocol satisfaction
def test_satisfies_code_parser_protocol(parser: TypeScriptParser) -> None:
    assert isinstance(parser, CodeParser)


# AC-2: 2 exported functions + 1 class → at least 3 entities
@pytest.mark.asyncio
async def test_entity_count_with_functions_and_class(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/greeter.ts", SAMPLE_SOURCE)
    assert len(result.entities) >= 3
    kinds = {e.kind for e in result.entities}
    assert "function" in kinds
    assert "class" in kinds


# AC-3: local import → ParsedRelationship DETERMINISTIC
@pytest.mark.asyncio
async def test_local_import_produces_deterministic_relationship(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/mod.ts", LOCAL_IMPORT_SOURCE)
    import_rels = [r for r in result.relationships if r.relationship == "IMPORTS"]
    assert len(import_rels) >= 1
    for rel in import_rels:
        assert rel.resolution == EdgeConfidence.DETERMINISTIC


# AC-4: package import → UnresolvedReference
@pytest.mark.asyncio
async def test_package_import_produces_unresolved_reference(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/mod.ts", PACKAGE_IMPORT_SOURCE)
    assert len(result.unresolved_references) >= 1
    refs = [u for u in result.unresolved_references if u.likely_kind == "import"]
    pkg_names = {u.reference_string for u in refs}
    assert "react" in pkg_names or "lodash" in pkg_names


# AC-5: pytest exits with code 0 — verified by running this test file
# AC-6: pyright clean — verified via pyright command

# ── Additional unit tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_module_entity_always_present(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/greeter.ts", SAMPLE_SOURCE)
    modules = [e for e in result.entities if e.kind == "module"]
    assert len(modules) == 1
    assert modules[0].qualified_name == "src.greeter"


@pytest.mark.asyncio
async def test_function_extraction_regular(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/greeter.ts", SAMPLE_SOURCE)
    fn_names = {e.name for e in result.entities if e.kind == "function"}
    assert "greet" in fn_names
    assert "fetchData" in fn_names


@pytest.mark.asyncio
async def test_async_function_detection(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/greeter.ts", SAMPLE_SOURCE)
    async_fns = [e for e in result.entities if e.kind == "function" and e.properties.get("is_async")]
    assert any(e.name == "fetchData" for e in async_fns)


@pytest.mark.asyncio
async def test_exported_function_flag(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/greeter.ts", SAMPLE_SOURCE)
    greet = next((e for e in result.entities if e.name == "greet"), None)
    assert greet is not None
    assert greet.properties.get("is_exported") is True


@pytest.mark.asyncio
async def test_arrow_function_extraction(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/arrows.ts", ARROW_SOURCE)
    fn_names = {e.name for e in result.entities if e.kind == "function"}
    assert "multiply" in fn_names
    assert "asyncFetch" in fn_names
    assert "privateHelper" in fn_names


@pytest.mark.asyncio
async def test_async_arrow_function_detection(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/arrows.ts", ARROW_SOURCE)
    async_fns = [e for e in result.entities if e.kind == "function" and e.properties.get("is_async")]
    assert any(e.name == "asyncFetch" for e in async_fns)


@pytest.mark.asyncio
async def test_class_extraction(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/classes.ts", CLASS_SOURCE)
    class_names = {e.name for e in result.entities if e.kind == "class"}
    assert "Animal" in class_names
    assert "Dog" in class_names
    assert "PrivateClass" in class_names


@pytest.mark.asyncio
async def test_class_base_classes(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/classes.ts", CLASS_SOURCE)
    dog = next((e for e in result.entities if e.name == "Dog"), None)
    assert dog is not None
    assert "Animal" in dog.properties.get("base_classes", [])


@pytest.mark.asyncio
async def test_class_no_base_classes(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/classes.ts", CLASS_SOURCE)
    animal = next((e for e in result.entities if e.name == "Animal"), None)
    assert animal is not None
    assert animal.properties.get("base_classes") == []


@pytest.mark.asyncio
async def test_exported_class_flag(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/classes.ts", CLASS_SOURCE)
    animal = next((e for e in result.entities if e.name == "Animal"), None)
    assert animal is not None
    assert animal.properties.get("is_exported") is True
    private_cls = next((e for e in result.entities if e.name == "PrivateClass"), None)
    assert private_cls is not None
    assert private_cls.properties.get("is_exported") is False


@pytest.mark.asyncio
async def test_local_import_not_unresolved(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/mod.ts", LOCAL_IMPORT_SOURCE)
    # Local imports should be ParsedRelationship, not UnresolvedReference
    import_rels = [r for r in result.relationships if r.relationship == "IMPORTS"]
    assert len(import_rels) >= 2  # ./utils and ../lib/helpers (and ./types)
    # No package names in unresolved
    unresolved_names = {u.reference_string for u in result.unresolved_references}
    assert "./utils" not in unresolved_names or len(unresolved_names) == 0


@pytest.mark.asyncio
async def test_package_import_not_in_relationships(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/mod.ts", PACKAGE_IMPORT_SOURCE)
    # Package imports should not appear as DETERMINISTIC relationships
    import_rels = [r for r in result.relationships if r.relationship == "IMPORTS"]
    for rel in import_rels:
        assert rel.resolution != EdgeConfidence.DETERMINISTIC or rel.target_qualified_name.startswith("./") or rel.target_qualified_name.startswith("../")


@pytest.mark.asyncio
async def test_syntax_error_resilience(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("broken.ts", SYNTAX_ERROR_SOURCE)
    assert len(result.entities) >= 1
    modules = [e for e in result.entities if e.kind == "module"]
    assert len(modules) == 1


@pytest.mark.asyncio
async def test_tsx_extension_parses(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/Button.tsx", TSX_SOURCE)
    fn_names = {e.name for e in result.entities if e.kind == "function"}
    assert "Button" in fn_names
    class_names = {e.name for e in result.entities if e.kind == "class"}
    assert "Form" in class_names


@pytest.mark.asyncio
async def test_dts_file_returns_module_only(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("types/index.d.ts", DTS_SOURCE)
    assert len(result.entities) == 1
    assert result.entities[0].kind == "module"


@pytest.mark.asyncio
async def test_parse_result_language(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("foo.ts", b"const x = 1;\n")
    assert result.language == "typescript"


@pytest.mark.asyncio
async def test_line_numbers_are_valid(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/greeter.ts", SAMPLE_SOURCE)
    for entity in result.entities:
        assert entity.line_start >= 1
        assert entity.line_end >= entity.line_start


@pytest.mark.asyncio
async def test_defines_relationships(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/greeter.ts", SAMPLE_SOURCE)
    defines_rels = [r for r in result.relationships if r.relationship == "DEFINES"]
    assert len(defines_rels) >= 1


@pytest.mark.asyncio
async def test_mixed_imports_sample(parser: TypeScriptParser) -> None:
    result = await parser.parse_file("src/arrows.ts", ARROW_SOURCE)
    # axios is a package import → unresolved
    unresolved_names = {u.reference_string for u in result.unresolved_references}
    assert "axios" in unresolved_names
    # ./format is local → deterministic relationship
    local_rels = [
        r for r in result.relationships
        if r.relationship == "IMPORTS" and r.resolution == EdgeConfidence.DETERMINISTIC
    ]
    assert len(local_rels) >= 1


@pytest.mark.asyncio
async def test_js_extension_supported(parser: TypeScriptParser) -> None:
    src = b"function add(a, b) { return a + b; }\n"
    result = await parser.parse_file("utils.js", src)
    fn_names = {e.name for e in result.entities if e.kind == "function"}
    assert "add" in fn_names


@pytest.mark.asyncio
async def test_jsx_extension_supported(parser: TypeScriptParser) -> None:
    src = b"export function App() { return null; }\n"
    result = await parser.parse_file("App.jsx", src)
    fn_names = {e.name for e in result.entities if e.kind == "function"}
    assert "App" in fn_names
