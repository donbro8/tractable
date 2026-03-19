"""TypeScriptParser — tree-sitter-based parser for TypeScript/JavaScript source files.

Extracts functions, classes, modules, and imports from ``.ts``, ``.tsx``,
``.js``, and ``.jsx`` files and returns a :class:`ParseResult`.

Scope (Phase 1 basic):
- ``function``: regular functions and arrow functions assigned to named constants
- ``class``: class declarations with optional ``extends`` clause
- ``module``: one per file
- ``IMPORTS``: local (``./`` / ``../``) → DETERMINISTIC; packages → UnresolvedReference
- ``DEFINES``: module → function/class

Not supported (Phase 4+): interface parsing, type extraction, generic resolution.
Not handled: ``.d.ts`` declaration files (returned as empty module-only results).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

import structlog
import tree_sitter_typescript as _tst
from tree_sitter import Language, Node, Parser

from tractable.protocols.graph_construction import (
    ParsedEntity,
    ParsedRelationship,
    ParseResult,
    UnresolvedReference,
)
from tractable.types.enums import EdgeConfidence

log = structlog.get_logger(__name__)

_TS_LANGUAGE = Language(_tst.language_typescript())
_TSX_LANGUAGE = Language(_tst.language_tsx())

# Extensions that should use the TSX grammar
_TSX_EXTENSIONS = frozenset({".tsx", ".jsx"})


def _language_for_extension(ext: str) -> Language:
    return _TSX_LANGUAGE if ext in _TSX_EXTENSIONS else _TS_LANGUAGE


def _file_path_to_module(file_path: str) -> str:
    """Derive a dotted module name from a file path.

    Examples::

        src/components/Button.tsx → src.components.Button
        ./lib/utils.ts            → lib.utils
    """
    p = file_path.replace("\\", "/").lstrip("./")
    for suffix in (".tsx", ".jsx", ".ts", ".js"):
        if p.endswith(suffix):
            p = p[: -len(suffix)]
            break
    return p.replace("/", ".")


def _node_text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _string_value(node: Node) -> str:
    """Return the unquoted string content of a tree-sitter ``string`` node."""
    # Prefer the string_fragment child which already lacks quotes
    for child in node.children:
        if child.type == "string_fragment":
            return _node_text(child)
    # Fallback: strip outer quote characters
    return _node_text(node).strip("'\"`")


def _is_local_import(source: str) -> bool:
    """Return True if *source* is a relative path (starts with ``./`` or ``../``)."""
    return source.startswith("./") or source.startswith("../")


class _ParseState:
    """Mutable accumulator threaded through the recursive walk."""

    def __init__(self, file_path: str, module_qn: str) -> None:
        self.file_path = file_path
        self.module_qn = module_qn
        self.entities: list[ParsedEntity] = []
        self.relationships: list[ParsedRelationship] = []
        self.unresolved: list[UnresolvedReference] = []

    def add_entity(self, entity: ParsedEntity) -> None:
        self.entities.append(entity)

    def add_rel(self, rel: ParsedRelationship) -> None:
        self.relationships.append(rel)

    def add_unresolved(self, u: UnresolvedReference) -> None:
        self.unresolved.append(u)


# ── Node handlers ──────────────────────────────────────────────────────────────


def _handle_import(node: Node, state: _ParseState) -> None:
    """Emit an IMPORTS relationship or UnresolvedReference for an import_statement."""
    source_node = node.child_by_field_name("source")
    if source_node is None:
        return
    source_str = _string_value(source_node)
    line = node.start_point[0] + 1

    if _is_local_import(source_str):
        state.add_rel(
            ParsedRelationship(
                source_qualified_name=state.module_qn,
                target_qualified_name=source_str,
                relationship="IMPORTS",
                confidence=1.0,
                resolution=EdgeConfidence.DETERMINISTIC,
            )
        )
    else:
        state.add_unresolved(
            UnresolvedReference(
                source_file=state.file_path,
                source_line=line,
                reference_string=source_str,
                context_snippet=_node_text(node)[:200],
                likely_kind="import",
            )
        )


def _has_async_child(node: Node) -> bool:
    """Return True if *node* has a direct child with type ``async``."""
    return any(child.type == "async" for child in node.children)


def _handle_function(
    node: Node,
    state: _ParseState,
    scope_qn: str,
    *,
    name: str | None = None,
    is_exported: bool = False,
) -> None:
    """Extract a function entity from a function_declaration or arrow_function node."""
    if name is None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(name_node)

    qn = f"{scope_qn}.{name}" if scope_qn else name
    is_async = _has_async_child(node)

    props: dict[str, Any] = {
        "is_async": is_async,
        "is_exported": is_exported,
    }

    state.add_entity(
        ParsedEntity(
            kind="function",
            name=name,
            qualified_name=qn,
            file_path=state.file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            properties=props,
        )
    )
    state.add_rel(
        ParsedRelationship(
            source_qualified_name=scope_qn,
            target_qualified_name=qn,
            relationship="DEFINES",
            resolution=EdgeConfidence.DETERMINISTIC,
        )
    )


def _handle_class(
    node: Node,
    state: _ParseState,
    scope_qn: str,
    *,
    is_exported: bool = False,
) -> None:
    """Extract a class entity from a class_declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node)
    qn = f"{scope_qn}.{name}" if scope_qn else name

    # Extract base classes from extends clause
    base_classes: list[str] = []
    for child in node.children:
        if child.type == "class_heritage":
            for heritage_child in child.children:
                if heritage_child.type == "extends_clause":
                    for extends_child in heritage_child.children:
                        if extends_child.type not in ("extends",) and extends_child.is_named:
                            text = _node_text(extends_child).strip()
                            if text:
                                base_classes.append(text)

    state.add_entity(
        ParsedEntity(
            kind="class",
            name=name,
            qualified_name=qn,
            file_path=state.file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            properties={
                "base_classes": base_classes,
                "is_exported": is_exported,
            },
        )
    )
    state.add_rel(
        ParsedRelationship(
            source_qualified_name=scope_qn,
            target_qualified_name=qn,
            relationship="DEFINES",
            resolution=EdgeConfidence.DETERMINISTIC,
        )
    )


def _handle_variable_declaration(
    node: Node,
    state: _ParseState,
    scope_qn: str,
    *,
    is_exported: bool = False,
) -> None:
    """Handle const/let/var declarations that may contain arrow functions."""
    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if (
                name_node is not None
                and value_node is not None
                and value_node.type == "arrow_function"
            ):
                _handle_function(
                    value_node,
                    state,
                    scope_qn,
                    name=_node_text(name_node),
                    is_exported=is_exported,
                )


def _walk_node(node: Node, state: _ParseState, scope_qn: str) -> None:
    """Dispatch on node type; recurse into compound statements."""
    match node.type:
        case "import_statement":
            _handle_import(node, state)

        case "function_declaration":
            _handle_function(node, state, scope_qn)

        case "class_declaration":
            _handle_class(node, state, scope_qn)

        case "lexical_declaration" | "variable_declaration":
            _handle_variable_declaration(node, state, scope_qn)

        case "export_statement":
            # Recurse into the exported declaration with is_exported=True
            for child in node.children:
                match child.type:
                    case "function_declaration":
                        _handle_function(child, state, scope_qn, is_exported=True)
                    case "class_declaration":
                        _handle_class(child, state, scope_qn, is_exported=True)
                    case "lexical_declaration" | "variable_declaration":
                        _handle_variable_declaration(child, state, scope_qn, is_exported=True)
                    case _:
                        pass

        case _:
            # Recurse into compound statements (if/while/etc.)
            for child in node.children:
                _walk_node(child, state, scope_qn)


# ── Public parser class ────────────────────────────────────────────────────────


class TypeScriptParser:
    """Concrete ``CodeParser`` for ``.ts``, ``.tsx``, ``.js``, and ``.jsx`` files.

    Uses ``tree-sitter-typescript`` to extract functions, classes, imports, and
    relationships from TypeScript and JavaScript source files.

    ``tsx`` grammar is used for ``.tsx`` and ``.jsx``; ``typescript`` grammar for
    ``.ts`` and ``.js``.

    ``.d.ts`` declaration files are skipped — only a module entity is returned.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".ts", ".tsx", ".js", ".jsx"})

    async def parse_file(self, file_path: str, content: bytes) -> ParseResult:
        """Parse *content* and return structured entities and relationships.

        If the file contains syntax errors, a minimal :class:`ParseResult` with
        only the module entity is returned (no exception is raised).
        """
        posix_path = PurePosixPath(file_path.replace("\\", "/"))
        ext = posix_path.suffix
        module_qn = _file_path_to_module(file_path)
        lang_name = "typescript"

        module_entity = ParsedEntity(
            kind="module",
            name=posix_path.stem,
            qualified_name=module_qn,
            file_path=file_path,
            line_start=1,
            line_end=max(1, content.count(b"\n") + 1),
            properties={},
        )

        # Skip .d.ts declaration files
        if file_path.endswith(".d.ts"):
            return ParseResult(
                file_path=file_path,
                language=lang_name,
                entities=[module_entity],
            )

        language = _language_for_extension(ext)
        parser = Parser(language)

        try:
            tree = parser.parse(content)
        except Exception:
            log.warning(
                "tree_sitter_parse_failed",
                file_path=file_path,
                reason="parser raised exception",
            )
            return ParseResult(
                file_path=file_path,
                language=lang_name,
                entities=[module_entity],
            )

        if tree.root_node.has_error:
            log.warning(
                "tree_sitter_syntax_error",
                file_path=file_path,
            )
            return ParseResult(
                file_path=file_path,
                language=lang_name,
                entities=[module_entity],
            )

        state = _ParseState(file_path, module_qn)
        state.add_entity(module_entity)

        for child in tree.root_node.children:
            _walk_node(child, state, module_qn)

        return ParseResult(
            file_path=file_path,
            language=lang_name,
            entities=state.entities,
            relationships=state.relationships,
            unresolved_references=state.unresolved,
        )
