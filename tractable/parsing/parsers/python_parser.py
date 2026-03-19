"""PythonParser — tree-sitter-based parser for Python source files.

Extracts functions, classes, modules, imports, and call relationships from
``.py`` and ``.pyi`` files and returns a :class:`ParseResult`.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

import structlog
import tree_sitter_python as _tsp
from tree_sitter import Language, Node, Parser

from tractable.protocols.graph_construction import (
    ParsedEntity,
    ParsedRelationship,
    ParseResult,
    UnresolvedReference,
)
from tractable.types.enums import EdgeConfidence

log = structlog.get_logger(__name__)

_PYTHON_LANGUAGE = Language(_tsp.language())


def _file_path_to_module(file_path: str) -> str:
    """Derive a dotted module name from a file path.

    Examples::

        tractable/parsing/parsers/python_parser.py → tractable.parsing.parsers.python_parser
        ./foo/bar.pyi → foo.bar
    """
    p = file_path.replace("\\", "/").lstrip("./")
    p = re.sub(r"\.pyi?$", "", p)
    return p.replace("/", ".")


def _node_text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _first_docstring(block: Node | None) -> str | None:
    """Return the first string literal in a block (docstring), if any."""
    if block is None:
        return None
    for child in block.children:
        if child.type == "expression_statement":
            inner = child.children[0] if child.children else None
            if inner and inner.type in ("string", "concatenated_string"):
                raw = _node_text(inner).strip("\"' \t\n")
                return raw.splitlines()[0][:200] if raw else None
        # Only the very first statement matters.
        break
    return None


class _ParseState:
    """Mutable accumulator threaded through the recursive walk."""

    def __init__(self, file_path: str, module_qn: str) -> None:
        self.file_path = file_path
        self.module_qn = module_qn
        self.entities: list[ParsedEntity] = []
        self.relationships: list[ParsedRelationship] = []
        self.unresolved: list[UnresolvedReference] = []
        # qualified names defined at top-level or within classes
        self._defined_names: set[str] = set()

    def add_entity(self, entity: ParsedEntity) -> None:
        self.entities.append(entity)
        self._defined_names.add(entity.qualified_name)

    def is_locally_defined(self, qualified_name: str) -> bool:
        return qualified_name in self._defined_names

    def add_rel(self, rel: ParsedRelationship) -> None:
        self.relationships.append(rel)

    def add_unresolved(self, u: UnresolvedReference) -> None:
        self.unresolved.append(u)


def _handle_import(node: Node, state: _ParseState, source_lines: list[bytes]) -> None:
    """Emit IMPORTS relationships or UnresolvedReferences for import nodes."""
    if node.type == "import_statement":
        # import foo, bar as baz
        for child in node.children:
            if child.type in ("dotted_name", "aliased_import"):
                name_node = (
                    child.child_by_field_name("name") or child
                    if child.type == "aliased_import"
                    else child
                )
                import_name = _node_text(name_node)
                state.add_rel(
                    ParsedRelationship(
                        source_qualified_name=state.module_qn,
                        target_qualified_name=import_name,
                        relationship="IMPORTS",
                        confidence=1.0,
                        resolution=EdgeConfidence.DETERMINISTIC,
                    )
                )
    elif node.type == "import_from_statement":
        # from module import name1, name2
        mod_node = node.child_by_field_name("module_name")
        mod_name = _node_text(mod_node) if mod_node else ""
        snippet = _node_text(node)
        line = node.start_point[0] + 1
        # Relative imports → always UnresolvedReference
        is_relative = mod_name.startswith(".") or mod_name == ""
        if is_relative:
            state.add_unresolved(
                UnresolvedReference(
                    source_file=state.file_path,
                    source_line=line,
                    reference_string=mod_name or ".",
                    context_snippet=snippet,
                    likely_kind="import",
                )
            )
        else:
            state.add_rel(
                ParsedRelationship(
                    source_qualified_name=state.module_qn,
                    target_qualified_name=mod_name,
                    relationship="IMPORTS",
                    confidence=1.0,
                    resolution=EdgeConfidence.DETERMINISTIC,
                )
            )


def _collect_decorators(decorated_node: Node) -> list[str]:
    """Collect decorator names from a decorated_definition node."""
    decorators: list[str] = []
    for child in decorated_node.children:
        if child.type == "decorator":
            # @name or @name(args)
            for sub in child.children:
                if sub.type not in ("@", "newline"):
                    decorators.append(_node_text(sub).strip())
                    break
    return decorators


def _walk_body(
    body: Node | None,
    state: _ParseState,
    scope_qn: str,
    *,
    source_lines: list[bytes],
) -> None:
    """Recursively walk a block/body node, extracting definitions and calls."""
    if body is None:
        return
    for child in body.children:
        _walk_node(child, state, scope_qn, source_lines=source_lines)


def _walk_node(
    node: Node,
    state: _ParseState,
    scope_qn: str,
    *,
    source_lines: list[bytes],
) -> None:
    """Dispatch on node type."""
    match node.type:
        case "import_statement" | "import_from_statement":
            _handle_import(node, state, source_lines)

        case "function_definition":
            _handle_function(node, state, scope_qn, [], source_lines=source_lines)

        case "class_definition":
            _handle_class(node, state, scope_qn, [], source_lines=source_lines)

        case "decorated_definition":
            decorators = _collect_decorators(node)
            inner = node.children[-1]  # last child is the actual definition
            if inner.type == "function_definition":
                _handle_function(inner, state, scope_qn, decorators, source_lines=source_lines)
            elif inner.type == "class_definition":
                _handle_class(inner, state, scope_qn, decorators, source_lines=source_lines)

        case "call":
            _handle_call(node, state, scope_qn, source_lines=source_lines)

        case _:
            # Recurse into compound statements
            for child in node.children:
                _walk_node(child, state, scope_qn, source_lines=source_lines)


def _handle_function(
    node: Node,
    state: _ParseState,
    scope_qn: str,
    decorators: list[str],
    *,
    source_lines: list[bytes],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node)
    qn = f"{scope_qn}.{name}" if scope_qn else name
    is_async = node.children[0].type == "async" if node.children else False
    is_method = "." in scope_qn and scope_qn != state.module_qn

    body = node.child_by_field_name("body")
    docstring = _first_docstring(body)

    props: dict[str, Any] = {
        "is_async": is_async,
        "is_method": is_method,
        "decorators": decorators,
    }
    if docstring is not None:
        props["docstring_first_line"] = docstring

    entity = ParsedEntity(
        kind="function",
        name=name,
        qualified_name=qn,
        file_path=state.file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        properties=props,
    )
    state.add_entity(entity)

    # DEFINES relationship: enclosing scope → function
    state.add_rel(
        ParsedRelationship(
            source_qualified_name=scope_qn,
            target_qualified_name=qn,
            relationship="DEFINES",
            resolution=EdgeConfidence.DETERMINISTIC,
        )
    )

    # Recurse into function body
    if body is not None:
        _walk_body(body, state, qn, source_lines=source_lines)


def _handle_class(
    node: Node,
    state: _ParseState,
    scope_qn: str,
    decorators: list[str],
    *,
    source_lines: list[bytes],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node)
    qn = f"{scope_qn}.{name}" if scope_qn else name

    # Base classes
    bases_node = node.child_by_field_name("superclasses")
    base_classes: list[str] = []
    if bases_node is not None:
        for child in bases_node.children:
            if child.type not in ("(", ")", ","):
                base_classes.append(_node_text(child))

    body = node.child_by_field_name("body")
    docstring = _first_docstring(body)

    props: dict[str, Any] = {
        "base_classes": base_classes,
        "decorators": decorators,
    }
    if docstring is not None:
        props["docstring_first_line"] = docstring

    entity = ParsedEntity(
        kind="class",
        name=name,
        qualified_name=qn,
        file_path=state.file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        properties=props,
    )
    state.add_entity(entity)

    # DEFINES relationship
    state.add_rel(
        ParsedRelationship(
            source_qualified_name=scope_qn,
            target_qualified_name=qn,
            relationship="DEFINES",
            resolution=EdgeConfidence.DETERMINISTIC,
        )
    )

    # Recurse into class body (methods, nested classes)
    if body is not None:
        _walk_body(body, state, qn, source_lines=source_lines)


def _handle_call(
    node: Node,
    state: _ParseState,
    scope_qn: str,
    *,
    source_lines: list[bytes],
) -> None:
    func_node = node.child_by_field_name("function")
    if func_node is None:
        return
    callee_text = _node_text(func_node)
    line = node.start_point[0] + 1

    # Simple name call: possibly within-file
    if func_node.type == "identifier":
        target_qn = f"{state.module_qn}.{callee_text}"
        if state.is_locally_defined(target_qn):
            state.add_rel(
                ParsedRelationship(
                    source_qualified_name=scope_qn,
                    target_qualified_name=target_qn,
                    relationship="CALLS",
                    resolution=EdgeConfidence.DETERMINISTIC,
                )
            )
        else:
            # Not found locally yet (may be defined later or in another module)
            snippet = _node_text(node)[:200]
            state.add_unresolved(
                UnresolvedReference(
                    source_file=state.file_path,
                    source_line=line,
                    reference_string=callee_text,
                    context_snippet=snippet,
                    likely_kind="function_call",
                )
            )
    else:
        # Attribute access or other complex call → unresolved
        snippet = _node_text(node)[:200]
        state.add_unresolved(
            UnresolvedReference(
                source_file=state.file_path,
                source_line=line,
                reference_string=callee_text,
                context_snippet=snippet,
                likely_kind="function_call",
            )
        )


class PythonParser:
    """Concrete ``CodeParser`` for ``.py`` and ``.pyi`` files.

    Uses ``tree-sitter-python`` to extract functions, classes, imports, and
    call relationships from Python source files.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".py", ".pyi"})

    async def parse_file(self, file_path: str, content: bytes) -> ParseResult:
        """Parse *content* and return structured entities and relationships.

        If the file contains syntax errors, a minimal :class:`ParseResult` with
        only the module entity is returned (no exception is raised).
        """
        module_qn = _file_path_to_module(file_path)

        # Module entity — always present
        module_entity = ParsedEntity(
            kind="module",
            name=PurePosixPath(file_path.replace("\\", "/")).stem,
            qualified_name=module_qn,
            file_path=file_path,
            line_start=1,
            line_end=max(1, content.count(b"\n") + 1),
            properties={},
        )

        parser = Parser(_PYTHON_LANGUAGE)
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
                language="python",
                entities=[module_entity],
            )

        if tree.root_node.has_error:
            log.warning(
                "tree_sitter_syntax_error",
                file_path=file_path,
            )
            return ParseResult(
                file_path=file_path,
                language="python",
                entities=[module_entity],
            )

        source_lines = content.split(b"\n")
        state = _ParseState(file_path, module_qn)
        state.add_entity(module_entity)

        # Walk top-level nodes
        for child in tree.root_node.children:
            _walk_node(child, state, module_qn, source_lines=source_lines)

        return ParseResult(
            file_path=file_path,
            language="python",
            entities=state.entities,
            relationships=state.relationships,
            unresolved_references=state.unresolved,
        )
