"""HCLParser — tree-sitter-based parser for HashiCorp Configuration Language.

Extracts Terraform entities (modules, resources, variables, outputs, data
sources) from ``.tf`` and ``.tfvars`` files and returns a :class:`ParseResult`.

Source: tech-spec.py §2.4, PLAN.md Phase 3 Week 12.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

import structlog
import tree_sitter_hcl as _tsh
from tree_sitter import Language, Node, Parser

from tractable.protocols.graph_construction import (
    ParsedEntity,
    ParsedRelationship,
    ParseResult,
    UnresolvedReference,
)
from tractable.types.enums import EdgeConfidence

log = structlog.get_logger(__name__)

_HCL_LANGUAGE = Language(_tsh.language())

# ── AST helpers ───────────────────────────────────────────────────────────────


def _node_text(node: Node) -> str:
    """Decode a node's raw bytes as UTF-8."""
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _string_lit_value(node: Node) -> str:
    """Extract the inner text from a ``string_lit`` node (strips quotes)."""
    for child in node.children:
        if child.type == "template_literal":
            return _node_text(child)
    return _node_text(node).strip('"')


def _attr_string_value(body: Node, attr_name: str) -> str | None:
    """Return the string value of *attr_name* inside *body*, or None."""
    for child in body.children:
        if child.type == "attribute":
            key_node = child.child_by_field_name("name") or _first_child_of_type(
                child, "identifier"
            )
            if key_node is None:
                continue
            if _node_text(key_node) != attr_name:
                continue
            val_node = child.child_by_field_name("val") or _first_child_of_type(
                child, "expression"
            )
            if val_node is None:
                continue
            # Drill: expression → literal_value → string_lit
            lit = _first_child_of_type(val_node, "literal_value")
            if lit is not None:
                slit = _first_child_of_type(lit, "string_lit")
                if slit is not None:
                    return _string_lit_value(slit)
            # expression → variable_expr → identifier (bare type ref like `string`)
            vexpr = _first_child_of_type(val_node, "variable_expr")
            if vexpr is not None:
                ident = _first_child_of_type(vexpr, "identifier")
                if ident is not None:
                    return _node_text(ident)
    return None


def _first_child_of_type(node: Node, node_type: str) -> Node | None:
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def _block_labels(block: Node) -> list[str]:
    """Return all ``string_lit`` label values from a block header."""
    labels: list[str] = []
    for child in block.children:
        if child.type == "string_lit":
            labels.append(_string_lit_value(child))
    return labels


def _block_body(block: Node) -> Node | None:
    """Return the inner ``body`` node of a block, if present."""
    for child in block.children:
        if child.type == "body":
            return child
    return None


def _has_expression_ref(body: Node, attr_name: str) -> bool:
    """Return True if *attr_name* in *body* contains a reference (not a literal)."""
    for child in body.children:
        if child.type == "attribute":
            key_node = child.child_by_field_name("name") or _first_child_of_type(
                child, "identifier"
            )
            if key_node is None or _node_text(key_node) != attr_name:
                continue
            val_node = child.child_by_field_name("val") or _first_child_of_type(
                child, "expression"
            )
            if val_node is None:
                return False
            # If the expression contains anything other than a literal_value,
            # treat it as a reference expression.
            lit = _first_child_of_type(val_node, "literal_value")
            if lit is None:
                return True
    return False


def _top_level_attr_names(body: Node) -> list[str]:
    """Return the names of all top-level attributes in *body*."""
    names: list[str] = []
    for child in body.children:
        if child.type == "attribute":
            key_node = child.child_by_field_name("name") or _first_child_of_type(
                child, "identifier"
            )
            if key_node is not None:
                name = _node_text(key_node)
                if name != "depends_on":
                    names.append(name)
    return names


def _is_remote_source(source: str) -> bool:
    """Return True for non-local module sources (registry or URL-based)."""
    return not (source.startswith("./") or source.startswith("../"))


def _depends_on_refs(body: Node) -> list[str]:
    """Extract resource reference strings from a ``depends_on`` attribute."""
    refs: list[str] = []
    for child in body.children:
        if child.type == "attribute":
            key_node = child.child_by_field_name("name") or _first_child_of_type(
                child, "identifier"
            )
            if key_node is None or _node_text(key_node) != "depends_on":
                continue
            val_node = child.child_by_field_name("val") or _first_child_of_type(
                child, "expression"
            )
            if val_node is None:
                continue
            coll = _first_child_of_type(val_node, "collection_value")
            if coll is None:
                continue
            tup = _first_child_of_type(coll, "tuple")
            if tup is None:
                continue
            for expr_child in tup.children:
                if expr_child.type == "expression":
                    # Reconstruct "type.name" from variable_expr + get_attr
                    parts: list[str] = []
                    vexpr = _first_child_of_type(expr_child, "variable_expr")
                    if vexpr:
                        ident = _first_child_of_type(vexpr, "identifier")
                        if ident:
                            parts.append(_node_text(ident))
                    for ga in expr_child.children:
                        if ga.type == "get_attr":
                            attr_ident = _first_child_of_type(ga, "identifier")
                            if attr_ident:
                                parts.append(_node_text(attr_ident))
                    if parts:
                        refs.append(".".join(parts))
    return refs


# ── Parse-state accumulator ───────────────────────────────────────────────────


class _ParseState:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.entities: list[ParsedEntity] = []
        self.relationships: list[ParsedRelationship] = []
        self.unresolved: list[UnresolvedReference] = []
        # Maps "resource_type.resource_name" → qualified_name for DEPENDS_ON edges
        self.resource_qns: dict[str, str] = {}


# ── Block handlers ────────────────────────────────────────────────────────────


def _handle_module(block: Node, state: _ParseState) -> None:
    labels = _block_labels(block)
    if not labels:
        return
    module_name = labels[0]
    body = _block_body(block)
    qn = f"{state.file_path}::module.{module_name}"

    props: dict[str, Any] = {}
    if body is not None:
        source = _attr_string_value(body, "source")
        if source is not None:
            props["source"] = source
        version = _attr_string_value(body, "version")
        if version is not None:
            props["version"] = version

    entity = ParsedEntity(
        kind="terraform_module",
        name=module_name,
        qualified_name=qn,
        file_path=state.file_path,
        line_start=block.start_point[0] + 1,
        line_end=block.end_point[0] + 1,
        properties=props,
    )
    state.entities.append(entity)

    source_val = props.get("source", "")
    if isinstance(source_val, str):
        if _is_remote_source(source_val):
            state.unresolved.append(
                UnresolvedReference(
                    source_file=state.file_path,
                    source_line=block.start_point[0] + 1,
                    reference_string=source_val,
                    context_snippet=f'module "{module_name}" source = "{source_val}"',
                    likely_kind="terraform_remote_module",
                )
            )
        else:
            # Local source → DEPENDS_ON edge to the path as qualified name
            local_qn = source_val
            state.relationships.append(
                ParsedRelationship(
                    source_qualified_name=qn,
                    target_qualified_name=local_qn,
                    relationship="DEPENDS_ON",
                    confidence=1.0,
                    resolution=EdgeConfidence.DETERMINISTIC,
                )
            )


def _handle_resource(block: Node, state: _ParseState) -> None:
    labels = _block_labels(block)
    if len(labels) < 2:  # noqa: PLR2004
        return
    resource_type, resource_name = labels[0], labels[1]
    qn = f"{state.file_path}::resource.{resource_type}.{resource_name}"
    # Register for depends_on resolution
    state.resource_qns[f"{resource_type}.{resource_name}"] = qn

    body = _block_body(block)
    config_keys: list[str] = []
    if body is not None:
        config_keys = _top_level_attr_names(body)

    entity = ParsedEntity(
        kind="terraform_resource",
        name=f"{resource_type}.{resource_name}",
        qualified_name=qn,
        file_path=state.file_path,
        line_start=block.start_point[0] + 1,
        line_end=block.end_point[0] + 1,
        properties={"type": resource_type, "config_keys": config_keys},
    )
    state.entities.append(entity)

    # depends_on edges — store pending until after all resources are parsed
    if body is not None:
        for ref_str in _depends_on_refs(body):
            target_qn = f"{state.file_path}::resource.{ref_str}"
            state.relationships.append(
                ParsedRelationship(
                    source_qualified_name=qn,
                    target_qualified_name=target_qn,
                    relationship="DEPENDS_ON",
                    confidence=1.0,
                    resolution=EdgeConfidence.DETERMINISTIC,
                )
            )


def _handle_variable(block: Node, state: _ParseState) -> None:
    labels = _block_labels(block)
    if not labels:
        return
    var_name = labels[0]
    qn = f"{state.file_path}::variable.{var_name}"
    body = _block_body(block)

    props: dict[str, Any] = {}
    if body is not None:
        type_val = _attr_string_value(body, "type")
        if type_val is not None:
            props["type"] = type_val
        desc = _attr_string_value(body, "description")
        if desc is not None:
            props["description"] = desc
        # Include `default` only if it is a literal (not a reference expression)
        if not _has_expression_ref(body, "default"):
            default_val = _attr_string_value(body, "default")
            if default_val is not None:
                props["default"] = default_val

    entity = ParsedEntity(
        kind="terraform_variable",
        name=var_name,
        qualified_name=qn,
        file_path=state.file_path,
        line_start=block.start_point[0] + 1,
        line_end=block.end_point[0] + 1,
        properties=props,
    )
    state.entities.append(entity)


def _handle_output(block: Node, state: _ParseState) -> None:
    labels = _block_labels(block)
    if not labels:
        return
    output_name = labels[0]
    qn = f"{state.file_path}::output.{output_name}"
    body = _block_body(block)

    props: dict[str, Any] = {}
    if body is not None:
        desc = _attr_string_value(body, "description")
        if desc is not None:
            props["description"] = desc

    entity = ParsedEntity(
        kind="terraform_output",
        name=output_name,
        qualified_name=qn,
        file_path=state.file_path,
        line_start=block.start_point[0] + 1,
        line_end=block.end_point[0] + 1,
        properties=props,
    )
    state.entities.append(entity)


def _handle_data(block: Node, state: _ParseState) -> None:
    labels = _block_labels(block)
    if len(labels) < 2:  # noqa: PLR2004
        return
    data_type, data_name = labels[0], labels[1]
    qn = f"{state.file_path}::data.{data_type}.{data_name}"

    entity = ParsedEntity(
        kind="terraform_data",
        name=f"{data_type}.{data_name}",
        qualified_name=qn,
        file_path=state.file_path,
        line_start=block.start_point[0] + 1,
        line_end=block.end_point[0] + 1,
        properties={"type": data_type},
    )
    state.entities.append(entity)


# ── Parser class ──────────────────────────────────────────────────────────────


class HCLParser:
    """Concrete ``CodeParser`` for ``.tf`` and ``.tfvars`` files.

    Extracts Terraform entities (modules, resources, variables, outputs, data
    sources) and ``DEPENDS_ON`` relationships using ``tree-sitter-hcl``.

    - Never raises ``FatalError`` or ``RecoverableError`` from
      :meth:`parse_file`.
    - Malformed HCL returns an empty entities list and a populated ``errors``
      list (via :attr:`ParseResult.unresolved_references`).
    - ``.tfvars`` files produce no entity values (variable assignments are
      not extracted).
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".tf", ".tfvars"})

    def _parse_tfvars(self, file_path: str, content: bytes) -> ParseResult:
        """Parse a ``.tfvars`` file — extract variable names only, never values."""
        parser = Parser(_HCL_LANGUAGE)
        try:
            tree = parser.parse(content)
        except Exception:
            return ParseResult(file_path=file_path, language="hcl")

        entities: list[ParsedEntity] = []
        root_body = _first_child_of_type(tree.root_node, "body") or tree.root_node
        for child in root_body.children:
            if child.type != "attribute":
                continue
            key_node = child.child_by_field_name("name") or _first_child_of_type(
                child, "identifier"
            )
            if key_node is None:
                continue
            var_name = _node_text(key_node)
            # Only store the name — deliberately omit the assigned value.
            entities.append(
                ParsedEntity(
                    kind="terraform_variable",
                    name=var_name,
                    qualified_name=f"{file_path}::variable.{var_name}",
                    file_path=file_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    properties={},
                )
            )
        return ParseResult(file_path=file_path, language="hcl", entities=entities)

    async def parse_file(self, file_path: str, content: bytes) -> ParseResult:
        """Parse *content* and return structured HCL entities and relationships."""
        suffix = PurePosixPath(file_path.replace("\\", "/")).suffix

        # .tfvars files: extract variable names only — never their values.
        if suffix == ".tfvars":
            return self._parse_tfvars(file_path, content)

        parser = Parser(_HCL_LANGUAGE)
        try:
            tree = parser.parse(content)
        except Exception as exc:
            log.warning(
                "hcl_parser.parse_failed",
                file_path=file_path,
                error=str(exc),
            )
            return ParseResult(
                file_path=file_path,
                language="hcl",
                entities=[],
                unresolved_references=[
                    UnresolvedReference(
                        source_file=file_path,
                        source_line=1,
                        reference_string="<malformed>",
                        context_snippet=str(exc),
                        likely_kind="parse_error",
                    )
                ],
            )

        if tree.root_node.has_error:
            log.warning("hcl_parser.syntax_error", file_path=file_path)
            return ParseResult(
                file_path=file_path,
                language="hcl",
                entities=[],
                unresolved_references=[
                    UnresolvedReference(
                        source_file=file_path,
                        source_line=1,
                        reference_string="<syntax_error>",
                        context_snippet="tree-sitter reported a syntax error",
                        likely_kind="parse_error",
                    )
                ],
            )

        state = _ParseState(file_path)

        # Walk top-level body blocks
        root_body = _first_child_of_type(tree.root_node, "body")
        if root_body is None:
            root_body = tree.root_node

        for child in root_body.children:
            if child.type != "block":
                continue
            block_type_node = _first_child_of_type(child, "identifier")
            if block_type_node is None:
                continue
            block_type = _node_text(block_type_node)
            match block_type:
                case "module":
                    _handle_module(child, state)
                case "resource":
                    _handle_resource(child, state)
                case "variable":
                    _handle_variable(child, state)
                case "output":
                    _handle_output(child, state)
                case "data":
                    _handle_data(child, state)
                case _:
                    pass  # terraform, locals, provider, etc. — skip

        return ParseResult(
            file_path=file_path,
            language="hcl",
            entities=state.entities,
            relationships=state.relationships,
            unresolved_references=state.unresolved,
        )
