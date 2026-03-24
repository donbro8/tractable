"""Unit tests for HCLParser.

Covers all acceptance criteria from TASK-3.2.1:
- Module extraction with local source → DEPENDS_ON edge
- Resource extraction: type + config_keys (no attribute values)
- Variable extraction: type + description (no default literal stored when reference)
- Output extraction: description
- DEPENDS_ON edge from resource with explicit depends_on
- Remote module source → UnresolvedReference (not ParsedRelationship)
- Malformed HCL returns empty entities without raising
- .tfvars parsing: variable names extracted, values never stored
"""

from __future__ import annotations

import pytest

from tractable.parsing.parsers.hcl_parser import HCLParser
from tractable.protocols.graph_construction import ParseResult


@pytest.fixture()
def parser() -> HCLParser:
    return HCLParser()


# ── Supported extensions ───────────────────────────────────────────────────────


def test_supported_extensions(parser: HCLParser) -> None:
    assert ".tf" in parser.supported_extensions
    assert ".tfvars" in parser.supported_extensions


# ── Module extraction ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_module_extraction_local_source(parser: HCLParser) -> None:
    """Local source module → entity + DEPENDS_ON relationship."""
    content = b"""
module "network" {
  source  = "./modules/network"
  version = "2.0.0"
}
"""
    result = await parser.parse_file("infra/main.tf", content)
    assert result.language == "hcl"

    modules = [e for e in result.entities if e.kind == "terraform_module"]
    assert len(modules) == 1
    mod = modules[0]
    assert mod.name == "network"
    assert mod.properties["source"] == "./modules/network"
    assert mod.properties["version"] == "2.0.0"

    # Local source → DEPENDS_ON relationship (not UnresolvedReference)
    deps = [r for r in result.relationships if r.relationship == "DEPENDS_ON"]
    assert len(deps) == 1
    assert deps[0].source_qualified_name == mod.qualified_name
    assert deps[0].target_qualified_name == "./modules/network"

    # No UnresolvedReferences for local module
    remote_refs = [
        u for u in result.unresolved_references if u.likely_kind == "terraform_remote_module"
    ]
    assert remote_refs == []


# ── Remote module source ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_module_source_produces_unresolved_reference(
    parser: HCLParser,
) -> None:
    """Remote/registry module source → UnresolvedReference, not ParsedRelationship."""
    content = b"""
module "consul" {
  source  = "hashicorp/consul/aws"
  version = "0.1.0"
}
"""
    result = await parser.parse_file("infra/main.tf", content)

    # Entity should still be created
    modules = [e for e in result.entities if e.kind == "terraform_module"]
    assert len(modules) == 1

    # No DEPENDS_ON relationship for remote source
    deps = [r for r in result.relationships if r.relationship == "DEPENDS_ON"]
    assert deps == []

    # But an UnresolvedReference with likely_kind="terraform_remote_module"
    remote_refs = [
        u for u in result.unresolved_references if u.likely_kind == "terraform_remote_module"
    ]
    assert len(remote_refs) == 1
    assert "hashicorp/consul/aws" in remote_refs[0].reference_string


# ── Resource extraction ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resource_extraction_type_and_config_keys(parser: HCLParser) -> None:
    """Resource entity has type + config_keys; no attribute values exposed."""
    content = b"""
resource "aws_instance" "web" {
  ami           = "ami-0abcdef1234567890"
  instance_type = "t3.micro"
  tags = {}
}
"""
    result = await parser.parse_file("infra/main.tf", content)

    resources = [e for e in result.entities if e.kind == "terraform_resource"]
    assert len(resources) == 1
    res = resources[0]
    assert res.properties["type"] == "aws_instance"
    # config_keys must list attribute names (not values)
    assert "ami" in res.properties["config_keys"]
    assert "instance_type" in res.properties["config_keys"]
    # Values must NOT appear in properties
    assert "ami-0abcdef1234567890" not in str(res.properties)
    assert "t3.micro" not in str(res.properties)


# ── Variable extraction ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_variable_extraction_type_and_description(parser: HCLParser) -> None:
    """Variable entity has type + description; literal default is present."""
    content = b"""
variable "environment" {
  type        = string
  description = "Deployment environment"
  default     = "dev"
}
"""
    result = await parser.parse_file("infra/main.tf", content)

    variables = [e for e in result.entities if e.kind == "terraform_variable"]
    assert len(variables) == 1
    var = variables[0]
    assert var.name == "environment"
    assert var.properties.get("type") == "string"
    assert var.properties.get("description") == "Deployment environment"


@pytest.mark.asyncio
async def test_variable_no_default_when_reference_expression(parser: HCLParser) -> None:
    """Variable with reference-expression default does not expose the default value."""
    content = b"""
variable "region" {
  type    = string
  default = var.some_other_var
}
"""
    result = await parser.parse_file("infra/main.tf", content)

    variables = [e for e in result.entities if e.kind == "terraform_variable"]
    assert len(variables) == 1
    var = variables[0]
    # Reference expression → `default` key must not appear in properties
    assert "default" not in var.properties


# ── Output extraction ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_extraction(parser: HCLParser) -> None:
    """Output entity is extracted with description."""
    content = b"""
output "vpc_id" {
  description = "The ID of the VPC"
  value       = aws_vpc.main.id
}
"""
    result = await parser.parse_file("infra/main.tf", content)

    outputs = [e for e in result.entities if e.kind == "terraform_output"]
    assert len(outputs) == 1
    out = outputs[0]
    assert out.name == "vpc_id"
    assert out.properties.get("description") == "The ID of the VPC"


# ── DEPENDS_ON between resources ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_depends_on_edge_between_resources(parser: HCLParser) -> None:
    """DEPENDS_ON edge from resource with explicit depends_on."""
    content = b"""
resource "aws_security_group" "sg" {
  name = "example-sg"
}

resource "aws_instance" "web" {
  ami           = "ami-0abcdef1234567890"
  instance_type = "t3.micro"
  depends_on    = [aws_security_group.sg]
}
"""
    result = await parser.parse_file("infra/main.tf", content)

    resources = {e.name: e for e in result.entities if e.kind == "terraform_resource"}
    assert "aws_security_group.sg" in resources
    assert "aws_instance.web" in resources

    web_qn = resources["aws_instance.web"].qualified_name
    sg_qn = resources["aws_security_group.sg"].qualified_name

    depends_on_rels = [
        r
        for r in result.relationships
        if r.relationship == "DEPENDS_ON" and r.source_qualified_name == web_qn
    ]
    assert len(depends_on_rels) == 1
    assert depends_on_rels[0].target_qualified_name == sg_qn


# ── Malformed HCL ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_malformed_hcl_returns_empty_entities_no_exception(
    parser: HCLParser,
) -> None:
    """Malformed HCL returns empty entities list; no exception is raised."""
    malformed = b"resource { this is not valid hcl !!! @@@"
    result: ParseResult = await parser.parse_file("infra/broken.tf", malformed)
    # Must not raise; entities must be empty
    assert isinstance(result, ParseResult)
    assert result.entities == []
    # errors are represented as unresolved references
    assert len(result.unresolved_references) >= 1


# ── .tfvars value safety ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tfvars_variable_name_extracted_no_value(parser: HCLParser) -> None:
    """Parsing a .tfvars file extracts variable names but never stores values."""
    content = b'secret_password = "s3cr3t_v4lu3_UNIQUE"\nenvironment = "prod"\n'
    result = await parser.parse_file("terraform.tfvars", content)

    # Entity names are extracted
    var_names = [e.name for e in result.entities if e.kind == "terraform_variable"]
    assert "secret_password" in var_names
    assert "environment" in var_names

    # The secret value must not appear anywhere in the ParseResult
    result_repr = repr(result)
    assert "s3cr3t_v4lu3_UNIQUE" not in result_repr, (
        "Secret value must never be stored in ParseResult"
    )

    # No properties should contain the value
    for entity in result.entities:
        for prop_val in entity.properties.values():
            assert "s3cr3t_v4lu3_UNIQUE" not in str(prop_val)
