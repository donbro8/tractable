"""Tests that all example registration files produce valid RepositoryRegistration models."""

import importlib.util
from pathlib import Path

import pytest

from tractable.types.config import RepositoryRegistration

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

EXAMPLE_FILES = [
    "register_python_api.py",
    "register_typescript_frontend.py",
    "register_terraform_infra.py",
]


def load_registration(filename: str) -> RepositoryRegistration:
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.registration  # type: ignore[no-any-return]


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_is_valid_registration(filename: str) -> None:
    registration = load_registration(filename)
    validated = RepositoryRegistration.model_validate(registration.model_dump())
    assert validated.name == registration.name
    assert validated.git_url == registration.git_url
    assert validated.primary_language == registration.primary_language


def test_python_api_example() -> None:
    reg = load_registration("register_python_api.py")
    assert reg.primary_language == "python"
    assert reg.agent_template == "api_maintainer"


def test_typescript_frontend_example() -> None:
    reg = load_registration("register_typescript_frontend.py")
    assert reg.primary_language == "typescript"
    assert reg.agent_template == "frontend_maintainer"
    assert reg.scope is not None
    assert "src/" in reg.scope.allowed_paths


def test_terraform_infra_example() -> None:
    from tractable.types.enums import AutonomyLevel

    reg = load_registration("register_terraform_infra.py")
    assert reg.primary_language == "hcl"
    assert reg.agent_template == "infra_maintainer"
    assert reg.autonomy_level == AutonomyLevel.SUPERVISED
