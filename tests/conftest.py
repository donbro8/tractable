"""Root pytest configuration."""


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Exit 0 when no tests are collected (empty scaffold is acceptable)."""
    if exitstatus == 5:
        session.exitstatus = 0  # type: ignore[union-attr]
