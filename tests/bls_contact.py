"""The contact address BLS asks automated clients for, as the live tests need it."""

from pathlib import Path

import annual_sources
import pytest


def require_bls_contact(
    monkeypatch: pytest.MonkeyPatch, env_file: Path = annual_sources.PROJECT_ENV
) -> str:
    """Export the contact for one test, or fail before any request reaches BLS.

    BLS answers a request that names no contact with 403 Forbidden, which a canary
    would otherwise report as the change it watches for. The failure names the
    setting, never its value.
    """
    contact = annual_sources.load_contact_email(env_file)
    if contact is None:
        pytest.fail(
            f"BLS_CONTACT_EMAIL is neither exported nor set in {env_file}, and BLS "
            "refuses a request that names no contact. A worktree has no copy of "
            ".project.env: export the variable or copy the file from the main checkout.",
            pytrace=False,
        )
    monkeypatch.setenv("BLS_CONTACT_EMAIL", contact)
    return contact
