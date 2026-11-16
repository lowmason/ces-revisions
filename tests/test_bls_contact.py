"""The contact the live BLS tests need before any request leaves the machine."""

from pathlib import Path

import archive_inventory
import pytest
import vintage_sources
from bls_contact import require_bls_contact

CESVINALL_URL = "https://www.bls.gov/web/empsit/cesvinall.zip"


def test_a_live_test_fails_before_reaching_bls_without_a_contact(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("BLS_CONTACT_EMAIL", raising=False)
    with pytest.raises(pytest.fail.Exception, match="BLS_CONTACT_EMAIL") as failure:
        require_bls_contact(monkeypatch, tmp_path / ".project.env")
    assert "@" not in str(failure.value)


def test_a_live_test_sends_the_project_contact_to_bls(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("BLS_CONTACT_EMAIL", raising=False)
    env_file = tmp_path / ".project.env"
    env_file.write_text("BLS_CONTACT_EMAIL=someone@example.org\n", encoding="utf-8")
    assert require_bls_contact(monkeypatch, env_file) == "someone@example.org"
    expected = "ces-revisions/0.1.0 (someone@example.org)"
    assert archive_inventory.user_agent() == expected
    assert vintage_sources.user_agent_for(CESVINALL_URL) == expected
