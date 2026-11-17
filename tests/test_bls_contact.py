"""The contact the live BLS tests need before any request leaves the machine."""

from pathlib import Path

import annual_sources
import archive_inventory
import pytest
import vintage_sources
from bls_contact import require_bls_contact
from fake_https import serve_https

CESVINALL_URL = "https://www.bls.gov/web/empsit/cesvinall.zip"
CONTACT = "someone@example.org"
WITH_CONTACT = f"ces-revisions/0.1.0 ({CONTACT})"
BARE = "ces-revisions/0.1.0"

# One call for each fetcher that opens a URL, so a test names none of them twice.
FETCHERS = {
    "archive_inventory.fetch": archive_inventory.fetch,
    "vintage_sources.download": vintage_sources.download,
    "annual_sources.download": lambda url: annual_sources.download(url, CONTACT),
}


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
    assert archive_inventory.user_agent(CESVINALL_URL) == expected
    assert vintage_sources.user_agent_for(CESVINALL_URL) == expected


@pytest.mark.parametrize("fetcher", FETCHERS.values(), ids=FETCHERS.keys())
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("https://www.example.com/moved", BARE),
        (
            "https://web.archive.org/web/20140719155542id_/http://www.bls.gov/moved",
            BARE,
        ),
        ("https://download.bls.gov/moved", WITH_CONTACT),
    ],
)
def test_every_fetcher_recomputes_the_contact_when_bls_redirects(
    monkeypatch, fetcher, target, expected
):
    # The environment supplies the contact to two fetchers and the argument to the third;
    # both name the same address, so the expected headers hold for each.
    monkeypatch.setenv("BLS_CONTACT_EMAIL", CONTACT)
    source = "https://www.bls.gov/moved-from"
    seen = serve_https(monkeypatch, {source: target})
    fetcher(source)
    assert seen == [(source, WITH_CONTACT), (target, expected)]
