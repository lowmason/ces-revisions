"""Fetch Stage 4 sources, refresh their manifest, or build the annual tables."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from urllib.parse import urlsplit

import polars as pl
from dotenv import load_dotenv

from ces_revisions.annual import raw

QCEW_REVISIONS_URL = "https://www.bls.gov/cew/revisions/qcew-revisions.csv"
PROJECT_ENV = raw.ROOT / ".project.env"
USER_AGENT_PRODUCT = "ces-revisions/0.1.0"
ATTEMPTS = 5
PAUSE_SECONDS = 3.0


def load_contact_email(env_file: Path = PROJECT_ENV) -> str | None:
    """Load project-local configuration; an exported value keeps precedence."""
    load_dotenv(env_file)
    value = os.environ.get("BLS_CONTACT_EMAIL", "").strip()
    return value if "@" in value and not value.startswith("@") else None


def user_agent(contact_email: str) -> str:
    """Return the descriptive identity sent with every Stage 4 request."""
    return f"{USER_AGENT_PRODUCT} ({contact_email})"


def download(url: str, contact_email: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent(contact_email)}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def retrying_download(url: str, contact_email: str) -> tuple[bytes, str]:
    """Retry transient network failures with the archive's proven backoff policy."""
    for number in range(1, ATTEMPTS):
        try:
            return download(url, contact_email)
        except OSError:
            sleep(PAUSE_SECONDS * number)
    try:
        return download(url, contact_email)
    except OSError as error:
        raise RuntimeError(f"{url} failed after {ATTEMPTS} attempts") from error


def fetch_catalog_source(url: str, contact_email: str) -> tuple[bytes, str]:
    """Respect Internet Archive burst limits before making a bounded request."""
    if urlsplit(url).hostname == "web.archive.org":
        sleep(PAUSE_SECONDS)
    return retrying_download(url, contact_email)


def validate_payload(url: str, payload: bytes) -> None:
    """Reject common BLS error pages before any committed source is replaced."""
    if not payload:
        raise ValueError(f"{url} returned an empty payload")
    suffix = Path(urlsplit(url).path).suffix.lower()
    prefix = payload.lstrip()[:4096].lower()
    if suffix == ".pdf" and not payload.startswith(b"%PDF-"):
        raise ValueError(f"{url} does not have a PDF signature")
    if suffix in {".htm", ".html"} and b"<html" not in prefix:
        raise ValueError(f"{url} does not have an HTML signature")
    if suffix == ".csv" and b"," not in payload.splitlines()[0]:
        raise ValueError(f"{url} does not have a CSV header")


def refresh_manifest(now: datetime, raw_dir: Path = raw.RAW_DIR) -> int:
    stamp = (
        now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    frame = raw.manifest_frame(raw_dir, fetched_at=stamp)
    frame.write_csv(raw_dir / raw.MANIFEST)
    return frame.height


def fetch_sources(now: datetime, env_file: Path = PROJECT_ENV) -> int:
    contact_email = load_contact_email(env_file)
    if not contact_email:
        print(
            "set BLS_CONTACT_EMAIL to the contact address BLS asks automated clients for",
            file=sys.stderr,
        )
        return 1
    if shutil.which("pdftotext") is None:
        print("pdftotext (poppler) is required: brew install poppler", file=sys.stderr)
        return 1
    catalog = raw.read_source_catalog()
    with tempfile.TemporaryDirectory(prefix="ces-stage4-fetch-") as directory:
        staging = Path(directory)
        for row in catalog.filter(pl.col("status") == "available").iter_rows(
            named=True
        ):
            target = staging / row["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            payload, _ = fetch_catalog_source(row["url"], contact_email)
            validate_payload(row["url"], payload)
            target.write_bytes(payload)
            if target.suffix.lower() == ".pdf":
                text_target = target.with_suffix(".txt")
                subprocess.run(
                    ["pdftotext", "-layout", str(target), str(text_target)],
                    check=True,
                )
                if not text_target.read_text(encoding="utf-8").strip():
                    raise ValueError(f"pdftotext produced no text for {row['url']}")
        for staged in sorted(path for path in staging.rglob("*") if path.is_file()):
            target = raw.RAW_DIR / staged.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)
    count = refresh_manifest(now)
    print(f"manifest: {count} source files")
    return 0


def build_tables() -> int:
    from ces_revisions.annual.build import build, write

    manifest = write(build())
    for name, entry in manifest["artifacts"].items():
        print(f"{name}: {entry['rows']} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "manifest", "build"])
    command = parser.parse_args(argv).command
    now = datetime.now(UTC)
    if command == "fetch":
        return fetch_sources(now)
    if command == "manifest":
        print(f"manifest: {refresh_manifest(now)} source files")
        return 0
    return build_tables()


if __name__ == "__main__":
    raise SystemExit(main())
