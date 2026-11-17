"""A stand-in for urllib's HTTPS layer that records each request's User-Agent, with no network.

It replaces only `HTTPSHandler.https_open`, the last step before a socket. Everything above it
runs for real, so a fetcher's redirect handling is exercised end to end: the first request, the
302 response, the redirect handler, and the request for the next hop.
"""

import email.message
import io
import urllib.request
import urllib.response

import pytest


def serve_https(
    monkeypatch: pytest.MonkeyPatch, redirects: dict[str, str] | None = None
) -> list[tuple[str, str | None]]:
    """Answer every HTTPS request, redirecting the URLs in `redirects` to their targets.

    Returns the list that fills with one `(url, user_agent)` pair for each request made.
    Any URL not in `redirects` gets an empty 200 response.
    """
    redirects = redirects or {}
    seen: list[tuple[str, str | None]] = []

    def https_open(self, request: urllib.request.Request):
        url = request.full_url
        seen.append((url, request.get_header("User-agent")))
        headers = email.message.Message()
        if url in redirects:
            headers["Location"] = redirects[url]
            code, message = 302, "Found"
        else:
            code, message = 200, "OK"
        response = urllib.response.addinfourl(io.BytesIO(b""), headers, url, code)
        response.msg = message
        return response

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", https_open)
    return seen
