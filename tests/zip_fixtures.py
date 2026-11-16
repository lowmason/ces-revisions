"""ZIP builders shared by the archive inventory and file store tests."""

import io
import zipfile


def make_zip(members: dict[str, bytes], date_time=(2026, 9, 1, 15, 33, 0)) -> bytes:
    return make_dated_zip(
        [(name, date_time, payload) for name, payload in members.items()]
    )


def make_dated_zip(members: list[tuple[str, tuple[int, ...], bytes]]) -> bytes:
    """A ZIP whose members, written in the order given, each carry their own date."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, date_time, payload in members:
            archive.writestr(zipfile.ZipInfo(name, date_time=date_time), payload)
    return buffer.getvalue()
