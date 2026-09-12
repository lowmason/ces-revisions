"""Smoke tests: the package imports, main() runs, and the console script is wired to it."""

from importlib.metadata import entry_points

import ces_revisions


def test_main_prints_greeting(capsys):
    ces_revisions.main()

    assert "ces-revisions" in capsys.readouterr().out


def test_console_script_resolves_to_main():
    (script,) = entry_points(group="console_scripts", name="ces-revisions")

    assert script.load() is ces_revisions.main
