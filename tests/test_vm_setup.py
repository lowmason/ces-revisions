"""infra/vm/setup.sh sets up the VM user's environment, and a rerun changes nothing."""

import shutil
import subprocess
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / "infra" / "vm" / "setup.sh"
BRANCH = "test/cloud-gpu-environment"
LOG = r"""#!/bin/sh
printf '%s\n' "$(basename "$0") $*" >> "$CALL_LOG"
"""
# git clone [--branch NAME] URL DEST: create the checkout and the files setup.sh reads.
GIT = (
    LOG
    + r"""for dest; do :; done
mkdir -p "$dest/.git"
case "$dest" in
  */ces-revisions) printf '[tool.uv]\nrequired-version = "~=0.12.13"\n' > "$dest/pyproject.toml" ;;
  */agent-skills)
    mkdir -p "$dest/skills/writing-plans" "$dest/agents" "$dest/commands" "$dest/hooks"
    touch "$dest/agents/code-reviewer.md" "$dest/hooks/readonly-agent-guard.py"
    ;;
esac
"""
)
# curl prints an installer script that puts a logging uv in ~/.local/bin.
CURL = (
    LOG
    + r"""cat <<'SCRIPT'
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/uv" <<'UV'
#!/bin/sh
printf '%s\n' "uv $* (in $PWD)" >> "$CALL_LOG"
if [ "$1" = --version ]; then echo "uv 0.12.13"; fi
UV
chmod +x "$HOME/.local/bin/uv"
SCRIPT
"""
)


def _stub(bin_dir, name, body):
    (bin_dir / name).write_text(body)
    (bin_dir / name).chmod(0o755)


@pytest.fixture
def setup(tmp_path):
    home = tmp_path / "home"
    links = home / ".config" / "ces-revisions" / "links"
    links.mkdir(parents=True)
    (links / "skills.txt").write_text("writing-plans\nretired-skill\n")
    (links / "agents.txt").write_text("code-reviewer.md\n")
    (links / "commands.txt").write_text("")
    (links / "hooks.txt").write_text("readonly-agent-guard.py\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "git", GIT)
    _stub(bin_dir, "curl", CURL)
    _stub(bin_dir, "sudo", LOG)
    log = tmp_path / "calls.log"

    def run():
        # A bare PATH keeps the real uv, git, and curl out of reach.
        environment = {
            "HOME": str(home),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "CALL_LOG": str(log),
            "CES_REVISIONS_BRANCH": BRANCH,
        }
        result = subprocess.run(
            ["bash", str(SETUP)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        calls = log.read_text().splitlines() if log.exists() else []
        log.unlink(missing_ok=True)
        return result, [call for call in calls if "--version" not in call]

    return home, bin_dir, run


def test_refuses_until_sync_config_has_sent_the_link_names(setup):
    home, _, run = setup
    shutil.rmtree(home / ".config")

    result, calls = run()

    assert result.returncode == 1
    assert "run infra/bin/vm sync-config" in result.stderr
    assert calls == []


def test_refuses_to_run_as_root(setup):
    _, bin_dir, run = setup
    _stub(bin_dir, "id", "#!/bin/sh\necho 0\n")

    result, calls = run()

    assert result.returncode == 1
    assert "not as root" in result.stderr
    assert calls == []


def test_the_first_run_clones_installs_syncs_links_and_installs_the_guards(
    setup, tmp_path
):
    home, _, run = setup
    projects = home / "Projects"
    skills = projects / "agent-skills"

    result, calls = run()

    assert result.returncode == 0, result.stderr
    assert calls == [
        f"git clone --branch {BRANCH} https://github.com/lowmason/ces-revisions.git {projects}/ces-revisions",
        f"git clone https://github.com/lowmason/agent-skills.git {skills}",
        "curl -LsSf https://astral.sh/uv/0.12.13/install.sh",
        f"uv python install 3.14 (in {tmp_path})",
        f"uv sync --locked --extra cuda (in {projects}/ces-revisions)",
        f"sudo {projects}/ces-revisions/infra/vm/guards/install.sh",
    ]
    claude = home / ".claude"
    assert (
        claude / "skills" / "writing-plans"
    ).readlink() == skills / "skills" / "writing-plans"
    assert (
        claude / "agents" / "code-reviewer.md"
    ).readlink() == skills / "agents" / "code-reviewer.md"
    assert (claude / "hooks" / "readonly-agent-guard.py").readlink() == (
        skills / "hooks" / "readonly-agent-guard.py"
    )
    assert list((claude / "commands").iterdir()) == []
    assert "skipping skills/retired-skill" in result.stderr


def test_a_rerun_skips_the_clones_and_the_uv_install(setup, tmp_path):
    home, _, run = setup
    projects = home / "Projects"
    skills = projects / "agent-skills"
    run()

    result, calls = run()

    assert result.returncode == 0, result.stderr
    assert calls == [
        f"uv python install 3.14 (in {tmp_path})",
        f"uv sync --locked --extra cuda (in {projects}/ces-revisions)",
        f"sudo {projects}/ces-revisions/infra/vm/guards/install.sh",
    ]
    link = home / ".claude" / "skills" / "writing-plans"
    assert link.readlink() == skills / "skills" / "writing-plans"
    # ln -n replaces the link rather than creating a second one inside its target.
    assert not (skills / "skills" / "writing-plans" / "writing-plans").exists()
