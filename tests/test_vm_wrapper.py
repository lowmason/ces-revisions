"""infra/bin/vm prints each command before running it, and refuses to run on the VM."""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VM = REPO / "infra" / "bin" / "vm"
INSTANCE = "--region us-east-1 --instance-ids i-0123456789abcdef0"
TARGET = "--region us-east-1 --target i-0123456789abcdef0"
VM_MEMORY = ".claude/projects/-home-ubuntu-Projects-ces-revisions/memory/"
# Stands in for aws, tofu, ssh, and rsync. It logs each call the way infra/bin/vm prints
# it, answers the two tofu output queries, copies rsync's local source directories aside,
# and fails where a test asks it to.
STUB = r"""#!/bin/sh
tool=$(basename "$0")
if [ "$tool" = tofu ]; then
  printf 'env AWS_PROFILE=%s %s\n' "$AWS_PROFILE" "$tool $*" >> "$CALL_LOG"
else
  printf '%s\n' "$tool $*" >> "$CALL_LOG"
fi
case "$tool $*" in
  *"output -raw instance_id") echo i-0123456789abcdef0 ;;
  *"output -raw region") echo us-east-1 ;;
  "tofu "*" apply "*) [ -z "$STUB_FAIL_APPLY" ] || exit 1 ;;
  "ssh "*" gh auth setup-git") [ -z "$STUB_GH_SIGNED_OUT" ] || exit 1 ;;
  "ssh "*" test -e "*) [ -n "$STUB_REMOTE_MEMORY" ] || exit 1 ;;
  "rsync "*)
    for arg in "$@"; do
      case "$arg" in
        *:*) ;;
        */) [ ! -d "$arg" ] || cp -R "$arg." "$STUB_CAPTURE" ;;
      esac
    done
    ;;
esac
exit 0
"""


@pytest.fixture
def vm(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("aws", "tofu", "ssh", "rsync"):
        (bin_dir / tool).write_text(STUB)
        (bin_dir / tool).chmod(0o755)
    (tmp_path / "env").mkdir()
    (tmp_path / "capture").mkdir()
    log = tmp_path / "calls.log"
    base = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "CALL_LOG": str(log),
        "STUB_CAPTURE": str(tmp_path / "capture"),
        "CES_INFRA_ENV_DIR": str(tmp_path / "env"),
        "CES_VM_MARKER": str(tmp_path / "not-the-vm"),
    }

    def run(*args, **env):
        result = subprocess.run(
            [VM, *args],
            env={**base, **env},
            capture_output=True,
            text=True,
            check=False,
        )
        calls = log.read_text().splitlines() if log.exists() else []
        log.unlink(missing_ok=True)
        return result, calls

    return run


def _printed(result):
    return [line[2:] for line in result.stderr.splitlines() if line.startswith("+ ")]


def _home(tmp_path):
    """A home whose ~/.claude links point into ~/Projects/agent-skills, plus two strays."""
    home = tmp_path / "home"
    claude = home / ".claude"
    for relative in (
        "skills/writing-plans",
        "agents/code-reviewer.md",
        "hooks/guard.py",
    ):
        target = home / "Projects" / "agent-skills" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        link = claude / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    (claude / "commands").mkdir()
    (claude / "skills" / "elsewhere").symlink_to(tmp_path / "outside")
    (claude / "hooks" / "local.sh").write_text("#!/bin/sh\n")
    (claude / "CLAUDE.md").write_text("# global\n")
    (claude / "settings.json").write_text("{}\n")
    (home / ".gitconfig").write_text("[init]\n")
    return home


def _mac_memory(home):
    common = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    key = re.sub(r"[^A-Za-z0-9-]", "-", str(Path(common).parent))
    memory = home / ".claude" / "projects" / key / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("# Memory index\n")
    return memory


def test_refuses_to_run_on_the_vm(vm, tmp_path):
    marker = tmp_path / "marker"
    marker.touch()

    result, calls = vm("status", CES_VM_MARKER=str(marker))

    assert result.returncode == 1
    assert "never on the VM" in result.stderr
    assert calls == []


def test_an_unknown_command_prints_the_usage(vm):
    result, calls = vm("reboot")

    assert result.returncode == 2
    assert "usage: infra/bin/vm" in result.stderr
    assert calls == []


@pytest.mark.parametrize(
    ("subcommand", "expected"),
    [
        ("start", f"aws ec2 start-instances --profile ces-revisions {INSTANCE}"),
        ("stop", f"aws ec2 stop-instances --profile ces-revisions {INSTANCE}"),
        ("status", f"aws ec2 describe-instances --profile ces-revisions {INSTANCE}"),
        ("connect", f"aws ssm start-session --profile ces-revisions {TARGET}"),
        (
            "forward",
            f"aws ssm start-session --profile ces-revisions {TARGET} --document-name AWS-StartPortForwardingSession",
        ),
    ],
)
def test_instance_commands_print_each_command_before_running_it(
    vm, tmp_path, subcommand, expected
):
    result, calls = vm(subcommand)

    assert result.returncode == 0, result.stderr
    assert _printed(result) == calls
    tofu = f"env AWS_PROFILE=ces-revisions tofu -chdir={tmp_path / 'env'} output -raw"
    assert calls[:2] == [f"{tofu} instance_id", f"{tofu} region"]
    assert calls[2].startswith(expected)


def test_profiles_can_be_overridden(vm):
    result, calls = vm(
        "stop", CES_AWS_PROFILE="other", CES_TOFU_PROFILE="ces-revisions-process"
    )

    assert result.returncode == 0, result.stderr
    assert calls[0].startswith("env AWS_PROFILE=ces-revisions-process tofu ")
    assert calls[2].startswith("aws ec2 stop-instances --profile other ")


def test_size_rejects_an_unknown_size(vm, tmp_path):
    result, calls = vm("size", "a100")

    assert result.returncode == 2
    assert calls == []
    assert not (tmp_path / "env" / "size.auto.tfvars").exists()


@pytest.mark.parametrize("size", ["l4", "l40s", "a10g"])
def test_size_applies_and_then_records_the_size(vm, tmp_path, size):
    env_dir = tmp_path / "env"

    result, calls = vm("size", size, "-auto-approve")

    assert result.returncode == 0, result.stderr
    assert calls == [
        f"env AWS_PROFILE=ces-revisions tofu -chdir={env_dir} apply -var size={size} -auto-approve"
    ]
    assert _printed(result) == calls
    assert (env_dir / "size.auto.tfvars").read_text() == f'size = "{size}"\n'


def test_a_failed_apply_keeps_the_recorded_size(vm, tmp_path):
    recorded = tmp_path / "env" / "size.auto.tfvars"
    recorded.write_text('size = "dev"\n')

    result, _ = vm("size", "h100", STUB_FAIL_APPLY="1")

    assert result.returncode != 0
    assert recorded.read_text() == 'size = "dev"\n'


def test_sync_config_copies_settings_and_sends_link_names(vm, tmp_path):
    home = _home(tmp_path)

    result, calls = vm("sync-config")

    assert result.returncode == 0, result.stderr
    assert _printed(result) == calls
    assert calls[:4] == [
        "ssh ces-revisions-vm mkdir -p .claude .config/ces-revisions/links",
        f"rsync -a {home}/.claude/CLAUDE.md {home}/.claude/settings.json ces-revisions-vm:.claude/",
        f"rsync -a {home}/.gitconfig ces-revisions-vm:",
        "ssh ces-revisions-vm gh auth setup-git",
    ]
    assert calls[4].endswith(" ces-revisions-vm:.config/ces-revisions/links/")
    assert len(calls) == 5
    capture = tmp_path / "capture"
    assert (capture / "skills.txt").read_text() == "writing-plans\n"
    assert (capture / "agents.txt").read_text() == "code-reviewer.md\n"
    assert (capture / "commands.txt").read_text() == ""
    assert (capture / "hooks.txt").read_text() == "guard.py\n"
    assert "skipping skills/elsewhere" in result.stderr
    assert "skipping hooks/local.sh" in result.stderr


def test_sync_config_continues_while_gh_is_signed_out_on_the_vm(vm, tmp_path):
    _home(tmp_path)

    result, calls = vm("sync-config", STUB_GH_SIGNED_OUT="1")

    assert result.returncode == 0, result.stderr
    assert "run gh auth setup-git" in result.stderr
    assert calls[-1].endswith(" ces-revisions-vm:.config/ces-revisions/links/")


def test_cutover_copies_the_project_memory_once(vm, tmp_path):
    memory = _mac_memory(_home(tmp_path))

    result, calls = vm("sync-config", "--cutover")

    assert result.returncode == 0, result.stderr
    assert calls[-3:] == [
        f"ssh ces-revisions-vm test -e {VM_MEMORY}MEMORY.md",
        f"ssh ces-revisions-vm mkdir -p {VM_MEMORY}",
        f"rsync -a {memory}/ ces-revisions-vm:{VM_MEMORY}",
    ]


def test_cutover_refuses_when_the_vm_already_has_project_memory(vm, tmp_path):
    _mac_memory(_home(tmp_path))

    result, calls = vm("sync-config", "--cutover", STUB_REMOTE_MEMORY="1")

    assert result.returncode == 1
    assert "runs once" in result.stderr
    assert not any(call.startswith("rsync") and VM_MEMORY in call for call in calls)


def test_cutover_refuses_without_project_memory_on_the_mac(vm, tmp_path):
    _home(tmp_path)

    result, calls = vm("sync-config", "--cutover")

    assert result.returncode == 1
    assert "no project memory" in result.stderr
    assert not any(VM_MEMORY in call for call in calls)
