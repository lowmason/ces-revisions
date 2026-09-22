# Cloud GPU Environment, Plan 4 — Infrastructure, VM Setup, Cost Guards, and Decision Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Spec: specs/cloud-gpu-environment.md, the second of its two implementation plans (Reqs 3–6 and
> 8–10). The spec
> sits outside specs/ces-revisions-roadmap.md, so no roadmap stage is ticked.

**Goal:** Build the cloud GPU development environment with OpenTofu (Reqs 3–5), set up the VM with the Mac's Claude Code and git configuration (Req 6), guard its cost (Req 8), operate it through `infra/bin/vm` and a runbook (Req 9), time the engine on `dev`, the `a10g` fallback, and `h100`, and write the decision record (Req 10), ending with project memory cut over to the VM. The permanent `l4` and `l40s` tiers remain available for later capacity retries.

**Architecture:** The work runs in four phases:
- Tasks 1–5 write and test the repo side with no AWS access: the cost guards, the operations wrapper, the VM's first-boot and setup scripts, the probe's driver check, and the two OpenTofu roots.
- Tasks 6–11 build the environment at `size = dev` with the human partner. They pin the location and image, create the state bucket, apply `infra/env`, open access, set up the VM, and run the `dev` checks.
- Tasks 12–14 wait on the calendar: the cost allocation tag and the first snapshot (Task 12), the G and VT quota and the `l4`/`l40s`/`a10g` capacity path (Task 13), and the P quota (Task 14).
- Tasks 15 and 16 write the runbook and the decision record, update the documentation, and cut project memory over to the VM.

Every AWS or VM output that the decision record cites lands in `docs/decisions/cloud-gpu-evidence/` or `docs/decisions/cloud-gpu-probe/`, narrowed so that no account identifier is committed.

**Tech Stack:** OpenTofu 1.12 (S3 backend with `use_lockfile`), the hashicorp/aws provider 6.64, AWS EC2, VPC, IAM, Systems Manager Session Manager, Budgets, Cost Explorer, and Data Lifecycle Manager, Canonical Ubuntu 24.04 with cloud-init and systemd 255, Ubuntu's NVIDIA 580 server driver with open kernel modules, GitHub CLI, uv 0.12, Python 3.14 for the project and 3.12 for the VM's guards, JAX 0.11.1 with `jax[cuda13]`, bash, jq, pytest, and ruff 0.16.7.

**Source:** [`specs/cloud-gpu-environment.md`](../cloud-gpu-environment.md) Reqs 3–6 and 8–10, the Rollout note, and Verification bullets 2–12 and 14, with bullet 13 rerun at the end.

**Retirement:** This is the spec's last plan. At Plan Completion Protocol step 5, retire this plan to `specs/plans/completed/` and `specs/cloud-gpu-environment.md` to `specs/completed/`, marked complete at its top, in one commit. Plan 3 has already retired, so no other live plan implements the spec. In that commit, also re-point the two links to the spec that stay behind, in `docs/decisions/cloud-gpu-evidence/README.md` and `docs/decisions/cloud-gpu.md`, from `specs/` to `specs/completed/`.

## Preconditions

Before dispatching Task 1, the controller confirms that plan 3 finished and left what this plan builds on:

```bash
if grep -qs '^\*\*Status: COMPLETE' specs/plans/completed/3-cloud-gpu-environment.md; then
  echo "plan 3 complete and retired"
else
  echo "STOP: plan 3 is not complete and retired"
fi
jq -e '.chosen.region and .chosen.zone' docs/decisions/cloud-gpu-evidence/zone-choice.json
if rg -Fq 'Outside this roadmap: the cloud GPU environment and its decision record.' \
  specs/ces-revisions-roadmap.md; then
  echo "roadmap Stage 6 cloud dependency present"
else
  echo "STOP: roadmap Stage 6 no longer consumes the cloud environment"
fi
command -v tofu aws session-manager-plugin jq
```

Expected: `plan 3 complete and retired`, `true`, `roadmap Stage 6 cloud dependency present`, and
four paths. If `zone-choice.json` names no zone, stop: Req 2 returned the provider choice to the
human partner, and this plan does not apply. The semantic roadmap check deliberately replaces the
stale branch-only `b7e2a85` ancestry check.

Then check the text this plan builds on:

```bash
uv run python - <<'EOF'
"""Check that the text from plan 3 that this plan edits or reads is where this plan expects it."""

import json
from pathlib import Path

ANCHORS = [
    ("pyproject.toml", 'required-version = "~=0.12.13"\n'),
    ("pyproject.toml", "cuda = [\"jax[cuda13]>=0.11.1; sys_platform == 'linux'\"]\n"),
    ("pyproject.toml", '    "fastexcel>=0.21.0",\n'),
    ("pyproject.toml", '    "python-dotenv>=1.0",\n'),
    ("pyproject.toml", "[tool.ruff.lint]\n"),
    (".gitignore", "backend.hcl\nterraform.tfvars\n"),
    (".gitignore", "data/annual/cache/\ndata/annual/panel/\n"),
    ("src/ces_revisions/devices.py", "def has_nvidia_device() -> bool:\n"),
    ("src/ces_revisions/engine_probe.py", "import json\nimport shutil\nimport subprocess\n"),
    ("src/ces_revisions/engine_probe.py", "from ces_revisions.kalman import LinearGaussianSSM, kalman_filter\n"),
    (
        "src/ces_revisions/engine_probe.py",
        'def _nvidia_driver() -> str | None:\n    """The NVIDIA driver version, or None on a host without nvidia-smi."""\n    if shutil.which("nvidia-smi") is None:\n        return None\n',
    ),
    ("tests/test_engine_probe.py", "import json\nimport shutil\nimport subprocess\n"),
    ("tests/test_engine_probe.py", "from ces_revisions.engine_probe import synthetic_problem\n"),
    ("tests/test_engine_probe.py", '    assert (record["nvidia_driver"] is None) == (shutil.which("nvidia-smi") is None)\n'),
    ("tests/test_stack.py", "def test_session_runs_float64_jax_on_the_expected_devices("),
    ("tests/test_stack.py", '    "fastexcel",\n'),
    (
        "docs/decisions/cloud-gpu-evidence/README.md",
        "Command outputs recorded by Plan 3, the first implementation plan for [`specs/cloud-gpu-environment.md`](../../../specs/cloud-gpu-environment.md), for its Reqs 1 and 2. The decision record `docs/decisions/cloud-gpu.md`, which Plan 4 writes, cites them. Every `aws` command ran as the IAM user through the `ces-revisions` profile.\n",
    ),
    ("docs/decisions/cloud-gpu-evidence/README.md", "Before each commit, this directory was searched for the account ID, for ARNs, and for at signs.\n"),
    ("CLAUDE.md", "- `src/ces_revisions/engine_probe.py` times the engine's value and gradient on the current host"),
    ("CLAUDE.md", "- `src/ces_revisions/annual/` implements roadmap Stage 4."),
    ("CLAUDE.md", "uv run python -m ces_revisions.engine_probe --help   # time the engine's value and gradient on this host; writes a JSON record\n"),
    ("CLAUDE.md", "and reformats Python code blocks inside it.\n"),
    (
        "README.md",
        "docs/decisions/                       decision records (engine.md: the Kalman engine)\n  cloud-gpu-evidence/                 AWS account, zone, and quota evidence for the cloud GPU environment\n  cloud-gpu-probe/                    engine timing records, starting with the Mac baseline\n",
    ),
    ("README.md", "### Annual benchmark-source tables\n"),
    ("README.md", "The modeling stack is JAX"),
]
problems = []
for name, anchor in ANCHORS:
    path = Path(name)
    count = path.read_text().count(anchor) if path.exists() else 0
    if count != 1:
        problems.append(f"STOP: {name} holds {count} copies of: {anchor.splitlines()[0][:70]}")
chosen = json.loads(Path("docs/decisions/cloud-gpu-evidence/zone-choice.json").read_text()).get("chosen")
if chosen is None:
    problems.append("STOP: zone-choice.json names no zone")
else:
    evidence = Path("docs/decisions/cloud-gpu-evidence")
    for path in (
        Path("docs/decisions/cloud-gpu-probe/mac.json"),
        evidence / f"service-quotas-get-service-quota-{chosen['region']}.json",
        evidence / f"service-quotas-request-increase-{chosen['region']}.json",
    ):
        if not path.exists():
            problems.append(f"STOP: {path} is missing")
print("\n".join(problems) or f"all {len(ANCHORS)} anchors found once, and plan 3's evidence files exist")
EOF
uv run pytest -m "not slow and not network" -q | tail -n 1
```

Expected on the reconciled `08ed203` baseline after Plan 3: `all 25 anchors found once, and plan
3's evidence files exist`, then `330 passed, 17 deselected`.

Underneath: Tasks 1, 2, 4, 6, and 16 edit text that Plan 3 committed, by exact replacement, and
Tasks 6, 7, 8, 15, and 16 read its evidence. The anchors now describe Plan 3's reconciled output,
including Stage 3's `fastexcel` stack import and the Stage 3/4 README and CLAUDE.md material. If the
script reports a STOP, show your human partner: compare the live file with the intended edit and
revise the matching step, noted as a deviation. Do not restore an old anchor. If a newer `main`
changes the preflight count, collect the live fast/slow/network baseline and recompute this plan's
totals from its explicit additions: Task 1 `+13`, Task 2 `+16`, Task 3 `+4`, and Task 4 `+2` fast
tests.

## Current-main reconciliation (2026-09-16)

Plan 3 leaves 330 fast tests on the `08ed203` baseline. Plan 4 adds 35 fast tests and no marked
tests:

| After task | Added in task | Fast-tier total | Fast-tier deselected |
|---|---:|---:|---:|
| Plan 3 precondition | — | 330 | 17 |
| Task 1: cost guards | 13 | 343 | 17 |
| Task 2: operations wrapper | 16 | 359 | 17 |
| Task 3: VM setup | 4 | 363 | 17 |
| Task 4: probe driver check | 2 | 365 | 17 |

The reconciled 2026-09-16 repository collected 382 cases: 365 fast, 13 slow, and 4 network. Its
slow selection was `13 passed, 369 deselected`, while each required full VM run expected 382 passed
when the four live network canaries are reachable and unchanged. A network-canary failure is a
live-source signal to diagnose, not a reason to remove the test or relabel it.

The original cloud branch hard-coded its own branch name and the 2026-09-13 counts of personal
Claude configuration links. This reconciliation makes the execution branch and link counts runtime
values: the branch comes from `git branch --show-current`, and the VM counts must equal a Mac-side
snapshot taken immediately before `sync-config`.

Current main has none of Plan 4's created `infra/`, guard, wrapper, VM-setup, or cloud-decision
paths, so those tasks have no path collision with Stages 2–4. Its only existing-file edits are the
Plan 3 outputs plus `pyproject.toml`, `.gitignore`, README, and CLAUDE.md; the 25-anchor precondition
protects the Stage 3/4 additions before any replacement runs.

## Capacity fallback amendment (2026-09-22)

Task 13's first `l4` start reached EC2 after the G and VT quota became 4 vCPUs, but EC2 returned
`InsufficientInstanceCapacity` for g6.xlarge in the pinned zone and left the instance stopped. The
user approved a permanent `l40s` tier backed by g6e.xlarge. A fresh, narrowed AWS check established
that the pinned zone offers g6e.xlarge, that it needs the same 4-vCPU G and VT quota, and that its
Linux On-Demand price is \$1.861/hr.

This amendment preserves the failed `l4` attempt as evidence and inserts a separate reviewed plan
and apply gate for `l40s`. Task 13's GPU checks, determinism comparison, and probe now run on
`l40s`; `l4` remains a supported tier for a later retry. Task 16 compares the Mac, `dev`, `l40s`,
and `h100`, and explains why `l40s` supplied the completed mid-tier measurement. Intervening
roadmap work had raised the live branch to 432 cases before this amendment: 415 fast, 13 slow, and
4 network, matching the 432-test `dev` evidence. The one added wrapper test raises the current
total to 433 cases: 416 fast, 13 slow, and 4 network.

## Second capacity fallback amendment (2026-09-22)

Task 13A's first `l40s` start reached EC2 under the approved quota, but EC2 again returned
`InsufficientInstanceCapacity` and left the instance stopped after changing its type to
g6e.xlarge. The user approved a permanent `a10g` tier backed by g5.xlarge. A fresh, narrowed AWS
check established that the pinned zone offers g5.xlarge, that it needs the same 4-vCPU G and VT
quota, and that its Linux On-Demand price is \$1.006/hr.

This amendment preserves both failed capacity attempts as evidence and inserts another separate
reviewed push and apply gate for `a10g`. Task 13's GPU checks, determinism comparison, and probe now
run on `a10g`; `l4` and `l40s` remain supported for later retries. Task 16 compares the Mac, `dev`,
`a10g`, and `h100`, and explains why `a10g` supplied the completed mid-tier measurement. The new
wrapper case raises the current total to 434 cases: 417 fast, 13 slow, and 4 network.

### Original planning evidence (2026-09-13)

This plan's scripts and tests ran before the plan was written, in a scratch clone holding plan 3's code. Its HCL, and everything that runs on AWS or Ubuntu, did not. Treat a deviation from these outcomes as a signal, not noise.

- **What ran.**
  - `tests/test_idle_stop.py`, `tests/test_vm_wrapper.py`, and `tests/test_vm_setup.py` passed their
    33 tests, and Task 4's two probe tests passed. In that scratch clone the fast tier passed 71
    tests, where Plan 3 left 36; those historical totals are superseded by the current-main table
    above; the capacity amendment adds one more wrapper case, for a `+36` delta.
  - `ruff check` and `ruff format --check` were clean.
  - `bash -n` passed for each shell script under Homebrew's bash 5 and macOS's bash 3.2.
  - Task 13's determinism script printed the same SHA-256 twice on the Mac's CPU.
  - Task 4's edits, replayed on plan 3's `engine_probe.py` and `tests/test_engine_probe.py` and formatted as its Step 6 formats them, reproduced the tested files, and the Preconditions' anchor check passed against the scratch clone's copies of plan 3's files.
- **What did not run.**
  - The HCL. `tofu` was not installed while planning, because plan 3's Task 1 installs it. The HCL follows the hashicorp/aws provider's v6.64.0 documentation and source. Task 5's `tofu fmt` and `tofu validate` are its first check, and `tofu fmt` may realign whitespace; commit what it writes.
  - cloud-init, the systemd units, `first-boot.sh`, `install.sh`, and the driver packages were checked against documentation and source only. Tasks 8–14 run them for the first time.
  - The user data's size. The seven embedded files total 7,615 bytes in 188 lines. With YAML's indentation and the SSH key, the user data should come to roughly 9 KB of EC2's 16 KB limit, and the precondition in `instance.tf` checks the real size at plan time.
- **Findings that shaped the plan, checked 2026-09-13.**
  - **Credentials.** OpenTofu's S3 backend reads `aws login` sessions from OpenTofu 1.12.0. The hashicorp/aws provider reads them from about 6.23, through aws-sdk-go-v2's config 1.32. Tasks 7 and 8 check both, and a `credential_process` profile is the fallback.
  - **Size switches start the instance.** Changing `instance_type` stops the instance, modifies it, and starts it, even when it was stopped before. A switch to a GPU size therefore starts that size's billing at once. Switching between x86_64 types keeps the instance and its root volume.
  - **Tags and the budget.**
    - `default_tags` reach the root volume at launch (provider 5.39.0 and later), so Data Lifecycle Manager can target the `project` tag.
    - A cost allocation tag can be activated only after a resource carries it, up to 24 hours later, and activation takes up to 24 hours more.
    - The tag-filtered budget cannot see public IPv4 hours (\$0.005 per hour), data transfer, or tax.
    - Forecast alerts need about five weeks of history, so the 50% and 80% alerts stay silent during this plan.
  - **Snapshots.** DLM's `times` is UTC, and each snapshot starts within an hour after it, so the first snapshot arrives the day after the apply.
  - **Shutdown.** systemd 255's `shutdown` powers off at once when it cannot reach logind, so the GPU cap starts after logind. `halt` leaves an EC2 instance running and billed; `shutdown -h` powers it off, which stops it.
  - **Driver.**
    - `ubuntu-drivers install --gpgpu` exits 1 on a GPU-less m7i, so first boot installs packages by name: `nvidia-headless-no-dkms-580-server-open`, `nvidia-utils-580-server`, and `linux-modules-nvidia-580-server-open-aws`. noble-updates had 580.173.02.
    - The module package can bring a newer AWS kernel, which runs from the next start.
    - NVIDIA recommends the open kernel modules for Ada (L4) and Hopper (H100).
  - **JAX without a GPU.** With `jax[cuda13]` installed and no GPU, JAX 0.11.1 logs an ERROR traceback from `cuInit` and then runs on the CPU. `JAX_PLATFORMS=cpu` silences it.
  - **gh.** Ubuntu's gh 2.45 is too old for GitHub's API, so first boot installs gh from GitHub's apt repository. gh's manual cautions that a fine-grained token given to `--with-token` can confuse commands that reach other repositories.
  - **Desktop app.** Claude Code issue #40967 reports `ProxyCommand` broken in the macOS desktop app, and #26809 reports that the app ignores `Port`. Both the spec's desktop path and its fallback may fail, so Task 9 tries each and records what worked. A terminal SSH session and a Session Manager shell work either way.
  - **State bucket.** OpenTofu's documentation suggests a lifecycle rule for the lock file's object versions. This plan omits it, because those versions cost about \$0.01 a month.
- **Ruff and the guard.** Ruff's formatter rewrote `except (OSError, ValueError):` into Python 3.14's unparenthesized form, which Ubuntu 24.04's Python 3.12 rejects. Task 1 guards against this three ways:
  - it pins `infra/vm/guards/*.py` to `py312` with `per-file-target-version`;
  - it writes the guard with `contextlib.suppress`, which reads the same under every target;
  - it tests that the guard parses as Python 3.12.
- **The Mac's settings, 2026-09-13.**
  - `~/.claude` had 32 skill, 7 agent, 3 command, and 1 hook links, all into `~/Projects/agent-skills/`.
  - `settings.json` holds no `/Users/` or `/opt/homebrew` path. Its one hook runs `$HOME/.claude/hooks/readonly-agent-guard.py`, a stdlib-only Python script, so the file works on Linux unchanged. It also carries `enabledPlugins` and `availableModels`, so the VM gets the same model pinning.
  - `~/.gitconfig` names `/opt/homebrew/bin/gh` as git's credential helper for github.com, and `gh auth setup-git` on the VM replaces it. Neither repository has a `.gitattributes`, so the git-lfs filter settings do nothing there.
- **An email already committed.** `pyproject.toml`'s `authors` entry, committed when the project was created, holds an email address, while Req 3 says no committed file holds one. Task 8's search reports that line and excludes it, and the handoff raises it with the human partner.
- **The probe on CPU sizes.** Plan 3's `_nvidia_driver()` runs `nvidia-smi` whenever it is installed. On `dev` the driver's `nvidia-smi` exits non-zero, which would stop the probe with `CalledProcessError`, so Task 4 asks it only where an NVIDIA device exists, as the idle stop does.
- **A check in plan 3.** `git check-ignore -q` refuses more than one path (`fatal: --quiet is only valid with a single pathname`), so a multi-path `if git check-ignore -q …` always takes its else branch. This plan's checks list the paths without `-q` and test the output instead.

## Execution notes

- **Where.** Continue from completed Plan 3 in its isolated execution worktree and `codex/` feature
  branch, or create a fresh worktree from that completed branch. Do not reuse
  `.claude/worktrees/cloud-gpu-ces-revisions-ee5804` or assume its branch name. Run `infra/` from
  the Mac only.
- **Who runs what.**
  - Tasks 1–5 suit subagent-driven-development. They need no AWS credentials; Task 5 needs `tofu`, which downloads the AWS provider.
  - Tasks 6–14 and 16 are operations with the human partner. The controller runs them itself, in order, and dispatches no implementer subagent for them. **STOP** marks a gate: say what the human partner must do or approve, wait for their answer, then run the check that follows.
  - Task 15, the runbook, suits a subagent once Task 11 is done.
- **Approvals.** Each of these waits for the human partner's explicit yes, given after they have seen what it does and what it costs. A yes covers only the action it names.
  - every `tofu apply`, including each size switch;
  - every `git push`;
  - appending a profile to `~/.aws/config`, if Task 7 needs the fallback.
- **Waiting.**
  - Task 12 waits for the tag activation and the first snapshot, Task 13 for the G and VT quota, and Task 14 for the P quota. Run each as soon as its condition holds, in any order.
  - While none holds, run Task 15 and keep the instance stopped with `infra/bin/vm stop`. The idle stop stops it after 45 idle minutes anyway, so if an `ssh` command times out, run `infra/bin/vm status` and then `infra/bin/vm start`.
  - Task 16 runs last.
- **AWS and OpenTofu.** Shell state does not persist between commands, so every block that calls `aws`, `tofu`, or `infra/bin/vm` starts with `export AWS_PROFILE=ces-revisions`.
  - When a command fails because the session expired, ask the human partner to run `aws login --profile ces-revisions` again.
  - If Task 7 or Task 8 finds that OpenTofu cannot read the login session, the fallback profile that step creates replaces `ces-revisions` in the `export` of every later block that runs `tofu`. `infra/bin/vm` then runs with `CES_TOFU_PROFILE=ces-revisions-process`.
- **On the VM.** From Task 10 on, the controller runs VM commands from the Mac through the SSH host entry, as `ssh ces-revisions-vm bash -l -s <<'EOF' … EOF`. The login shell puts uv on `PATH`. Steps that need a person at the VM's keyboard, such as entering the GitHub token, are the human partner's, in their own terminal.
- **Long commands.** The OpenTofu size switches, the first `setup.sh` run, the full test runs, the probes, and the waits for a stopped or reachable instance can take longer than a foreground command may. Run them in the background and continue when they finish.
- **Secrets.** Never write an account ID, ARN, bucket name, email address, or token into a repo file, and never print `terraform.tfvars`, `backend.hcl`, or a state file. The evidence commands narrow their outputs with `--query`. Every evidence commit first runs the evidence scan in Task 6, Step 7.
- **Commits.** Every commit step first runs `uv run ruff format` and `uv run ruff check`, then stages files by explicit path. Never run `git add -A`.
- **Additions to the spec.** Each is small, and each is flagged here:
  - `infra/bin/vm forward`, beside Req 9's six subcommands, runs Req 4's `AWS-StartPortForwardingSession` fallback.
  - `infra/bin/vm sync-config` also runs `gh auth setup-git` on the VM, because the copied `~/.gitconfig` names the Mac's gh. Before the token exists, it prints a reminder instead of failing.
  - Req 6 copies `~/.claude/hooks/`. This plan sends the hook's link name, as for the skill, agent, and command links, because the Mac's one hook is a link into `~/Projects/agent-skills/hooks/` that a copy would carry over broken.
  - `budget_enabled`, which defaults to `false`, holds back the budget, its action, and the action's role until Task 12 activates the `project` tag. Task 8's first apply and its no-change plan therefore do not depend on activation.
  - The root volume is encrypted with AWS's managed EBS key. Encryption is chosen only at creation, and `prevent_destroy` forbids replacing the instance, so this choice is permanent.
  - The GPU cap looks for an NVIDIA PCI device (vendor `0x10de`) instead of `/dev/nvidia0`, because those device files may not exist at boot. It also starts after logind, and it first runs `nvidia-smi -L`, which creates the `/dev/nvidia*` files that `devices.py` and the idle stop look for.
  - `infra/vm/first-boot.sh` holds cloud-init's root steps as a script, which OpenTofu embeds with the guards. It installs the guards before any package step, so a failed install cannot leave the VM unguarded. It holds the AWS kernel packages along with the driver, so a kernel upgrade cannot leave the modules behind.
  - `[tool.ruff] per-file-target-version` stops ruff from rewriting the guards into Python 3.14-only syntax.
  - Task 4 changes plan 3's `engine_probe.py` so it asks `nvidia-smi` for the driver version only where an NVIDIA device exists.
  - `pinned.auto.tfvars` (committed) holds the region, zone, and image ID. `size.auto.tfvars` (gitignored) records the last applied size, so a later plain `tofu plan` does not switch back to `dev`.
  - The state bucket is named with `bucket_prefix`, so no bucket name is ever written into the repo.
  - A precondition on the instance checks EC2's 16 KB user-data limit.
  - Task 7 backs up the gitignored operational files to `~/.config/ces-revisions/infra/`, outside the repo, because removing this worktree would delete them.
  - Task 9 compares the SSH host key's fingerprint with the one read through Session Manager before trusting it.
  - Task 12 runs IAM's policy simulator on the budget action's role, because a budget stop cannot be triggered on demand.
  - Tasks 13 and 14 compare two GPU runs bit for bit, with and without XLA's deterministic flag, which settles the item plan 3 handed on.
- **Stop conditions.** Stop and report rather than improvise when:
  - a plan adds, changes, or destroys a resource its step does not list;
  - `tofu plan` reports changes right after an apply;
  - AWS returns an error other than expired credentials, including capacity and quota errors;
  - a VM check, test, or scan differs from its step's expected output.

  Never loosen a check to make it pass.

## Global Constraints

- **State (Req 3):** "`infra/state/` creates the state bucket with versioning on, all public access blocked, default encryption, and `prevent_destroy`. Its own state stays local and gitignored." "`infra/env/` creates everything else and stores its state in that bucket through the S3 backend with `use_lockfile = true`, which locks with S3 conditional writes instead of a DynamoDB table. Bucket, key, and region come from a gitignored `infra/env/backend.hcl` (partial backend configuration); a committed `backend.hcl.example` shows its shape. Personal values, such as the budget email address, live in a gitignored `terraform.tfvars`."
- **Lock file, tags, and secrets (Req 3):** "The provider dependency lock file, `.terraform.lock.hcl`, is committed. The AWS provider's `default_tags` put `project = ces-revisions` on every resource." "No committed file contains an account ID, a bucket name, an email address, or a token."
- **Network (Req 4):** "A dedicated VPC has one public subnet in the Req 2 zone, an internet gateway, and a route table. The instance gets an auto-assigned public IPv4 address, used only for outbound traffic. Its security group has no ingress rules and unrestricted egress. No NAT gateway, VPC endpoint, or Elastic IP exists." "An IAM instance profile grants only `AmazonSSMManagedInstanceCore`. The instance requires IMDSv2."
- **Access (Req 4):** "Shell access is `aws ssm start-session`. SSH runs through Session Manager's `AWS-StartSSHSession` document from a `~/.ssh/config` host entry that uses `ProxyCommand` (the desktop app does not support `ProxyJump`), names the AWS CLI by absolute path so it resolves outside a login shell, and passes the AWS profile. A dedicated ed25519 key pair authenticates SSH; its private half stays on the Mac." The desktop app's fallback "is `AWS-StartPortForwardingSession` to `localhost:2222`, with the port typed into the connection dialog, because the app ignores `Port` in `~/.ssh/config`."
- **Sizes (Req 5):**

  | `size` | Instance type | vCPU / memory | GPU | On-Demand |
  |---|---|---|---|---|
  | `dev` (default) | m7i.xlarge | 4 / 16 GiB | none | \$0.20/hr |
  | `l4` | g6.xlarge | 4 / 16 GiB | L4, 24 GB | \$0.81/hr |
  | `l40s` | g6e.xlarge | 4 / 32 GiB | L40S, 48 GB | \$1.861/hr |
  | `a10g` | g5.xlarge | 4 / 16 GiB | A10G, 24 GB | \$1.006/hr |
  | `h100` | p5.4xlarge | 16 / 256 GiB | H100, 80 GB | \$6.88/hr |

- **Instance (Req 5):**
  - "Changing `size` updates the instance in place (stop, modify, start); the instance ID and root volume do not change."
  - "The AMI is Canonical Ubuntu 24.04 LTS for amd64. Its ID is read once from Canonical's public Systems Manager parameter and pinned as a variable, never looked up at plan time".
  - "The root volume is gp3, 100 GB, and is deleted with the instance."
  - "`instance_initiated_shutdown_behavior = "stop"`".
  - "The instance's `lifecycle` block sets `prevent_destroy = true` and ignores changes to `user_data`".
- **First boot (Req 6):** "At first boot, as root, cloud-init installs Ubuntu's packaged NVIDIA server driver from a branch numbered 580 or later (the minimum for `jax[cuda13]`) and holds it with `apt-mark hold`; installs `git` and `gh`; installs the SSH public key; and installs the Req 8 guards, which OpenTofu embeds in the user data from `infra/vm/guards/`."
- **Setup (Req 6):** "`infra/vm/setup.sh` runs as `ubuntu` and is idempotent. It installs uv from the Req 7 version series, runs `uv python install 3.14`, clones `lowmason/ces-revisions` and `lowmason/agent-skills` into `~/Projects/`, runs `uv sync --locked --extra cuda`, recreates the Mac's skill, agent, and command links (32, 7, and 3 on 2026-09-13) as links into `~/Projects/agent-skills/`, using the link names that `sync-config` sent, and reinstalls the guards from the checkout. `sync-config` therefore runs before `setup.sh`."
- **Carried configuration (Req 6):** "`infra/bin/vm sync-config` copies from the Mac with rsync over the SSH host entry, the Mac being the source until cutover: `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, `~/.claude/hooks/`, `~/.gitconfig`, and the names of the Mac's skill, agent, and command links. At cutover it copies this project's memory folder once from the Mac to the VM's memory folder for `~/Projects/ces-revisions`; the VM owns project memory afterward."
- **Token (Req 6):** "The user creates a fine-grained GitHub token scoped to `lowmason/ces-revisions` (contents and pull requests, read and write) and enters it on the VM with `gh auth login --with-token`; `gh auth setup-git` makes `git push` use it. The agent never handles the token."
- **Idle stop (Req 8):** "a systemd timer runs every 5 minutes and calls `shutdown -h now` once the 1-minute load average has stayed below 0.3 for 45 consecutive minutes and, where an NVIDIA device exists, GPU utilization has stayed below 5% over the same window. The window and thresholds come from an environment file."
- **GPU cap (Req 8):** "at boot with an NVIDIA device present, a unit schedules `shutdown -h +480`. The runbook shows how to inspect, cancel, and reschedule it."
- **Budget (Req 8):** "an AWS Budgets monthly cost budget of \$150, filtered on the cost allocation tag `project = ces-revisions` (activated once in the Billing console), emails at 50% and 80% of forecast spend and at 100% of actual spend. At 100% of actual spend, an `aws_budgets_budget_action` of type `RUN_SSM_DOCUMENTS` with sub-type `STOP_EC2_INSTANCES` and `approval_model = "AUTOMATIC"` stops the instance through an execution role limited to that action. Instance-targeted actions do not reset at the next budget period, so the runbook resets it."
- **Snapshots (Req 8):** "an `aws_dlm_lifecycle_policy` snapshots the project's volume daily, keeps 7 snapshots, and copies tags. Restoring uses EC2's replace-root-volume task with a snapshot."
- **Wrapper (Req 9):** "`infra/` is operated from the Mac, never from the VM: resizing stops the instance, which would kill an apply running on it, and the instance role has no AWS permission beyond Systems Manager. `infra/bin/vm` exits with that explanation when run on the VM." It "provides `start`, `stop`, `status`, `connect` (a Session Manager shell), `size <dev|l4|l40s|a10g|h100>` (runs `tofu apply` in `infra/env` with that size), and `sync-config` (Req 6). Every subcommand prints the AWS CLI or OpenTofu command it runs before running it."
- **Runbook (Req 9):** "`docs/cloud-gpu-runbook.md` covers one-time setup, daily use, switching sizes, running a long GPU job, checking spend, snapshots and restore, token rotation, updating the held driver or pinned image, recovery after a budget stop, teardown (removing `prevent_destroy`, then destroying `infra/env` and `infra/state`), and troubleshooting (insufficient capacity, quota errors, Session Manager not connecting, JAX not seeing the GPU). Each step gives its commands and a short note on what happens underneath. The document follows CLAUDE.md's Markdown conventions."
- **Decision record (Req 10):** "`docs/decisions/cloud-gpu.md` takes the shape of `docs/decisions/engine.md`. It is written after the Verification bullets pass, from their recorded evidence, and contains no placeholder." Its Context, Decision, Evidence, Alternatives considered, and Revisit triggers hold what Req 10 lists. The probe table covers the Mac, `dev`, `a10g`, and `h100` "at T=280, n=150, p=70 with batch sizes 1, 4, and 16 everywhere and 64 on `h100`". It records the failed `l4` and `l40s` capacity attempts separately. "Stage 6's measurement at Stage 7–9 dimensions belongs in `docs/decisions/seasonal-state.md`, as the roadmap assigns it, not in this record."
- **Who does what (Rollout note):** "Steps that enter credentials or change account security belong to the user: root MFA, creating the IAM user and registering its MFA device, the `aws login` browser sign-in, activating the cost allocation tag, and creating and entering the GitHub token."
- **BLS (Verification bullet 12):** "One small `download.bls.gov` fetch, run once by hand from a VM shell with a User-Agent naming the project and a contact, is recorded in the decision record only as allowed or blocked with its HTTP status; no committed file holds the User-Agent string."
- **Tests:** the fast hermetic tier is `uv run pytest -m "not slow and not network"`. MCMC tests carry the `slow` marker, and an unregistered marker is a collection error.
- **Ruff:** the default rules plus `extend-select = ["I", "B", "UP"]`; never switch to `select`. `uv run ruff format` also formats Python blocks inside Markdown, including this plan's.
- **Markdown** (the runbook, the decision record, the evidence README, `CLAUDE.md`, `README.md`) follows CLAUDE.md's conventions:
  - literal dollars written as `\$`;
  - real headings;
  - hard line breaks as a trailing `\`;
  - pseudo-math containing `_` in code spans;
  - no bare `$…$` math.
- **Scope:** Reqs 3–6 and 8–10. Out of scope, per the spec:
  - Azure;
  - the dev-box-plus-runners topology and S3 artifacts;
  - Spot instances, Capacity Blocks, and Savings Plans;
  - containers, GPU continuous integration, and multi-GPU instances;
  - Stage 6's measurement and Stage 3's data layout;
  - editing `specs/ces-revisions-roadmap.md`.

---

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `infra/vm/guards/idle_stop.py` | The idle decision, and the sampling of load and GPU utilization | 1 |
| `infra/vm/guards/idle-stop.env` | The idle stop's window and thresholds | 1 |
| `infra/vm/guards/ces-idle-stop.service`, `ces-idle-stop.timer` | Run the idle stop every 5 minutes | 1 |
| `infra/vm/guards/ces-gpu-cap.service` | Schedule the poweroff 8 hours after boot on GPU hosts | 1 |
| `infra/vm/guards/install.sh` | Install and enable the guards, as root | 1 |
| `tests/test_idle_stop.py` | The idle decision, the streak, the guard end to end, and Python 3.12 syntax | 1 |
| `pyproject.toml` | `per-file-target-version` for the guards | 1 |
| `infra/bin/vm` | `start`, `stop`, `status`, `connect`, `forward`, `size`, `sync-config` | 2 |
| `tests/test_vm_wrapper.py` | Printed commands, the refusal on the VM, size recording, settings and memory copies | 2 |
| `.gitignore` | `size.auto.tfvars` | 2 |
| `infra/vm/setup.sh` | The `ubuntu` user's idempotent setup | 3 |
| `tests/test_vm_setup.py` | `setup.sh`'s first run, rerun, and refusals | 3 |
| `infra/vm/first-boot.sh` | Root's first-boot steps: driver, holds, gh, guards | 3 |
| `src/ces_revisions/engine_probe.py` | The driver version, asked for only on NVIDIA hosts | 4 |
| `tests/test_engine_probe.py` | The driver version with and without an NVIDIA device | 4 |
| `infra/state/main.tf`, `infra/state/.terraform.lock.hcl` | The state bucket | 5 |
| `infra/env/{versions,variables,network,instance,budget,snapshots,outputs}.tf` | Provider and backend, inputs, network, instance, budget, snapshots, outputs | 5 |
| `infra/env/.terraform.lock.hcl`, `infra/env/backend.hcl.example` | The provider lock and the backend file's shape | 5 |
| `infra/state/pinned.auto.tfvars`, `infra/env/pinned.auto.tfvars` | Region, zone, and image ID, pinned from evidence | 6 |
| `docs/decisions/cloud-gpu-evidence/` (new files and README sections) | Command outputs for the decision record | 6–14 |
| `docs/decisions/cloud-gpu-probe/{dev,a10g,h100}.json` | Probe records for each measured size | 11, 13, 14 |
| `docs/cloud-gpu-runbook.md` | The runbook | 15 |
| `docs/decisions/cloud-gpu.md` | The decision record | 16 |
| `CLAUDE.md`, `README.md` | `infra/`, the runbook, and the decision record | 16 |

Tasks 7, 8, and 13 also create gitignored files on the Mac: `infra/state/terraform.tfstate` and `infra/env/backend.hcl` (Task 7), `infra/env/terraform.tfvars` (Task 8), and `infra/env/size.auto.tfvars` (Task 13).

---

### Task 1: Cost guards (Req 8)

**Files:**
- Create: `infra/vm/guards/idle_stop.py`
- Create: `infra/vm/guards/idle-stop.env`
- Create: `infra/vm/guards/ces-idle-stop.service`
- Create: `infra/vm/guards/ces-idle-stop.timer`
- Create: `infra/vm/guards/ces-gpu-cap.service`
- Create: `infra/vm/guards/install.sh`
- Modify: `pyproject.toml` (a `[tool.ruff]` table before `[tool.ruff.lint]`)
- Test: `tests/test_idle_stop.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `is_idle(load: float, gpu_utilizations: list[float], load_threshold: float, gpu_threshold: float) -> bool` and `next_state(now: float, idle: bool, idle_since: float | None, window_seconds: float) -> tuple[float | None, bool]` in `idle_stop.py`, which runs as a script;
  - the settings `IDLE_WINDOW_MINUTES`, `LOAD_THRESHOLD`, and `GPU_UTILIZATION_THRESHOLD` from `/etc/ces-revisions/idle-stop.env`, and the overrides `IDLE_STATE_FILE`, `IDLE_LOADAVG_FILE`, `IDLE_NVIDIA_DEVICE`, and `IDLE_SHUTDOWN_COMMAND` that the tests use;
  - `install.sh`, which runs as root from the directory holding the six guard files. It installs `/usr/local/lib/ces-revisions/idle_stop.py`, installs `/etc/ces-revisions/idle-stop.env` when that file is absent, installs the three units in `/etc/systemd/system/`, and creates the marker `/etc/ces-revisions/vm` that `infra/bin/vm` refuses on. It enables and starts `ces-idle-stop.timer` and enables `ces-gpu-cap.service`.

  Task 3's `first-boot.sh` and `setup.sh` run `install.sh`, and Task 5 embeds all six files in the instance's user data.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_idle_stop.py`:

```python
"""The idle-stop guard shuts down only after a whole idle window, never on a busy sample."""

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "infra" / "vm" / "guards" / "idle_stop.py"
_spec = importlib.util.spec_from_file_location("idle_stop", GUARD)
idle_stop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(idle_stop)

WINDOW_SECONDS = 45 * 60
LONG_AGO = 1.0  # one second after the epoch: far more than a window ago


def test_the_guard_parses_as_python_3_12():
    # The VM runs the guard with Ubuntu 24.04's system Python, not this project's 3.14.
    ast.parse(GUARD.read_text(), feature_version=(3, 12))


@pytest.mark.parametrize(
    ("load", "gpus", "expected"),
    [
        (0.1, [], True),
        (0.3, [], False),  # at the threshold is not below it
        (0.1, [0.0, 4.0], True),
        (0.1, [0.0, 5.0], False),
        (2.0, [0.0], False),
    ],
)
def test_a_sample_is_idle_only_below_both_thresholds(load, gpus, expected):
    assert (
        idle_stop.is_idle(load, gpus, load_threshold=0.3, gpu_threshold=5) is expected
    )


def test_an_idle_streak_shuts_down_once_it_spans_the_window():
    start, shut_down = idle_stop.next_state(1000.0, True, None, WINDOW_SECONDS)

    assert (start, shut_down) == (1000.0, False)
    almost = 1000.0 + WINDOW_SECONDS - 1
    assert idle_stop.next_state(almost, True, start, WINDOW_SECONDS) == (1000.0, False)
    whole = 1000.0 + WINDOW_SECONDS
    assert idle_stop.next_state(whole, True, start, WINDOW_SECONDS) == (1000.0, True)


def test_a_busy_sample_ends_the_streak():
    assert idle_stop.next_state(5000.0, False, 1000.0, WINDOW_SECONDS) == (None, False)


def _run_guard(tmp_path, *, load, idle_since, device, nvidia_smi="exit 9"):
    """Run the guard as its timer does, with a stub nvidia-smi whose body is nvidia_smi."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(f"#!/bin/sh\n{nvidia_smi}\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    nvidia0 = tmp_path / "nvidia0"
    if device:
        nvidia0.touch()
    loadavg = tmp_path / "loadavg"
    loadavg.write_text(f"{load} 0.10 0.20 1/100 1234\n")
    state = tmp_path / "run" / "idle-since"
    if idle_since is not None:
        state.parent.mkdir()
        state.write_text(f"{idle_since}\n")
    marker = tmp_path / "shutdown-ran"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "IDLE_LOADAVG_FILE": str(loadavg),
        "IDLE_NVIDIA_DEVICE": str(nvidia0),
        "IDLE_STATE_FILE": str(state),
        "IDLE_SHUTDOWN_COMMAND": f"touch {marker}",
    }
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return state, marker, result.stderr


def test_an_old_idle_streak_shuts_the_host_down(tmp_path):
    state, marker, _ = _run_guard(
        tmp_path,
        load=0.05,
        idle_since=LONG_AGO,
        device=True,
        nvidia_smi="printf '2\\n0\\n'",
    )

    assert marker.exists()
    assert float(state.read_text()) == LONG_AGO


def test_a_first_idle_sample_starts_a_streak_without_shutting_down(tmp_path):
    state, marker, _ = _run_guard(tmp_path, load=0.05, idle_since=None, device=False)

    assert not marker.exists()
    assert state.exists()


def test_a_busy_gpu_ends_the_streak(tmp_path):
    state, marker, _ = _run_guard(
        tmp_path, load=0.05, idle_since=LONG_AGO, device=True, nvidia_smi="echo 50"
    )

    assert not marker.exists()
    assert not state.exists()


def test_an_unreadable_gpu_counts_as_busy(tmp_path):
    state, marker, stderr = _run_guard(
        tmp_path, load=0.05, idle_since=LONG_AGO, device=True, nvidia_smi="exit 9"
    )

    assert not marker.exists()
    assert not state.exists()
    assert "unreadable sample" in stderr


def test_without_an_nvidia_device_nvidia_smi_is_not_asked(tmp_path):
    # A CPU size has the driver's nvidia-smi, which fails without a GPU.
    _, marker, _ = _run_guard(
        tmp_path, load=0.05, idle_since=LONG_AGO, device=False, nvidia_smi="exit 9"
    )

    assert marker.exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_idle_stop.py -q`

Expected: `1 error` while collecting, from `FileNotFoundError: [Errno 2] No such file or directory: '…/infra/vm/guards/idle_stop.py'`. Underneath: the test loads the guard from its path instead of importing it, because `infra/` is not a Python package and the VM runs the file as a script.

- [ ] **Step 3: Write the guard**

Create `infra/vm/guards/idle_stop.py`:

```python
#!/usr/bin/env python3
"""Stop the instance once it has been idle for a whole window.

ces-idle-stop.timer runs this as root every five minutes. A sample is idle when the
1-minute load average is below LOAD_THRESHOLD and, where an NVIDIA device exists, every
GPU's utilization is below GPU_UTILIZATION_THRESHOLD percent. The first idle sample of a
streak is recorded under /run, which a reboot clears, so a streak never spans a stop and
start. A busy sample, or one that cannot be read, ends the streak. Once a streak spans
IDLE_WINDOW_MINUTES, the host shuts down, and the instance's shutdown behavior turns that
into a stopped instance. Written for the VM's system Python (3.12 on Ubuntu 24.04).
"""

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

STATE_FILE = Path(os.environ.get("IDLE_STATE_FILE", "/run/ces-revisions/idle-since"))
LOADAVG_FILE = Path(os.environ.get("IDLE_LOADAVG_FILE", "/proc/loadavg"))
NVIDIA_DEVICE = Path(os.environ.get("IDLE_NVIDIA_DEVICE", "/dev/nvidia0"))
SHUTDOWN_COMMAND = os.environ.get("IDLE_SHUTDOWN_COMMAND", "shutdown -h now").split()


def is_idle(
    load: float,
    gpu_utilizations: list[float],
    load_threshold: float,
    gpu_threshold: float,
) -> bool:
    """Whether one sample is idle: low load, and every GPU (if any) below its threshold."""
    return load < load_threshold and all(u < gpu_threshold for u in gpu_utilizations)


def next_state(
    now: float, idle: bool, idle_since: float | None, window_seconds: float
) -> tuple[float | None, bool]:
    """The streak start to record (None ends the streak), and whether to shut down."""
    if not idle:
        return None, False
    start = now if idle_since is None else idle_since
    return start, now - start >= window_seconds


def main() -> None:
    window_seconds = 60 * float(os.environ.get("IDLE_WINDOW_MINUTES", "45"))
    load_threshold = float(os.environ.get("LOAD_THRESHOLD", "0.3"))
    gpu_threshold = float(os.environ.get("GPU_UTILIZATION_THRESHOLD", "5"))
    try:
        idle = is_idle(_load(), _gpu_utilizations(), load_threshold, gpu_threshold)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(
            f"idle-stop: treating an unreadable sample as busy: {error}",
            file=sys.stderr,
        )
        idle = False
    start, shut_down = next_state(time.time(), idle, _idle_since(), window_seconds)
    if start is None:
        STATE_FILE.unlink(missing_ok=True)
    else:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(f"{start}\n")
    if shut_down:
        print(f"idle-stop: idle since {start:.0f}; shutting down", file=sys.stderr)
        subprocess.run(SHUTDOWN_COMMAND, check=True)


def _load() -> float:
    return float(LOADAVG_FILE.read_text().split()[0])


def _gpu_utilizations() -> list[float]:
    # On a CPU size the driver's nvidia-smi is installed but fails, so ask it only
    # where the device exists.
    if not NVIDIA_DEVICE.exists():
        return []
    query = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    return [float(value) for value in output.split()]


def _idle_since() -> float | None:
    # No file, or an unreadable one, means no streak is under way.
    with contextlib.suppress(OSError, ValueError):
        return float(STATE_FILE.read_text())
    return None


if __name__ == "__main__":
    main()
```

Then make it executable:

```bash
chmod +x infra/vm/guards/idle_stop.py
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_idle_stop.py -q`

Expected: `13 passed`. Underneath: the pure functions carry the rule, and the five end-to-end tests run the script the way its timer does, with a stub `nvidia-smi` on `PATH` and a `touch` in place of `shutdown`. The last test covers `dev`, where the driver's `nvidia-smi` is installed but fails because no GPU exists.

- [ ] **Step 5: Keep ruff from rewriting the guard into Python 3.14 syntax**

In `pyproject.toml`, replace:

```toml
[tool.ruff.lint]
```

with:

```toml
[tool.ruff]
# The cloud VM runs its cost guards with Ubuntu 24.04's system Python, so ruff must not
# rewrite them into Python 3.14-only syntax.
per-file-target-version = { "infra/vm/guards/*.py" = "py312" }

[tool.ruff.lint]
```

Then run:

```bash
uv run ruff format --check infra/vm/guards/idle_stop.py
uv run ruff check infra/vm/guards/idle_stop.py
```

Expected: `1 file already formatted` and `All checks passed!`. Underneath: under the project's `py314` target, `ruff format` turns `except (A, B):` into `except A, B:`, which only Python 3.14 parses. The per-file target stops that rewrite, the guard avoids the construct anyway, and `test_the_guard_parses_as_python_3_12` fails if it ever returns.

- [ ] **Step 6: Write the environment file and the units**

Create `infra/vm/guards/idle-stop.env`:

```text
# The idle stop's window and thresholds, read by ces-idle-stop.service. install.sh copies
# this file to /etc/ces-revisions/idle-stop.env only when that file is absent.
IDLE_WINDOW_MINUTES=45
LOAD_THRESHOLD=0.3
GPU_UTILIZATION_THRESHOLD=5
```

Create `infra/vm/guards/ces-idle-stop.service`:

```ini
[Unit]
Description=Power off after a whole idle window (ces-revisions cost guard)

[Service]
Type=oneshot
EnvironmentFile=/etc/ces-revisions/idle-stop.env
ExecStart=/usr/bin/python3 /usr/local/lib/ces-revisions/idle_stop.py
```

Create `infra/vm/guards/ces-idle-stop.timer`:

```ini
[Unit]
Description=Check for idleness every 5 minutes (ces-revisions cost guard)

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
```

Create `infra/vm/guards/ces-gpu-cap.service`:

```ini
[Unit]
Description=Power off 8 hours after boot on hosts with an NVIDIA GPU (ces-revisions cost guard)
# shutdown powers off at once when it cannot reach logind, so logind comes first.
Wants=systemd-logind.service
After=systemd-logind.service dbus.socket

[Service]
Type=oneshot
# Skip hosts without an NVIDIA PCI device (vendor 0x10de), whatever the driver's state.
ExecCondition=/bin/sh -c 'grep -qs 0x10de /sys/bus/pci/devices/*/vendor'
# nvidia-smi creates the /dev/nvidia* files that devices.py and the idle stop look for.
ExecStartPre=-/usr/bin/nvidia-smi -L
ExecStart=/usr/sbin/shutdown -h +480

[Install]
WantedBy=multi-user.target
```

Underneath:
- The timer needs `OnBootSec`, because `OnUnitActiveSec` alone never fires for a service that has not run yet.
- systemd reads `EnvironmentFile=` each time the service starts, so an edit to the window applies at the next check.
- `ExecCondition=` skips the GPU cap cleanly on `dev`.
- The `-` before `nvidia-smi` lets the cap still be scheduled if `nvidia-smi` fails.
- `shutdown -h +480` only asks logind to schedule a poweroff, and systemd 255 powers off at once when it cannot reach logind; hence `After=systemd-logind.service`.

The units cannot run on the Mac; Task 10 checks them with `systemd-analyze verify` on the VM.

- [ ] **Step 7: Write the installer**

Create `infra/vm/guards/install.sh`:

```bash
#!/usr/bin/env bash
# Install the cost guards as root: the idle stop and the GPU runtime cap. cloud-init runs
# this at first boot from the copy it wrote to /opt/ces-revisions/guards, and setup.sh
# reruns it from the checkout; rerunning it is harmless.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install -d -m 0755 /etc/ces-revisions /usr/local/lib/ces-revisions
# infra/bin/vm refuses to run where this marker exists.
touch /etc/ces-revisions/vm
install -m 0755 "$here/idle_stop.py" /usr/local/lib/ces-revisions/idle_stop.py
# Installed once, so a local edit to the window or thresholds survives a rerun.
if [ ! -e /etc/ces-revisions/idle-stop.env ]; then
  install -m 0644 "$here/idle-stop.env" /etc/ces-revisions/idle-stop.env
fi
install -m 0644 "$here/ces-idle-stop.service" "$here/ces-idle-stop.timer" \
  "$here/ces-gpu-cap.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now ces-idle-stop.timer
# Enabled, not started: the cap is scheduled at boot, so a rerun never pushes it back.
systemctl enable ces-gpu-cap.service
```

Then:

```bash
chmod +x infra/vm/guards/install.sh
bash -n infra/vm/guards/install.sh && echo "install.sh parses"
```

Expected: `install.sh parses`. Underneath: `install.sh` runs at first boot, from the copy cloud-init writes, and again at every `setup.sh` run, from the checkout. It rewrites the script and the units each time, but it installs the environment file only once, so a local change to the thresholds survives.

- [ ] **Step 8: Run the fast tier**

Run: `uv run pytest -m "not slow and not network" -q`

Expected on base `08ed203`: `343 passed, 17 deselected`.

- [ ] **Step 9: Commit**

```bash
uv run ruff format && uv run ruff check
git add infra/vm/guards/idle_stop.py infra/vm/guards/idle-stop.env \
  infra/vm/guards/ces-idle-stop.service infra/vm/guards/ces-idle-stop.timer \
  infra/vm/guards/ces-gpu-cap.service infra/vm/guards/install.sh \
  tests/test_idle_stop.py pyproject.toml
git commit -m "Add the cloud VM's idle stop and GPU runtime cap"
```

### Task 2: Operations wrapper (Req 9 and Req 6's `sync-config`)

**Files:**
- Create: `infra/bin/vm`
- Modify: `.gitignore` (one line in the OpenTofu block)
- Test: `tests/test_vm_wrapper.py`

**Interfaces:**
- Consumes: Task 5's `infra/env` outputs `instance_id` and `region`, read with `tofu -chdir=infra/env output -raw`. The tests stub them.
- Produces:
  - `infra/bin/vm start | stop | status | connect | forward | size <dev|l4|l40s|a10g|h100> [apply arguments] | sync-config [--cutover]`, exiting 2 on a usage error and 1 on the VM or a refused cutover;
  - the overrides `CES_INFRA_ENV_DIR`, `CES_VM_MARKER`, `CES_VM_SSH_HOST` (default `ces-revisions-vm`), `CES_AWS_PROFILE` (default `ces-revisions`), and `CES_TOFU_PROFILE` (default the AWS profile);
  - `infra/env/size.auto.tfvars`, written after each successful size apply;
  - on the VM, `~/.config/ces-revisions/links/skills.txt`, `agents.txt`, `commands.txt`, and `hooks.txt`, one link name per line, which Task 3's `setup.sh` reads;
  - at cutover, the Mac's memory folder for this project copied to `~/.claude/projects/-home-ubuntu-Projects-ces-revisions/memory/` on the VM.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vm_wrapper.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_vm_wrapper.py -q`

Expected: `17 failed`, each with `FileNotFoundError: [Errno 2] No such file or directory: PosixPath('…/infra/bin/vm')`.

- [ ] **Step 3: Write the wrapper**

Create `infra/bin/vm`:

```bash
#!/usr/bin/env bash
# Operate the ces-revisions cloud GPU VM from the Mac; docs/cloud-gpu-runbook.md explains
# each command. Every AWS CLI, OpenTofu, ssh, and rsync command is printed before it runs.
# Written for macOS's bash 3.2: ${@+"$@"} guards empty argument lists under set -u.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${CES_INFRA_ENV_DIR:-$REPO_ROOT/infra/env}"
VM_MARKER="${CES_VM_MARKER:-/etc/ces-revisions/vm}"
VM_HOST="${CES_VM_SSH_HOST:-ces-revisions-vm}"
AWS_PROFILE_NAME="${CES_AWS_PROFILE:-ces-revisions}"
TOFU_PROFILE_NAME="${CES_TOFU_PROFILE:-$AWS_PROFILE_NAME}"
VM_PROJECT_KEY="-home-ubuntu-Projects-ces-revisions"
LINK_SETS="skills agents commands hooks"
STAGING=""

usage() {
  cat >&2 <<'EOF'
usage: infra/bin/vm <command>
  start                       start the instance and wait until it is running
  stop                        stop the instance
  status                      show the instance's state, type, and zone
  connect                     open a Session Manager shell on the instance
  forward                     forward the instance's SSH port to localhost:2222
  size <dev|l4|l40s|a10g|h100> [ARGS]
                              switch the instance type with tofu apply (ARGS go to apply)
  sync-config [--cutover]     copy Claude Code and git settings to the VM
                              (--cutover also copies this project's memory, once)
EOF
  exit 2
}

run() {
  printf '+ %s\n' "$*" >&2
  "$@"
}

tofu_env() { run env AWS_PROFILE="$TOFU_PROFILE_NAME" tofu -chdir="$ENV_DIR" "$@"; }
output() { tofu_env output -raw "$1"; }

# The directory name Claude Code gives a project: every character other than a letter,
# digit, or hyphen becomes a hyphen.
project_key() { printf '%s' "$1" | sed 's/[^A-Za-z0-9-]/-/g'; }

# Claude Code keys a worktree's memory by its main checkout.
main_checkout() {
  dirname "$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
}

sync_config() {
  local cutover=false link_set link name target
  case "${1:-}" in
    "") ;;
    --cutover) cutover=true ;;
    *) usage ;;
  esac
  STAGING="$(mktemp -d)"
  trap 'rm -rf "$STAGING"' EXIT
  for link_set in $LINK_SETS; do
    : > "$STAGING/$link_set.txt"
    for link in "$HOME/.claude/$link_set"/*; do
      [ -e "$link" ] || [ -L "$link" ] || continue
      name="$(basename "$link")"
      target="$(readlink "$link" || true)"
      if [ "$target" = "$HOME/Projects/agent-skills/$link_set/$name" ]; then
        printf '%s\n' "$name" >> "$STAGING/$link_set.txt"
      else
        echo "skipping $link_set/$name: not a link into ~/Projects/agent-skills/$link_set" >&2
      fi
    done
  done
  run ssh "$VM_HOST" mkdir -p .claude .config/ces-revisions/links
  run rsync -a "$HOME/.claude/CLAUDE.md" "$HOME/.claude/settings.json" "$VM_HOST:.claude/"
  run rsync -a "$HOME/.gitconfig" "$VM_HOST:"
  # The Mac's gitconfig names the Mac's gh as git's credential helper; the VM's gh
  # takes that role back once it holds a token.
  run ssh "$VM_HOST" gh auth setup-git ||
    echo "gh is not signed in on the VM yet; after gh auth login there, run gh auth setup-git" >&2
  run rsync -a "$STAGING/" "$VM_HOST:.config/ces-revisions/links/"
  if [ "$cutover" = true ]; then
    local memory remote_memory
    memory="$HOME/.claude/projects/$(project_key "$(main_checkout)")/memory/"
    remote_memory=".claude/projects/$VM_PROJECT_KEY/memory/"
    if [ ! -f "${memory}MEMORY.md" ]; then
      echo "no project memory at $memory" >&2
      exit 1
    fi
    if run ssh "$VM_HOST" test -e "${remote_memory}MEMORY.md"; then
      echo "the VM already has this project's memory; the cutover copy runs once" >&2
      exit 1
    fi
    run ssh "$VM_HOST" mkdir -p "$remote_memory"
    run rsync -a "$memory" "$VM_HOST:$remote_memory"
  fi
}

if [ -e "$VM_MARKER" ]; then
  echo "infra/bin/vm runs on the Mac, never on the VM: resizing stops the instance, which" \
    "would kill an apply running on it, and the instance role has no AWS permission" \
    "beyond Systems Manager." >&2
  exit 1
fi

command="${1:-}"
[ $# -eq 0 ] || shift
case "$command" in
  start)
    id="$(output instance_id)"
    region="$(output region)"
    run aws ec2 start-instances --profile "$AWS_PROFILE_NAME" --region "$region" \
      --instance-ids "$id" --query 'StartingInstances[0].CurrentState.Name' --output text
    run aws ec2 wait instance-running --profile "$AWS_PROFILE_NAME" --region "$region" \
      --instance-ids "$id"
    ;;
  stop)
    id="$(output instance_id)"
    region="$(output region)"
    run aws ec2 stop-instances --profile "$AWS_PROFILE_NAME" --region "$region" \
      --instance-ids "$id" --query 'StoppingInstances[0].CurrentState.Name' --output text
    ;;
  status)
    id="$(output instance_id)"
    region="$(output region)"
    run aws ec2 describe-instances --profile "$AWS_PROFILE_NAME" --region "$region" \
      --instance-ids "$id" --output table \
      --query 'Reservations[0].Instances[0].{State: State.Name, Type: InstanceType, Zone: Placement.AvailabilityZone}'
    ;;
  connect)
    id="$(output instance_id)"
    region="$(output region)"
    run aws ssm start-session --profile "$AWS_PROFILE_NAME" --region "$region" --target "$id"
    ;;
  forward)
    id="$(output instance_id)"
    region="$(output region)"
    run aws ssm start-session --profile "$AWS_PROFILE_NAME" --region "$region" --target "$id" \
      --document-name AWS-StartPortForwardingSession \
      --parameters 'portNumber=22,localPortNumber=2222'
    ;;
  size)
    [ $# -ge 1 ] || usage
    size="$1"
    shift
    case "$size" in
      dev | l4 | l40s | a10g | h100) ;;
      *)
        echo "unknown size: $size (expected dev, l4, l40s, a10g, or h100)" >&2
        exit 2
        ;;
    esac
    tofu_env apply -var "size=$size" ${@+"$@"}
    # Recorded only after a successful apply, so a later plain tofu plan keeps this size.
    printf 'size = "%s"\n' "$size" > "$ENV_DIR/size.auto.tfvars"
    ;;
  sync-config) sync_config ${@+"$@"} ;;
  *) usage ;;
esac
```

Then make it executable:

```bash
chmod +x infra/bin/vm
```

Underneath:
- `run` prints each command with a `+` before running it.
- OpenTofu commands run under `env AWS_PROFILE=…`, so the printed line is the whole command.
- The wrapper reads the instance ID and region from OpenTofu's outputs, so it holds no identifier of its own.
- `size` records the size only after `tofu apply` succeeds. A failed or declined apply leaves the last good size in `size.auto.tfvars`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_vm_wrapper.py -q`

Expected: `17 passed`.

- [ ] **Step 5: Run them again under macOS's bash 3.2**

```bash
/bin/bash -n infra/bin/vm && echo "parses under bash 3.2"
PATH="/bin:$PATH" uv run pytest tests/test_vm_wrapper.py -q
```

Expected: `parses under bash 3.2` and `17 passed`. Underneath: `#!/usr/bin/env bash` runs whichever bash comes first on `PATH`, and putting `/bin` first selects macOS's bash 3.2. Under `set -u`, bash before 4.4 treats an empty `"$@"` as unbound, which `${@+"$@"}` avoids.

- [ ] **Step 6: Ignore the recorded size**

In `.gitignore`, replace:

```gitignore
backend.hcl
terraform.tfvars
```

with:

```gitignore
backend.hcl
terraform.tfvars
size.auto.tfvars
```

Then check both directions:

```bash
git check-ignore -v infra/env/size.auto.tfvars
if [ -n "$(git check-ignore infra/env/pinned.auto.tfvars infra/state/pinned.auto.tfvars infra/env/backend.hcl.example)" ]; then
  echo "STOP: a file this plan commits is ignored"
else
  echo "pinned.auto.tfvars and backend.hcl.example stay tracked"
fi
```

Expected: one line naming `.gitignore`'s `size.auto.tfvars` rule and
`infra/env/size.auto.tfvars`, then `pinned.auto.tfvars and backend.hcl.example stay tracked`. Do
not assert a `.gitignore` line number: Stages 2–4 added rules before the cloud block. Underneath:
`git check-ignore` prints only the paths it ignores. With `-q` it refuses more than one path, which
is why this check reads its output instead.

- [ ] **Step 7: Run the fast tier**

Run: `uv run pytest -m "not slow and not network" -q`

Expected after the capacity amendment: `360 passed, 17 deselected`.

- [ ] **Step 8: Commit**

```bash
uv run ruff format && uv run ruff check
git add infra/bin/vm tests/test_vm_wrapper.py .gitignore
git commit -m "Add infra/bin/vm to operate the cloud VM from the Mac"
```

### Task 3: VM first-boot and setup scripts (Req 6)

**Files:**
- Create: `infra/vm/setup.sh`
- Create: `infra/vm/first-boot.sh`
- Test: `tests/test_vm_setup.py`

**Interfaces:**
- Consumes:
  - Task 1's `infra/vm/guards/install.sh`;
  - Task 2's link lists in `~/.config/ces-revisions/links/`;
  - plan 3's `required-version = "~=0.12.13"` line and `cuda` extra in `pyproject.toml`.
- Produces:
  - `setup.sh`, run as `ubuntu`. `CES_REVISIONS_BRANCH` names the branch for a first clone. The script creates `~/Projects/ces-revisions`, `~/Projects/agent-skills`, `~/.local/bin/uv`, and the links in `~/.claude/skills`, `agents`, `commands`, and `hooks`.
  - `first-boot.sh`, run once as root by cloud-init. Task 5 embeds it at `/opt/ces-revisions/first-boot.sh`, beside the guards in `/opt/ces-revisions/guards/`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vm_setup.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_vm_setup.py -q`

Expected: `4 failed`. bash cannot open the missing script and exits 127, where the tests expect 1 or 0.

- [ ] **Step 3: Write the setup script**

Create `infra/vm/setup.sh`:

```bash
#!/usr/bin/env bash
# Set up the ubuntu user's environment on the ces-revisions VM; docs/cloud-gpu-runbook.md
# explains each step. Idempotent: rerun it after pulling changes. Run it as ubuntu, after
# infra/bin/vm sync-config has sent the link names. The first run is piped from the Mac,
# with CES_REVISIONS_BRANCH naming the branch to clone; later runs use the checkout's copy,
# and an existing clone keeps whatever branch it has.
set -euo pipefail

PROJECTS="$HOME/Projects"
REPO="$PROJECTS/ces-revisions"
AGENT_SKILLS="$PROJECTS/agent-skills"
LINKS="$HOME/.config/ces-revisions/links"

step() { printf '\n== %s\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then
  echo "run setup.sh as ubuntu, not as root" >&2
  exit 1
fi
if [ ! -d "$LINKS" ]; then
  echo "no link names in $LINKS: run infra/bin/vm sync-config on the Mac first" >&2
  exit 1
fi

step "clone the repositories into $PROJECTS"
mkdir -p "$PROJECTS"
if [ ! -d "$REPO/.git" ]; then
  if [ -n "${CES_REVISIONS_BRANCH:-}" ]; then
    git clone --branch "$CES_REVISIONS_BRANCH" https://github.com/lowmason/ces-revisions.git "$REPO"
  else
    git clone https://github.com/lowmason/ces-revisions.git "$REPO"
  fi
fi
if [ ! -d "$AGENT_SKILLS/.git" ]; then
  git clone https://github.com/lowmason/agent-skills.git "$AGENT_SKILLS"
fi

step "install uv from the series that pyproject.toml pins"
uv_version="$(sed -n 's/^required-version = "~=\([0-9.]*\)"$/\1/p' "$REPO/pyproject.toml")"
if [ -z "$uv_version" ]; then
  echo "no required-version = \"~=X.Y.Z\" line in $REPO/pyproject.toml" >&2
  exit 1
fi
export PATH="$HOME/.local/bin:$PATH"
if ! uv --version 2>/dev/null | grep -q "^uv ${uv_version%.*}\."; then
  curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" | sh
fi

step "install Python 3.14 and sync the environment with the cuda extra"
uv python install 3.14
(cd "$REPO" && uv sync --locked --extra cuda)

step "link the personal skills, agents, commands, and hooks into ~/.claude"
for link_set in skills agents commands hooks; do
  if [ ! -f "$LINKS/$link_set.txt" ]; then
    echo "no $LINKS/$link_set.txt: rerun infra/bin/vm sync-config on the Mac" >&2
    exit 1
  fi
  mkdir -p "$HOME/.claude/$link_set"
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    target="$AGENT_SKILLS/$link_set/$name"
    if [ -e "$target" ]; then
      ln -sfn "$target" "$HOME/.claude/$link_set/$name"
    else
      echo "skipping $link_set/$name: agent-skills has no such entry" >&2
    fi
  done < "$LINKS/$link_set.txt"
  echo "$link_set: $(find "$HOME/.claude/$link_set" -maxdepth 1 -type l | wc -l | tr -d ' ') links"
done

step "reinstall the cost guards from the checkout"
sudo "$REPO/infra/vm/guards/install.sh"
```

Then make it executable:

```bash
chmod +x infra/vm/setup.sh
```

Underneath:
- Each step checks before it acts: a clone that exists is kept, and uv is installed only when the pinned series is missing.
- `uv sync --locked` fails rather than rewrite `uv.lock`, so the VM runs exactly the locked packages.
- `ln -sfn` replaces a link in place instead of following it into its target directory.
- The last step reinstalls the guards from the checkout, so a guard change on the branch reaches the VM with `git pull` and `setup.sh`, even though cloud-init never runs again.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_vm_setup.py -q`

Expected: `4 passed`. Underneath: the tests give the script a bare `PATH`, so the real git, curl, and uv stay out of reach and stubs record each call.

- [ ] **Step 5: Write the first-boot script**

Create `infra/vm/first-boot.sh`:

```bash
#!/usr/bin/env bash
# The VM's first boot, run once as root by cloud-init (infra/env/instance.tf embeds this
# file): install the cost guards first, so they run even if a package step fails; then
# install git, Ubuntu's 580 server driver with NVIDIA's open kernel modules, and gh; and
# hold the driver and the AWS kernel.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

/opt/ces-revisions/guards/install.sh

# The modules are prebuilt for one AWS kernel, so installing them may add a newer kernel,
# which runs from the next start. The kernel is held with the driver so an upgrade never
# leaves a kernel without its modules.
driver_packages="nvidia-headless-no-dkms-580-server-open nvidia-utils-580-server linux-modules-nvidia-580-server-open-aws"
apt-get update
# shellcheck disable=SC2086 # the package list splits on spaces
apt-get install -y --no-install-recommends git $driver_packages
# shellcheck disable=SC2086
apt-mark hold $driver_packages
for package in linux-aws linux-image-aws linux-headers-aws; do
  if dpkg-query -W -f='${db:Status-Status}' "$package" 2> /dev/null | grep -qx installed; then
    apt-mark hold "$package"
  fi
done

# gh from GitHub's apt repository: Ubuntu's own gh 2.45 is too old for GitHub's API.
install -d -m 0755 /etc/apt/keyrings
curl -fsSL -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  https://cli.github.com/packages/githubcli-archive-keyring.gpg
chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  > /etc/apt/sources.list.d/github-cli.list
apt-get update
apt-get install -y gh
```

Then:

```bash
chmod +x infra/vm/first-boot.sh
bash -n infra/vm/first-boot.sh && echo "first-boot.sh parses"
```

Expected: `first-boot.sh parses`. Underneath:
- cloud-init runs `runcmd` in its final stage, after its own package stage, as root and once per instance.
- The guards come first, so a package step that fails cannot leave the VM without its idle stop.
- `apt-mark hold` stops unattended upgrades from moving the driver, its modules, or the AWS kernel, which keeps the environment fixed between measurements. Of the kernel packages, only those the image has installed are held.
- If a step fails, cloud-init reports `status: error`, and Task 10 shows the log. Every step can safely run again with `sudo /opt/ces-revisions/first-boot.sh`.

- [ ] **Step 6: Run the fast tier**

Run: `uv run pytest -m "not slow and not network" -q`

Expected after the capacity amendment: `364 passed, 17 deselected`.

- [ ] **Step 7: Commit**

```bash
uv run ruff format && uv run ruff check
git add infra/vm/setup.sh infra/vm/first-boot.sh tests/test_vm_setup.py
git commit -m "Add the cloud VM's first-boot and setup scripts"
```

### Task 4: The probe's driver check on CPU sizes (Req 7 on the VM)

**Files:**
- Modify: `src/ces_revisions/engine_probe.py` (its imports and `_nvidia_driver`)
- Test: `tests/test_engine_probe.py` (its imports, one assertion, and two new tests)

**Interfaces:**
- Consumes: plan 3's `has_nvidia_device() -> bool` in `src/ces_revisions/devices.py`, and its `_nvidia_driver() -> str | None` in `src/ces_revisions/engine_probe.py`.
- Produces: `_nvidia_driver()`, which returns `None` wherever `has_nvidia_device()` is false and otherwise asks `nvidia-smi`. Task 11's probe on `dev` then records `"nvidia_driver": null` instead of stopping.

- [ ] **Step 1: Write the failing tests**

Make three edits to `tests/test_engine_probe.py`.

First, replace:

```python
import json
import shutil
import subprocess
```

with:

```python
import json
import os
import subprocess
```

and replace:

```python
from ces_revisions.engine_probe import synthetic_problem
```

with:

```python
from ces_revisions import engine_probe
from ces_revisions.devices import has_nvidia_device
from ces_revisions.engine_probe import synthetic_problem
```

Second, in `test_probe_module_writes_a_timing_record`, replace:

```text
    assert (record["nvidia_driver"] is None) == (shutil.which("nvidia-smi") is None)
```

with:

```text
    assert (record["nvidia_driver"] is None) == (not has_nvidia_device())
```

The assertion now follows `has_nvidia_device()` instead of `shutil.which("nvidia-smi")`, so it also holds on `dev`, where `nvidia-smi` is installed but no GPU exists.

Third, append to the end of the file:

```python
def _stub_nvidia_smi(monkeypatch, tmp_path, body):
    stub = tmp_path / "nvidia-smi"
    stub.write_text(f"#!/bin/sh\n{body}\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def test_the_driver_is_not_asked_for_without_an_nvidia_device(monkeypatch, tmp_path):
    # A CPU size can have the driver's nvidia-smi, which fails there without a GPU.
    _stub_nvidia_smi(monkeypatch, tmp_path, "exit 9")
    monkeypatch.setattr(engine_probe, "has_nvidia_device", lambda: False)

    assert engine_probe._nvidia_driver() is None


def test_the_driver_version_comes_from_nvidia_smi_on_an_nvidia_host(
    monkeypatch, tmp_path
):
    _stub_nvidia_smi(monkeypatch, tmp_path, "echo 580.173.02")
    monkeypatch.setattr(engine_probe, "has_nvidia_device", lambda: True)

    assert engine_probe._nvidia_driver() == "580.173.02"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_probe.py -q`

Expected: `2 failed, 3 passed`. Both failures raise `AttributeError`, because `ces_revisions.engine_probe` has no attribute `has_nvidia_device` yet.

- [ ] **Step 3: Ask for the driver version only on NVIDIA hosts**

In `src/ces_revisions/engine_probe.py`, replace:

```python
import importlib.metadata
import json
import shutil
import subprocess
```

with:

```python
import importlib.metadata
import json
import subprocess
```

Then replace:

```python
from ces_revisions.kalman import LinearGaussianSSM, kalman_filter
```

with:

```python
from ces_revisions.devices import has_nvidia_device
from ces_revisions.kalman import LinearGaussianSSM, kalman_filter
```

Then replace:

```python
def _nvidia_driver() -> str | None:
    """The NVIDIA driver version, or None on a host without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
```

with:

```python
def _nvidia_driver() -> str | None:
    """The NVIDIA driver version, or None on a host without an NVIDIA device."""
    # A CPU size can have the driver's nvidia-smi, which fails there without a GPU.
    if not has_nvidia_device():
        return None
```

Underneath: `has_nvidia_device()` checks for `/dev/nvidia0`, the same test that `tests/conftest.py` and the idle stop use. On `dev`, first boot installs the driver's `nvidia-smi`, and it exits non-zero because no GPU exists, which `check=True` would turn into a `CalledProcessError` that ends the probe.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine_probe.py -q`

Expected: `5 passed`.

- [ ] **Step 5: Run the fast tier**

Run: `uv run pytest -m "not slow and not network" -q`

Expected after the capacity amendment: `366 passed, 17 deselected`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check
git add src/ces_revisions/engine_probe.py tests/test_engine_probe.py
git commit -m "Ask nvidia-smi for the driver version only where an NVIDIA device exists"
```

### Task 5: OpenTofu roots (Reqs 3–5 and 8)

**Files:**
- Create: `infra/state/main.tf`
- Create: `infra/env/versions.tf`, `infra/env/variables.tf`, `infra/env/network.tf`, `infra/env/instance.tf`, `infra/env/budget.tf`, `infra/env/snapshots.tf`, `infra/env/outputs.tf`
- Create: `infra/env/backend.hcl.example`
- Create (generated by `tofu init`): `infra/state/.terraform.lock.hcl`, `infra/env/.terraform.lock.hcl`

**Interfaces:**
- Consumes: Task 1's guard files and Task 3's `first-boot.sh`, which `instance.tf` embeds from `infra/vm/`.
- Produces:
  - `infra/state`: variable `region`; output `bucket`.
  - `infra/env` variables:
    - `region`, `availability_zone`, and `ami_id` (Task 6 pins them);
    - `size` (default `dev`) and `ssh_public_key_path` (default `~/.ssh/ces-revisions-vm.pub`);
    - `budget_email` (sensitive, no default), `budget_enabled` (default `false`), and `monthly_budget_usd` (default `"150"`).
  - `infra/env` outputs: `instance_id`, `region`, `availability_zone`, `instance_type`, `root_volume_id`, `security_group_id`, `budget_name`, `budget_action_id`.
  - Names later tasks query: the IAM roles `ces-revisions-vm`, `ces-revisions-snapshots`, and `ces-revisions-budget-stop`; the budget `ces-revisions-monthly`; and the security group `ces-revisions-vm`.

- [ ] **Step 1: Write the state root**

Create `infra/state/main.tf`:

```hcl
# The S3 bucket that holds infra/env's state. This root's own state stays local, in the
# gitignored terraform.tfstate beside this file.

terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.64"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project = "ces-revisions"
    }
  }
}

variable "region" {
  description = "The region Req 2 chose, pinned in pinned.auto.tfvars."
  type        = string
}

resource "aws_s3_bucket" "state" {
  # A prefix rather than a name: AWS appends a unique suffix, so no bucket name is committed.
  bucket_prefix = "ces-revisions-tofu-state-"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "bucket" {
  description = "The state bucket's generated name, for the gitignored infra/env/backend.hcl."
  value       = aws_s3_bucket.state.bucket
  # Kept out of plan and apply output; tofu output -raw bucket still prints it.
  sensitive = true
}
```

Underneath: this root keeps its own state in a local file, because the bucket that would hold it is what it creates. The bucket's versioning keeps every earlier state of `infra/env`, which is the recovery path for a bad apply.

- [ ] **Step 2: Write the environment's provider, backend, and inputs**

Create `infra/env/versions.tf`:

```hcl
terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.64"
    }
  }

  # Bucket, key, and region come from the gitignored backend.hcl (backend.hcl.example shows
  # its shape): tofu init -backend-config=backend.hcl. The lock is an S3 object written with
  # a conditional put, so no DynamoDB table exists.
  backend "s3" {
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project = "ces-revisions"
    }
  }
}

# The account the trust policies name, read at plan time so no account ID is committed.
data "aws_caller_identity" "current" {}
```

Create `infra/env/variables.tf`:

```hcl
variable "region" {
  description = "The region Req 2 chose, pinned in pinned.auto.tfvars."
  type        = string
}

variable "availability_zone" {
  description = "The zone Req 2 chose, pinned in pinned.auto.tfvars."
  type        = string
}

variable "ami_id" {
  description = "Canonical's Ubuntu 24.04 LTS amd64 image, read once from its public SSM parameter and pinned in pinned.auto.tfvars."
  type        = string
}

variable "size" {
  description = "dev (m7i.xlarge), l4 (g6.xlarge), l40s (g6e.xlarge), a10g (g5.xlarge), or h100 (p5.4xlarge). infra/bin/vm size records the last applied size in the gitignored size.auto.tfvars."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "l4", "l40s", "a10g", "h100"], var.size)
    error_message = "The size must be dev, l4, l40s, a10g, or h100."
  }
}

variable "ssh_public_key_path" {
  description = "The public half of the VM's dedicated SSH key pair; the private half stays on the Mac."
  type        = string
  default     = "~/.ssh/ces-revisions-vm.pub"
}

variable "budget_email" {
  description = "Where AWS Budgets sends alerts. A personal value, so it lives in the gitignored terraform.tfvars."
  type        = string
  sensitive   = true
}

variable "budget_enabled" {
  description = "Whether the budget and its stop action exist. A budget filters on an activated cost allocation tag, and the project tag can be activated only after resources carry it, so this starts false."
  type        = bool
  default     = false
}

variable "monthly_budget_usd" {
  description = "The monthly cost ceiling in US dollars, revisited at roadmap Stage 6."
  type        = string
  default     = "150"
}
```

Underneath: `backend "s3"` holds only `use_lockfile`, and `tofu init -backend-config=backend.hcl` merges in the rest, which is what keeps the bucket name out of git. `default_tags` stamps `project = ces-revisions` on every resource that accepts tags, including the root volume at launch.

- [ ] **Step 3: Write the network**

Create `infra/env/network.tf`:

```hcl
# One public subnet and no inbound rules. The instance's public IPv4 address carries only
# outbound traffic; Session Manager reaches the instance over a connection its agent opens.

resource "aws_vpc" "main" {
  cidr_block = "10.42.0.0/16"

  tags = {
    Name = "ces-revisions"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true

  tags = {
    Name = "ces-revisions-public"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "ces-revisions"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "ces-revisions-public"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "vm" {
  name        = "ces-revisions-vm"
  description = "No ingress: Session Manager connects over the agent's outbound connection"
  vpc_id      = aws_vpc.main.id

  tags = {
    Name = "ces-revisions-vm"
  }
}

# OpenTofu removes the allow-all egress rule AWS gives a new group, so it is declared here.
resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.vm.id
  description       = "All outbound IPv4: packages, GitHub, PyPI, and Systems Manager"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}
```

- [ ] **Step 4: Write the instance**

Create `infra/env/instance.tf`:

```hcl
# The VM: one instance on one root volume, whose type follows var.size.

locals {
  instance_types = {
    dev  = "m7i.xlarge"
    l4   = "g6.xlarge"
    l40s = "g6e.xlarge"
    a10g = "g5.xlarge"
    h100 = "p5.4xlarge"
  }

  # The files cloud-init writes under /opt/ces-revisions at first boot, with their modes,
  # named by their paths under infra/vm.
  vm_files = {
    "first-boot.sh"                = "0755"
    "guards/install.sh"            = "0755"
    "guards/idle_stop.py"          = "0755"
    "guards/idle-stop.env"         = "0644"
    "guards/ces-idle-stop.service" = "0644"
    "guards/ces-idle-stop.timer"   = "0644"
    "guards/ces-gpu-cap.service"   = "0644"
  }

  # cloud-init applies this once per instance: the SSH key for ubuntu, the files above, and
  # first-boot.sh, which runs after cloud-init's own package stage.
  cloud_config = {
    ssh_authorized_keys = [trimspace(file(pathexpand(var.ssh_public_key_path)))]
    write_files = [
      for name, permissions in local.vm_files : {
        path        = "/opt/ces-revisions/${name}"
        permissions = permissions
        content     = file("${path.module}/../vm/${name}")
      }
    ]
    runcmd = [["/opt/ces-revisions/first-boot.sh"]]
  }

  user_data = "#cloud-config\n${yamlencode(local.cloud_config)}"
}

resource "aws_iam_role" "vm" {
  name = "ces-revisions-vm"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

# The instance's only AWS permission: registering with Systems Manager.
resource "aws_iam_role_policy_attachment" "vm_ssm" {
  role       = aws_iam_role.vm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "vm" {
  name = "ces-revisions-vm"
  role = aws_iam_role.vm.name
}

resource "aws_instance" "vm" {
  ami                                  = var.ami_id
  instance_type                        = local.instance_types[var.size]
  subnet_id                            = aws_subnet.public.id
  vpc_security_group_ids               = [aws_security_group.vm.id]
  iam_instance_profile                 = aws_iam_instance_profile.vm.name
  instance_initiated_shutdown_behavior = "stop"
  user_data                            = local.user_data

  metadata_options {
    http_tokens = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 100
    encrypted             = true
    delete_on_termination = true
  }

  tags = {
    Name = "ces-revisions-vm"
  }

  lifecycle {
    prevent_destroy = true
    # cloud-init runs once per instance; later changes reach the VM through setup.sh.
    ignore_changes = [user_data]

    precondition {
      condition     = length(local.user_data) <= 16384
      error_message = "EC2 accepts at most 16 KB of user data."
    }
  }
}
```

Underneath:
- `yamlencode` turns the `cloud_config` object into the YAML cloud-init reads, and every guard file's text is read from `infra/vm/` at plan time.
- `ignore_changes = [user_data]` keeps later edits to those files from asking to stop the instance. The edits reach the VM through `setup.sh` instead.
- `prevent_destroy` turns any plan that would replace the instance into an error.

- [ ] **Step 5: Write the budget and the snapshots**

Create `infra/env/budget.tf`:

```hcl
# A monthly cost budget over the costs tagged project = ces-revisions. At 100% of actual
# spend, a budget action stops the instance through a role that can do only that. All four
# resources wait for budget_enabled, which follows the tag's activation.

resource "aws_budgets_budget" "monthly" {
  count = var.budget_enabled ? 1 : 0

  name         = "ces-revisions-monthly"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Budgets name a user-defined cost allocation tag as user:<key>$<value>.
  cost_filter {
    name   = "TagKeyValue"
    values = ["user:project$ces-revisions"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}

resource "aws_iam_role" "budget_action" {
  count = var.budget_enabled ? 1 : 0

  name = "ces-revisions-budget-stop"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "budgets.amazonaws.com" }
      Condition = {
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
        ArnLike      = { "aws:SourceArn" = "arn:aws:budgets::${data.aws_caller_identity.current.account_id}:budget/*" }
      }
    }]
  })
}

# The EC2 stop half of AWS's AWSBudgetsActions_RolePolicyForResourceAdministrationWithSSM,
# narrowed to this instance.
resource "aws_iam_role_policy" "budget_action" {
  count = var.budget_enabled ? 1 : 0

  name = "stop-the-vm"
  role = aws_iam_role.budget_action[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "RunTheStopAutomation"
        Effect = "Allow"
        Action = "ssm:StartAutomationExecution"
        Resource = [
          "arn:aws:ssm:*:*:document/AWS-StopEC2Instance",
          "arn:aws:ssm:*:*:automation-definition/AWS-StopEC2Instance:*",
          "arn:aws:ssm:*:*:automation-execution/*",
        ]
      },
      {
        Sid       = "StopTheVm"
        Effect    = "Allow"
        Action    = "ec2:StopInstances"
        Resource  = aws_instance.vm.arn
        Condition = { "ForAnyValue:StringEquals" = { "aws:CalledVia" = ["ssm.amazonaws.com"] } }
      },
      {
        Sid       = "ReadInstanceStatus"
        Effect    = "Allow"
        Action    = "ec2:DescribeInstanceStatus"
        Resource  = "*"
        Condition = { "ForAnyValue:StringEquals" = { "aws:CalledVia" = ["ssm.amazonaws.com"] } }
      },
    ]
  })
}

resource "aws_budgets_budget_action" "stop_vm" {
  count = var.budget_enabled ? 1 : 0

  budget_name        = aws_budgets_budget.monthly[0].name
  action_type        = "RUN_SSM_DOCUMENTS"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"
  execution_role_arn = aws_iam_role.budget_action[0].arn

  action_threshold {
    action_threshold_type  = "PERCENTAGE"
    action_threshold_value = 100
  }

  definition {
    ssm_action_definition {
      action_sub_type = "STOP_EC2_INSTANCES"
      region          = var.region
      instance_ids    = [aws_instance.vm.id]
    }
  }

  subscriber {
    address           = var.budget_email
    subscription_type = "EMAIL"
  }
}
```

Create `infra/env/snapshots.tf`:

```hcl
# Daily snapshots of every volume tagged project = ces-revisions, which default_tags puts on
# the root volume at launch. Seven are kept.

resource "aws_iam_role" "snapshots" {
  name = "ces-revisions-snapshots"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "dlm.amazonaws.com" }
      Condition = {
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "snapshots" {
  role       = aws_iam_role.snapshots.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
}

resource "aws_dlm_lifecycle_policy" "daily" {
  # Data Lifecycle Manager allows only letters, digits, spaces, hyphens, and underscores here.
  description        = "ces-revisions daily root volume snapshots keeping 7"
  execution_role_arn = aws_iam_role.snapshots.arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]

    target_tags = {
      project = "ces-revisions"
    }

    schedule {
      name      = "daily-keep-7"
      copy_tags = true

      create_rule {
        interval      = 24
        interval_unit = "HOURS"
        times         = ["05:00"]
      }

      retain_rule {
        count = 7
      }
    }
  }
}
```

Underneath:
- `count = var.budget_enabled ? 1 : 0` creates none of the budget's resources until Task 12 sets `budget_enabled = true`.
- A budget action runs the `AWS-StopEC2Instance` automation, which calls EC2 with the action role's credentials. That call carries `aws:CalledVia = ssm.amazonaws.com`, and the policy requires it.
- DLM's `target_tags` match EC2 tags, not cost allocation tags, so snapshots need no activation.

- [ ] **Step 6: Write the outputs and the backend example**

Create `infra/env/outputs.tf`:

```hcl
output "instance_id" {
  description = "The instance, which keeps this ID across sizes."
  value       = aws_instance.vm.id
}

output "region" {
  description = "The region, for AWS CLI commands."
  value       = var.region
}

output "availability_zone" {
  description = "The instance's zone."
  value       = aws_instance.vm.availability_zone
}

output "instance_type" {
  description = "The instance type the current size maps to."
  value       = aws_instance.vm.instance_type
}

output "root_volume_id" {
  description = "The root volume, which keeps this ID across sizes."
  value       = aws_instance.vm.root_block_device[0].volume_id
}

output "security_group_id" {
  description = "The instance's security group, which has no ingress rules."
  value       = aws_security_group.vm.id
}

output "budget_name" {
  description = "The monthly cost budget; null until budget_enabled is true."
  value       = one(aws_budgets_budget.monthly[*].name)
}

output "budget_action_id" {
  description = "The budget action that stops the instance, for resetting it; null until budget_enabled is true."
  value       = one(aws_budgets_budget_action.stop_vm[*].action_id)
}
```

Create `infra/env/backend.hcl.example`:

```hcl
# Partial backend configuration for infra/env. Copy this file to backend.hcl, which git
# ignores, and set bucket to the output of: tofu -chdir=infra/state output -raw bucket
bucket = "ces-revisions-tofu-state-example"
key    = "env/terraform.tfstate"
region = "us-east-1"
```

- [ ] **Step 7: Format the HCL**

```bash
tofu fmt -recursive infra
tofu fmt -check -recursive infra && echo "HCL formatted"
```

Expected: the first command lists any file it rewrote, and the second prints `HCL formatted`. This HCL was written without `tofu`, so a rewrite of whitespace is expected. `tofu fmt` changes layout only.

- [ ] **Step 8: Initialize each root without AWS, and validate it**

```bash
tofu -chdir=infra/state init -input=false
tofu -chdir=infra/state validate
tofu -chdir=infra/env init -backend=false -input=false
tofu -chdir=infra/env validate
```

Expected:
- `OpenTofu has been successfully initialized!` twice, each after downloading `hashicorp/aws` 6.64 or a later 6.x and writing `.terraform.lock.hcl`;
- `Success! The configuration is valid.` twice.

If `validate` reports an error, change only what its message names, keeping every resource and argument the spec requires, and add a `> Deviation:` note to this step. Underneath: `-backend=false` installs the provider without reaching the S3 backend, which does not exist until Task 7. `validate` checks the configuration against the provider's schema and treats variables as unknown, so it needs neither credentials nor `pinned.auto.tfvars`.

- [ ] **Step 9: Check the lock files and the caches**

```bash
git check-ignore -v infra/state/.terraform infra/env/.terraform
if [ -n "$(git check-ignore infra/state/.terraform.lock.hcl infra/env/.terraform.lock.hcl)" ]; then
  echo "STOP: a provider lock file is ignored"
else
  echo "lock files stay tracked"
fi
grep -A1 'provider "registry.opentofu.org/hashicorp/aws"' infra/env/.terraform.lock.hcl
```

Expected: two lines matching `.terraform/`, then `lock files stay tracked`, then the provider line followed by `version = "6.N.M"` with N of 64 or more.

- [ ] **Step 10: Commit**

```bash
uv run ruff format && uv run ruff check
git add infra/state/main.tf infra/state/.terraform.lock.hcl \
  infra/env/versions.tf infra/env/variables.tf infra/env/network.tf \
  infra/env/instance.tf infra/env/budget.tf infra/env/snapshots.tf \
  infra/env/outputs.tf infra/env/backend.hcl.example infra/env/.terraform.lock.hcl
git commit -m "Add the OpenTofu roots for the state bucket and the cloud environment"
```

### Task 6: Location, image, and SSH key (Reqs 4 and 5)

**Mode:** the controller runs this task inline. It makes read-only AWS calls, and Step 5 is the human partner's.

**Files:**
- Create: `infra/state/pinned.auto.tfvars`
- Create: `infra/env/pinned.auto.tfvars`
- Create: `docs/decisions/cloud-gpu-evidence/ssm-get-parameters-ubuntu-24.04-<region>.json`
- Create: `docs/decisions/cloud-gpu-evidence/ec2-describe-images-<region>.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (its opening paragraphs; append a section)

**Interfaces:**
- Consumes: `.chosen.region` and `.chosen.zone` in plan 3's `zone-choice.json`, and the `ces-revisions` profile.
- Produces:
  - `region` in `infra/state/pinned.auto.tfvars`, and `region`, `availability_zone`, and `ami_id` in `infra/env/pinned.auto.tfvars`;
  - the key pair `~/.ssh/ces-revisions-vm` and `~/.ssh/ces-revisions-vm.pub` on the Mac, which `ssh_public_key_path` names by default;
  - the evidence scan in Step 7, which every later evidence commit runs.

- [ ] **Step 1: Pin the state root's region**

```bash
REGION=$(jq -r .chosen.region docs/decisions/cloud-gpu-evidence/zone-choice.json)
ZONE=$(jq -r .chosen.zone docs/decisions/cloud-gpu-evidence/zone-choice.json)
echo "region $REGION, zone $ZONE"
cat > infra/state/pinned.auto.tfvars <<EOF
# Pinned by plan 4 from docs/decisions/cloud-gpu-evidence/zone-choice.json.
region = "$REGION"
EOF
cat infra/state/pinned.auto.tfvars
```

Expected: the chosen region and zone, then the file with the region. Underneath: OpenTofu loads every `*.auto.tfvars` file in a root without being told, so the committed file supplies the variable, and no command needs a `-var`.

- [ ] **Step 2: Read the image ID from Canonical's parameter**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r .chosen.region "$EVIDENCE/zone-choice.json")
OUT="$EVIDENCE/ssm-get-parameters-ubuntu-24.04-$REGION.json"
for NAME in \
  /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  /aws/service/canonical/ubuntu/server/noble/stable/current/amd64/hvm/ebs-gp3/ami-id; do
  aws ssm get-parameters --region "$REGION" --names "$NAME" --output json \
    --query '{Parameters: Parameters[].{Name: Name, Value: Value, Version: Version, LastModifiedDate: LastModifiedDate}, InvalidParameters: InvalidParameters}' \
    > "$OUT"
  if jq -e '.Parameters | length == 1' "$OUT" > /dev/null; then break; fi
done
if jq -e '.Parameters | length == 1' "$OUT" > /dev/null; then
  jq -r '.Parameters[0] | "\(.Name) \(.Value)"' "$OUT"
else
  echo "STOP: neither Canonical parameter resolved in $REGION"
fi
```

Expected: one line with the parameter's name and an `ami-…` ID. Underneath: Canonical publishes the current image for each release under both its version number and its codename. The `current` alias moves whenever Canonical rebuilds the image, which is why the ID is copied into a committed file instead of being looked up at plan time. `--query` leaves out the parameter's ARN.

- [ ] **Step 3: Describe the image**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r .chosen.region "$EVIDENCE/zone-choice.json")
AMI=$(jq -r '.Parameters[0].Value' "$EVIDENCE/ssm-get-parameters-ubuntu-24.04-$REGION.json")
aws ec2 describe-images --region "$REGION" --image-ids "$AMI" --output json \
  --query 'Images[0].{ImageId: ImageId, Name: Name, CreationDate: CreationDate, Architecture: Architecture, RootDeviceType: RootDeviceType, VirtualizationType: VirtualizationType, BootMode: BootMode}' \
  > "$EVIDENCE/ec2-describe-images-$REGION.json"
jq -e '(.Name | startswith("ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-")) and .Architecture == "x86_64" and .RootDeviceType == "ebs"' \
  "$EVIDENCE/ec2-describe-images-$REGION.json"
jq -r .Name "$EVIDENCE/ec2-describe-images-$REGION.json"
```

Expected: `true`, then the image name, which ends in its build date. `--query` leaves out `OwnerId`, which is Canonical's account ID.

- [ ] **Step 4: Pin the environment's location and image**

```bash
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r .chosen.region "$EVIDENCE/zone-choice.json")
ZONE=$(jq -r .chosen.zone "$EVIDENCE/zone-choice.json")
AMI=$(jq -r .ImageId "$EVIDENCE/ec2-describe-images-$REGION.json")
cat > infra/env/pinned.auto.tfvars <<EOF
# Pinned by plan 4 from committed evidence; change these only on purpose (docs/cloud-gpu-runbook.md).
# region and availability_zone: docs/decisions/cloud-gpu-evidence/zone-choice.json
# ami_id: docs/decisions/cloud-gpu-evidence/ec2-describe-images-$REGION.json
region            = "$REGION"
availability_zone = "$ZONE"
ami_id            = "$AMI"
EOF
tofu fmt -check infra/env/pinned.auto.tfvars infra/state/pinned.auto.tfvars && echo "tfvars formatted"
cat infra/env/pinned.auto.tfvars
```

Expected: `tfvars formatted`, then the file. If `tofu fmt -check` names a file, run `tofu fmt` on it, which only realigns the `=` signs.

- [ ] **Step 5: STOP — the human partner creates the SSH key pair**

Ask your human partner to run this in their own terminal:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/ces-revisions-vm -C ces-revisions-vm
```

The passphrase is their choice. The key reaches the VM only through Session Manager, which also needs their `aws login` session. If they set a passphrase, they also run `ssh-add --apple-use-keychain ~/.ssh/ces-revisions-vm`, so this plan's non-interactive SSH commands can use the key. Wait until they confirm. Underneath: a key pair is a credential, so creating it is the human partner's step. OpenTofu reads only the public half.

- [ ] **Step 6: Check the key**

```bash
ssh-keygen -lf ~/.ssh/ces-revisions-vm.pub
stat -f '%Lp' ~/.ssh/ces-revisions-vm
```

Expected: `256 SHA256:… ces-revisions-vm (ED25519)`, then `600`.

- [ ] **Step 7: Update the evidence README, scan the evidence, and commit**

In `docs/decisions/cloud-gpu-evidence/README.md`, replace:

````markdown
Command outputs recorded by Plan 3, the first implementation plan for [`specs/cloud-gpu-environment.md`](../../../specs/cloud-gpu-environment.md), for its Reqs 1 and 2. The decision record `docs/decisions/cloud-gpu.md`, which Plan 4 writes, cites them. Every `aws` command ran as the IAM user through the `ces-revisions` profile.
````

with:

````markdown
Command outputs recorded by plans 3 and 4, the two plans of [`specs/cloud-gpu-environment.md`](../../../specs/cloud-gpu-environment.md), for its Reqs 1–6 and 8. The decision record `docs/decisions/cloud-gpu.md`, which plan 4 writes, cites them. Every `aws` command ran as the IAM user through the `ces-revisions` profile, and every command on the VM ran through the `ces-revisions-vm` SSH host entry.
````

Then replace:

````markdown
Before each commit, this directory was searched for the account ID, for ARNs, and for at signs.
````

with:

````markdown
Before each commit, this directory and `../cloud-gpu-probe/` were searched for the account ID, the state bucket's name, ARNs, and at signs.
````

Then append:

````markdown

## Reqs 4 and 5: image

- `ssm-get-parameters-ubuntu-24.04-<region>.json` — `aws ssm get-parameters` for Canonical's parameter that names the current Ubuntu 24.04 LTS amd64 gp3 image, narrowed to its name, value, version, and date. The value is the image pinned in `infra/env/pinned.auto.tfvars`.
- `ec2-describe-images-<region>.json` — `aws ec2 describe-images` for that image, narrowed to its ID, name, creation date, architecture, root device type, virtualization type, and boot mode. The owner's account ID is left out.
````

Scan the evidence. This is the evidence scan that every later evidence commit runs:

```bash
export AWS_PROFILE=ces-revisions
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET=$(tofu -chdir=infra/state output -raw bucket 2> /dev/null)
if [ -z "$ACCOUNT_ID" ]; then
  echo "STOP: no account ID (expired credentials?), so the scan cannot run"
elif grep -rn -e "$ACCOUNT_ID" -e "${BUCKET:-no-state-bucket-yet}" -e "arn:aws" -e "@" \
  docs/decisions/cloud-gpu-evidence docs/decisions/cloud-gpu-probe; then
  echo "STOP: the lines above hold an account ID, bucket name, ARN, or address; remove them first"
else
  echo "evidence scan clean"
fi
```

Expected: `evidence scan clean`. Before Task 7 creates the bucket, `BUCKET` is empty and the scan searches for a string that never occurs instead. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r .chosen.region "$EVIDENCE/zone-choice.json")
git add infra/state/pinned.auto.tfvars infra/env/pinned.auto.tfvars "$EVIDENCE/README.md" \
  "$EVIDENCE/ssm-get-parameters-ubuntu-24.04-$REGION.json" "$EVIDENCE/ec2-describe-images-$REGION.json"
git commit -m "Pin the cloud environment's region, zone, and Ubuntu image"
```

### Task 7: State bucket (Req 3; Verification bullets 2 and 3)

**Mode:** the controller runs this task inline. Step 3 is an approval gate, and Step 2 runs only if Step 1 fails on credentials.

**Files:**
- Create: `docs/decisions/cloud-gpu-evidence/s3-state-bucket.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)
- Create (gitignored): `infra/state/terraform.tfstate`, `infra/env/backend.hcl`

**Interfaces:**
- Consumes: Task 5's `infra/state` root and Task 6's `infra/state/pinned.auto.tfvars`.
- Produces:
  - the state bucket, whose name is `tofu -chdir=infra/state output -raw bucket`;
  - `infra/env/backend.hcl`, which Task 8's `tofu init` reads;
  - backups in `~/.config/ces-revisions/infra/`;
  - which profile OpenTofu uses, `ces-revisions` or `ces-revisions-process`, which every later `tofu` block exports.

- [ ] **Step 1: Plan the state root**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/state plan -input=false -out=/tmp/ces-revisions-state.tfplan
```

Expected: `Plan: 4 to add, 0 to change, 0 to destroy.`, for `aws_s3_bucket.state`, `aws_s3_bucket_versioning.state`, `aws_s3_bucket_public_access_block.state`, and `aws_s3_bucket_server_side_encryption_configuration.state`. A plan that runs at all shows that the AWS provider read the `aws login` session, which is half of Verification bullet 3; skip Step 2. If the plan fails with a credentials error, such as `No valid credential sources found`, run Step 2.

- [ ] **Step 2: Only if Step 1 failed on credentials — STOP — the fallback profile**

Tell your human partner that OpenTofu could not read the login session, and ask them to approve appending a `ces-revisions-process` profile to `~/.aws/config`. The profile holds no secret: it tells the AWS SDK to run `aws configure export-credentials`. On a yes:

```bash
REGION=$(jq -r .chosen.region docs/decisions/cloud-gpu-evidence/zone-choice.json)
if grep -q '^\[profile ces-revisions-process\]$' ~/.aws/config; then
  echo "the ces-revisions-process profile already exists"
else
  printf '\n[profile ces-revisions-process]\ncredential_process = /opt/homebrew/bin/aws configure export-credentials --profile ces-revisions --format process\nregion = %s\n' "$REGION" >> ~/.aws/config
fi
export AWS_PROFILE=ces-revisions-process
tofu -chdir=infra/state plan -input=false -out=/tmp/ces-revisions-state.tfplan
```

Expected: the plan that Step 1 describes. From here on, every block that runs `tofu` exports `AWS_PROFILE=ces-revisions-process` instead of `ces-revisions`, and `infra/bin/vm` runs with `CES_TOFU_PROFILE=ces-revisions-process`; the `aws` commands keep `ces-revisions`. Underneath: `credential_process` makes the SDK run a command that prints temporary credentials as JSON. `aws configure export-credentials` prints the login session's current credentials, refreshing them when needed.

- [ ] **Step 3: STOP — the human partner approves the apply**

Show your human partner Step 1's plan summary. It creates one private, versioned, encrypted S3 bucket, which will hold a few small state files for a few cents a month. Proceed only on a clear yes.

- [ ] **Step 4: Apply the saved plan**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/state apply -input=false /tmp/ces-revisions-state.tfplan
rm /tmp/ces-revisions-state.tfplan
```

Expected: `Apply complete! Resources: 4 added, 0 changed, 0 destroyed.` and `bucket = <sensitive>`. Underneath: applying a saved plan runs exactly what was approved, without a second prompt. The output is marked sensitive only to keep the bucket's name out of logs.

- [ ] **Step 5: Confirm that a new plan finds nothing to change (bullet 2)**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/state plan -input=false -detailed-exitcode > /dev/null
case $? in
  0) echo "infra/state: no changes" ;;
  2) echo "STOP: the infra/state plan shows changes right after the apply" ;;
  *) echo "STOP: the infra/state plan failed" ;;
esac
```

Expected: `infra/state: no changes`. Underneath: `-detailed-exitcode` makes `plan` exit 0 when nothing would change and 2 when something would.

- [ ] **Step 6: Record the bucket's settings, and check `prevent_destroy`**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
BUCKET=$(tofu -chdir=infra/state output -raw bucket)
jq -n \
  --argjson versioning "$(aws s3api get-bucket-versioning --bucket "$BUCKET" --output json)" \
  --argjson public_access_block "$(aws s3api get-public-access-block --bucket "$BUCKET" --query PublicAccessBlockConfiguration --output json)" \
  --argjson default_encryption "$(aws s3api get-bucket-encryption --bucket "$BUCKET" --query 'ServerSideEncryptionConfiguration.Rules[].ApplyServerSideEncryptionByDefault' --output json)" \
  '{versioning: $versioning, public_access_block: $public_access_block, default_encryption: $default_encryption}' \
  > "$EVIDENCE/s3-state-bucket.json"
jq -e '.versioning.Status == "Enabled" and ([.public_access_block[]] | all) and .default_encryption[0].SSEAlgorithm == "AES256"' \
  "$EVIDENCE/s3-state-bucket.json"
if tofu -chdir=infra/state plan -input=false -destroy > /tmp/ces-revisions-destroy-check.txt 2>&1; then
  echo "STOP: a destroy plan succeeded, so prevent_destroy is missing"
else
  grep -c "Instance cannot be destroyed" /tmp/ces-revisions-destroy-check.txt
fi
rm /tmp/ces-revisions-destroy-check.txt
```

Expected: `true`, then a count of 1 or more. Underneath: a destroy plan against a resource with `prevent_destroy` fails before it touches AWS, so this check is safe.

- [ ] **Step 7: Write the backend file**

```bash
export AWS_PROFILE=ces-revisions
BUCKET=$(tofu -chdir=infra/state output -raw bucket)
REGION=$(jq -r .chosen.region docs/decisions/cloud-gpu-evidence/zone-choice.json)
printf 'bucket = "%s"\nkey    = "env/terraform.tfstate"\nregion = "%s"\n' "$BUCKET" "$REGION" > infra/env/backend.hcl
git check-ignore -v infra/env/backend.hcl
```

Expected: one line naming `.gitignore`'s `backend.hcl` rule and `infra/env/backend.hcl`; the line
number is intentionally unconstrained.

- [ ] **Step 8: Back up the operational files**

```bash
install -d -m 700 ~/.config/ces-revisions/infra
cp infra/state/terraform.tfstate ~/.config/ces-revisions/infra/state-terraform.tfstate
cp infra/env/backend.hcl ~/.config/ces-revisions/infra/backend.hcl
ls ~/.config/ces-revisions/infra
```

Expected: `backend.hcl` and `state-terraform.tfstate`. Underneath: git ignores both files, so removing this worktree after the merge would delete them. The runbook's "Where the operational files live" explains how to put them in another checkout.

- [ ] **Step 9: Append to the evidence README, scan the evidence, and commit**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Req 3: state bucket

- `s3-state-bucket.json` — `aws s3api get-bucket-versioning`, `get-public-access-block`, and `get-bucket-encryption` for the bucket that `infra/state` created, combined without the bucket's name: versioning enabled, all four public access blocks on, and AES256 default encryption. A destroy plan for `infra/state` failed with `Instance cannot be destroyed`, which confirms `prevent_destroy`.
````

Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
git add docs/decisions/cloud-gpu-evidence/README.md docs/decisions/cloud-gpu-evidence/s3-state-bucket.json
git commit -m "Create the OpenTofu state bucket and record its settings"
```

### Task 8: The environment at `dev` (Reqs 3–5 and 8; Verification bullets 2–4)

**Mode:** the controller runs this task inline. Step 1 is the human partner's, and Step 5 is an approval gate.

**Files:**
- Create: `docs/decisions/cloud-gpu-evidence/opentofu-credentials.json`, `ec2-instance-dev.json`, `ec2-network-dev.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)
- Create (gitignored): `infra/env/terraform.tfvars`

**Interfaces:**
- Consumes: Task 5's `infra/env` root, Task 6's `pinned.auto.tfvars` and key pair, and Task 7's `backend.hcl` and profile.
- Produces:
  - the running instance at `dev`, with the outputs `instance_id`, `region`, `root_volume_id`, and `security_group_id`;
  - `ec2-instance-dev.json`, whose `InstanceId`, `InstanceType`, and `RootDevice.VolumeId` Task 13 compares every size switch against;
  - the committed-tree search in Step 11, which Task 16 reruns.

- [ ] **Step 1: STOP — the human partner writes `terraform.tfvars`**

Ask your human partner to create `infra/env/terraform.tfvars` in their editor. It holds one line of the form `budget_email = "…"`, with the address that should receive budget alerts between the quotes. Wait until they confirm. Underneath: the address is a personal value, so it lives in a file git ignores, and `variables.tf` marks the variable sensitive, so plans print `(sensitive value)` in its place.

- [ ] **Step 2: Check the file without printing it, and back it up**

```bash
git check-ignore -v infra/env/terraform.tfvars
grep -c '^budget_email = ".*@.*"$' infra/env/terraform.tfvars
cp infra/env/terraform.tfvars ~/.config/ces-revisions/infra/terraform.tfvars
```

Expected: one line naming `.gitignore`'s `terraform.tfvars` rule and
`infra/env/terraform.tfvars`, then `1`; the line number is intentionally unconstrained.

- [ ] **Step 3: Initialize the S3 backend**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env init -input=false -backend-config=backend.hcl
```

Expected: `Successfully configured the backend "s3"!` and `OpenTofu has been successfully initialized!`. The backend reaches the bucket while it initializes, so success shows that it read the login session, which is the other half of Verification bullet 3. If it fails with a credentials error while Task 7 used `ces-revisions`, run Task 7's Step 2 with your human partner's approval, then rerun this step under `export AWS_PROFILE=ces-revisions-process`.

- [ ] **Step 4: Plan the environment**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -out=/tmp/ces-revisions-env.tfplan
tofu -chdir=infra/env show -json /tmp/ces-revisions-env.tfplan |
  jq -r '.resource_changes[] | select(.mode == "managed" and .change.actions != ["no-op"]) | "\(.change.actions | join(",")) \(.address)"'
```

Expected: `Plan: 14 to add, 0 to change, 0 to destroy.`, then 14 lines beginning `create` for:
- `aws_vpc.main`, `aws_subnet.public`, `aws_internet_gateway.main`, `aws_route_table.public`, and `aws_route_table_association.public`;
- `aws_security_group.vm` and `aws_vpc_security_group_egress_rule.all`;
- `aws_iam_role.vm`, `aws_iam_role_policy_attachment.vm_ssm`, `aws_iam_instance_profile.vm`, and `aws_instance.vm`;
- `aws_iam_role.snapshots`, `aws_iam_role_policy_attachment.snapshots`, and `aws_dlm_lifecycle_policy.daily`.

No budget resource appears, because `budget_enabled` is `false`. Underneath: the user-data precondition runs during this plan, with every guard file read from `infra/vm/`.

- [ ] **Step 5: STOP — the human partner approves the apply**

Show your human partner the plan summary and these costs:
- While running, `dev` bills \$0.20/hr for the instance and \$0.005/hr for its public IPv4 address.
- The 100 GB volume bills about \$8 a month, running or not.
- First boot installs packages for about 10–15 minutes.
- After that, the idle stop stops the instance once it has been idle for 45 minutes.

Proceed only on a clear yes.

- [ ] **Step 6: Apply the saved plan**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env apply -input=false /tmp/ces-revisions-env.tfplan
rm /tmp/ces-revisions-env.tfplan
```

Expected: `Apply complete! Resources: 14 added, 0 changed, 0 destroyed.`, then outputs that include `instance_type = "m7i.xlarge"`.

- [ ] **Step 7: Confirm that a new plan finds nothing to change (bullet 2)**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -detailed-exitcode > /dev/null
case $? in
  0) echo "infra/env: no changes" ;;
  2) echo "STOP: the infra/env plan shows changes right after the apply" ;;
  *) echo "STOP: the infra/env plan failed" ;;
esac
```

Expected: `infra/env: no changes`.

- [ ] **Step 8: Record the instance, its volume, and its role (Reqs 4 and 5)**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
VOLUME_ID=$(tofu -chdir=infra/env output -raw root_volume_id)
jq -n \
  --argjson instance "$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output json \
    --query 'Reservations[0].Instances[0].{InstanceId: InstanceId, InstanceType: InstanceType, Zone: Placement.AvailabilityZone, ImageId: ImageId, HttpTokens: MetadataOptions.HttpTokens, HasPublicIp: PublicIpAddress != `null`, HasInstanceProfile: IamInstanceProfile != `null`, SecurityGroups: SecurityGroups[].GroupName, RootDevice: BlockDeviceMappings[0].{DeviceName: DeviceName, VolumeId: Ebs.VolumeId, DeleteOnTermination: Ebs.DeleteOnTermination}}')" \
  --arg shutdown_behavior "$(aws ec2 describe-instance-attribute --region "$REGION" --instance-id "$INSTANCE_ID" \
    --attribute instanceInitiatedShutdownBehavior --query 'InstanceInitiatedShutdownBehavior.Value' --output text)" \
  --argjson volume "$(aws ec2 describe-volumes --region "$REGION" --volume-ids "$VOLUME_ID" --output json \
    --query 'Volumes[0].{VolumeType: VolumeType, Size: Size, Encrypted: Encrypted, ProjectTag: Tags[?Key==`project`] | [0].Value}')" \
  --argjson attached "$(aws iam list-attached-role-policies --role-name ces-revisions-vm --query 'AttachedPolicies[].PolicyName' --output json)" \
  --argjson inline "$(aws iam list-role-policies --role-name ces-revisions-vm --query PolicyNames --output json)" \
  '$instance + {ShutdownBehavior: $shutdown_behavior, RootVolume: $volume, RoleAttachedPolicies: $attached, RoleInlinePolicies: $inline}' \
  > "$EVIDENCE/ec2-instance-dev.json"
jq -e --arg zone "$(jq -r .chosen.zone "$EVIDENCE/zone-choice.json")" \
  --arg ami "$(jq -r .ImageId "$EVIDENCE/ec2-describe-images-$REGION.json")" \
  '.InstanceType == "m7i.xlarge" and .Zone == $zone and .ImageId == $ami and .HttpTokens == "required" and .HasPublicIp and .HasInstanceProfile and .ShutdownBehavior == "stop" and .RootDevice.DeleteOnTermination and .RootVolume == {VolumeType: "gp3", Size: 100, Encrypted: true, ProjectTag: "ces-revisions"} and .RoleAttachedPolicies == ["AmazonSSMManagedInstanceCore"] and .RoleInlinePolicies == []' \
  "$EVIDENCE/ec2-instance-dev.json"
```

Expected: `true`. Underneath: `--query` keeps the public address and the instance profile only as yes-or-no values, because the profile's ARN holds the account ID. The volume's `project` tag shows that `default_tags` reached it at launch, which is what the snapshot policy selects on.

- [ ] **Step 9: Check the network (bullet 4, first half)**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(tofu -chdir=infra/env output -raw region)
SG_ID=$(tofu -chdir=infra/env output -raw security_group_id)
VPC_ID=$(aws ec2 describe-security-groups --region "$REGION" --group-ids "$SG_ID" --query 'SecurityGroups[0].VpcId' --output text)
jq -n \
  --argjson security_group "$(aws ec2 describe-security-groups --region "$REGION" --group-ids "$SG_ID" --output json \
    --query 'SecurityGroups[0].{GroupName: GroupName, IngressRuleCount: length(IpPermissions), Egress: IpPermissionsEgress[].{Protocol: IpProtocol, Ranges: IpRanges[].CidrIp}}')" \
  --argjson nat_gateways "$(aws ec2 describe-nat-gateways --region "$REGION" --filter Name=vpc-id,Values="$VPC_ID" --query 'length(NatGateways)' --output json)" \
  --argjson vpc_endpoints "$(aws ec2 describe-vpc-endpoints --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" --query 'length(VpcEndpoints)' --output json)" \
  --argjson elastic_ips "$(aws ec2 describe-addresses --region "$REGION" --filters Name=tag:project,Values=ces-revisions --query 'length(Addresses)' --output json)" \
  '{security_group: $security_group, nat_gateways: $nat_gateways, vpc_endpoints: $vpc_endpoints, elastic_ips: $elastic_ips}' \
  > "$EVIDENCE/ec2-network-dev.json"
jq -e '.security_group.IngressRuleCount == 0 and .security_group.Egress == [{Protocol: "-1", Ranges: ["0.0.0.0/0"]}] and .nat_gateways == 0 and .vpc_endpoints == 0 and .elastic_ips == 0' \
  "$EVIDENCE/ec2-network-dev.json"
```

Expected: `true`. Underneath: protocol `-1` means all protocols. With zero ingress rules, nothing on the internet can open a connection to the instance, and the SSM agent's outbound connections are the only way in.

- [ ] **Step 10: Record how OpenTofu authenticated (bullet 3)**

Set `PROFILE` to the profile the `tofu` blocks used: `ces-revisions`, or `ces-revisions-process` if Task 7's Step 2 or this task's Step 3 needed the fallback.

```bash
PROFILE=ces-revisions
PROVIDER=$(grep -A1 'provider "registry.opentofu.org/hashicorp/aws"' infra/env/.terraform.lock.hcl | sed -n 's/.*version *= *"\(.*\)"/\1/p')
jq -n --arg opentofu "$(tofu version | head -n 1)" --arg provider "$PROVIDER" --arg profile "$PROFILE" \
  '{opentofu: $opentofu, aws_provider: $provider, profile: $profile, credential_process_needed: ($profile != "ces-revisions")}' \
  > docs/decisions/cloud-gpu-evidence/opentofu-credentials.json
cat docs/decisions/cloud-gpu-evidence/opentofu-credentials.json
```

Expected: `OpenTofu v1.12.N` or later, the provider version, the profile, and `credential_process_needed` matching it.

- [ ] **Step 11: Search the committed tree (bullet 4, second half)**

This is the committed-tree search that Task 16 reruns:

```bash
export AWS_PROFILE=ces-revisions
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET=$(tofu -chdir=infra/state output -raw bucket)
EMAIL=$(sed -n 's/^budget_email *= *"\(.*\)"$/\1/p' infra/env/terraform.tfvars)
if [ -z "$ACCOUNT_ID" ] || [ -z "$BUCKET" ] || [ -z "$EMAIL" ]; then
  echo "STOP: a value to search for is missing"
elif git grep --untracked -l -F -e "$ACCOUNT_ID" -e "$BUCKET" -e "$EMAIL" -- . ':(exclude)pyproject.toml'; then
  echo "STOP: the files above hold the account ID, the bucket name, or the budget email address"
elif git grep --untracked -l -E -e 'github_pat_[A-Za-z0-9_]{20,}' -e 'gh[pousr]_[A-Za-z0-9]{30,}' -e '(AKIA|ASIA)[0-9A-Z]{16}'; then
  echo "STOP: the files above hold what looks like a token or an access key"
else
  echo "committed tree clean"
fi
git grep -c -F -e "$EMAIL" -- pyproject.toml || true
```

Expected: `committed tree clean`, then either nothing or `pyproject.toml:1`. That one line is `pyproject.toml`'s `authors` entry, committed when the project was created. Record which of the two appeared, for the handoff. Underneath:
- `git grep --untracked` searches tracked files and new files that git does not ignore, so evidence waiting to be committed is covered too.
- `-l` prints only file names, so a match never echoes the address or the ID.
- The token patterns need 20 or more characters after the prefix, which keeps this plan's own patterns from matching themselves.

- [ ] **Step 12: Append to the evidence README, scan the evidence, and commit**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Reqs 3 to 5: the environment at `dev`

- `opentofu-credentials.json` — the OpenTofu and AWS provider versions, and the AWS profile OpenTofu used: `ces-revisions` when it read the `aws login` session directly, or `ces-revisions-process` when it needed a `credential_process` profile.
- `ec2-instance-dev.json` — `aws ec2 describe-instances`, `describe-instance-attribute`, and `describe-volumes`, with `aws iam list-attached-role-policies` and `list-role-policies`, for the instance at `dev`. It records the type, zone, image, IMDSv2 setting, whether a public address and an instance profile exist, the shutdown behavior, the root volume, and the role's policies. The address and the profile's ARN are left out.
- `ec2-network-dev.json` — `aws ec2 describe-security-groups`, `describe-nat-gateways`, `describe-vpc-endpoints`, and `describe-addresses`. It records the security group's ingress rule count and egress rules, the counts of NAT gateways and VPC endpoints in the VPC, and the count of Elastic IPs tagged `project = ces-revisions`.
````

Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
git add "$EVIDENCE/README.md" "$EVIDENCE/opentofu-credentials.json" \
  "$EVIDENCE/ec2-instance-dev.json" "$EVIDENCE/ec2-network-dev.json"
git commit -m "Apply the cloud environment at dev and record its instance and network"
```

### Task 9: Access (Req 4; Verification bullet 5)

**Mode:** the controller runs this task inline. Steps 3 and 6 are the human partner's.

**Files:**
- Create: `docs/decisions/cloud-gpu-evidence/ssm-instance-information.json`, `access.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)
- Modify (on the Mac, outside the repo): `~/.ssh/config`, `~/.ssh/known_hosts`

**Interfaces:**
- Consumes: Task 8's outputs `instance_id` and `region`, and Task 6's key pair.
- Produces:
  - the SSH host entry `ces-revisions-vm`, which `infra/bin/vm sync-config` and every later VM command use;
  - `access.json`, whose `desktop_app` is `host-entry`, `port-forward`, or `neither`, for Task 16's decision record and runbook update.

- [ ] **Step 1: Wait for the instance to register with Systems Manager**

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
aws ssm describe-instance-information --region "$REGION" --filters Key=InstanceIds,Values="$INSTANCE_ID" --output json \
  --query 'InstanceInformationList[0].{PingStatus: PingStatus, AgentVersion: AgentVersion, PlatformName: PlatformName, PlatformVersion: PlatformVersion}' |
  tee docs/decisions/cloud-gpu-evidence/ssm-instance-information.json
```

Expected: `"PingStatus": "Online"`, `"PlatformName": "Ubuntu"`, and `"PlatformVersion": "24.04"`. If it prints `null`, the agent has not registered yet. Wait two minutes and run it again, for up to 15 minutes after the apply. Underneath: Ubuntu's image ships the SSM agent as a snap. With the instance profile's `AmazonSSMManagedInstanceCore`, the agent registers itself and keeps an outbound connection open.

- [ ] **Step 2: Read the host key's fingerprint through Session Manager**

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
aws ssm start-session --region "$REGION" --target "$INSTANCE_ID" \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters '{"command":["ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub"]}' |
  tr -d '\r' | grep '(ED25519)'
```

Expected: one line of the form `256 SHA256:… root@ip-10-42-1-N (ED25519)`, which also shows that Session Manager reaches the instance. Underneath: the session's own start and end lines name its session ID, which includes the IAM user's name, so `grep` drops them. Systems Manager authenticates the instance through AWS, so a fingerprint read this way can be trusted in a way that a first SSH connection's cannot.

- [ ] **Step 3: STOP — the human partner opens a Session Manager shell (bullet 5)**

Ask your human partner to run `infra/bin/vm connect` in their own terminal, then `whoami` and `exit` in the shell it opens. Expected: the printed `+ … aws ssm start-session …` lines, a shell prompt, and `ssm-user`. Wait until they confirm. Underneath: an interactive session needs a terminal, which the controller's shell does not have.

- [ ] **Step 4: Add the SSH host entry**

```bash
export AWS_PROFILE=ces-revisions
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
REGION=$(tofu -chdir=infra/env output -raw region)
if [ "$(command -v aws)" != /opt/homebrew/bin/aws ]; then
  echo "STOP: aws is not at /opt/homebrew/bin/aws, the path the host entry names"
elif grep -qs '^Host ces-revisions-vm$' ~/.ssh/config; then
  echo "STOP: ~/.ssh/config already has a ces-revisions-vm entry; show it to your human partner"
else
  cat >> ~/.ssh/config <<EOF

# ces-revisions cloud VM, reached through AWS Systems Manager (docs/cloud-gpu-runbook.md)
Host ces-revisions-vm
  HostName $INSTANCE_ID
  User ubuntu
  IdentityFile ~/.ssh/ces-revisions-vm
  IdentitiesOnly yes
  ProxyCommand sh -c "PATH=/opt/homebrew/bin:/usr/bin:/bin /opt/homebrew/bin/aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p' --profile ces-revisions --region $REGION"
EOF
  chmod 600 ~/.ssh/config
  ssh -G ces-revisions-vm | grep -e '^hostname ' -e '^user ' -e '^proxycommand '
fi
```

Expected: `hostname i-…`, `user ubuntu`, and the `proxycommand` line. Underneath:
- ssh runs the `ProxyCommand` instead of opening a TCP connection itself. `%h` becomes the `HostName`, which is the instance ID, and `%p` becomes the port, 22.
- The AWS CLI asks Systems Manager for a stream to that port, and the Session Manager plugin carries ssh's bytes over it.
- Setting `PATH` inside the command lets the CLI find the plugin even when the calling app starts no login shell.
- `ssh -G` prints the configuration ssh would use, without connecting.

- [ ] **Step 5: Connect over SSH and compare host keys**

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
SSM_FP=$(aws ssm start-session --region "$REGION" --target "$INSTANCE_ID" \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters '{"command":["ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub"]}' |
  tr -d '\r' | awk '/\(ED25519\)/ {print $2}')
ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes ces-revisions-vm 'lsb_release -ds; uname -r'
SSH_FP=$(ssh-keygen -l -F "$INSTANCE_ID" | awk '$2 == "ED25519" {print $3}')
if [ -n "$SSM_FP" ] && [ "$SSM_FP" = "$SSH_FP" ]; then
  echo "host key matches Session Manager"
else
  ssh-keygen -R "$INSTANCE_ID"
  echo "STOP: the SSH host key does not match the one read through Session Manager; its known_hosts entry was removed"
fi
```

Expected: `Ubuntu 24.04.N LTS`, a kernel ending in `-aws`, and `host key matches Session Manager`. Underneath: `accept-new` records a host key the first time it sees one and refuses a changed key afterward. `ssh-keygen -l -F` prints the fingerprint that ssh just recorded, so a match shows that nothing sat between the Mac and the instance on that first connection. If the connection is refused with `Permission denied (publickey)`, the key has a passphrase that is not loaded; see Task 6, Step 5.

- [ ] **Step 6: STOP — the human partner tries the Claude Code desktop app (bullet 5)**

Ask your human partner to try these in order, stopping at the first that opens a Claude Code session on the VM:
1. **The host entry.** In the desktop app, add an SSH connection to the host `ces-revisions-vm` and connect. On its first connection, the app installs Claude Code on the VM.
2. **Port forwarding.**
   - In a terminal, run `infra/bin/vm forward` and leave it running.
   - In a second terminal, accept the forwarded host key once, with `ssh -p 2222 -i ~/.ssh/ces-revisions-vm -o StrictHostKeyChecking=accept-new ubuntu@localhost true`.
   - Connect the desktop app to host `localhost`, port `2222`, user `ubuntu`, with the identity file `~/.ssh/ces-revisions-vm`.
3. **Neither.** A terminal SSH session and Session Manager shells, which Steps 3 and 5 showed working, remain the way in.

Ask which one worked: `host-entry`, `port-forward`, or `neither`. Underneath: Claude Code issue #40967 reports that the macOS desktop app mishandles `ProxyCommand`, and #26809 that it ignores `Port` in `~/.ssh/config`, which is why the fallback types the port into the dialog. The forwarded port reaches the VM's sshd through the same Session Manager channel.

- [ ] **Step 7: Record the access outcomes, and commit**

Set `DESKTOP_APP` to what the human partner reported:

```bash
DESKTOP_APP=host-entry
jq -n --arg desktop_app "$DESKTOP_APP" \
  '{session_manager_shell: true, ssh_through_session_manager: true, ssh_host_key_matched_session_manager: true, desktop_app: $desktop_app}' \
  > docs/decisions/cloud-gpu-evidence/access.json
jq -e '.desktop_app | IN("host-entry", "port-forward", "neither")' docs/decisions/cloud-gpu-evidence/access.json
```

Expected: `true`. `neither` does not stop the plan: the decision record names the methods that work, and the desktop half of Verification bullet 5 goes to the Plan Completion Protocol's gate as unmet.

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Req 4: access

- `ssm-instance-information.json` — `aws ssm describe-instance-information` for the instance after first boot: the agent's ping status and version, and the platform.
- `access.json` — which access methods opened a session on the VM. It covers a Session Manager shell; SSH through the `ces-revisions-vm` host entry, with the host key checked against the fingerprint read through Session Manager; and how the Claude Code desktop app connected: `host-entry`, `port-forward`, or `neither`.
````

Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
git add "$EVIDENCE/README.md" "$EVIDENCE/ssm-instance-information.json" "$EVIDENCE/access.json"
git commit -m "Record access to the cloud VM through Session Manager and SSH"
```

### Task 10: VM setup (Req 6; Verification bullet 11, first half)

**Mode:** the controller runs this task inline. Step 3 is an approval gate, and Step 6 is the human partner's.

**Files:**
- Create: `docs/decisions/cloud-gpu-evidence/vm-first-boot.txt`, `vm-carried-config.txt`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)

**Interfaces:**
- Consumes:
  - Task 9's host entry `ces-revisions-vm`;
  - the commits of Tasks 1–5;
  - Task 2's `infra/bin/vm sync-config` and Task 3's `infra/vm/setup.sh`.
- Produces, on the VM:
  - `~/Projects/ces-revisions` on the current execution branch, derived with
    `git branch --show-current`, and synced with the `cuda` extra;
  - `~/Projects/agent-skills`, uv in `~/.local/bin`, and the links in `~/.claude/`;
  - `gh` signed in with the human partner's token, and set up as git's credential helper.

  Tasks 11, 13, and 14 run the tests and the probe there.

- [ ] **Step 1: Wait for first boot to finish**

```bash
ssh ces-revisions-vm cloud-init status --wait --long
```

Expected: `status: done` and `errors: []`, possibly after several minutes while first boot downloads the driver packages. If it reports `status: error`, show the log and stop:

```bash
ssh ces-revisions-vm sudo tail -n 60 /var/log/cloud-init-output.log
```

Underneath: cloud-init runs in stages at first boot, and `--wait` returns after the last one, which ran `first-boot.sh`.

- [ ] **Step 2: Record what first boot installed**

```bash
ssh ces-revisions-vm bash -s <<'EOF' 2>&1 | tee docs/decisions/cloud-gpu-evidence/vm-first-boot.txt
echo "== release and kernels"
lsb_release -ds
uname -r
ls /boot | grep '^vmlinuz-'
echo "== held packages"
apt-mark showhold
echo "== driver packages"
dpkg-query -W -f='${Package} ${Version}\n' nvidia-headless-no-dkms-580-server-open nvidia-utils-580-server linux-modules-nvidia-580-server-open-aws
echo "== git and gh"
git --version
gh --version | head -n 1
echo "== guards"
systemctl is-enabled ces-idle-stop.timer ces-gpu-cap.service
systemctl is-active ces-idle-stop.timer
ls /etc/ces-revisions /usr/local/lib/ces-revisions
if sudo systemd-analyze verify /etc/systemd/system/ces-idle-stop.service /etc/systemd/system/ces-idle-stop.timer /etc/systemd/system/ces-gpu-cap.service 2> /tmp/ces-units.txt; then
  echo "units verified"
else
  echo "systemd-analyze verify failed:"
  cat /tmp/ces-units.txt
fi
echo "== nvidia-smi without a GPU"
nvidia-smi > /dev/null 2>&1
echo "nvidia-smi exit status $?"
EOF
```

Expected:
- `Ubuntu 24.04.N LTS`, the running kernel ending in `-aws`, and one or two `vmlinuz-` files (two when the module package brought a newer kernel);
- the held packages: the three driver packages and `linux-image-aws`, plus `linux-aws` and `linux-headers-aws` where the image has them installed;
- the three driver packages at `580.` versions;
- a `git version 2.` line and a `gh version 2.N.M` line with N above 45;
- `enabled`, `enabled`, `active`, the listings `idle-stop.env vm` and `idle_stop.py`, and `units verified`;
- a non-zero `nvidia-smi` exit status. Status 9 means the driver is not loaded, which is expected without a GPU.

If a driver package or `linux-image-aws` is not held, stop: first boot did not finish its holds. Underneath: `systemd-analyze verify` checks the three unit files and the units they name. Its warnings go to a temporary file, so notes about unrelated units stay out of the evidence.

- [ ] **Step 3: STOP — the human partner approves pushing the branch**

```bash
CLOUD_BRANCH="$(git branch --show-current)"
if [ -z "$CLOUD_BRANCH" ]; then echo "STOP: detached HEAD has no branch to push"; fi
git log --oneline main..HEAD
```

Show your human partner that list and the value of `CLOUD_BRANCH`, and ask them to approve pushing
that branch to `origin`, the public `lowmason/ces-revisions`, so the VM can clone it. The branch is
public once pushed. On a clear yes, recompute the name so the command does not depend on shell state:

```bash
CLOUD_BRANCH="$(git branch --show-current)"
git push -u origin "$CLOUD_BRANCH"
```

Underneath: the VM clones over HTTPS without credentials, because the repository is public. The token in Step 6 is only for pushing from the VM.

- [ ] **Step 4: Send the Mac's settings**

```bash
export AWS_PROFILE=ces-revisions
MAC_LINK_COUNTS=/tmp/ces-revisions-mac-link-counts.txt
VM_LINK_COUNTS=/tmp/ces-revisions-vm-link-counts.txt
: > "$MAC_LINK_COUNTS"
for link_set in skills agents commands hooks; do
  links="$(find ~/.claude/$link_set -maxdepth 1 -type l | wc -l | tr -d ' ')"
  broken="$(find -L ~/.claude/$link_set -maxdepth 1 -type l | wc -l | tr -d ' ')"
  if [ "$broken" -ne 0 ]; then echo "STOP: $link_set has $broken broken links"; fi
  printf '%s %s\n' "$link_set" "$links" | tee -a "$MAC_LINK_COUNTS"
done
infra/bin/vm sync-config
ssh ces-revisions-vm 'for link_set in skills agents commands hooks; do
  links=$(wc -l < ".config/ces-revisions/links/$link_set.txt" | tr -d " ")
  printf "%s %s\n" "$link_set" "$links"
done' > "$VM_LINK_COUNTS"
cat "$VM_LINK_COUNTS"
diff "$MAC_LINK_COUNTS" "$VM_LINK_COUNTS"
ssh ces-revisions-vm 'ls .claude/CLAUDE.md .claude/settings.json'
```

Expected:
- the `+ ssh` and `+ rsync` lines, with no `skipping` line, ending with `gh is not signed in on the VM yet; after gh auth login there, run gh auth setup-git`;
- four Mac link-count lines with no `STOP`, the same four counts from the VM's copied link lists,
  and a silent successful `diff`; the actual counts are deliberately captured at execution time
  rather than fixed to the 2026-09-13 values;
- `CLAUDE.md` and `settings.json`.

- [ ] **Step 5: Run `setup.sh` for the first time**

Run in the background:

```bash
CLOUD_BRANCH="$(git branch --show-current)"
if [ -z "$CLOUD_BRANCH" ]; then echo "STOP: detached HEAD has no branch to clone"; fi
ssh ces-revisions-vm "CES_REVISIONS_BRANCH=$CLOUD_BRANCH bash -s" < infra/vm/setup.sh
```

Expected, in order:
- `== clone the repositories into /home/ubuntu/Projects`, with two `Cloning into` lines;
- `== install uv from the series that pyproject.toml pins`, with the installer naming `/home/ubuntu/.local/bin`;
- `== install Python 3.14 and sync the environment with the cuda extra`, then `uv sync` installing packages that include `jax-cuda13-plugin` and the `nvidia-*` wheels;
- `== link the personal skills, agents, commands, and hooks into ~/.claude`, with each count equal
  to the Mac snapshot from Step 4;
- `== reinstall the cost guards from the checkout`, with no error.

Underneath: piping the script through `bash -s` runs the Mac's copy, because the VM has no checkout yet. From here on, `setup.sh` runs from the VM's own checkout.

- [ ] **Step 6: STOP — the human partner creates and enters the GitHub token**

Ask your human partner to:
1. On github.com, open **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**. Choose an expiration. Set **Repository access** to **Only select repositories** and pick `lowmason/ces-revisions`. Under **Repository permissions**, set **Contents** and **Pull requests** to **Read and write**.
2. In their own terminal, run `ssh ces-revisions-vm`. On the VM, run `gh auth login --with-token`, paste the token, press Return, then press Control-D.
3. Still on the VM, run `gh auth setup-git`, then `exit`.

Wait until they confirm. Never ask for the token, and never run a command that prints it. Underneath: on a server with no keyring, `gh` keeps the token in `~/.config/gh/hosts.yml`. `setup-git` makes `gh` git's credential helper for github.com, so `git push` on the VM uses the token.

- [ ] **Step 7: Check the token without reading it**

```bash
ssh ces-revisions-vm gh auth status
ssh ces-revisions-vm git -C Projects/ces-revisions push --dry-run origin HEAD
```

Expected: `Logged in to github.com account lowmason`, with the token shown masked, then `Everything up-to-date`. Underneath: a dry-run push authenticates and compares refs, but sends nothing.

- [ ] **Step 8: Rerun both scripts, to confirm they are idempotent**

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm sync-config
ssh ces-revisions-vm bash -l -s <<'EOF'
~/Projects/ces-revisions/infra/vm/setup.sh
git config --global --get-all credential.https://github.com.helper
EOF
```

Expected:
- `sync-config` ends without the `gh` reminder;
- `setup.sh` clones nothing and installs no uv, its `uv sync` reports nothing to install, and it prints the same link counts;
- the last command prints an empty line, then `!/usr/bin/gh auth git-credential`.

Underneath: `sync-config` replaced `~/.gitconfig` with the Mac's copy, which names `/opt/homebrew/bin/gh`, and then its `gh auth setup-git` pointed git back at the VM's own `gh`.

- [ ] **Step 9: Record the carried configuration**

```bash
ssh ces-revisions-vm bash -s <<'EOF' | tee docs/decisions/cloud-gpu-evidence/vm-carried-config.txt
for link_set in skills agents commands hooks; do
  links=$(find ~/.claude/$link_set -maxdepth 1 -type l | wc -l)
  broken=$(find -L ~/.claude/$link_set -maxdepth 1 -type l | wc -l)
  echo "$link_set: $links links, $broken broken"
done
ls ~/.claude/CLAUDE.md ~/.claude/settings.json | sed "s|^$HOME|~|"
git -C ~/Projects/ces-revisions branch --show-current
EOF
sed -n '1,4p' docs/decisions/cloud-gpu-evidence/vm-carried-config.txt \
  | sed -E 's/^([^:]+): ([0-9]+) links, 0 broken$/\1 \2/' \
  > /tmp/ces-revisions-vm-carried-link-counts.txt
diff /tmp/ces-revisions-mac-link-counts.txt /tmp/ces-revisions-vm-carried-link-counts.txt
CLOUD_BRANCH="$(git branch --show-current)"
test "$(tail -n 1 docs/decisions/cloud-gpu-evidence/vm-carried-config.txt)" = "$CLOUD_BRANCH"
```

Expected: the four link lines exactly match the Mac snapshot and report zero broken links; then
`~/.claude/CLAUDE.md` and `~/.claude/settings.json`; then the current execution branch. Both
`diff` and `test` exit 0. Underneath: `find -L` follows links, so a link it still reports as type
`l` is one whose target is missing.

- [ ] **Step 10: Append to the evidence README, scan the evidence, and commit**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Req 6: machine setup

- `vm-first-boot.txt` — on the VM after cloud-init finished: the release and kernels, the held packages, the driver packages' versions, the git and gh versions, the guards' unit states and `systemd-analyze verify`, and `nvidia-smi`'s exit status on `dev`.
- `vm-carried-config.txt` — on the VM after `setup.sh` and the GitHub token: the skill, agent, command, and hook link counts, with broken links counted; the copied `CLAUDE.md` and `settings.json`; and the checkout's branch.
````

Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
git add "$EVIDENCE/README.md" "$EVIDENCE/vm-first-boot.txt" "$EVIDENCE/vm-carried-config.txt"
git commit -m "Set up the cloud VM and record its first boot and carried configuration"
```

### Task 11: `dev` checks (Verification bullets 6, 9, 10, and 12)

**Mode:** the controller runs this task inline. Step 5 is the human partner's.

**Files:**
- Create: `docs/decisions/cloud-gpu-probe/dev.json`
- Create: `docs/decisions/cloud-gpu-evidence/vm-dev-checks.txt`, `bls-canary.json`, `vm-idle-stop.txt`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)

**Interfaces:**
- Consumes: Task 10's checkout on the VM, Task 4's probe, Task 1's idle stop, and plan 3's device expectation in `tests/test_stack.py`.
- Produces: `dev.json`, and the outcomes Task 16 reports: JAX on `dev` (Req 6's open item), the idle stop, and the BLS fetch.

- [ ] **Step 1: Watch JAX fall back to the CPU (Req 6's open item)**

```bash
ssh ces-revisions-vm bash -l -s <<'EOF' | tee docs/decisions/cloud-gpu-evidence/vm-dev-checks.txt
cd ~/Projects/ces-revisions
echo "== JAX with the cuda extra and no GPU"
uv run python -c 'import jax; print("backend", jax.default_backend(), "devices", jax.local_device_count())' 2> /tmp/ces-jax-stderr.txt
echo "python exit status $?"
echo "stderr lines: $(wc -l < /tmp/ces-jax-stderr.txt)"
MESSAGE=$(grep -m 1 -i -e 'cuinit' -e 'plugin configuration' /tmp/ces-jax-stderr.txt | cut -c 1-160)
echo "${MESSAGE:-no CUDA message on stderr}"
EOF
```

Expected: `backend cpu devices 1` and `python exit status 0`. After those come a count of stderr lines and either the first message from JAX's CUDA plugin or `no CUDA message on stderr`. Either way, JAX ran on the CPU, which discharges Req 6's open item without a separate CPU environment. If the exit status is not 0, stop: Req 6's fallback would be needed, a `UV_PROJECT_ENVIRONMENT` synced without the extra on CPU sizes, and that is a plan change for the human partner. Underneath: the evidence keeps one message line, not the whole traceback.

- [ ] **Step 2: Run the whole test suite on `dev` (bullet 6)**

Run in the background:

```bash
ssh ces-revisions-vm bash -l -s <<'EOF' | tee -a docs/decisions/cloud-gpu-evidence/vm-dev-checks.txt
cd ~/Projects/ces-revisions
echo "== uv sync --locked --extra cuda"
uv sync --locked --extra cuda 2>&1 | tail -n 1
echo "== uv run pytest"
uv run pytest -q -p no:cacheprovider 2>&1 | tail -n 1
EOF
```

Execution recorded a line reporting the packages audited, then `432 passed`, with warnings. Four of
those cases are live network canaries and 13 are slow cases added before the cloud plans;
`test_session_runs_float64_jax_on_the_expected_devices` asserts the `cpu` backend and four host
devices here, so a full pass discharges bullet 6.

- [ ] **Step 3: Run the probe on `dev` (bullet 9)**

Ask your human partner not to use the VM while the probe runs, since it is a timing. Then run in the background:

```bash
ssh ces-revisions-vm bash -l -s <<'EOF'
cd ~/Projects/ces-revisions
uv run python -m ces_revisions.engine_probe --steps 280 --states 150 --cells 70 --batch 1 4 16 --label dev --out docs/decisions/cloud-gpu-probe/dev.json
echo "probe exit status $?"
EOF
```

Expected: `probe exit status 0`, after several minutes. Batch 16 peaked at 11.9 GB on the Mac, and `dev` has 16 GiB. If the status is 137, or `ssh ces-revisions-vm sudo dmesg | grep -i 'killed process'` names python, batch 16 ran out of memory. Stop and report, because the decision record needs batch 16 on every host.

- [ ] **Step 4: Bring the record to the Mac and check it**

```bash
rsync -a ces-revisions-vm:Projects/ces-revisions/docs/decisions/cloud-gpu-probe/dev.json docs/decisions/cloud-gpu-probe/dev.json
jq -e '.label == "dev" and .backend == "cpu" and ([.batches[].batch] == [1, 4, 16]) and .dimensions == {steps: 280, states: 150, cells: 70} and .nvidia_driver == null' \
  docs/decisions/cloud-gpu-probe/dev.json
jq -r '.batches[] | "batch \(.batch): median \(.median_seconds) s, compile \(.compile_seconds) s"' docs/decisions/cloud-gpu-probe/dev.json
```

Expected: `true`, then three lines. Underneath: rsync runs over the same SSH host entry, so the file comes back through Session Manager.

- [ ] **Step 5: STOP — the human partner runs the BLS fetch (bullet 12)**

Ask your human partner to open a shell on the VM with `ssh ces-revisions-vm` and run:

```bash
read -r CONTACT
curl -sS -o /dev/null -w '%{http_code}\n' -A "ces-revisions research ($CONTACT)" https://download.bls.gov/pub/time.series/ce/ce.datatype
```

After the first command, they type the contact they want BLS to see and press Return. They report only the three-digit status that the second command prints, then `exit`. Underneath: `ce.datatype` is the few-hundred-byte list of CES data types. BLS asks automated clients to name themselves and a contact in the User-Agent, and answers requests it refuses with 403. The contact stays in that shell, and no file records the User-Agent.

Then set `STATUS` to the status they reported, and record it:

```bash
STATUS=200
case "$STATUS" in
  2??) OUTCOME=allowed ;;
  *) OUTCOME=blocked ;;
esac
jq -n --argjson http_status "$STATUS" --arg outcome "$OUTCOME" --arg date "$(date -u +%F)" \
  '{url: "https://download.bls.gov/pub/time.series/ce/ce.datatype", from: "the VM at dev", http_status: $http_status, outcome: $outcome, date: $date}' \
  > docs/decisions/cloud-gpu-evidence/bls-canary.json
cat docs/decisions/cloud-gpu-evidence/bls-canary.json
```

- [ ] **Step 6: Shorten the idle window**

```bash
ssh ces-revisions-vm "sudo sed -i 's/^IDLE_WINDOW_MINUTES=.*/IDLE_WINDOW_MINUTES=10/' /etc/ces-revisions/idle-stop.env && grep '^IDLE_WINDOW_MINUTES' /etc/ces-revisions/idle-stop.env"
```

Expected: `IDLE_WINDOW_MINUTES=10`. Ask your human partner to close any shell or desktop session they have open on the VM.

- [ ] **Step 7: Wait for the idle stop (bullet 10, first part)**

Run in the background:

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
{
  echo "== waiting for the idle stop, with a 10-minute window"
  echo "started $(date -u +%FT%TZ)"
  for MINUTE in $(seq 45); do
    STATE=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
      --query 'Reservations[0].Instances[0].State.Name' --output text)
    if [ "$STATE" = stopped ]; then break; fi
    sleep 60
  done
  echo "state $STATE after $MINUTE minutes, $(date -u +%FT%TZ)"
} | tee docs/decisions/cloud-gpu-evidence/vm-idle-stop.txt
```

Expected: `state stopped after N minutes`, with N between 10 and 25. If the state is still `running` after 45 minutes, stop and report. Underneath:
- The first idle sample starts a streak under `/run`.
- The timer's later checks see the streak grow, and the check that finds it 10 minutes old calls `shutdown -h now`.
- EC2 turns that poweroff into a stop, because `instance_initiated_shutdown_behavior` is `stop`.

- [ ] **Step 8: Start the instance, record the stop, and restore the window**

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm start
until ssh -o ConnectTimeout=20 ces-revisions-vm true 2> /dev/null; do sleep 15; done
ssh ces-revisions-vm bash -s <<'EOF' | tee -a docs/decisions/cloud-gpu-evidence/vm-idle-stop.txt
echo "== the idle stop's message in the boot it ended"
sudo journalctl -b -1 -u ces-idle-stop.service -o short-iso --no-pager | grep 'idle-stop:'
echo "== the window, restored"
sudo install -m 0644 ~/Projects/ces-revisions/infra/vm/guards/idle-stop.env /etc/ces-revisions/idle-stop.env
grep '^IDLE_WINDOW_MINUTES' /etc/ces-revisions/idle-stop.env
EOF
```

Run the `until` loop in the background if the instance is slow to come back. Expected: a journal line ending in `idle-stop: idle since 17…; shutting down`, then `IDLE_WINDOW_MINUTES=45`. Underneath: Ubuntu keeps the journal on disk, so `-b -1` reads the boot that the idle stop ended.

- [ ] **Step 9: Append to the evidence README, scan the evidence, and commit**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Reqs 6 and 8 on `dev`

- `vm-dev-checks.txt` — on `dev`: JAX's backend and device count with the `cuda` extra installed and no GPU, the number of stderr lines, and the first message from JAX's CUDA plugin if any; then `uv sync --locked --extra cuda` and the full test run's summary line.
- `bls-canary.json` — one by-hand fetch of `https://download.bls.gov/pub/time.series/ce/ce.datatype` from the VM, recorded as its HTTP status and whether BLS allowed it. The User-Agent is not recorded.
- `vm-idle-stop.txt` — the wait for the idle stop with its window shortened to 10 minutes, its journal line from the boot it ended, and the window restored to 45 minutes.
- `../cloud-gpu-probe/dev.json` — the engine probe on `dev` at T=280, n=150, p=70, with batch sizes 1, 4, and 16.
````

Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
git add docs/decisions/cloud-gpu-probe/dev.json "$EVIDENCE/README.md" \
  "$EVIDENCE/vm-dev-checks.txt" "$EVIDENCE/bls-canary.json" "$EVIDENCE/vm-idle-stop.txt"
git commit -m "Check the cloud VM at dev: tests, probe, idle stop, and BLS fetch"
```

### Task 12: Tag, budget, and snapshots (Req 8; Verification bullet 10)

**Mode:** the controller runs this task inline. Steps 2 and 6 are gates.
- Step 1 needs a lifecycle run after Task 8's apply, so it can start after 06:00 UTC on the next day.
- Step 2 needs the `project` tag to appear in Billing, up to 24 hours after Task 7's apply.

**Files:**
- Create: `docs/decisions/cloud-gpu-evidence/ec2-dlm-snapshots.json`, `ce-cost-allocation-tags.json`, `budgets-budget.json`, `budgets-notifications.json`, `budgets-actions.json`, `iam-simulate-budget-action-role.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)
- Modify (gitignored): `infra/env/terraform.tfvars`

**Interfaces:**
- Consumes: Task 5's `budget_enabled` and budget resources, and Task 8's instance, root volume, and snapshot policy.
- Produces: the budget `ces-revisions-monthly` and its stop action, which make the outputs `budget_name` and `budget_action_id` non-null. The runbook's recovery section uses `budget_action_id`.

- [ ] **Step 1: Check the first lifecycle snapshot**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(tofu -chdir=infra/env output -raw region)
VOLUME_ID=$(tofu -chdir=infra/env output -raw root_volume_id)
aws ec2 describe-snapshots --region "$REGION" --owner-ids self --output json \
  --filters Name=volume-id,Values="$VOLUME_ID" Name=tag-key,Values=aws:dlm:lifecycle-policy-id \
  --query 'Snapshots[].{SnapshotId: SnapshotId, VolumeId: VolumeId, StartTime: StartTime, State: State, ProjectTag: Tags[?Key==`project`] | [0].Value}' \
  > "$EVIDENCE/ec2-dlm-snapshots.json"
jq -e 'length >= 1 and all(.[]; .State == "completed" and .ProjectTag == "ces-revisions")' "$EVIDENCE/ec2-dlm-snapshots.json"
```

Expected: `true`. If it prints `false` because the list is empty, the policy has not run yet; try again after the next 06:00 UTC. Underneath: Data Lifecycle Manager finds the volume by its `project` tag and tags each snapshot with its policy's ID. `copy_tags` also copies the volume's `project` tag, which puts snapshot storage in the budget's view.

- [ ] **Step 2: STOP — the human partner activates the cost allocation tag**

Ask your human partner to sign in to the console as the IAM user and open **Billing and Cost Management** → **Cost allocation tags**. Under **User-defined cost allocation tags**, they select `project` and choose **Activate**. If `project` is not listed, AWS has not collected it yet; they try again later. Wait until they confirm. Underneath: activation tells AWS's billing pipeline to carry the tag onto cost data from then on, and the budget's filter reads that data.

- [ ] **Step 3: Confirm the activation**

```bash
export AWS_PROFILE=ces-revisions
aws ce list-cost-allocation-tags --region us-east-1 --tag-keys project --output json \
  --query 'CostAllocationTags[].{TagKey: TagKey, Type: Type, Status: Status}' \
  > docs/decisions/cloud-gpu-evidence/ce-cost-allocation-tags.json
jq -e '.[0].Status == "Active" and .[0].Type == "UserDefined"' docs/decisions/cloud-gpu-evidence/ce-cost-allocation-tags.json
```

Expected: `true`. The status can take a while to change after activation; if it still reads `Inactive`, try again later. Cost Explorer's API bills \$0.01 per request.

- [ ] **Step 4: Enable the budget**

```bash
if grep -q '^budget_enabled' infra/env/terraform.tfvars; then
  echo "STOP: budget_enabled is already set"
else
  printf 'budget_enabled = true\n' >> infra/env/terraform.tfvars
  echo "budget_enabled set"
fi
cp infra/env/terraform.tfvars ~/.config/ces-revisions/infra/terraform.tfvars
```

Expected: `budget_enabled set`.

- [ ] **Step 5: Plan the budget**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -out=/tmp/ces-revisions-budget.tfplan
tofu -chdir=infra/env show -json /tmp/ces-revisions-budget.tfplan |
  jq -r '.resource_changes[] | select(.mode == "managed" and .change.actions != ["no-op"]) | "\(.change.actions | join(",")) \(.address)"'
```

Expected: `Plan: 4 to add, 0 to change, 0 to destroy.`, then `create` lines for `aws_budgets_budget.monthly[0]`, `aws_budgets_budget_action.stop_vm[0]`, `aws_iam_role.budget_action[0]`, and `aws_iam_role_policy.budget_action[0]`.

- [ ] **Step 6: STOP — the human partner approves the apply**

Show your human partner the plan summary: a \$150 monthly cost budget with three email alerts, a budget action that stops the instance at 100% of actual spend, and that action's role, which can do nothing but stop this instance. None of these runs anything until the budget is reached. Proceed only on a clear yes.

- [ ] **Step 7: Apply, and confirm that a new plan finds nothing to change**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env apply -input=false /tmp/ces-revisions-budget.tfplan
rm /tmp/ces-revisions-budget.tfplan
tofu -chdir=infra/env plan -input=false -detailed-exitcode > /dev/null
case $? in
  0) echo "infra/env: no changes" ;;
  2) echo "STOP: the infra/env plan shows changes right after the apply" ;;
  *) echo "STOP: the infra/env plan failed" ;;
esac
tofu -chdir=infra/env output -raw budget_name
echo
```

Expected: `Apply complete! Resources: 4 added, 0 changed, 0 destroyed.`, `infra/env: no changes`, and `ces-revisions-monthly`.

- [ ] **Step 8: Record the budget, its alerts, and its action**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws budgets describe-budget --account-id "$ACCOUNT_ID" --budget-name ces-revisions-monthly --output json \
  --query 'Budget.{BudgetName: BudgetName, BudgetType: BudgetType, TimeUnit: TimeUnit, BudgetLimit: BudgetLimit, CostFilters: CostFilters}' \
  > "$EVIDENCE/budgets-budget.json"
aws budgets describe-notifications-for-budget --account-id "$ACCOUNT_ID" --budget-name ces-revisions-monthly --output json \
  --query 'Notifications[].{NotificationType: NotificationType, ComparisonOperator: ComparisonOperator, Threshold: Threshold, ThresholdType: ThresholdType}' \
  > "$EVIDENCE/budgets-notifications.json"
aws budgets describe-budget-actions-for-budget --account-id "$ACCOUNT_ID" --budget-name ces-revisions-monthly --output json \
  --query 'Actions[].{ActionType: ActionType, ApprovalModel: ApprovalModel, NotificationType: NotificationType, Status: Status, ActionThreshold: ActionThreshold, ActionSubType: Definition.SsmActionDefinition.ActionSubType, InstanceCount: length(Definition.SsmActionDefinition.InstanceIds)}' \
  > "$EVIDENCE/budgets-actions.json"
jq -e '(.BudgetLimit.Amount | tonumber) == 150 and .BudgetLimit.Unit == "USD" and .TimeUnit == "MONTHLY" and .CostFilters.TagKeyValue == ["user:project$ces-revisions"]' \
  "$EVIDENCE/budgets-budget.json"
jq -e 'map("\(.NotificationType) \(.Threshold | floor)") | sort == ["ACTUAL 100", "FORECASTED 50", "FORECASTED 80"]' "$EVIDENCE/budgets-notifications.json"
jq -e 'length == 1 and .[0].ActionType == "RUN_SSM_DOCUMENTS" and .[0].ActionSubType == "STOP_EC2_INSTANCES" and .[0].ApprovalModel == "AUTOMATIC" and .[0].NotificationType == "ACTUAL" and .[0].InstanceCount == 1' \
  "$EVIDENCE/budgets-actions.json"
```

Expected: `true` three times. `--query` leaves out the alerts' subscribers, the action's role ARN, and the instance IDs. Underneath: since version 1.7, jq keeps a number as it was written, so a threshold of `50.0` prints as `50.0`; `floor` makes each one a whole number before the comparison.

- [ ] **Step 9: Simulate the action role's permissions**

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN=$(aws iam get-role --role-name ces-revisions-budget-stop --query Role.Arn --output text)
VIA_SSM=ContextKeyName=aws:CalledVia,ContextKeyValues=ssm.amazonaws.com,ContextKeyType=stringList
decision() {
  aws iam simulate-principal-policy --policy-source-arn "$ROLE_ARN" --action-names "$1" \
    --resource-arns "$2" --context-entries "$VIA_SSM" \
    --query 'EvaluationResults[0].EvalDecision' --output text
}
jq -n \
  --arg stop_this_vm "$(decision ec2:StopInstances "arn:aws:ec2:$REGION:$ACCOUNT_ID:instance/$INSTANCE_ID")" \
  --arg run_stop_automation "$(decision ssm:StartAutomationExecution "arn:aws:ssm:$REGION::automation-definition/AWS-StopEC2Instance:\$DEFAULT")" \
  --arg stop_another_instance "$(decision ec2:StopInstances "arn:aws:ec2:$REGION:$ACCOUNT_ID:instance/i-00000000000000000")" \
  --arg terminate_this_vm "$(decision ec2:TerminateInstances "arn:aws:ec2:$REGION:$ACCOUNT_ID:instance/$INSTANCE_ID")" \
  '{stop_this_vm: $stop_this_vm, run_stop_automation: $run_stop_automation, stop_another_instance: $stop_another_instance, terminate_this_vm: $terminate_this_vm}' \
  > docs/decisions/cloud-gpu-evidence/iam-simulate-budget-action-role.json
cat docs/decisions/cloud-gpu-evidence/iam-simulate-budget-action-role.json
```

Expected: `allowed`, `allowed`, `implicitDeny`, and `implicitDeny`, in that order. Underneath: the simulator evaluates the role's policies against a described request, with `aws:CalledVia` set as Systems Manager would set it. It cannot run the action itself, which fires only at 100% of actual spend, so this is the closest check available.

- [ ] **Step 10: Append to the evidence README, scan the evidence, and commit**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Req 8: budget and snapshots

- `ec2-dlm-snapshots.json` — `aws ec2 describe-snapshots` for the root volume's lifecycle snapshots, narrowed to their IDs, start times, states, and copied `project` tags.
- `ce-cost-allocation-tags.json` — `aws ce list-cost-allocation-tags` for `project`: its type and status after activation.
- `budgets-budget.json` — `aws budgets describe-budget`: the name, type, period, limit, and tag filter.
- `budgets-notifications.json` — `aws budgets describe-notifications-for-budget`: each alert's type and threshold, without its subscriber.
- `budgets-actions.json` — `aws budgets describe-budget-actions-for-budget`: the action's type, sub-type, approval model, threshold, status, and instance count, without its role ARN.
- `iam-simulate-budget-action-role.json` — `aws iam simulate-principal-policy` for the action's role, called through Systems Manager. Stopping this VM and starting the stop automation are allowed; stopping another instance and terminating this one are not.
````

Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
git add "$EVIDENCE/README.md" "$EVIDENCE/ec2-dlm-snapshots.json" "$EVIDENCE/ce-cost-allocation-tags.json" \
  "$EVIDENCE/budgets-budget.json" "$EVIDENCE/budgets-notifications.json" "$EVIDENCE/budgets-actions.json" \
  "$EVIDENCE/iam-simulate-budget-action-role.json"
git commit -m "Enable the cloud budget and record it, its stop action, and the first snapshots"
```

### Task 13: `l4` (Verification bullets 7–10)

**Mode:** the controller runs this task inline once the G and VT quota is approved. Step 2 and the switch in Step 11 are approval gates.

**Files:**
- Create: `docs/decisions/cloud-gpu-probe/l4.json`
- Create: `docs/decisions/cloud-gpu-evidence/vm-l4-checks.txt`, `size-switches.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)
- Create (gitignored): `infra/env/size.auto.tfvars`
- Create (on the VM): `~/ces-determinism.py`

**Interfaces:**
- Consumes: plan 3's request for `L-DB2E81BA`, Task 2's `infra/bin/vm size`, and Task 8's `ec2-instance-dev.json`.
- Produces:
  - `size-switches.json`, a JSON array whose entries hold `size`, `InstanceId`, `InstanceType`, and `RootVolumeId`, plus `date` for every switch after the first entry. Task 14 extends it.
  - `~/ces-determinism.py` on the VM, which Task 14 reuses.

- [ ] **Step 1: Check the quota**

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code L-DB2E81BA \
  --query 'Quota.Value' --output text
aws service-quotas list-requested-service-quota-change-history-by-quota --region "$REGION" \
  --service-code ec2 --quota-code L-DB2E81BA \
  --query 'RequestedQuotas[].{DesiredValue: DesiredValue, Status: Status, Created: Created}' --output table
```

Expected: `4.0` or more. While the value is below 4 and the latest request reads `PENDING` or `CASE_OPENED`, this task waits, and another waiting task can run. If the request reads `DENIED` or `NOT_APPROVED`, stop and report.

- [ ] **Step 2: STOP — the human partner approves running `l4`**

Tell your human partner:
- switching to `l4` starts the instance as a g6.xlarge at \$0.81 per hour;
- the checks, the tests, and the probe take about an hour;
- the GPU cap powers the instance off 8 hours after boot, and the idle stop after 45 idle minutes.

Proceed only on a clear yes.

- [ ] **Step 3: Plan the switch**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -var size=l4 -out=/tmp/ces-revisions-size.tfplan
tofu -chdir=infra/env show -json /tmp/ces-revisions-size.tfplan |
  jq -r '.resource_changes[] | select(.mode == "managed" and .change.actions != ["no-op"]) | "\(.change.actions | join(",")) \(.address) \(.change.before.instance_type // "") -> \(.change.after.instance_type // "")"'
rm /tmp/ces-revisions-size.tfplan
```

Expected: `Plan: 0 to add, 1 to change, 0 to destroy.`, then `update aws_instance.vm m7i.xlarge -> g6.xlarge`.

- [ ] **Step 4: Switch to `l4`**

Run in the background:

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm size l4 -auto-approve
cat infra/env/size.auto.tfvars
```

Expected: the printed `+ env AWS_PROFILE=ces-revisions tofu -chdir=… apply -var size=l4 -auto-approve`, then `Apply complete! Resources: 0 added, 1 changed, 0 destroyed.`, then `size = "l4"`. If the apply fails with `VcpuLimitExceeded`, stop and report. If it fails with `InsufficientInstanceCapacity`, leave the instance stopped, record the reduced failure evidence, skip Steps 5–11, and continue at Task 13A only after the human partner chooses the permanent `l40s` fallback. The runbook's troubleshooting section covers both errors.

- [ ] **Step 5: Confirm that the instance and its root volume kept their IDs (bullet 8)**

```bash
export AWS_PROFILE=ces-revisions
SIZE=l4
TYPE=g6.xlarge
EVIDENCE=docs/decisions/cloud-gpu-evidence
SWITCHES="$EVIDENCE/size-switches.json"
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
if [ ! -f "$SWITCHES" ]; then
  jq '[{size: "dev", InstanceId, InstanceType, RootVolumeId: .RootDevice.VolumeId}]' "$EVIDENCE/ec2-instance-dev.json" > "$SWITCHES"
fi
aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output json \
  --query 'Reservations[0].Instances[0].{InstanceId: InstanceId, InstanceType: InstanceType, RootVolumeId: BlockDeviceMappings[0].Ebs.VolumeId}' \
  > /tmp/ces-revisions-instance.json
jq --arg size "$SIZE" --arg date "$(date -u +%F)" --slurpfile now /tmp/ces-revisions-instance.json \
  '. + [{size: $size, date: $date} + $now[0]]' "$SWITCHES" > /tmp/ces-revisions-switches.json
mv /tmp/ces-revisions-switches.json "$SWITCHES"
rm /tmp/ces-revisions-instance.json
jq -e --arg type "$TYPE" '(map(.InstanceId) | unique | length == 1) and (map(.RootVolumeId) | unique | length == 1) and .[-1].InstanceType == $type' "$SWITCHES"
```

Expected: `true`. Underneath: EC2 changes an instance's type only while it is stopped. The instance, and the EBS root volume attached to it, stay the same objects throughout.

- [ ] **Step 6: Check the GPU, the driver, and the cap (bullets 7 and 10)**

```bash
until ssh -o ConnectTimeout=20 ces-revisions-vm true 2> /dev/null; do sleep 15; done
ssh ces-revisions-vm bash -l -s <<'EOF' 2>&1 | tee docs/decisions/cloud-gpu-evidence/vm-l4-checks.txt
echo "== kernel, GPU, and driver"
uname -r
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ls -l /dev/nvidia0
echo "== GPU cap"
shutdown --show
systemctl show ces-gpu-cap.service -p Result -p ExecMainStatus
cd ~/Projects/ces-revisions
echo "== JAX"
uv run python -c 'import jax; from ces_revisions.devices import chain_method; print("backend", jax.default_backend(), "devices", jax.local_device_count(), "chain_method(4)", chain_method(4))'
EOF
```

Run the `until` loop in the background if the instance is slow to come back. Expected:
- a kernel ending in `-aws`;
- an `NVIDIA L4` line with a driver version starting `580.`;
- the `/dev/nvidia0` device file;
- a shutdown scheduled about 8 hours after boot, with `Result=success` and `ExecMainStatus=0`;
- `backend gpu devices 1 chain_method(4) vectorized`.

Underneath: this boot is the first one on an NVIDIA GPU, so the kernel loads the prebuilt open modules, and the GPU cap's `nvidia-smi -L` creates the device files before `tests/conftest.py` looks for `/dev/nvidia0`.

- [ ] **Step 7: Run the whole test suite on `l4` (bullet 7)**

Run in the background:

```bash
ssh ces-revisions-vm bash -l -s <<'EOF' | tee -a docs/decisions/cloud-gpu-evidence/vm-l4-checks.txt
cd ~/Projects/ces-revisions
echo "== uv run pytest"
uv run pytest -q -p no:cacheprovider 2>&1 | tail -n 1
EOF
```

Expected after the capacity amendment: `433 passed`, with any warnings. `tests/test_stack.py` asserts the `gpu` backend and the
deterministic flag in `XLA_FLAGS` here, and the slow pilot tests run with `chain_method(4)`'s
`"vectorized"`, so a full pass discharges bullet 7 on `l4`.

- [ ] **Step 8: Compare two GPU runs bit for bit**

```bash
ssh ces-revisions-vm 'cat > ces-determinism.py' <<'EOF'
"""Hash one batched value and gradient of the engine, so two runs compare bit for bit."""

import hashlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from ces_revisions.engine_probe import LOG_SCALE_SD, _log_likelihood, synthetic_problem
from ces_revisions.kalman import LinearGaussianSSM

numpyro.enable_x64()
steps, states, cells, batch = (int(value) for value in sys.argv[1:5])
ssm, panel = synthetic_problem(steps, states, cells, 0.2, 0)
ssm = LinearGaussianSSM(*(jnp.asarray(array) for array in ssm))
value_and_grad = jax.jit(
    jax.vmap(jax.value_and_grad(_log_likelihood), in_axes=(0, None, None))
)
rng = np.random.default_rng(0)
log_scales = jnp.asarray(rng.normal(scale=LOG_SCALE_SD, size=(batch, 2)))
values, grads = value_and_grad(log_scales, ssm, jnp.asarray(panel))
digest = hashlib.sha256(np.asarray(values).tobytes() + np.asarray(grads).tobytes())
print(jax.default_backend(), np.asarray(values).dtype, digest.hexdigest())
EOF
ssh ces-revisions-vm bash -l -s <<'EOF' | tee -a docs/decisions/cloud-gpu-evidence/vm-l4-checks.txt
cd ~/Projects/ces-revisions
echo "== determinism at T=280, n=150, p=70, batch 16, with --xla_gpu_deterministic_ops=true"
for run in 1 2; do XLA_FLAGS=--xla_gpu_deterministic_ops=true uv run python ~/ces-determinism.py 280 150 70 16; done
echo "== the same, without the flag"
for run in 1 2; do uv run python ~/ces-determinism.py 280 150 70 16; done
EOF
```

Expected: four lines of the form `gpu float64` followed by 64 hexadecimal digits. Neither outcome fails the step: the decision record says whether each pair matched. If a line says `cpu`, stop, because JAX did not use the GPU. Underneath: the script hashes the bytes of one batched log likelihood and its gradient. Identical hashes mean identical results to the last bit, which run-to-run nondeterministic GPU kernels would break.

- [ ] **Step 9: Run the probe on `l4` (bullet 9)**

Run in the background:

```bash
ssh ces-revisions-vm bash -l -s <<'EOF'
cd ~/Projects/ces-revisions
uv run python -m ces_revisions.engine_probe --steps 280 --states 150 --cells 70 --batch 1 4 16 --label l4 --out docs/decisions/cloud-gpu-probe/l4.json
echo "probe exit status $?"
EOF
```

Then:

```bash
rsync -a ces-revisions-vm:Projects/ces-revisions/docs/decisions/cloud-gpu-probe/l4.json docs/decisions/cloud-gpu-probe/l4.json
jq -e '.label == "l4" and .backend == "gpu" and ([.batches[].batch] == [1, 4, 16]) and .dimensions == {steps: 280, states: 150, cells: 70} and (.nvidia_driver | startswith("580."))' \
  docs/decisions/cloud-gpu-probe/l4.json
jq -r '.batches[] | "batch \(.batch): median \(.median_seconds) s, compile \(.compile_seconds) s"' docs/decisions/cloud-gpu-probe/l4.json
```

Expected: `probe exit status 0`, `true`, then three lines.

- [ ] **Step 10: Append to the evidence README, scan the evidence, and commit**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Sizes: `l4`

- `size-switches.json` — after each size switch, `aws ec2 describe-instances` narrowed to the instance ID, type, and root volume ID, with the size and date. The first entry is the `dev` instance from `ec2-instance-dev.json`. Every entry has the same instance ID and the same root volume ID.
- `vm-l4-checks.txt` — on `l4`: the kernel; the GPU and driver from `nvidia-smi`; the device file; the GPU cap's scheduled poweroff; JAX's backend, device count, and `chain_method(4)`; the full test run's summary; and four hashes of one batched value and gradient, two with XLA's deterministic flag and two without.
- `../cloud-gpu-probe/l4.json` — the engine probe on `l4` at T=280, n=150, p=70, with batch sizes 1, 4, and 16.
````

Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
git add docs/decisions/cloud-gpu-probe/l4.json "$EVIDENCE/README.md" "$EVIDENCE/size-switches.json" "$EVIDENCE/vm-l4-checks.txt"
git commit -m "Check the cloud VM at l4: driver, GPU cap, tests, determinism, and probe"
```

- [ ] **Step 11: Leave `l4`**

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code L-417A185B \
  --query 'Quota.Value' --output text
```

If this prints 16.0 or more and your human partner wants to run `h100` now, go to Task 14, Step 2, from `l4`. Otherwise, switch back to `dev`. First, plan:

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -var size=dev -out=/tmp/ces-revisions-size.tfplan
tofu -chdir=infra/env show -json /tmp/ces-revisions-size.tfplan |
  jq -r '.resource_changes[] | select(.mode == "managed" and .change.actions != ["no-op"]) | "\(.change.actions | join(",")) \(.address) \(.change.before.instance_type // "") -> \(.change.after.instance_type // "")"'
rm /tmp/ces-revisions-size.tfplan
```

Expected: `update aws_instance.vm g6.xlarge -> m7i.xlarge`. **STOP:** your human partner approves the switch, which starts the instance at `dev`. On a yes, run in the background:

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm size dev -auto-approve
```

Then record the switch, stop the instance, and commit:

```bash
export AWS_PROFILE=ces-revisions
SIZE=dev
TYPE=m7i.xlarge
EVIDENCE=docs/decisions/cloud-gpu-evidence
SWITCHES="$EVIDENCE/size-switches.json"
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output json \
  --query 'Reservations[0].Instances[0].{InstanceId: InstanceId, InstanceType: InstanceType, RootVolumeId: BlockDeviceMappings[0].Ebs.VolumeId}' \
  > /tmp/ces-revisions-instance.json
jq --arg size "$SIZE" --arg date "$(date -u +%F)" --slurpfile now /tmp/ces-revisions-instance.json \
  '. + [{size: $size, date: $date} + $now[0]]' "$SWITCHES" > /tmp/ces-revisions-switches.json
mv /tmp/ces-revisions-switches.json "$SWITCHES"
rm /tmp/ces-revisions-instance.json
jq -e --arg type "$TYPE" '(map(.InstanceId) | unique | length == 1) and (map(.RootVolumeId) | unique | length == 1) and .[-1].InstanceType == $type' "$SWITCHES"
infra/bin/vm stop
git add "$SWITCHES"
git commit -m "Record the cloud VM's switch back to dev"
```

Expected: `size = "dev"` in `infra/env/size.auto.tfvars`, `true`, and the printed stop command.

### Task 13A: `l40s` capacity fallback (Verification bullets 7–10)

**Mode:** the controller runs this task inline after an `l4` start fails with
`InsufficientInstanceCapacity` and the human partner chooses the fallback. Steps 2 and 5 are
separate push and apply gates even when the user already approved adding the tier.

**Files:**
- Modify: `infra/bin/vm`, `infra/env/variables.tf`, `infra/env/instance.tf`
- Modify: `tests/test_vm_wrapper.py`
- Modify: `specs/cloud-gpu-environment.md`, this plan, `docs/cloud-gpu-runbook.md`
- Create: `docs/decisions/cloud-gpu-evidence/ec2-l4-capacity-failure.json`
- Create: `docs/decisions/cloud-gpu-evidence/ec2-l40s-fallback.json`
- Create after the run: `docs/decisions/cloud-gpu-probe/l40s.json`
- Create after the run: `docs/decisions/cloud-gpu-evidence/vm-l40s-checks.txt`,
  `size-switches.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md`

**Interfaces:**
- Consumes: Task 13's approved G and VT quota and its failed g6.xlarge start.
- Produces: `size = "l40s"`, backed by g6e.xlarge; the same checks and probe Task 13 would have
  produced for `l4`; and dated evidence for the capacity decision. `l4` remains a supported tier.

- [ ] **Step 1: Qualify and implement the fallback, test first**

Record the failed start without the request ID, and query the chosen zone for g6e.xlarge's
offering, public instance metadata, the G and VT quota, and its Linux On-Demand price. Reduce those
facts to `ec2-l4-capacity-failure.json` and `ec2-l40s-fallback.json`; neither file may contain an
account ID, bucket name, ARN, email address, or token.

Parameterize the wrapper's successful-size test over `l4` and `l40s`; observe the `l40s` case fail
because the wrapper rejects it. Then add `l40s` to the wrapper, variable validation, and instance
map as g6e.xlarge. Update the spec, runbook, active plan, evidence README, and Task 16's decision
generator and template. Historical Plan 3 evidence and its original three-type location rule stay
unchanged.

Run:

```bash
uv run pytest tests/test_vm_wrapper.py -q
tofu fmt -check infra/env/*.tf infra/env/pinned.auto.tfvars
tofu -chdir=infra/env validate
uv run ruff format --check
uv run ruff check
```

Expected: `17 passed`, `Success! The configuration is valid.`, and clean formatter and linter
results. Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`.

Commit the supported tier and its qualification evidence before planning the live switch:

```bash
git add infra/bin/vm infra/env/variables.tf infra/env/instance.tf tests/test_vm_wrapper.py \
  specs/cloud-gpu-environment.md specs/plans/4-cloud-gpu-environment.md \
  docs/cloud-gpu-runbook.md docs/decisions/cloud-gpu-evidence/README.md \
  docs/decisions/cloud-gpu-evidence/ec2-l4-capacity-failure.json \
  docs/decisions/cloud-gpu-evidence/ec2-l40s-fallback.json
git commit -m "Add the L40S capacity fallback"
```

- [ ] **Step 2: STOP — the human partner approves pushing the fallback commit**

Show the commit and its verification. Explain that pushing updates the existing feature branch so
the VM can run the exact amended code during the GPU checks. Proceed only on a clear yes that names
this push; it does not approve the later OpenTofu apply.

- [ ] **Step 3: Push the fallback commit**

```bash
BRANCH=$(git branch --show-current)
git push -u origin "$BRANCH"
```

Expected: the remote feature branch advances to the fallback commit.

- [ ] **Step 4: Plan the `l40s` switch**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -var size=l40s -out=/tmp/ces-revisions-size.tfplan
tofu -chdir=infra/env show -json /tmp/ces-revisions-size.tfplan |
  jq -r '.resource_changes[] | select(.mode == "managed" and .change.actions != ["no-op"]) | "\(.change.actions | join(",")) \(.address) \(.change.before.instance_type // "") -> \(.change.after.instance_type // "")"'
rm /tmp/ces-revisions-size.tfplan
```

Expected: `Plan: 0 to add, 1 to change, 0 to destroy.`, then
`update aws_instance.vm g6.xlarge -> g6e.xlarge`. A different resource change stops the task.

- [ ] **Step 5: STOP — the human partner approves the concrete apply**

Show the Step 4 summary. Explain that the apply changes only the stopped instance from g6.xlarge
to g6e.xlarge, starts it at \$1.861/hr, and leaves the instance ID and root volume in place. The
checks, full test suite, determinism comparison, and probe take about an hour; the GPU cap and idle
stop remain active. Proceed only on a clear yes given after this plan is shown.

- [ ] **Step 6: Switch to `l40s`**

Run in the background:

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm size l40s -auto-approve
cat infra/env/size.auto.tfvars
```

Expected: `Apply complete! Resources: 0 added, 1 changed, 0 destroyed.`, then
`size = "l40s"`. If EC2 returns a quota error, stop and report it. If it returns
`InsufficientInstanceCapacity`, leave the instance stopped, record the reduced failure evidence,
skip Steps 7–8, and continue at Task 13B only after the human partner chooses the permanent
`a10g` fallback.

- [ ] **Step 7: Update the VM checkout, then run Task 13's checks on `l40s`**

Before the checks, update the VM to the pushed commit. Task 11 generated `dev.json` on the VM
before the Mac committed it, so verify that any untracked copy is byte-identical, move it aside,
pull, and compare it with the tracked copy:

```bash
ssh ces-revisions-vm bash -l -s <<'EOF'
set -euo pipefail
cd ~/Projects/ces-revisions
BRANCH=$(git branch --show-current)
PROBE=docs/decisions/cloud-gpu-probe/dev.json
BACKUP=/tmp/ces-revisions-dev-probe.json
rm -f "$BACKUP"
git fetch origin
if [ -e "$PROBE" ] && ! git ls-files --error-unmatch "$PROBE" > /dev/null 2>&1; then
  git show "origin/$BRANCH:$PROBE" > "$BACKUP"
  cmp "$PROBE" "$BACKUP"
  mv "$PROBE" "$BACKUP"
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "STOP: the VM checkout has changes other than the verified generated probe"
  git status --short
  exit 1
fi
git pull --ff-only
if [ -e "$BACKUP" ]; then
  cmp "$BACKUP" "$PROBE"
  rm "$BACKUP"
fi
git rev-parse --short HEAD
EOF
```

Expected: both comparisons succeed, the pull fast-forwards, and the printed commit is the pushed
fallback commit. Any other checkout change stops the task for review.

Run Task 13, Steps 5–10 with this exact substitution table:

| Task 13 value | Task 13A value |
|---|---|
| `SIZE=l4` | `SIZE=l40s` |
| `TYPE=g6.xlarge` | `TYPE=g6e.xlarge` |
| `NVIDIA L4` | `NVIDIA L40S` |
| `vm-l4-checks.txt` | `vm-l40s-checks.txt` |
| `cloud-gpu-probe/l4.json` | `cloud-gpu-probe/l40s.json` |
| `--label l4` and `.label == "l4"` | `--label l40s` and `.label == "l40s"` |
| evidence heading ``## Sizes: `l4` `` | evidence heading ``## Sizes: `l40s` `` |
| commit message's `l4` | `l40s` |

The batch sizes remain 1, 4, and 16. Expected: the instance and root volume IDs remain constant;
`nvidia-smi` reports L40S with a 580-series driver; JAX reports one GPU and a vectorized
four-chain method; all 433 tests pass; all four determinism lines report `gpu float64`; and the
probe JSON passes the same schema and dimension checks.

- [ ] **Step 8: Leave `l40s`**

Run Task 13, Step 11 from `l40s`. If the P quota is at least 16 and the human partner proceeds to
Task 14, its plan must show `g6e.xlarge -> p5.4xlarge`. Otherwise plan `g6e.xlarge ->
m7i.xlarge`, obtain the existing explicit apply approval, switch to `dev`, record the switch, and
stop the instance.

### Task 13B: `a10g` capacity fallback (Verification bullets 7–10)

**Mode:** the controller runs this task inline after the `l40s` start fails with
`InsufficientInstanceCapacity` and the human partner chooses the fallback. Steps 2 and 5 are
separate push and apply gates even when the user already approved adding the tier.

**Files:**
- Modify: `infra/bin/vm`, `infra/env/variables.tf`, `infra/env/instance.tf`
- Modify: `tests/test_vm_wrapper.py`
- Modify: `specs/cloud-gpu-environment.md`, this plan, `docs/cloud-gpu-runbook.md`
- Create: `docs/decisions/cloud-gpu-evidence/ec2-l40s-capacity-failure.json`
- Create: `docs/decisions/cloud-gpu-evidence/ec2-a10g-fallback.json`
- Create after the run: `docs/decisions/cloud-gpu-probe/a10g.json`
- Create after the run: `docs/decisions/cloud-gpu-evidence/vm-a10g-checks.txt`
- Modify after the run: `docs/decisions/cloud-gpu-evidence/size-switches.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md`

**Interfaces:**
- Consumes: Task 13A's approved G and VT quota, its failed g6e.xlarge start, and the stopped
  instance whose actual type is now g6e.xlarge.
- Produces: `size = "a10g"`, backed by g5.xlarge; the same checks and probe Task 13 would have
  produced for `l4`; and dated evidence for both the failed L40S start and the A10G decision. `l4`
  and `l40s` remain supported tiers.

- [ ] **Step 1: Qualify and implement the fallback, test first**

Record the failed start without the request ID, and query the chosen zone for g5.xlarge's
offering, public instance metadata, the G and VT quota, and its Linux On-Demand price. Reduce those
facts to `ec2-l40s-capacity-failure.json` and `ec2-a10g-fallback.json`; neither file may contain an
account ID, bucket name, ARN, email address, or token.

Parameterize the wrapper's successful-size test over `l4`, `l40s`, and `a10g`; observe the `a10g`
case fail because the wrapper rejects it. Then add `a10g` to the wrapper, variable validation, and
instance map as g5.xlarge. Update the spec, runbook, active plan, evidence README, and Task 16's
decision generator and template. Historical Plan 3 evidence and its original three-type location
rule stay unchanged.

Run:

```bash
uv run pytest tests/test_vm_wrapper.py -q
tofu fmt -check infra/env/*.tf infra/env/pinned.auto.tfvars
tofu -chdir=infra/env validate
uv run ruff format --check
uv run ruff check
```

Expected: `18 passed`, `Success! The configuration is valid.`, and clean formatter and linter
results. Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`.

Commit the supported tier and its qualification evidence before planning the live switch:

```bash
git add infra/bin/vm infra/env/variables.tf infra/env/instance.tf tests/test_vm_wrapper.py \
  specs/cloud-gpu-environment.md specs/plans/4-cloud-gpu-environment.md \
  docs/cloud-gpu-runbook.md docs/decisions/cloud-gpu-evidence/README.md \
  docs/decisions/cloud-gpu-evidence/ec2-l40s-capacity-failure.json \
  docs/decisions/cloud-gpu-evidence/ec2-a10g-fallback.json
git commit -m "Add the A10G capacity fallback"
```

- [ ] **Step 2: STOP — the human partner approves pushing the fallback commit**

Show the commit and its verification. Explain that pushing updates the existing feature branch so
the VM can run the exact amended code during the GPU checks. Proceed only on a clear yes that names
this push; it does not approve the later OpenTofu apply.

- [ ] **Step 3: Push the fallback commit**

```bash
BRANCH=$(git branch --show-current)
git push -u origin "$BRANCH"
```

Expected: the remote feature branch advances to the fallback commit.

- [ ] **Step 4: Plan the `a10g` switch**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -var size=a10g -out=/tmp/ces-revisions-size.tfplan
tofu -chdir=infra/env show -json /tmp/ces-revisions-size.tfplan |
  jq -r '.resource_changes[] | select(.mode == "managed" and .change.actions != ["no-op"]) | "\(.change.actions | join(",")) \(.address) \(.change.before.instance_type // "") -> \(.change.after.instance_type // "")"'
rm /tmp/ces-revisions-size.tfplan
```

Expected: `Plan: 0 to add, 1 to change, 0 to destroy.`, then
`update aws_instance.vm g6e.xlarge -> g5.xlarge`. A different resource change stops the task.

- [ ] **Step 5: STOP — the human partner approves the concrete apply**

Show the Step 4 summary. Explain that the apply changes only the stopped instance from
g6e.xlarge to g5.xlarge, starts it at \$1.006/hr, and leaves the instance ID and root volume in
place. The checks, full test suite, determinism comparison, and probe take about an hour; the GPU
cap and idle stop remain active. Proceed only on a clear yes given after this plan is shown.

- [ ] **Step 6: Switch to `a10g`**

Run in the background:

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm size a10g -auto-approve
cat infra/env/size.auto.tfvars
```

Expected: `Apply complete! Resources: 0 added, 1 changed, 0 destroyed.`, then
`size = "a10g"`. If EC2 returns a capacity or quota error, stop and report it.

- [ ] **Step 7: Update the VM checkout, then run Task 13's checks on `a10g`**

Run Task 13A, Step 7's checkout-update block. Expected: both comparisons succeed, the pull
fast-forwards, and the printed commit is the pushed A10G fallback commit. Any other checkout change
stops the task for review.

Then run Task 13, Steps 5–10 with this exact substitution table:

| Task 13 value | Task 13B value |
|---|---|
| `SIZE=l4` | `SIZE=a10g` |
| `TYPE=g6.xlarge` | `TYPE=g5.xlarge` |
| `NVIDIA L4` | `NVIDIA A10G` |
| `vm-l4-checks.txt` | `vm-a10g-checks.txt` |
| `cloud-gpu-probe/l4.json` | `cloud-gpu-probe/a10g.json` |
| `--label l4` and `.label == "l4"` | `--label a10g` and `.label == "a10g"` |
| evidence heading ``## Sizes: `l4` `` | evidence heading ``## Sizes: `a10g` `` |
| commit message's `l4` | `a10g` |

The batch sizes remain 1, 4, and 16. Expected: the instance and root volume IDs remain constant;
`nvidia-smi` reports A10G with a 580-series driver; JAX reports one GPU and a vectorized four-chain
method; all 434 tests pass; all four determinism lines report `gpu float64`; and the probe JSON
passes the same schema and dimension checks.

- [ ] **Step 8: Leave `a10g`**

Run Task 13, Step 11 from `a10g`. If the P quota is at least 16 and the human partner proceeds to
Task 14, its plan must show `g5.xlarge -> p5.4xlarge`. Otherwise plan `g5.xlarge -> m7i.xlarge`,
obtain the existing explicit apply approval, switch to `dev`, record the switch, and stop the
instance.

### Task 14: `h100` (Verification bullets 7–9; Req 2's open item)

**Mode:** the controller runs this task inline once the P quota is approved. Step 2 and the switch in Step 10 are approval gates.

**Files:**
- Create: `docs/decisions/cloud-gpu-probe/h100.json`
- Create: `docs/decisions/cloud-gpu-evidence/vm-h100-checks.txt`
- Modify: `docs/decisions/cloud-gpu-evidence/size-switches.json`, `docs/decisions/cloud-gpu-evidence/README.md` (append a section)

**Interfaces:**
- Consumes: plan 3's request for `L-417A185B`, and Task 13B's `size-switches.json` and `~/ces-determinism.py`.
- Produces: `h100.json`; the first p5.4xlarge On-Demand start in the account, dated in `size-switches.json`; and the environment back at `dev`, stopped.

- [ ] **Step 1: Check the quota**

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code L-417A185B \
  --query 'Quota.Value' --output text
aws service-quotas list-requested-service-quota-change-history-by-quota --region "$REGION" \
  --service-code ec2 --quota-code L-417A185B \
  --query 'RequestedQuotas[].{DesiredValue: DesiredValue, Status: Status, Created: Created}' --output table
```

Expected: `16.0` or more. While the value is below 16 and the latest request reads `PENDING` or `CASE_OPENED`, this task waits. If the request reads `DENIED` or `NOT_APPROVED`, stop and report.

- [ ] **Step 2: STOP — the human partner approves running `h100`**

Tell your human partner:
- switching to `h100` starts the instance as a p5.4xlarge at \$6.88 per hour;
- the checks, the tests, and the probe take about an hour, roughly \$7–10 with the switches;
- this is the account's first p5.4xlarge On-Demand start, which can fail for lack of capacity in the zone.

Proceed only on a clear yes.

- [ ] **Step 3: Plan the switch**

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -var size=h100 -out=/tmp/ces-revisions-size.tfplan
tofu -chdir=infra/env show -json /tmp/ces-revisions-size.tfplan |
  jq -r '.resource_changes[] | select(.mode == "managed" and .change.actions != ["no-op"]) | "\(.change.actions | join(",")) \(.address) \(.change.before.instance_type // "") -> \(.change.after.instance_type // "")"'
rm /tmp/ces-revisions-size.tfplan
```

Expected: `Plan: 0 to add, 1 to change, 0 to destroy.`, then `update aws_instance.vm m7i.xlarge -> p5.4xlarge`, or `g5.xlarge -> p5.4xlarge` when coming straight from Task 13B.

- [ ] **Step 4: Switch to `h100` (Req 2's open item)**

Run in the background:

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm size h100 -auto-approve
cat infra/env/size.auto.tfvars
```

Expected: `Apply complete! Resources: 0 added, 1 changed, 0 destroyed.`, then `size = "h100"`. That success is the first p5.4xlarge On-Demand start, which discharges Req 2's open item. If the apply fails with `InsufficientInstanceCapacity`, stop and report it to your human partner. The instance is then stopped, possibly with the new type already set, and the choices are theirs: retry later, or return to `dev` by planning and applying `-var size=dev` as in Task 13, Step 11.

- [ ] **Step 5: Record the switch (bullet 8)**

```bash
export AWS_PROFILE=ces-revisions
SIZE=h100
TYPE=p5.4xlarge
EVIDENCE=docs/decisions/cloud-gpu-evidence
SWITCHES="$EVIDENCE/size-switches.json"
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output json \
  --query 'Reservations[0].Instances[0].{InstanceId: InstanceId, InstanceType: InstanceType, RootVolumeId: BlockDeviceMappings[0].Ebs.VolumeId}' \
  > /tmp/ces-revisions-instance.json
jq --arg size "$SIZE" --arg date "$(date -u +%F)" --slurpfile now /tmp/ces-revisions-instance.json \
  '. + [{size: $size, date: $date} + $now[0]]' "$SWITCHES" > /tmp/ces-revisions-switches.json
mv /tmp/ces-revisions-switches.json "$SWITCHES"
rm /tmp/ces-revisions-instance.json
jq -e --arg type "$TYPE" '(map(.InstanceId) | unique | length == 1) and (map(.RootVolumeId) | unique | length == 1) and .[-1].InstanceType == $type' "$SWITCHES"
```

Expected: `true`.

- [ ] **Step 6: Check the GPU, the driver, and the cap (bullets 7 and 10)**

```bash
until ssh -o ConnectTimeout=20 ces-revisions-vm true 2> /dev/null; do sleep 15; done
ssh ces-revisions-vm bash -l -s <<'EOF' 2>&1 | tee docs/decisions/cloud-gpu-evidence/vm-h100-checks.txt
echo "== kernel, GPU, and driver"
uname -r
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ls -l /dev/nvidia0
echo "== GPU cap"
shutdown --show
systemctl show ces-gpu-cap.service -p Result -p ExecMainStatus
cd ~/Projects/ces-revisions
echo "== JAX"
uv run python -c 'import jax; from ces_revisions.devices import chain_method; print("backend", jax.default_backend(), "devices", jax.local_device_count(), "chain_method(4)", chain_method(4))'
EOF
```

Run the `until` loop in the background if the instance is slow to come back. Expected:
- a kernel ending in `-aws`;
- an `NVIDIA H100` line with a driver version starting `580.`;
- the `/dev/nvidia0` device file;
- a shutdown scheduled about 8 hours after boot, with `Result=success` and `ExecMainStatus=0`;
- `backend gpu devices 1 chain_method(4) vectorized`.

- [ ] **Step 7: Run the whole test suite on `h100` (bullet 7)**

Run in the background:

```bash
ssh ces-revisions-vm bash -l -s <<'EOF' | tee -a docs/decisions/cloud-gpu-evidence/vm-h100-checks.txt
cd ~/Projects/ces-revisions
echo "== uv run pytest"
uv run pytest -q -p no:cacheprovider 2>&1 | tail -n 1
EOF
```

Expected: `434 passed`, with any warnings.

- [ ] **Step 8: Compare two GPU runs bit for bit**

```bash
ssh ces-revisions-vm test -f ces-determinism.py && echo "determinism script present"
ssh ces-revisions-vm bash -l -s <<'EOF' | tee -a docs/decisions/cloud-gpu-evidence/vm-h100-checks.txt
cd ~/Projects/ces-revisions
echo "== determinism at T=280, n=150, p=70, batch 16, with --xla_gpu_deterministic_ops=true"
for run in 1 2; do XLA_FLAGS=--xla_gpu_deterministic_ops=true uv run python ~/ces-determinism.py 280 150 70 16; done
echo "== the same, without the flag"
for run in 1 2; do uv run python ~/ces-determinism.py 280 150 70 16; done
EOF
```

Expected: `determinism script present`, then four lines of the form `gpu float64` followed by 64 hexadecimal digits. If the script is missing, copy it with Task 13, Step 8's first command.

- [ ] **Step 9: Run the probe on `h100`, with batch 64 (bullet 9)**

Run in the background:

```bash
ssh ces-revisions-vm bash -l -s <<'EOF'
cd ~/Projects/ces-revisions
uv run python -m ces_revisions.engine_probe --steps 280 --states 150 --cells 70 --batch 1 4 16 64 --label h100 --out docs/decisions/cloud-gpu-probe/h100.json
echo "probe exit status $?"
EOF
```

Then:

```bash
rsync -a ces-revisions-vm:Projects/ces-revisions/docs/decisions/cloud-gpu-probe/h100.json docs/decisions/cloud-gpu-probe/h100.json
jq -e '.label == "h100" and .backend == "gpu" and ([.batches[].batch] == [1, 4, 16, 64]) and .dimensions == {steps: 280, states: 150, cells: 70} and (.nvidia_driver | startswith("580."))' \
  docs/decisions/cloud-gpu-probe/h100.json
jq -r '.batches[] | "batch \(.batch): median \(.median_seconds) s, compile \(.compile_seconds) s"' docs/decisions/cloud-gpu-probe/h100.json
```

Expected: `probe exit status 0`, `true`, then four lines.

- [ ] **Step 10: Switch back to `dev`, and check the whole switch history**

Plan:

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env plan -input=false -var size=dev -out=/tmp/ces-revisions-size.tfplan
tofu -chdir=infra/env show -json /tmp/ces-revisions-size.tfplan |
  jq -r '.resource_changes[] | select(.mode == "managed" and .change.actions != ["no-op"]) | "\(.change.actions | join(",")) \(.address) \(.change.before.instance_type // "") -> \(.change.after.instance_type // "")"'
rm /tmp/ces-revisions-size.tfplan
```

Expected: `update aws_instance.vm p5.4xlarge -> m7i.xlarge`. **STOP:** your human partner approves the switch. On a yes, run in the background:

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm size dev -auto-approve
```

Then record the switch, stop the instance, and check the history:

```bash
export AWS_PROFILE=ces-revisions
SIZE=dev
TYPE=m7i.xlarge
EVIDENCE=docs/decisions/cloud-gpu-evidence
SWITCHES="$EVIDENCE/size-switches.json"
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output json \
  --query 'Reservations[0].Instances[0].{InstanceId: InstanceId, InstanceType: InstanceType, RootVolumeId: BlockDeviceMappings[0].Ebs.VolumeId}' \
  > /tmp/ces-revisions-instance.json
jq --arg size "$SIZE" --arg date "$(date -u +%F)" --slurpfile now /tmp/ces-revisions-instance.json \
  '. + [{size: $size, date: $date} + $now[0]]' "$SWITCHES" > /tmp/ces-revisions-switches.json
mv /tmp/ces-revisions-switches.json "$SWITCHES"
rm /tmp/ces-revisions-instance.json
jq -e --arg type "$TYPE" '(map(.InstanceId) | unique | length == 1) and (map(.RootVolumeId) | unique | length == 1) and .[-1].InstanceType == $type' "$SWITCHES"
infra/bin/vm stop
jq -r 'map(.size) | join(" -> ")' "$SWITCHES"
jq -e '(map(.size) | index("a10g") != null and index("h100") != null) and .[-1].size == "dev"' "$SWITCHES"
```

Expected: `true`, the printed stop command, a sequence such as `dev -> a10g -> h100 -> dev`, and
`true`. The separate capacity evidence records the unsuccessful `l4` and `l40s` attempts. With
every successful entry sharing one instance ID and one root volume ID, this discharges Verification
bullet 8.

- [ ] **Step 11: Append to the evidence README, scan the evidence, and commit**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Sizes: `h100`

- `vm-h100-checks.txt` — the checks of `vm-a10g-checks.txt`, on `h100`.
- `../cloud-gpu-probe/h100.json` — the engine probe on `h100` at T=280, n=150, p=70, with batch sizes 1, 4, 16, and 64.
- `size-switches.json` gains the switch to `h100`, the account's first p5.4xlarge On-Demand start, and the switch back to `dev`.
````

Run the evidence scan from Task 6, Step 7. Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
git add docs/decisions/cloud-gpu-probe/h100.json "$EVIDENCE/README.md" "$EVIDENCE/size-switches.json" "$EVIDENCE/vm-h100-checks.txt"
git commit -m "Check the cloud VM at h100 and record the switches back to dev"
```

### Task 15: Runbook (Req 9)

**Mode:** suitable for a subagent once Task 11 is done. The controller may run it while Tasks 12–14 wait.

**Files:**
- Create: `docs/cloud-gpu-runbook.md`

**Interfaces:**
- Consumes the names and commands of Tasks 1–14:
  - `infra/bin/vm`'s subcommands, the roots `infra/state` and `infra/env`, and their outputs;
  - `backend.hcl`, `terraform.tfvars`, `pinned.auto.tfvars`, and `size.auto.tfvars`;
  - the host entry `ces-revisions-vm`;
  - the units `ces-idle-stop.timer`, `ces-idle-stop.service`, and `ces-gpu-cap.service`, and `/etc/ces-revisions/idle-stop.env`;
  - the budget `ces-revisions-monthly`.
- Produces: `docs/cloud-gpu-runbook.md`. Task 16 links it from `CLAUDE.md`, `README.md`, and the decision record, and adds the recorded desktop connection method to its Daily use section.

- [ ] **Step 1: Write the runbook**

Create `docs/cloud-gpu-runbook.md`:

````markdown
# Cloud GPU environment runbook

How to build, use, and take down the ces-revisions cloud GPU development environment: one EC2 instance on one disk, whose instance type switches between a CPU size for daily work and four GPU sizes for fits. The decision behind it, with its evidence, is [`docs/decisions/cloud-gpu.md`](decisions/cloud-gpu.md).

Everything under `infra/` runs on the Mac, from the repository root, never on the VM. Resizing stops the instance, which would kill an apply running on it, and the instance's role has no AWS permission beyond Systems Manager. `infra/bin/vm` refuses to run on the VM.

## Conventions

- Each block that calls `aws`, `tofu`, or `infra/bin/vm` starts with `export AWS_PROFILE=ces-revisions`, the profile that `aws login` signed in. When a command reports expired credentials, run `aws login --profile ces-revisions` again; a session lasts up to 12 hours.
- `infra/bin/vm` prints each AWS CLI, OpenTofu, ssh, and rsync command before running it, so every subcommand also shows the command underneath.
- The sizes:
  - `dev` is m7i.xlarge: 4 vCPU and 16 GiB, \$0.20/hr.
  - `l4` is g6.xlarge: an NVIDIA L4 with 24 GB, \$0.81/hr.
  - `l40s` is g6e.xlarge: an NVIDIA L40S with 48 GB, \$1.861/hr.
  - `a10g` is g5.xlarge: an NVIDIA A10G with 24 GB, \$1.006/hr.
  - `h100` is p5.4xlarge: an NVIDIA H100 with 80 GB, \$6.88/hr.

  The `dev`, `l4`, and `h100` prices were checked in us-east-1 on 2026-09-13; `l40s` and `a10g`
  were checked on 2026-09-22.

## One-time setup

Plans 3 and 4 in `specs/plans/completed/` built the environment this way. Repeat these steps only for a new account or after a teardown.

### Mac tooling and sign-in

```bash
brew install awscli opentofu
brew install --cask session-manager-plugin
aws login --profile ces-revisions
```

Underneath:
- `aws login` turns a console sign-in as the IAM user into temporary credentials. It caches them in `~/.aws/login/cache` and refreshes them for up to 12 hours; the user has no access keys.
- The Session Manager plugin is the Mac's half of every shell, SSH, and port-forwarding session.
- The decision record says whether OpenTofu read the login session directly or needed a `credential_process` profile.

### Location, image, and quotas

The region and zone come from plan 3's evidence rule, and the GPU quota requests from its Task 4; both are recorded in `docs/decisions/cloud-gpu-evidence/`. The image ID comes from Canonical's public parameter:

```bash
export AWS_PROFILE=ces-revisions
REGION=$(jq -r .chosen.region docs/decisions/cloud-gpu-evidence/zone-choice.json)
aws ssm get-parameters --region "$REGION" \
  --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query 'Parameters[0].Value' --output text
```

The region, zone, and image ID are committed in `infra/env/pinned.auto.tfvars`, and the region in `infra/state/pinned.auto.tfvars`. Underneath: OpenTofu loads every `*.auto.tfvars` file in a root without being told. Pinning the image keeps a Canonical rebuild from changing the kernel between measurements.

### SSH key

```bash
ssh-keygen -t ed25519 -f ~/.ssh/ces-revisions-vm -C ces-revisions-vm
```

Underneath: OpenTofu reads the public half, and cloud-init installs it for `ubuntu` at first boot. The private half never leaves the Mac. The key reaches the VM only through Session Manager, which also needs the `aws login` session.

### State bucket

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/state init
tofu -chdir=infra/state apply
BUCKET=$(tofu -chdir=infra/state output -raw bucket)
REGION=$(jq -r .chosen.region docs/decisions/cloud-gpu-evidence/zone-choice.json)
printf 'bucket = "%s"\nkey    = "env/terraform.tfstate"\nregion = "%s"\n' "$BUCKET" "$REGION" > infra/env/backend.hcl
```

Underneath: `infra/state` creates a versioned, private, encrypted S3 bucket, named from the prefix `ces-revisions-tofu-state-` plus a unique suffix. It keeps its own state in the gitignored `infra/state/terraform.tfstate`. The gitignored `backend.hcl` tells `infra/env` where its state lives.

### The environment

Create `infra/env/terraform.tfvars`, which git ignores, with one line that sets `budget_email` to the address that receives budget alerts. Then:

```bash
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env init -backend-config=backend.hcl
tofu -chdir=infra/env plan
tofu -chdir=infra/env apply
```

Underneath:
- The apply creates:
  - the VPC, its public subnet, internet gateway, and route table;
  - a security group with no ingress rules;
  - the instance, its Systems Manager-only role, and its encrypted 100 GB gp3 root volume;
  - the daily snapshot policy.
- At first boot, cloud-init installs the SSH key and runs `infra/vm/first-boot.sh`. That script installs the cost guards, then git, the NVIDIA 580 server driver, and gh, and holds the driver and kernel packages.
- OpenTofu writes the state to the bucket, and a `.tflock` object locks it while an operation runs.

### SSH host entry

```bash
export AWS_PROFILE=ces-revisions
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
REGION=$(tofu -chdir=infra/env output -raw region)
cat >> ~/.ssh/config <<EOF

# ces-revisions cloud VM, reached through AWS Systems Manager (docs/cloud-gpu-runbook.md)
Host ces-revisions-vm
  HostName $INSTANCE_ID
  User ubuntu
  IdentityFile ~/.ssh/ces-revisions-vm
  IdentitiesOnly yes
  ProxyCommand sh -c "PATH=/opt/homebrew/bin:/usr/bin:/bin /opt/homebrew/bin/aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p' --profile ces-revisions --region $REGION"
EOF
```

Underneath: ssh runs the `ProxyCommand` instead of opening a TCP connection itself. The AWS CLI asks Systems Manager for a stream to port 22 on the instance, and the Session Manager plugin carries ssh's bytes over HTTPS through the agent's outbound connection. The absolute path and `PATH` let apps that start no login shell, such as the desktop app, find `aws` and the plugin.

### VM setup

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm sync-config
ssh ces-revisions-vm 'bash -s' < infra/vm/setup.sh
```

Then create a fine-grained GitHub token for the repository `lowmason/ces-revisions`, with Contents and Pull requests set to read and write. Enter it on the VM:

```bash
gh auth login --with-token
gh auth setup-git
```

Paste the token after the first command, then press Control-D. Finally, run `infra/bin/vm sync-config` from the Mac once more.

Underneath:
- `sync-config` copies `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, and `~/.gitconfig`, and sends the names of the Mac's skill, agent, command, and hook links.
- `setup.sh` clones this repository and `lowmason/agent-skills` into `~/Projects/`, installs uv and Python 3.14, and runs `uv sync --locked --extra cuda`. It then recreates the links into `~/Projects/agent-skills/` and reinstalls the guards.
- Both are safe to rerun. The second `sync-config` runs `gh auth setup-git` again, because the copied `~/.gitconfig` names the Mac's gh as git's credential helper.

### Cost allocation tag and budget

About a day after the first apply, activate the `project` tag in the Billing and Cost Management console, under **Cost allocation tags**. Once it shows as active:

```bash
echo 'budget_enabled = true' >> infra/env/terraform.tfvars
export AWS_PROFILE=ces-revisions
tofu -chdir=infra/env apply
```

Underneath:
- A budget reads cost data, and cost data carries only activated tags, from the activation onward.
- The budget emails at 50% and 80% of forecast spend, and forecasts need about five weeks of history. It also emails at 100% of actual spend, when its action stops the instance.

### Where the operational files live

Git does not carry these files, so removing a checkout deletes them:

| File | Holds |
|---|---|
| `infra/state/terraform.tfstate` | The state bucket's own state |
| `infra/env/backend.hcl` | The state bucket's name and region |
| `infra/env/terraform.tfvars` | The budget email address and `budget_enabled` |
| `infra/env/size.auto.tfvars` | The last applied size |

Copies of the first three are on the Mac in `~/.config/ces-revisions/infra/`, as `state-terraform.tfstate`, `backend.hcl`, and `terraform.tfvars`. To operate `infra/` from another checkout:
1. Copy them to their paths there.
2. Write `size = "dev"`, or the current size, to `infra/env/size.auto.tfvars`.
3. Run `tofu -chdir=infra/env init -backend-config=backend.hcl`.

Underneath: `infra/env`'s real state is in the bucket, so these files only point at it. A lost `infra/state/terraform.tfstate` can be rebuilt with `tofu -chdir=infra/state import aws_s3_bucket.state "$BUCKET"` and the same command for the bucket's three settings resources.

## Daily use

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm start
infra/bin/vm status
ssh ces-revisions-vm
```

Work in `~/Projects/ces-revisions` on the VM, in any of three ways:
- a terminal over SSH;
- the Claude Code desktop app connected to the host `ces-revisions-vm`;
- a Session Manager shell (`infra/bin/vm connect`), which opens as `ssm-user`; run `sudo -iu ubuntu` there.

If the desktop app cannot connect through the host entry:
1. Run `infra/bin/vm forward` in a terminal and leave it running.
2. Once, run `ssh -p 2222 -i ~/.ssh/ces-revisions-vm ubuntu@localhost true` in another terminal to accept the host key.
3. Connect the app to host `localhost`, port `2222`, user `ubuntu`, with the key `~/.ssh/ces-revisions-vm`.

When done, run `infra/bin/vm stop`. Underneath: a stopped instance bills nothing for compute, while its 100 GB volume, about \$8 a month, and its snapshots still bill. Each start gives the instance a new public IPv4 address, which nothing depends on.

The idle stop powers the VM off after 45 minutes in which the 1-minute load average stays below 0.3 and, on a GPU size, GPU utilization stays below 5%. Light editing can look idle to it. To keep the VM up until its next boot, run this on the VM:

```bash
sudo systemctl stop ces-idle-stop.timer
```

To change the window or the thresholds for good, edit `/etc/ces-revisions/idle-stop.env` on the VM; `setup.sh` leaves an existing file alone. Underneath: `ces-idle-stop.timer` runs `ces-idle-stop.service` every 5 minutes. The service records the start of an idle streak under `/run` and calls `shutdown -h now` once the streak spans the window, and EC2 turns that poweroff into a stop.

After pulling changes on the VM, rerun `~/Projects/ces-revisions/infra/vm/setup.sh`. Until cutover, rerun `infra/bin/vm sync-config` after changing `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, the links, or `~/.gitconfig` on the Mac. After cutover, the VM's own copies are the ones to edit.

## Switching sizes

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm size l4
```

Review the plan, which should change only `aws_instance.vm`'s `instance_type`, and answer `yes`.
Switch back with `infra/bin/vm size dev`, or switch to `l40s`, `a10g`, or `h100` the same way. The
first `l4` and `l40s` starts both failed with `InsufficientInstanceCapacity`; `a10g` is the next
qualified fallback in the pinned zone.

Underneath:
- OpenTofu stops the instance, changes its type, and starts it again, even if it was stopped. A switch to a GPU size therefore starts billing at that size's rate.
- The instance ID, its root volume, and everything on it stay.
- `infra/bin/vm` records the size in `infra/env/size.auto.tfvars` after a successful apply, so a later `tofu plan` keeps it.
- A GPU size needs its vCPU quota in the region: 4 in "Running On-Demand G and VT instances" for
  `l4`, `l40s`, or `a10g`, and 16 in "Running On-Demand P instances" for `h100`.
- A stop erases any local NVMe instance-store disk on the selected GPU size; nothing here uses
  those disks.

## Running a long GPU job

On a GPU size, `ces-gpu-cap.service` schedules a poweroff 8 hours after boot. Before a longer job, run on the VM:

```bash
shutdown --show
sudo shutdown -c
sudo shutdown -h +720
```

The first command shows when the poweroff is scheduled. The second cancels it, and the third reschedules it for 720 minutes from now. Run the job inside `tmux new -s fit` so it survives a dropped connection, and return to it with `tmux attach -t fit`. Keep outputs on the root volume, for example under `~/Projects/ces-revisions`.

Underneath:
- `shutdown -h +N` asks systemd-logind to power off in N minutes, replacing any earlier schedule, and logins are refused for the last five minutes.
- A cancelled cap returns at the next boot.
- The idle stop leaves a busy GPU or CPU alone, so a running job keeps the VM up, and the VM stops 45 minutes after the job ends.

## Checking spend

```bash
export AWS_PROFILE=ces-revisions
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws budgets describe-budget --account-id "$ACCOUNT_ID" --budget-name ces-revisions-monthly \
  --query 'Budget.CalculatedSpend' --output json
aws ce get-cost-and-usage --region us-east-1 --granularity MONTHLY --metrics UnblendedCost \
  --time-period Start="$(date -u +%Y-%m-01)",End="$(date -u -v+1d +%F)" \
  --query 'ResultsByTime[0].Total.UnblendedCost' --output json
```

The first command shows this month's actual and forecast spend as the budget sees it. The second shows the whole account's cost for the month so far.

Underneath:
- The budget sees only costs tagged `project = ces-revisions` since the tag's activation. It cannot see public IPv4 hours (\$0.005 per hour while running), data transfer, tax, or anything untagged, so check the account total too.
- Budget data refreshes at least daily. Cost Explorer's API bills \$0.01 per request, while the Billing console's pages are free.
- Costs from before the activation stay outside the budget unless backfilled with `aws ce start-cost-allocation-tag-backfill --region us-east-1 --backfill-from 2026-09-01T00:00:00Z`. That command takes the first of a month and can run once a day.

## Snapshots and restore

Data Lifecycle Manager snapshots the root volume every day, starting within an hour after 05:00 UTC, and keeps the latest 7:

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
VOLUME_ID=$(tofu -chdir=infra/env output -raw root_volume_id)
aws ec2 describe-snapshots --region "$REGION" --owner-ids self \
  --filters Name=volume-id,Values="$VOLUME_ID" \
  --query 'sort_by(Snapshots, &StartTime)[].{SnapshotId: SnapshotId, StartTime: StartTime, State: State}' \
  --output table
```

To restore the whole disk to the latest snapshot, with the instance running:

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
VOLUME_ID=$(tofu -chdir=infra/env output -raw root_volume_id)
SNAPSHOT_ID=$(aws ec2 describe-snapshots --region "$REGION" --owner-ids self \
  --filters Name=volume-id,Values="$VOLUME_ID" \
  --query 'sort_by(Snapshots, &StartTime)[-1].SnapshotId' --output text)
aws ec2 create-replace-root-volume-task --region "$REGION" --instance-id "$INSTANCE_ID" \
  --snapshot-id "$SNAPSHOT_ID"
aws ec2 describe-replace-root-volume-tasks --region "$REGION" \
  --filters Name=instance-id,Values="$INSTANCE_ID"
```

For an earlier snapshot, change `[-1]` to `[-2]`, `[-3]`, and so on. Once the task's state is `succeeded`:

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
tofu -chdir=infra/env apply -refresh-only
aws ec2 describe-volumes --region "$REGION" --filters Name=attachment.instance-id,Values="$INSTANCE_ID" \
  --query 'Volumes[].{VolumeId: VolumeId, Tags: Tags}' --output json
```

If the new volume lacks the `project = ces-revisions` tag, add it with `aws ec2 create-tags --region "$REGION" --resources` followed by the volume's ID and `--tags Key=project,Value=ces-revisions`. The snapshot policy and the budget both find volumes by that tag.

Underneath:
- EC2 reboots the instance onto a new root volume built from the snapshot. The instance ID, network, and role stay, and memory contents are lost.
- The snapshot must come from this instance's root volume.
- The old volume is detached and kept, and it bills until deleted.
- The new volume has a new ID, which `-refresh-only` records in OpenTofu's state.

## Token rotation

Before the GitHub token expires, create a new fine-grained token with the same scope: the repository `lowmason/ces-revisions`, with Contents and Pull requests set to read and write. On the VM:

```bash
gh auth login --with-token
gh auth status
```

Paste the new token after the first command, then press Control-D. Then revoke the old token on github.com, under Settings → Developer settings → Personal access tokens → Fine-grained tokens.

Underneath: `gh` replaces the stored token for github.com in `~/.config/gh/hosts.yml`. git keeps using `gh` as its credential helper, so nothing else changes. The token opens only `lowmason/ces-revisions`, so `gh` and `git push` against other repositories fail on the VM by design.

## Updating the held driver or the pinned image

The driver, its kernel modules, and the AWS kernel are held so they change only on purpose. To update them, run on the VM:

```bash
sudo apt-mark unhold nvidia-headless-no-dkms-580-server-open nvidia-utils-580-server \
  linux-modules-nvidia-580-server-open-aws linux-aws linux-image-aws linux-headers-aws
sudo apt-get update
sudo apt-get install --only-upgrade nvidia-headless-no-dkms-580-server-open nvidia-utils-580-server \
  linux-modules-nvidia-580-server-open-aws linux-aws linux-image-aws linux-headers-aws
sudo apt-mark hold nvidia-headless-no-dkms-580-server-open nvidia-utils-580-server \
  linux-modules-nvidia-580-server-open-aws linux-aws linux-image-aws linux-headers-aws
```

Then, from the Mac:
1. Run `infra/bin/vm stop` and `infra/bin/vm start`.
2. On a GPU size, check `nvidia-smi`, and rerun the tests and the engine probe.
3. Record the new versions in the decision record.

Underneath: the modules are prebuilt for one kernel, so the kernel and the modules have to move together, and the new kernel runs only after the next boot.

The pinned image matters only when an instance is created. Changing `ami_id` makes OpenTofu plan a replacement of the instance, which `prevent_destroy` refuses. To move to a newer image, rebuild:
1. Keep a snapshot of the old root volume; snapshots outlive their volume.
2. Read the new image ID as in One-time setup, and update `infra/env/pinned.auto.tfvars`.
3. Set `prevent_destroy = false` in `infra/env/instance.tf` and apply, which replaces the instance and its root volume.
4. Set `prevent_destroy = true` again, update the host entry's `HostName`, and repeat the VM setup.

## Recovery after a budget stop

When the month's actual spend reaches 100% of the budget, AWS Budgets sends an email and its action stops the instance. To see the action's state:

```bash
export AWS_PROFILE=ces-revisions
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws budgets describe-budget-actions-for-budget --account-id "$ACCOUNT_ID" \
  --budget-name ces-revisions-monthly --query 'Actions[].{Status: Status, ActionId: ActionId}' --output table
```

Decide whether to wait for the next month or raise the ceiling. To raise it, add a line such as `monthly_budget_usd = "200"` to `infra/env/terraform.tfvars` and run `tofu -chdir=infra/env apply`. Then re-arm the action and start the instance:

```bash
export AWS_PROFILE=ces-revisions
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ACTION_ID=$(tofu -chdir=infra/env output -raw budget_action_id)
aws budgets execute-budget-action --account-id "$ACCOUNT_ID" --budget-name ces-revisions-monthly \
  --action-id "$ACTION_ID" --execution-type RESET_BUDGET_ACTION
infra/bin/vm start
```

Underneath: an action aimed at specific instances runs once and does not re-arm at the next budget period. Without the reset, the next month has no automatic stop. Roadmap Stage 6 revisits the ceiling itself.

## Teardown

Teardown deletes the VM's disk, so copy off anything worth keeping first. Snapshots outlive the volume until step 3 deletes them.

1. In `infra/env/instance.tf`, set `prevent_destroy = false`. Then destroy the environment:

   ```bash
   export AWS_PROFILE=ces-revisions
   tofu -chdir=infra/env destroy
   ```

2. Remove the Mac's SSH host entry, the `Host ces-revisions-vm` block in `~/.ssh/config`, and the key pair `~/.ssh/ces-revisions-vm` and `~/.ssh/ces-revisions-vm.pub`. On github.com, revoke the VM's token.
3. Delete the lifecycle snapshots, which outlive their policy:

   ```bash
   export AWS_PROFILE=ces-revisions
   REGION=$(jq -r .chosen.region docs/decisions/cloud-gpu-evidence/zone-choice.json)
   for snapshot in $(aws ec2 describe-snapshots --region "$REGION" --owner-ids self \
     --filters Name=tag:project,Values=ces-revisions --query 'Snapshots[].SnapshotId' --output text); do
     aws ec2 delete-snapshot --region "$REGION" --snapshot-id "$snapshot"
   done
   ```

4. In `infra/state/main.tf`, set `prevent_destroy = false`. Then empty the state bucket, including every object version, and destroy it:

   ```bash
   export AWS_PROFILE=ces-revisions
   BUCKET=$(tofu -chdir=infra/state output -raw bucket)
   aws s3api list-object-versions --bucket "$BUCKET" --output json \
     --query '{Objects: [Versions, DeleteMarkers][] | [].{Key: Key, VersionId: VersionId}}' \
     > /tmp/ces-revisions-versions.json
   aws s3api delete-objects --bucket "$BUCKET" --delete file:///tmp/ces-revisions-versions.json
   tofu -chdir=infra/state destroy
   ```

5. Optionally, deactivate the `project` cost allocation tag in the Billing console.

Underneath:
- `prevent_destroy` makes OpenTofu refuse any plan that deletes the protected resource, so it has to come out of the configuration first. Commit neither change.
- `infra/env` goes before `infra/state`, because the bucket holds `infra/env`'s state.
- S3 refuses to delete a bucket that holds objects, and a versioned bucket keeps every earlier version and delete marker, which `list-object-versions` returns.

## Troubleshooting

### Insufficient capacity

A size switch fails with `InsufficientInstanceCapacity`: AWS has no spare instance of that type in
the zone at the moment. The instance is left stopped, possibly with the new type already set.
Switch back with `infra/bin/vm size dev`, try the GPU size again later, or try another configured
GPU tier. The first `l4` and `l40s` starts both failed this way, so `a10g` is the next qualified
fallback.

Underneath: On-Demand capacity is counted per zone and per type, and AWS does not queue a request that finds none. The environment stays in one zone, because its subnet and its volume do.

### Quota errors

A size switch fails with `VcpuLimitExceeded`: the account's vCPU quota for that instance family in the region is below the size's vCPU count.

```bash
export AWS_PROFILE=ces-revisions
REGION=$(tofu -chdir=infra/env output -raw region)
aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code L-DB2E81BA \
  --query 'Quota.Value'
aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code L-417A185B \
  --query 'Quota.Value'
aws service-quotas list-requested-service-quota-change-history --region "$REGION" --service-code ec2 \
  --query 'RequestedQuotas[].{Quota: QuotaName, Desired: DesiredValue, Status: Status}' --output table
```

`l4`, `l40s`, and `a10g` each need at least 4 in `L-DB2E81BA` ("Running On-Demand G and VT
instances"), and `h100` needs at least 16 in `L-417A185B` ("Running On-Demand P instances").
Request an increase in the Service Quotas console, and switch back to `dev` meanwhile. Underneath:
EC2 quotas count the vCPUs of running instances per family, not the instances themselves.

### Session Manager not connecting

Work down this list:

1. The instance is running: `infra/bin/vm status`. The idle stop or the GPU cap may have stopped it.
2. The login session is current: `aws sts get-caller-identity --profile ces-revisions`. If it is not, run `aws login --profile ces-revisions`.
3. The agent is online:

   ```bash
   export AWS_PROFILE=ces-revisions
   REGION=$(tofu -chdir=infra/env output -raw region)
   INSTANCE_ID=$(tofu -chdir=infra/env output -raw instance_id)
   aws ssm describe-instance-information --region "$REGION" \
     --filters Key=InstanceIds,Values="$INSTANCE_ID" \
     --query 'InstanceInformationList[0].{PingStatus: PingStatus, LastPingDateTime: LastPingDateTime}'
   aws ec2 get-console-output --region "$REGION" --instance-id "$INSTANCE_ID" --latest --output text | tail -n 40
   ```

   An empty result, or `ConnectionLost` a few minutes after a start, points at the instance, and its console output shows how far the boot got.
4. The plugin is installed: `session-manager-plugin --version`.
5. For SSH, `ssh -v ces-revisions-vm` shows whether the `ProxyCommand`, the host key, or the user key fails.

Underneath: the SSM agent on the instance opens outbound HTTPS connections to Systems Manager and waits, and every session rides those connections. The security group's lack of ingress rules therefore never matters, but the instance's outbound route through the internet gateway does.

### JAX not seeing the GPU

On a GPU size, in `~/Projects/ces-revisions`:

```bash
nvidia-smi
ls -l /dev/nvidia0
uv run python -c 'import jax; print(jax.default_backend(), jax.devices())'
uv pip list | grep -i -e jax -e nvidia
```

- `nvidia-smi` reports a driver/library version mismatch: the driver packages or the kernel moved without a boot. See Updating the held driver or the pinned image.
- `nvidia-smi` works but `/dev/nvidia0` is missing: run `sudo nvidia-smi -L`, which the GPU cap also runs at boot, and try again.
- `jax.default_backend()` prints `cpu` on a GPU size: the `cuda` extra is missing; run `uv sync --locked --extra cuda`.
- On `dev`, an ERROR traceback from the CUDA plugin before JAX falls back to the CPU is expected, and `export JAX_PLATFORMS=cpu` silences it.

Underneath: JAX's CUDA plugin needs NVIDIA driver 580 or later and a device file. When either is missing, it logs the failure and falls back to the CPU instead of raising, so check for a `gpu` backend rather than assuming it.
````

- [ ] **Step 2: Check the runbook's commands and Markdown**

```bash
grep -o 'infra/bin/vm [a-z-]*' docs/cloud-gpu-runbook.md | sort -u
grep -n '^## ' docs/cloud-gpu-runbook.md
if grep -n '[^\\]\$[0-9]' docs/cloud-gpu-runbook.md; then
  echo "STOP: an unescaped dollar amount"
else
  echo "dollar amounts escaped"
fi
uv run ruff format --check docs/cloud-gpu-runbook.md
```

Expected:
- the subcommands `connect`, `forward`, `size`, `start`, `status`, `stop`, and `sync-config`, each after `infra/bin/vm`;
- the headings Conventions, One-time setup, Daily use, Switching sizes, Running a long GPU job, Checking spend, Snapshots and restore, Token rotation, Updating the held driver or the pinned image, Recovery after a budget stop, Teardown, and Troubleshooting;
- `dollar amounts escaped`;
- `1 file already formatted`.

The spec's list maps onto those headings, and the troubleshooting subsections cover its four problems.

- [ ] **Step 3: Commit**

```bash
uv run ruff format && uv run ruff check
git add docs/cloud-gpu-runbook.md
git commit -m "Add the cloud GPU environment runbook"
```

### Task 16: Decision record, documentation, and cutover (Req 10; Verification bullets 2, 4, 11, 13, and 14)

**Mode:** the controller runs this task inline. Step 2 is the human partner's.

**Files:**
- Create: `docs/decisions/cloud-gpu.md`
- Create: `docs/decisions/cloud-gpu-evidence/vm-claude-session.json`
- Modify: `docs/cloud-gpu-runbook.md` (Daily use), `CLAUDE.md`, `README.md`, `docs/decisions/cloud-gpu-evidence/README.md`

**Interfaces:**
- Consumes:
  - every evidence file from plan 3 and Tasks 6–14, and the four probe records;
  - Task 15's runbook;
  - Task 2's `infra/bin/vm sync-config --cutover`;
  - the committed-tree search in Task 8, Step 11, and the evidence scan in Task 6, Step 7.
- Produces:
  - `docs/decisions/cloud-gpu.md`, which roadmap Stage 6 consumes;
  - project memory on the VM, which the VM owns from here on;
  - `CLAUDE.md` and `README.md` entries for `infra/`, the runbook, and the decision record.

- [ ] **Step 1: Copy project memory to the VM (cutover)**

```bash
export AWS_PROFILE=ces-revisions
infra/bin/vm start
until ssh -o ConnectTimeout=20 ces-revisions-vm true 2> /dev/null; do sleep 15; done
infra/bin/vm sync-config --cutover
ls ~/.claude/projects/-Users-lowell-Projects-ces-revisions/memory
ssh ces-revisions-vm 'ls .claude/projects/-home-ubuntu-Projects-ces-revisions/memory'
```

Run the `until` loop in the background if the instance is slow to come back. Expected: the printed commands, ending with `+ rsync -a …/memory/ ces-revisions-vm:.claude/projects/-home-ubuntu-Projects-ces-revisions/memory/`. The two listings match and include `MEMORY.md`.

Underneath:
- Claude Code keys a project's memory by the project's path, with every character other than a letter, digit, or hyphen turned into a hyphen. It keys a worktree by its main checkout.
- On the VM, the project's path is `/home/ubuntu/Projects/ces-revisions`.
- The copy runs once: `sync-config --cutover` refuses when the VM already has a `MEMORY.md`.
- From here on, the VM owns project memory, and the Mac's copy stays as it was.

- [ ] **Step 2: STOP — the human partner checks a Claude Code session on the VM (bullet 11)**

Ask your human partner to open a Claude Code session on the VM in `~/Projects/ces-revisions`, the way `docs/decisions/cloud-gpu-evidence/access.json` records: through the desktop app, or by running `claude` in a terminal SSH session. If Claude Code is not yet installed on the VM because the desktop app never connected, they install it there first, following Claude Code's setup documentation. In that session, they ask two questions:
1. "Which personal skills do you have?" A list that includes `writing-plans` passes.
2. "What does your project memory say about concurrent sessions and the main checkout?" An answer drawn from the memory note on sessions sharing the main checkout passes.

They report yes or no for each. Set the two values to their answers and record them:

```bash
SKILLS=true
MEMORY=true
jq -n --argjson skills "$SKILLS" --argjson memory "$MEMORY" \
  '{lists_personal_skills: $skills, reads_project_memory: $memory}' \
  > docs/decisions/cloud-gpu-evidence/vm-claude-session.json
jq -e '.lists_personal_skills and .reads_project_memory' docs/decisions/cloud-gpu-evidence/vm-claude-session.json
```

Expected: `true`. If it prints `false`, stop and report which check failed.

- [ ] **Step 3: Print the decision record's facts**

```bash
uv run python - <<'EOF'
"""Print the decision record's facts, ready to paste, from the committed evidence."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

EVIDENCE = Path("docs/decisions/cloud-gpu-evidence")
PROBE = Path("docs/decisions/cloud-gpu-probe")
HOSTS = ("mac", "dev", "a10g", "h100")
GPU_HOSTS = ("a10g", "h100")


def load(name):
    return json.loads((EVIDENCE / name).read_text())


def listing(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def median(host, batch):
    return next(e["median_seconds"] for e in records[host]["batches"] if e["batch"] == batch)


def determinism(host):
    text = (EVIDENCE / f"vm-{host}-checks.txt").read_text()
    found = re.findall(r"^gpu float64 ([0-9a-f]{64})$", text, flags=re.MULTILINE)
    if len(found) != 4:
        raise SystemExit(f"vm-{host}-checks.txt holds {len(found)} hashes, not 4")
    with_flag = "matched" if found[0] == found[1] else "differed"
    without = "matched" if found[2] == found[3] else "differed"
    return f"on `{host}`, two runs {with_flag} bit for bit with XLA's deterministic flag and {without} without it"


choice = load("zone-choice.json")
region, zone = choice["chosen"]["region"], choice["chosen"]["zone"]
chosen = next(r for r in choice["evaluated"] if r["region"] == region)
image = load(f"ec2-describe-images-{region}.json")
credentials = load("opentofu-credentials.json")
switches = load("size-switches.json")
bls = load("bls-canary.json")
capacity_failure = load("ec2-l4-capacity-failure.json")
l40s_fallback = load("ec2-l40s-fallback.json")
l40s_capacity_failure = load("ec2-l40s-capacity-failure.json")
a10g_fallback = load("ec2-a10g-fallback.json")
records = {host: json.loads((PROBE / f"{host}.json").read_text()) for host in HOSTS}
a10g_checks = (EVIDENCE / "vm-a10g-checks.txt").read_text()
dev_checks = (EVIDENCE / "vm-dev-checks.txt").read_text()
idle = re.search(r"^state stopped after (\d+) minutes", (EVIDENCE / "vm-idle-stop.txt").read_text(), re.MULTILINE)
drivers = {records[host]["nvidia_driver"] for host in GPU_HOSTS}
access = {
    "host-entry": "The Claude Code desktop app connects through the `ces-revisions-vm` SSH host entry, whose `ProxyCommand` opens Session Manager's `AWS-StartSSHSession`.",
    "port-forward": "The Claude Code desktop app could not use the host entry's `ProxyCommand`, so it connects to `localhost:2222`, which `infra/bin/vm forward` forwards to the VM's SSH port through `AWS-StartPortForwardingSession`.",
    "neither": "The Claude Code desktop app connected neither through the host entry nor through a forwarded port, so work on the VM runs in terminal SSH sessions through the host entry and in Session Manager shells.",
}

print("date:", datetime.now(UTC).date().isoformat())
print("zone:", zone)
print("region:", region)
print("regions evaluated:", listing(r["region"] for r in choice["evaluated"]))
print("p5 price:", f"{min(chosen['p5_4xlarge_on_demand_usd_per_hour']):.2f}")
print("zones offering all three:", listing(f"`{z}`" for z in chosen["zones_offering_all_three"]))
print("l4 capacity date:", capacity_failure["date"])
print("l40s capacity date:", l40s_capacity_failure["date"])
print("l40s price:", l40s_fallback["on_demand_usd_per_hour"])
print("a10g price:", a10g_fallback["on_demand_usd_per_hour"])
print("image name:", image["Name"])
print("ami id:", image["ImageId"])
print("kernel:", re.search(r"^\S+-aws$", a10g_checks, re.MULTILINE).group(0))
print("driver:", drivers.pop() if len(drivers) == 1 else listing(f"{records[host]['nvidia_driver']} on `{host}`" for host in GPU_HOSTS))
print("opentofu version:", credentials["opentofu"].removeprefix("OpenTofu v"))
print("provider version:", credentials["aws_provider"])
print("credential phrase:", "through a `credential_process` profile that runs `aws configure export-credentials`" if credentials["credential_process_needed"] else "directly")
print("access sentence:", access[load("access.json")["desktop_app"]])
print("first p5 start date:", next(s["date"] for s in switches if s["InstanceType"] == "p5.4xlarge"))
print("switch sequence:", " → ".join(f"`{s['size']}`" for s in switches))
print("jax sentence:", "ran on the CPU without a message from its CUDA plugin" if "no CUDA message on stderr" in dev_checks else "logged an error from its CUDA plugin, then ran on the CPU")
print("determinism sentence:", "; ".join(determinism(host) for host in GPU_HOSTS))
print("idle stop minutes:", idle.group(1))
print("bls sentence:", f"was {bls['outcome']} (HTTP {bls['http_status']}) on {bls['date']}")
print("mac batch 1:", f"{median('mac', 1):.3g}")
print("mac batch 16:", f"{median('mac', 16):.3g}")
for batch in (1, 16):
    mac = median("mac", batch)
    parts = [f"`{host}` {median(host, batch):.3g} s ({mac / median(host, batch):.2g}× the Mac's speed)" for host in HOSTS[1:]]
    print(f"probe summary {batch}: At batch {batch}, the Mac took {mac:.3g} s per call, {listing(parts)}.")
print(f"probe summary 64: At batch 64, `h100` took {median('h100', 64):.3g} s per call.")
quotas = load(f"service-quotas-get-service-quota-{region}.json")
requests = load(f"service-quotas-request-increase-{region}.json")
for quota in quotas:
    asked = [f"{r['DesiredValue']:g}" for r in requests if r["QuotaCode"] == quota["QuotaCode"]]
    print(f"quota row: | {quota['QuotaName']} (`{quota['QuotaCode']}`) | {quota['Value']:g} | {', '.join(asked) or 'not requested'} |")
for host in HOSTS:
    for entry in records[host]["batches"]:
        print(f"probe row: | `{host}` | {records[host]['backend']} | {entry['batch']} | {entry['compile_seconds']:.2f} | {entry['median_seconds']:.3g} | {entry['iqr_seconds']:.2g} |")
EOF
```

Expected: one line for each slot named in Step 4, then three `quota row` lines and 13 `probe row` lines. A `SystemExit` or `StopIteration` names evidence that is missing or malformed; stop and report it. Underneath: every measured value in the record comes from this output, so no number is copied by hand.

- [ ] **Step 4: Write the decision record**

Create `docs/decisions/cloud-gpu.md` from the text below. Replace each `{{name}}` slot with the text Step 3 printed after `name:`. Replace the `{{quota rows}}` slot with the `quota row` lines and the `{{probe rows}}` slot with the `probe row` lines, in each case without the prefix.

````markdown
# Develop on one AWS instance that switches between a CPU size and four GPU sizes

- **Status:** Accepted
- **Date:** {{date}}
- **Deciders:** Lowell Mason
- **Blast radius:** where the project's code runs and what that costs: `infra/`, the `cuda` extra, `src/ces_revisions/devices.py` and `engine_probe.py`, and roadmap Stage 6, which measures the engine on this environment

## Context

The engine runs in float64 by mandate: Req 17 of [`specs/ces-revisions.md`](../../specs/ces-revisions.md) requires it, and `tests/conftest.py` enables it for every test. NVIDIA's Ampere (GA102) and Ada (AD102) architecture whitepapers state that those chips carry two FP64 units per streaming multiprocessor and run FP64 at 1/64 of their FP32 rate. NVIDIA rates the A100 at 9.7 and the H100 at 34 FP64 TFLOPS, and by the 1/64 rule the L4, A10G, and L40S run float64 at about 0.5, 0.5, and 1.4.

A single chain's `lax.scan` launches each operation as its own GPU kernel, so a cheap GPU may keep pace there. Batched work, `vmap` over chains or simulation-based calibration replicates, multiplies the arithmetic in each kernel, and there the FP64 rating decides. The design of [`specs/cloud-gpu-environment.md`](../../specs/cloud-gpu-environment.md) left that crossover to measurement.

Roadmap Stage 6 consumes this environment and this record, and measures the engine on it at Stage 7–9 dimensions. That measurement belongs in `docs/decisions/seasonal-state.md`, not here.

The Mac came first. On the M4 Max, the engine probe (`python -m ces_revisions.engine_probe`) times `jax.value_and_grad` of the Kalman filter's log likelihood, vmapped over a batch of two log-scale parameters. At T=280, n=150, p=70 it took {{mac batch 1}} s per call at batch 1 and {{mac batch 16}} s at batch 16. At that pace, a four-chain fit at Stage 7 scale projects to 10–40 hours before simulation-based calibration multiplies it.

Learning the cloud stack is also a project goal, so [`docs/cloud-gpu-runbook.md`](../cloud-gpu-runbook.md) explains what happens underneath each step.

## Decision

We will develop on one EC2 instance in `{{zone}}` ({{region}}), built with OpenTofu, whose instance type switches among five sizes on one root volume:

| Size | Instance type | GPU | On-Demand in us-east-1 |
|---|---|---|---|
| `dev` | m7i.xlarge | none | \$0.20/hr |
| `l4` | g6.xlarge | L4, 24 GB | \$0.81/hr |
| `l40s` | g6e.xlarge | L40S, 48 GB | \${{l40s price}}/hr |
| `a10g` | g5.xlarge | A10G, 24 GB | \${{a10g price}}/hr |
| `h100` | p5.4xlarge | H100, 80 GB | \$6.88/hr |

The original three prices were checked on 2026-09-13; `l40s` and `a10g` were checked on 2026-09-22.

- **Image and driver:** Canonical's `{{image name}}` (`{{ami id}}`), pinned, with Ubuntu's NVIDIA 580 server driver {{driver}} and NVIDIA's open kernel modules on kernel `{{kernel}}`, held with `apt-mark hold`.
- **Access:** Session Manager, with no inbound port. {{access sentence}}
- **Infrastructure as code:** OpenTofu {{opentofu version}} with the hashicorp/aws provider {{provider version}}. State lives in a versioned S3 bucket, locked with `use_lockfile`, and OpenTofu reads the `aws login` session {{credential phrase}}.
- **Guards:**
  - an idle stop after 45 idle minutes;
  - a poweroff 8 hours after boot on the GPU sizes;
  - a \$150 monthly budget on the `project` tag, whose action stops the instance at 100% of actual spend;
  - daily snapshots, kept for 7 days.

## Evidence

The command outputs are in [`docs/decisions/cloud-gpu-evidence/`](cloud-gpu-evidence/), whose README says which command produced each file, and the probe records are in [`docs/decisions/cloud-gpu-probe/`](cloud-gpu-probe/).

### Location and quotas

Req 2's original rule evaluated {{regions evaluated}} and chose `{{zone}}`. In {{region}}, the Price List API had a Linux On-Demand p5.4xlarge price of \${{p5 price}} per hour, and {{zones offering all three}} offered the original three instance types. On {{l4 capacity date}}, the first g6.xlarge start failed with `InsufficientInstanceCapacity`; a live check established that the chosen zone offered g6e.xlarge at \${{l40s price}} per hour under the approved 4-vCPU G and VT quota. Its first start failed with the same error on {{l40s capacity date}}. A second live check established that the zone offered g5.xlarge at \${{a10g price}} per hour under that quota. The account's first p5.4xlarge On-Demand start succeeded on {{first p5 start date}}, which settles the question AWS's August 2025 announcement raised about single-GPU P5 On-Demand in US regions.

| Quota | Prior value (vCPUs) | Requested |
|---|---|---|
{{quota rows}}

### Engine probe

At T=280, n=150, p=70, with 20% of panel cells missing, each batch size ran one compiling call and then 10 timed calls:

| Host | Backend | Batch | Compile (s) | Median per call (s) | IQR (s) |
|---|---|---|---|---|---|
{{probe rows}}

{{probe summary 1}} {{probe summary 16}} {{probe summary 64}}

### Machine checks

- On `dev`, with the `cuda` extra installed and no GPU, JAX {{jax sentence}}. No separate CPU environment was needed, and the full test suite passed with the `cpu` backend and four host devices.
- On `a10g` and `h100`, `nvidia-smi` reported the GPU with driver {{driver}}. The full test suite passed with the `gpu` backend and `chain_method(4) == "vectorized"`, and the GPU cap scheduled a poweroff 8 hours after boot. The `l4` and `l40s` tiers remain configured for later capacity retries.
- Repeated runs of one batched value and gradient at batch 16: {{determinism sentence}}.
- The size switches {{switch sequence}} kept one instance ID and one root volume ID.
- With its window shortened to 10 minutes, the idle stop stopped the instance within {{idle stop minutes}} minutes of the VM going idle.
- The budget exists with its three alerts and its stop action, and lifecycle snapshots of the root volume exist. IAM's policy simulator allows the action's role to stop this instance, and denies stopping another instance or terminating this one.

### BLS

One by-hand fetch of a small file from `download.bls.gov`, run from the VM with a User-Agent naming the project and a contact, {{bls sentence}}.

## Consequences

- **Positive:**
  - One environment on one disk: the checkout, the uv environment, and Claude Code's settings and memory persist across sizes, and a size switch is one `infra/bin/vm size` command.
  - Nothing listens on the internet. The security group has no ingress rules, and every session rides the SSM agent's outbound connection.
  - Cost is bounded three ways: the idle stop, the GPU cap, and the budget action.
  - Every AWS resource is in OpenTofu, so the runbook can rebuild the environment or tear it down.
- **Negative:**
  - A size switch stops the VM and whatever runs on it, and a switch to a GPU size starts billing as soon as the instance starts.
  - The budget sees only costs tagged since the tag's activation. Public IPv4 hours, data transfer, and tax fall outside it.
  - GPU capacity is per zone and per instance type. The first g6.xlarge and g6e.xlarge starts failed for lack of capacity, and any GPU type can do the same; the environment cannot move zones without a rebuild.
  - The held driver and kernel receive no security updates until someone updates them by hand.
- **Neutral / follow-on:**
  - Roadmap Stage 6 measures the engine at Stage 7–9 dimensions on this environment, and revisits the budget and the default GPU size.

## Alternatives considered

- **Azure NC24ads A100 v4 (\$3.67/hr).** Rejected. An account with AWS billing history was the better bet for quick GPU quota approval, and p5.4xlarge gives more float64 per dollar: 34 FP64 TFLOPS at \$6.88/hr against 9.7 at \$3.67/hr. Azure returns only under Req 2's stopping rule.
- **A CPU dev box plus disposable GPU runners.** Rejected for now: two environments to keep current, and an S3 job flow before any stage needs parallel fits.
- **The Mac as the dev machine, with cloud GPUs only for fits.** Rejected: it reverses the move to remote development.
- **AWS's Deep Learning Base AMI.** Rejected: it lists only GPU instance families as supported, so it cannot serve `dev`, and JAX's CUDA wheels bring their own CUDA libraries.
- **IAM Identity Center.** Rejected: it would make the account an AWS Organizations management account, while an IAM user with MFA and `aws login` already gives temporary credentials.
- **A private subnet with a NAT gateway.** Rejected: about \$33 a month, or Systems Manager VPC endpoints, for no benefit when nothing listens on the public subnet.
- **SageMaker and AWS Batch.** Rejected: both run jobs rather than a development machine, and neither keeps a checkout and a Claude Code session between fits.
- **T4 (g4dn) fallback size.** Rejected for this capacity fallback: it has only 16 GB of GPU memory and lower estimated float64 throughput than A10G, while A10G fits the existing 4-vCPU G and VT quota.

## Trade-offs & reversibility

This is a two-way door. Everything is in `infra/`, the instance type is a variable, and the runbook's teardown removes the environment. The repository's lasting changes are the `cuda` extra, `devices.py`, and the probe, which run on any host.

Revisit this decision if any of these occurs:
- Stage 6's measurement, which also revisits the \$150 budget and the default GPU size;
- a later stage needs parallel fits, which is the case for the dev-box-plus-runners topology;
- p5.4xlarge loses On-Demand availability in `{{zone}}`;
- the probe shows `dev` competitive with the GPU sizes.
````

Underneath: the record follows `docs/decisions/engine.md`'s shape: status lines, Context, Decision, Evidence, Consequences, Alternatives considered, and Trade-offs & reversibility with its revisit triggers.

- [ ] **Step 5: Check the decision record**

```bash
if grep -n '{{' docs/decisions/cloud-gpu.md; then
  echo "STOP: a slot is still unfilled"
else
  echo "no unfilled slots"
fi
if grep -n '[^\\]\$[0-9]' docs/decisions/cloud-gpu.md; then
  echo "STOP: an unescaped dollar amount"
else
  echo "dollar amounts escaped"
fi
grep -c '^| `' docs/decisions/cloud-gpu.md
uv run ruff format --check docs/decisions/cloud-gpu.md
```

Expected: `no unfilled slots`, `dollar amounts escaped`, `18` (the five size rows plus the 13 probe rows), and `1 file already formatted`.

- [ ] **Step 6: Name the verified connection method in the runbook**

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

method = json.loads(Path("docs/decisions/cloud-gpu-evidence/access.json").read_text())["desktop_app"]
sentence = {
    "host-entry": "When the environment was built, the desktop app connected through the host entry.",
    "port-forward": "When the environment was built, the desktop app could not use the host entry and connected through the forwarded port.",
    "neither": "When the environment was built, the desktop app connected neither way, so work ran in terminal SSH sessions and Session Manager shells.",
}[method]
path = Path("docs/cloud-gpu-runbook.md")
text = path.read_text()
anchor = "3. Connect the app to host `localhost`, port `2222`, user `ubuntu`, with the key `~/.ssh/ces-revisions-vm`.\n"
if text.count(anchor) != 1:
    raise SystemExit("the runbook's port-forwarding steps were not found once")
path.write_text(text.replace(anchor, f"{anchor}\n{sentence} The decision record gives the details.\n"))
print(sentence)
EOF
```

Expected: the sentence for the recorded method.

- [ ] **Step 7: Update `CLAUDE.md`**

Make three edits to `CLAUDE.md`.

First, under **Layout and tooling**, insert this bullet directly after the bullet that begins `` - `src/ces_revisions/engine_probe.py` times the engine's value and gradient ``:

````markdown
- `infra/` builds and operates the cloud GPU development environment, always from the Mac. It holds the OpenTofu roots `infra/state` (the state bucket) and `infra/env` (network, instance, budget, snapshots); the VM's first-boot script, setup script, and cost guards in `infra/vm/`; and `infra/bin/vm`, which starts, stops, resizes, and connects to the VM and copies settings to it, printing each AWS CLI and OpenTofu command it runs. [`docs/cloud-gpu-runbook.md`](docs/cloud-gpu-runbook.md) is the operating manual, and [`docs/decisions/cloud-gpu.md`](docs/decisions/cloud-gpu.md) records the decision with its evidence. On the VM, this checkout lives at `~/Projects/ces-revisions`, and `infra/bin/vm` refuses to run there.
````

Second, in **Commands**, insert this line directly after the `uv run python -m ces_revisions.engine_probe --help` line:

````text
infra/bin/vm status                             # the cloud VM's state, type, and zone (run on the Mac; see docs/cloud-gpu-runbook.md)
````

Its comment starts at column 49, like its neighbors'.

Third, at the end of the paragraph after the Commands block, which ends `and reformats Python code blocks inside it.`, append this sentence:

````text
 `[tool.ruff] per-file-target-version` holds `infra/vm/guards/*.py` to Python 3.12, because the VM's system Python runs the cost guards.
````

- [ ] **Step 8: Update `README.md`**

Make two edits to `README.md`.

First, in the Status tree, replace:

````text
docs/decisions/                       decision records (engine.md: the Kalman engine)
  cloud-gpu-evidence/                 AWS account, zone, and quota evidence for the cloud GPU environment
  cloud-gpu-probe/                    engine timing records, starting with the Mac baseline
````

with:

````text
docs/cloud-gpu-runbook.md             runbook: build, use, and tear down the cloud GPU environment
docs/decisions/                       decision records (engine.md: the Kalman engine;
                                      cloud-gpu.md: the cloud GPU environment)
  cloud-gpu-evidence/                 AWS and VM evidence for the cloud GPU environment
  cloud-gpu-probe/                    engine timing records for the Mac and each cloud size
infra/                                OpenTofu roots, VM scripts and cost guards, and bin/vm (run from the Mac)
````

Second, insert this paragraph, followed by a blank line, directly before the paragraph that begins `The modeling stack is JAX`:

````markdown
The cloud GPU development environment is one AWS instance that switches between a CPU size and four GPU sizes. It is built and operated from the Mac with `infra/`, as [`docs/cloud-gpu-runbook.md`](docs/cloud-gpu-runbook.md) describes, and [`docs/decisions/cloud-gpu.md`](docs/decisions/cloud-gpu.md) records why, with engine timings for each measured size.
````

- [ ] **Step 9: Append to the evidence README**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Cutover

- `vm-claude-session.json` — after project memory was copied to the VM, whether a Claude Code session there listed the personal skills and answered from the project memory, as the human partner reported.
````

- [ ] **Step 10: Run the final verification (bullets 2, 4, and 13)**

```bash
uv run pytest -m "not slow and not network" -q
uv run pytest -m slow -q
uv run ruff check
uv run ruff format --check
uv lock --check
tofu fmt -check -recursive infra && echo "HCL formatted"
```

Expected after the second capacity amendment: `417 passed, 17 deselected`, `13 passed, 421 deselected`, `All checks
passed!`, every file already formatted, `uv lock --check` exiting 0, and `HCL formatted`. Then
confirm that both roots still match AWS:

```bash
export AWS_PROFILE=ces-revisions
for ROOT in infra/state infra/env; do
  tofu -chdir="$ROOT" plan -input=false -detailed-exitcode > /dev/null
  case $? in
    0) echo "$ROOT: no changes" ;;
    *) echo "STOP: the $ROOT plan shows changes or failed" ;;
  esac
done
```

Expected: `infra/state: no changes` and `infra/env: no changes`. Last, rerun the committed-tree search from Task 8, Step 11, and the evidence scan from Task 6, Step 7. Expected: `committed tree clean`, with the same `pyproject.toml` result as in Task 8, and `evidence scan clean`.

- [ ] **Step 11: Commit**

```bash
uv run ruff format && uv run ruff check
git add docs/decisions/cloud-gpu.md docs/cloud-gpu-runbook.md CLAUDE.md README.md \
  docs/decisions/cloud-gpu-evidence/README.md docs/decisions/cloud-gpu-evidence/vm-claude-session.json
git commit -m "Record the cloud GPU decision and document infra/ and its runbook"
```

## After this plan

- **Completion.** The Plan Completion Protocol retires this plan and the spec, and re-points the spec's links in `docs/decisions/cloud-gpu-evidence/README.md` and `docs/decisions/cloud-gpu.md`, as the Retirement note says.
- **The VM's checkout.** After the branch merges, switch the VM's checkout to `main`:

  ```bash
  ssh ces-revisions-vm 'git -C Projects/ces-revisions switch main && git -C Projects/ces-revisions pull --ff-only'
  ```

- **The operational files.** Before this worktree is removed, copy its gitignored `infra/` files into the checkout that will operate `infra/` from then on, as the runbook's "Where the operational files live" describes. The copies in `~/.config/ces-revisions/infra/` are the fallback.
- **Questions for the human partner.**
  - `pyproject.toml`'s `authors` entry holds an email address, which Req 3's rule excludes. Task 8's search reported whether it matches the budget address.
  - Plan 3's Task 10, Step 2 checks with `git check-ignore -q` and several paths, which always takes the "stay tracked" branch.
  - Any Verification bullet left unmet, such as the desktop half of bullet 5 when `access.json` says `neither`, goes to the Plan Completion Protocol's gate.
- **Roadmap Stage 6** can start: it consumes this environment and `docs/decisions/cloud-gpu.md`.
