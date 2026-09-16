# Cloud GPU Environment, Plan 3 — Account, Zone, Quotas, and Repo Changes Implementation Plan

**Status: COMPLETE (2026-09-16)** — executed via executing-plans; nothing deferred

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Spec: specs/cloud-gpu-environment.md, the first of its two implementation plans (Reqs 1, 2, and
> 7). The spec
> sits outside specs/ces-revisions-roadmap.md, so no roadmap stage is ticked.

**Goal:** Prepare the AWS account and the Mac (Req 1), choose the region and availability zone by evidence and file the GPU quota requests (Req 2), and make the repo run on GPU hosts (Req 7). Req 7 brings a Linux-only `cuda` extra under a pinned uv series, device-aware tests and chain method, and an engine timing probe with its Mac baseline. No AWS resource is created.

**Architecture:** Tasks 1–4 are operations run with the human partner. Each human step is a gate followed by a machine check. Every AWS command output lands in `docs/decisions/cloud-gpu-evidence/`, narrowed so that no account identifier is committed. The GPU quota requests go out in Task 4, as soon as credentials and the zone choice exist, because approval may take days and sets Plan 4's timeline. Tasks 5–10 need no AWS access and run while approval is pending:
- `src/ces_revisions/devices.py` holds the device facts that `tests/conftest.py`, `tests/test_stack.py`, and the synthetic pilot consult.
- `src/ces_revisions/engine_probe.py` times `jax.value_and_grad` of the Kalman filter's log likelihood and writes one JSON record per host.

**Tech Stack:** Homebrew (awscli 2.32.0 or later, opentofu 1.10 or later, the `session-manager-plugin` cask, uv 0.12), AWS CLI v2 with `aws login`, the AWS Price List Query, EC2, and Service Quotas APIs, jq, Python 3.14, JAX 0.11.1 (float64; `jax[cuda13]` on Linux), NumPyro 0.21.0, pytest, and ruff 0.16.7.

**Source:** [`specs/cloud-gpu-environment.md`](../../cloud-gpu-environment.md) Reqs 1, 2, and 7, Req 3's secrets rule, the Rollout note, and Verification bullets 1 and 13 and the Mac run of bullet 9.

**Retirement:** When this plan retires to `specs/plans/completed/`, `specs/cloud-gpu-environment.md` does **not** retire with it. At Plan Completion Protocol step 5, move only this plan. Plan 4 (`specs/plans/4-cloud-gpu-environment.md`) still implements Reqs 3–6 and 8–10, and the spec retires when Plan 4 completes.

## Current-main reconciliation (2026-09-16)

This plan was transplanted without the stale roadmap commit from
`claude/cloud-gpu-ces-revisions-ee5804` onto `main` at `08ed203`. The current-main baseline was
measured before this file was revised:

| Selection | Current-main result | Plan 3 final result |
|---|---:|---:|
| `pytest -m "not slow and not network"` | 324 passed, 17 deselected | 330 passed, 17 deselected |
| `pytest --collect-only -m slow` | 13 selected, 328 deselected | 13 selected, 334 deselected |
| `pytest --collect-only -m network` | 4 selected, 337 deselected | 4 selected, 343 deselected |
| All collected tests | 341 | 347 |

The six new fast tests are three in `tests/test_devices.py` and three in
`tests/test_engine_probe.py`; this plan adds no slow or network tests. Stage 3 added `fastexcel` to
the stack-import parametrization and Stage 4 added `python-dotenv`, annual-source code, and more
slow/network tests. Task 7 therefore preserves every current stack module instead of replacing the
file with the six-module 2026-09-13 version. Tasks 5 and 10 anchor against the current
`pyproject.toml`, `.gitignore`, README, and CLAUDE.md text and preserve all Stage 2–4 documentation.
The path-level diff from the original `cd835c5` base through `08ed203` confirms that
`src/ces_revisions/kalman.py`, `tests/conftest.py`, and `tests/test_synthetic_pilot.py` did not
change, so their code-level plan anchors remain current; `tests/test_stack.py` did change and is
the whole-file example revised below.

These exact counts are evidence for base `08ed203`, not magic constants. If execution starts after
another roadmap stage lands, first rerun the three collection commands above, record the new
baseline in the deviation note, and recompute each later total from the explicit `+3`, `+3`,
`+13`, `+16`, `+4`, and `+2` test deltas in Plans 3 and 4. Never delete, deselect, or loosen an
unrelated test to recover a historical count.

The roadmap now carries the Stage 6 dependency semantically. No cloud-branch commit hash is a
precondition: Plan 4 checks the live Stage 6 `Consumes` text instead.

### Original planning evidence (2026-09-13)

This plan's code was run before the plan was written, in a scratch clone of `cd835c5` under uv 0.9.5, and its filters were run against synthetic AWS outputs. Deviations from these outcomes are signals, not noise:

- **Roadmap amendment.** The Stage 2 session's roadmap resume made Stage 6 consume this environment. The current-main reconciliation preserves that dependency in `specs/ces-revisions-roadmap.md`; the old cloud-branch commit identity is deliberately not a precondition.
- **Versions.** Homebrew offered:
  - awscli 2.36.44;
  - opentofu 1.12.6;
  - uv 0.12.13, also the newest on PyPI, released 2026-09-10;
  - the `session-manager-plugin` cask 1.2.835.0, a `.pkg` installer.

  uv 0.12.0's changelog lists no build-backend configuration change and says to raise any `uv_build` upper bound to allow 0.12.
- **The `cuda` extra.** jax 0.11.1's `cuda13` extra pins `jaxlib==0.11.1` and `jax-cuda13-plugin[with-cuda]==0.11.1`, which has cp314 manylinux wheels for x86_64. With the extra, `uv lock` added 17 CUDA packages: `jax-cuda13-plugin`, `jax-cuda13-pjrt`, and 15 `nvidia-*` packages. The default environment's `uv export` was unchanged, and `uv sync --locked --extra cuda` installed nothing on the Mac.
- **XLA's determinism flag.** jaxlib 0.11.1 pins openxla/xla `dcf304bc`. Its `xla/debug_options_flags.cc` defines `xla_gpu_deterministic_ops`, described as "Guarantees run-to-run determinism on GPU." On the Mac, a JAX operation ran under `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, while an unknown flag aborted with `Unknown flag in XLA_FLAGS`. That settles the flag's name. Whether the flag makes GPU runs deterministic is Plan 4's check on `l4` and `h100`.
- **NumPyro.** `numpyro.set_host_device_count` rewrites `XLA_FLAGS` by removing only its own flag and prepending it, so a flag set beforehand survives.
- **Tests and lint.** In the original scratch clone, with the code of Tasks 6–8:
  - the fast tier passed 36 tests (30 before); these are historical counts, superseded by the
    current-main table above;
  - the then-two-test slow tier passed in 16 s; current main has 13 slow cases because Stage 3's
    build cases also carry the marker;
  - `ruff check` and `ruff format --check` were clean.

  Ruff formats Python blocks inside Markdown, including this plan's.
- **Probe.** The probe ran at T=280, n=150, p=70 on the M4 Max while other test runs shared the CPU.

  | Batch | Median per call | Compile |
  |---|---|---|
  | 1 | 0.567 s | 0.89 s |
  | 4 | 1.33 s | 1.78 s |
  | 16 | 3.77 s | 6.02 s |

  The run took 77 s of wall time. Batch 16 peaked at 11.9 GB resident memory, a figure Plan 4 needs for its `dev` size, which has 16 GiB.
- **Filters.** Task 3's region filter was run on synthetic outputs:
  - it rejected a region whose only p5.4xlarge price was a Capacity Block price;
  - it rejected a region with no zone offering all three types;
  - it rejected a zero price;
  - it chose `eu-west-2a` over `eu-west-2b`.

  Task 4's request plan skipped a quota already at its target and one with an open request.
- **Shell.** In this harness's Bash tool (zsh), `set -e; false; echo hi` still prints `hi`. Every guard in this plan is therefore an explicit `if …; then echo "STOP: …"; fi`.

## Execution notes

- **Where.** After this reconciled plan is on `main`, execute it in a fresh isolated worktree and a
  new `codex/` feature branch created from then-current `main`; do not reuse
  `.claude/worktrees/cloud-gpu-ces-revisions-ee5804` or its detached checkout. Record the execution
  branch with `git branch --show-current`; Plan 4 derives that name at runtime rather than assuming
  one.
- **Who runs what.** Tasks 1–4 contain the human partner's steps and an approval gate:
  - The controller runs them itself, in order, with the human partner present, and dispatches no implementer subagent for them.
  - **STOP** marks a gate. Say what the human partner must do, wait for their confirmation, then run the check that follows.
  - Tasks 5–10 suit subagent-driven-development.
- **Ordering.** Credentials (Task 2) come before any AWS call, and the quota requests (Task 4) follow the zone choice (Task 3) with nothing in between. If Task 3 finds no qualifying region, skip Task 4, report to the human partner (Req 2 returns the provider choice to them), and continue with Task 5.
- **Heads-ups.** Before dispatching Task 5, tell the human partner that it upgrades Homebrew's uv,
  which every checkout on this Mac shares. Before Task 9, ask them to pause other heavy work on the
  Mac, because the probe's baseline is a timing.
- **AWS commands.** Shell state does not persist between commands, so every block that calls `aws` starts with `export AWS_PROFILE=ces-revisions`. When a command fails because the credentials expired, ask the human partner to run `aws login --profile ces-revisions` again.
- **Secrets.** Never write an account ID, ARN, email address, or token into a repo file. For that reason the evidence commands narrow their outputs with `--query`, and every evidence commit first runs the scan in Task 2, Step 7.
- **Commits.** Every commit step first runs `uv run ruff format` and `uv run ruff check`, then stages files by explicit path. Never run `git add -A`.
- **Additions to the spec.** Each is small, and each is flagged here:
  - Task 2 has the human partner activate IAM access to Billing information while signed in as root, so Plan 4's cost-allocation-tag and budget steps never need root again.
  - Task 2 also checks the IAM user's MFA device count, and that root has no access keys.
  - Task 4 reads the Standard On-Demand quota that `dev` (m7i.xlarge) needs, and requests 4 vCPUs only if the quota is below that.
  - `devices.py` exposes `has_nvidia_device()` and `DETERMINISTIC_GPU_FLAG` beside the spec's `chain_method`, so `tests/conftest.py` and `tests/test_stack.py` share one device path and one flag string.
- **Stop conditions.** Stop and report rather than improvise when:
  - a tool version is still below its floor after upgrading;
  - a Task 2 check prints `false` after the human partner's fix;
  - AWS returns an error other than expired credentials;
  - `brew upgrade uv` yields a version outside the 0.12 series;
  - `uv lock` changes an existing package's version;
  - a test outcome differs from this plan.

  Never loosen a check to make it pass.

## Global Constraints

- **Account (Req 1):** "Work happens in the user's existing personal AWS account. The root user has MFA and is not used after setup. Daily human access is an IAM user with MFA and administrator permissions, used from the console and from `aws login` (AWS CLI 2.32.0 or later); that user has no access keys."
- **Mac tooling (Req 1):** "The Mac gets AWS CLI v2, the Session Manager plugin, and OpenTofu 1.10 or later from Homebrew."
- **Who does what (Rollout note):** "Steps that enter credentials or change account security belong to the user: root MFA, creating the IAM user and registering its MFA device, the `aws login` browser sign-in, activating the cost allocation tag, and creating and entering the GitHub token. The agent runs read-only checks and submits quota requests only with the user's explicit approval."
- **Region rule (Req 2):**
  - Candidates, in order: us-east-1, us-east-2, us-west-2, eu-west-2.
  - "A region qualifies when the AWS Price List API has a Linux On-Demand price for p5.4xlarge there and at least one of its availability zones offers m7i.xlarge, g6.xlarge, and p5.4xlarge."
  - The environment uses "the first qualifying region in that order and, within it, the alphabetically first zone that offers all three types."
  - "If no candidate qualifies, work stops and the provider choice returns to the user."
- **Quotas (Req 2):** "Immediately after the choice, Service Quotas increases are requested in that region: 'Running On-Demand G and VT instances' to at least 4 vCPUs and 'Running On-Demand P instances' to at least 16, recording the prior values and the request IDs."
- **No resources (Rollout note):** this plan creates no AWS resource.
- **Secrets (Req 3):** "No committed file contains an account ID, a bucket name, an email address, or a token."
- **`cuda` extra (Req 7):** `cuda = ["jax[cuda13]>=0.11.1; sys_platform == 'linux'"]`, "so the Mac's plain `uv sync` installs nothing new."
- **uv (Req 7):** "`[tool.uv] required-version` pins a compatible-release range (`~=`) of the newest uv minor series at plan time, which both the Mac and the VM install; `uv.lock` is regenerated once under it; and the `uv_build` range in `[build-system]` widens to include that series." At plan time that series is 0.12, so the values are `required-version = "~=0.12.13"` and `uv_build>=0.9.5,<0.13.0`.
- **Devices (Req 7):** "Float64 holds everywhere; where `/dev/nvidia0` exists, `jax.default_backend()` is `"gpu"`; elsewhere, `jax.local_device_count() == 4`." `chain_method(num_chains: int) -> str` returns `"parallel"` when `jax.local_device_count() >= num_chains`, and `"vectorized"` otherwise.
- **Probe (Req 7):** `python -m ces_revisions.engine_probe` takes `--steps`, `--states`, `--cells`, `--batch` (one or more sizes), `--missing-share` (default 0.2), `--repeats` (default 10), `--seed` (default 0), `--label`, and `--out`. The decision record's dimensions are T=280, n=150, and p=70, with batch sizes 1, 4, and 16 on the Mac.
- **Tests:** the fast hermetic tier is `uv run pytest -m "not slow and not network"`. MCMC tests carry the `slow` marker, and an unregistered marker is a collection error.
- **Ruff:** the default rules plus `extend-select = ["I", "B", "UP"]`; never switch to `select`. `uv run ruff format` also formats Python blocks inside Markdown.
- **Markdown** (the evidence README, `CLAUDE.md`, `README.md`) follows CLAUDE.md's conventions:
  - literal dollars written as `\$`;
  - real headings;
  - hard line breaks as a trailing `\`;
  - pseudo-math containing `_` in code spans;
  - no bare `$…$` math.
- **Scope:** Reqs 1, 2, and 7 only. Plan 4 owns:
  - `infra/`, the guards, and `setup.sh`;
  - the runbook, and the decision record `docs/decisions/cloud-gpu.md`;
  - the probe runs on `dev`, `l4`, and `h100`;
  - the `infra/` and runbook references in CLAUDE.md and the README.

  Spot instances, Capacity Blocks, Savings Plans, containers, and Azure are out of scope.

---

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `docs/decisions/cloud-gpu-evidence/README.md` | What each evidence file holds and which command produced it | 2, 3, 4 |
| `docs/decisions/cloud-gpu-evidence/iam-*.json` | Req 1 checks: no access keys, root MFA, the IAM user's MFA device | 2 |
| `docs/decisions/cloud-gpu-evidence/{pricing,ec2,region}-*.json`, `zone-choice.json` | Req 2 region and zone evidence, and the choice | 3 |
| `docs/decisions/cloud-gpu-evidence/service-quotas-*.json` | Prior quota values, earlier requests, the request plan, the submitted requests | 4 |
| `pyproject.toml`, `uv.lock` | The `cuda` extra, the uv series pin, the `uv_build` range | 5 |
| `src/ces_revisions/devices.py` | `has_nvidia_device`, `chain_method`, `DETERMINISTIC_GPU_FLAG` | 6, 7 |
| `tests/test_devices.py` | Both branches of `chain_method`, and the four-chain expectation per host | 6 |
| `tests/test_synthetic_pilot.py` | The slow pilot tests call `chain_method` | 6 |
| `tests/conftest.py` | XLA's deterministic GPU flag on NVIDIA hosts, set before the host device count | 7 |
| `tests/test_stack.py` | The device expectation: `gpu` backend or four host devices | 7 |
| `src/ces_revisions/engine_probe.py` | Synthetic model, timing, JSON record, command line | 8 |
| `tests/test_engine_probe.py` | The synthetic model's properties and the record's fields | 8 |
| `docs/decisions/cloud-gpu-probe/mac.json` | The Mac baseline at the decision record's dimensions | 9 |
| `.gitignore` | OpenTofu local state, provider caches, per-user backend and variable files | 10 |
| `CLAUDE.md`, `README.md` | The `cuda` extra, `devices.py`, `engine_probe.py`, the uv series | 10 |

---

### Task 1: Mac tooling (Req 1)

**Mode:** the controller runs this task inline. Step 2 is the human partner's.

**Files:** none in the repo; Homebrew installs on the Mac.

**Interfaces:**
- Consumes: Homebrew.
- Produces:
  - `aws` (2.32.0 or later) at `/opt/homebrew/bin/aws`;
  - `tofu` (1.10 or later);
  - `session-manager-plugin` on `PATH`.

  Tasks 2–4 call `aws`. Plan 4 calls all three, and its SSH `ProxyCommand` names `aws` by its absolute path.

- [x] **Step 1: Install the AWS CLI and OpenTofu**

```bash
brew install awscli opentofu
```

Expected: both formulae install from bottles, with no password prompt. Underneath: a bottle is a prebuilt binary that Homebrew unpacks under `/opt/homebrew`.

- [x] **Step 2: STOP — the human partner installs the Session Manager plugin**

Ask your human partner to run this in their own terminal:

```bash
brew install --cask session-manager-plugin
```

The cask wraps AWS's signed `.pkg` installer, so macOS asks for their password, and an agent's shell cannot answer that prompt. Wait for them to confirm that it finished. Underneath: the plugin is the local half of a Session Manager connection. The AWS CLI hands it a session, and the plugin relays the shell or SSH stream over HTTPS.

- [x] **Step 3: Check the versions and the CLI's path**

```bash
aws --version
tofu version
session-manager-plugin --version
command -v aws
```

Expected:
- `aws-cli/2.N.M …` with N of 32 or more (2.32.0 is the first release with `aws login`);
- `OpenTofu v1.N.M` with N of 10 or more;
- a plugin version such as `1.2.835.0`;
- `/opt/homebrew/bin/aws`.

If a version is below its floor, run `brew upgrade` for that formula or cask and check again. Stop if it stays below.

No commit: nothing in the repo changed.

### Task 2: Account identity and hygiene (Req 1)

**Mode:** the controller runs this task inline. Steps 1–3 are the human partner's.

**Files:**
- Create: `docs/decisions/cloud-gpu-evidence/README.md`
- Create: `docs/decisions/cloud-gpu-evidence/iam-list-access-keys.json`
- Create: `docs/decisions/cloud-gpu-evidence/iam-get-account-summary.json`
- Create: `docs/decisions/cloud-gpu-evidence/iam-list-mfa-devices.json`

**Interfaces:**
- Consumes: Task 1's `aws`.
- Produces:
  - the AWS CLI profile `ces-revisions`, backed by an `aws login` session for the IAM user, under which every later `aws` command in this plan and in Plan 4 runs;
  - the evidence directory and its README, which Tasks 3 and 4 extend.

- [x] **Step 1: STOP — the human partner secures the root user**

Ask your human partner to sign in to the AWS console as the root user and:
1. Open the account menu, choose **Security credentials** → **Multi-factor authentication (MFA)**, and assign an MFA device if none is listed.
2. On the same page, under **Access keys**, delete any root access keys.
3. Open the account menu, choose **Account** → **IAM user and role access to Billing information** → **Edit**, select **Activate IAM Access**, and choose **Update**.

The third setting lets the IAM user open Billing. Without it, Plan 4's cost-allocation-tag and budget steps would need root, which is not used after setup. Wait until they confirm all three. Underneath: `aws iam get-account-summary` reports root MFA and root access keys as the flags `AccountMFAEnabled` and `AccountAccessKeysPresent`, which Step 5 checks.

- [x] **Step 2: STOP — the human partner creates the IAM user**

Ask your human partner, still in the console, to:
1. Open **IAM** → **Users** → **Create user** and choose a user name. Tick **Provide user access to the AWS Management Console**, choose **I want to create an IAM user**, and set a custom password.
2. Under **Set permissions**, choose **Attach policies directly** and attach `AdministratorAccess`. It includes the `signin` actions that `aws login` needs, the same ones AWS's narrower `SignInLocalDevelopmentAccess` managed policy grants.
3. Create the user. Create no access keys for it, now or later.
4. Sign out, then sign in as the new user through the account's sign-in URL. Open the account menu, choose **Security credentials** → **Assign MFA device**.

Wait until they confirm. Underneath: an IAM user is a long-lived identity inside the account. Without access keys, its only credentials are its console password and MFA device, and `aws login` turns a console sign-in into temporary CLI credentials.

- [x] **Step 3: STOP — the human partner signs in the CLI**

Ask your human partner to run this in their own terminal, because it opens a browser:

```bash
aws login --profile ces-revisions
```

At the region prompt, they enter `us-east-1`; Tasks 3 and 4 pass `--region` explicitly, so this is only the profile's default. In the browser, they choose the IAM user's session, never root's. Wait until they confirm.

Underneath: the CLI writes `login_session` (the user's ARN) and `region` under `[profile ces-revisions]` in `~/.aws/config`. It caches temporary credentials in `~/.aws/login/cache` and refreshes them every 15 minutes for up to 12 hours, after which the same command signs in again.

- [x] **Step 4: Confirm the session belongs to the IAM user**

```bash
export AWS_PROFILE=ces-revisions
ARN=$(aws sts get-caller-identity --query Arn --output text)
case "$ARN" in
  *:user/*) echo "IAM user session" ;;
  *) echo "STOP: not an IAM user session (it ends in ${ARN##*:}); rerun Step 3 and choose the IAM user" ;;
esac
```

Expected: `IAM user session`. The ARN holds the account ID, so the command prints only the verdict.

- [x] **Step 5: Record and check the identity evidence**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
mkdir -p "$EVIDENCE"
aws iam list-access-keys --output json > "$EVIDENCE/iam-list-access-keys.json"
aws iam get-account-summary --output json > "$EVIDENCE/iam-get-account-summary.json"
aws iam list-mfa-devices --query '{MFADeviceCount: length(MFADevices)}' --output json \
  > "$EVIDENCE/iam-list-mfa-devices.json"
jq -e '.AccessKeyMetadata | length == 0' "$EVIDENCE/iam-list-access-keys.json"
jq -e '.SummaryMap.AccountMFAEnabled == 1 and .SummaryMap.AccountAccessKeysPresent == 0' \
  "$EVIDENCE/iam-get-account-summary.json"
jq -e '.MFADeviceCount >= 1' "$EVIDENCE/iam-list-mfa-devices.json"
```

Expected: three lines of `true`. A `false` names the failing check:
- First line: the IAM user has access keys. The human partner deletes them under **IAM** → **Users** → their user → **Security credentials**.
- Second line: root has no MFA, or still has access keys. Go back to Step 1.
- Third line: the IAM user has no MFA device. Go back to Step 2.

After a fix, rerun this step. While the first line prints `false`, `iam-list-access-keys.json` holds the user name and key IDs, so never commit it in that state.

Underneath: `list-access-keys` without `--user-name` lists the caller's own keys. `get-account-summary` holds only account-wide counts and flags, which is why its full output is safe to commit.

- [x] **Step 6: Write the evidence README**

Create `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown
# Cloud GPU environment: account and zone evidence

Command outputs recorded by Plan 3, the first implementation plan for [`specs/cloud-gpu-environment.md`](../../../specs/cloud-gpu-environment.md), for its Reqs 1 and 2. The decision record `docs/decisions/cloud-gpu.md`, which Plan 4 writes, cites them. Every `aws` command ran as the IAM user through the `ces-revisions` profile.

No file here holds an account ID, an ARN, an email address, or a token. Outputs that carry one were narrowed with `--query` before they were written. Before each commit, this directory was searched for the account ID, for ARNs, and for at signs.

## Req 1: identity

- `iam-list-access-keys.json` — `aws iam list-access-keys`: the IAM user's access keys. Req 1 requires none.
- `iam-get-account-summary.json` — `aws iam get-account-summary`: account-wide counts and flags. `AccountMFAEnabled` is 1 when the root user has MFA, and `AccountAccessKeysPresent` is 0 when root has no access keys.
- `iam-list-mfa-devices.json` — `aws iam list-mfa-devices`, narrowed to `MFADeviceCount`: how many MFA devices the IAM user has, without their serial-number ARNs.
````

- [x] **Step 7: Scan the evidence, then commit**

```bash
export AWS_PROFILE=ces-revisions
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$ACCOUNT_ID" ]; then
  echo "STOP: no account ID (expired credentials?), so the scan cannot run"
elif grep -rn -e "$ACCOUNT_ID" -e "arn:aws" -e "@" docs/decisions/cloud-gpu-evidence; then
  echo "STOP: the lines above hold an account ID, ARN, or address; remove them first"
else
  echo "evidence scan clean"
fi
```

Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
git add docs/decisions/cloud-gpu-evidence/README.md \
  docs/decisions/cloud-gpu-evidence/iam-list-access-keys.json \
  docs/decisions/cloud-gpu-evidence/iam-get-account-summary.json \
  docs/decisions/cloud-gpu-evidence/iam-list-mfa-devices.json
git commit -m "Record the cloud GPU account's identity checks"
```

### Task 3: Region and availability zone (Req 2)

**Mode:** the controller runs this task inline. It makes read-only AWS calls only.

**Files:**
- Create, for each region evaluated: `docs/decisions/cloud-gpu-evidence/pricing-get-products-p5.4xlarge-<region>.json`, `ec2-describe-instance-type-offerings-<region>.json`, and `region-<region>.json`
- Create: `docs/decisions/cloud-gpu-evidence/zone-choice.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)

**Interfaces:**
- Consumes: Task 2's `ces-revisions` profile.
- Produces: `zone-choice.json`, shaped `{"evaluated": [region records, in order], "chosen": {"region": …, "zone": …} or null}`. Each region record is `{"region", "p5_4xlarge_on_demand_usd_per_hour": [numbers], "zones_offering_all_three": [zone names], "qualifies": bool}`. Task 4 reads `.chosen.region`. Plan 4 reads `.chosen.region` and `.chosen.zone` into its OpenTofu variables.

- [x] **Step 1: Evaluate the candidates in order and record the choice**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
region_files=()
failed=""
for REGION in us-east-1 us-east-2 us-west-2 eu-west-2; do
  aws pricing get-products --region us-east-1 --service-code AmazonEC2 --output json \
    --filters Type=TERM_MATCH,Field=instanceType,Value=p5.4xlarge \
    Type=TERM_MATCH,Field=regionCode,Value="$REGION" \
    Type=TERM_MATCH,Field=operatingSystem,Value=Linux \
    Type=TERM_MATCH,Field=tenancy,Value=Shared \
    Type=TERM_MATCH,Field=preInstalledSw,Value=NA \
    Type=TERM_MATCH,Field=capacitystatus,Value=Used |
    jq '.PriceList | map(fromjson | {sku: .product.sku, attributes: .product.attributes, onDemand: .terms.OnDemand})' \
      > "$EVIDENCE/pricing-get-products-p5.4xlarge-$REGION.json"
  aws ec2 describe-instance-type-offerings --region "$REGION" --location-type availability-zone \
    --filters Name=instance-type,Values=m7i.xlarge,g6.xlarge,p5.4xlarge --output json \
    > "$EVIDENCE/ec2-describe-instance-type-offerings-$REGION.json"
  if ! jq -e 'type == "array"' "$EVIDENCE/pricing-get-products-p5.4xlarge-$REGION.json" > /dev/null 2>&1 ||
    ! jq -e '.InstanceTypeOfferings | type == "array"' \
      "$EVIDENCE/ec2-describe-instance-type-offerings-$REGION.json" > /dev/null 2>&1; then
    failed="$REGION"
    break
  fi
  jq -n --arg region "$REGION" \
    --slurpfile prices "$EVIDENCE/pricing-get-products-p5.4xlarge-$REGION.json" \
    --slurpfile offerings "$EVIDENCE/ec2-describe-instance-type-offerings-$REGION.json" '
    {
      region: $region,
      p5_4xlarge_on_demand_usd_per_hour: [
        $prices[0][]
        | select((.attributes.marketoption // "OnDemand") == "OnDemand")
        | (.onDemand // {})[] | .priceDimensions[] | .pricePerUnit.USD | tonumber
        | select(. > 0)
      ],
      zones_offering_all_three: ([
        $offerings[0].InstanceTypeOfferings
        | group_by(.Location)[]
        | select((map(.InstanceType) | unique) == ["g6.xlarge", "m7i.xlarge", "p5.4xlarge"])
        | .[0].Location
      ] | sort)
    }
    | .qualifies = ((.p5_4xlarge_on_demand_usd_per_hour | length) > 0
                    and (.zones_offering_all_three | length) > 0)' \
    > "$EVIDENCE/region-$REGION.json"
  region_files+=("$EVIDENCE/region-$REGION.json")
  jq -c '{region, qualifies, p5_4xlarge_on_demand_usd_per_hour, zones_offering_all_three}' \
    "$EVIDENCE/region-$REGION.json"
  if jq -e '.qualifies' "$EVIDENCE/region-$REGION.json" > /dev/null; then break; fi
done
if [ -n "$failed" ]; then
  echo "STOP: an AWS call for $failed failed (error above); fix it, then rerun this step"
else
  jq -s '{evaluated: ., chosen: (map(select(.qualifies)) | first
          | if . == null then null else {region, zone: .zones_offering_all_three[0]} end)}' \
    "${region_files[@]}" > "$EVIDENCE/zone-choice.json"
  jq -c '.chosen' "$EVIDENCE/zone-choice.json"
fi
```

Expected: one compact line per region evaluated, the last with `"qualifies":true` unless none qualified, then the choice, such as `{"region":"us-east-1","zone":"us-east-1a"}`, or `null`.

A failed AWS call stops the loop before it can skip past an unevaluated region, which would break Req 2's in-order rule.

Underneath:
- The Price List Query API is served from a few regions only, us-east-1 among them, so every pricing call goes there with the target region as a filter. It returns each product as a JSON string, which `fromjson` decodes.
- AWS's August 2025 announcement sold single-GPU P5 On-Demand only outside the US, so a US region may list the type with only a Capacity Block price. The `marketoption` filter keeps such a price from qualifying the region. The first successful p5.4xlarge On-Demand start, in Plan 4, settles that question for good.

- [x] **Step 2: STOP if no region qualified**

```bash
if jq -e '.chosen == null' docs/decisions/cloud-gpu-evidence/zone-choice.json > /dev/null; then
  echo "STOP: no candidate region qualifies; Req 2 returns the provider choice to the human partner"
else
  jq -r '"chosen: \(.chosen.region) \(.chosen.zone)"' docs/decisions/cloud-gpu-evidence/zone-choice.json
fi
```

Expected: `chosen: <region> <zone>`.

Before acting on a STOP, rule out filters that matched nothing, which would also leave every region without a price:

```bash
export AWS_PROFILE=ces-revisions
for f in docs/decisions/cloud-gpu-evidence/pricing-get-products-p5.4xlarge-*.json; do
  echo "$f: $(jq length "$f") products"
done
aws pricing describe-services --service-code AmazonEC2 --region us-east-1 \
  --query 'Services[0].AttributeNames' --output json |
  jq -c '[ "instanceType", "regionCode", "operatingSystem", "tenancy", "preInstalledSw", "capacitystatus" ] - .'
```

The last line prints `[]` when every filtered attribute exists. If it names an attribute, correct that filter in Step 1 and rerun it. If the attributes all exist, the STOP stands: tell your human partner that work on the environment stops and the provider choice is theirs. Still finish Steps 3 and 4 so the evidence is committed, then skip Task 4 and continue with Task 5.

- [x] **Step 3: Append the zone section to the evidence README**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Req 2: region and zone

Req 2 evaluates us-east-1, us-east-2, us-west-2, and eu-west-2, in that order, and stops at the first region that qualifies. A region qualifies when the AWS Price List API has a Linux On-Demand price for p5.4xlarge there and at least one of its availability zones offers m7i.xlarge, g6.xlarge, and p5.4xlarge. Within that region, the environment uses the alphabetically first such zone.

For each region evaluated:

- `pricing-get-products-p5.4xlarge-<region>.json` — `aws pricing get-products`, called on the Price List Query API endpoint in us-east-1. It is filtered to p5.4xlarge in the region, with Linux, shared tenancy, no preinstalled software, and used capacity. Each product is decoded from its JSON string and reduced to its SKU, attributes, and On-Demand terms.
- `ec2-describe-instance-type-offerings-<region>.json` — `aws ec2 describe-instance-type-offerings --location-type availability-zone` for the three instance types.
- `region-<region>.json` — derived from the two files above. It holds the positive On-Demand hourly prices of products whose `marketoption` attribute is absent or `OnDemand`, so a Capacity Block price does not count. It also lists the zones that offer all three types, and says whether the region qualifies.

`zone-choice.json` lists the regions evaluated, in order, and `chosen`: the first qualifying region with its first zone, or `null` when none qualified. Each account maps zone names such as `us-east-1a` to physical zones in its own way, so the choice holds for this account only.
````

- [x] **Step 4: Scan the evidence, then commit**

```bash
export AWS_PROFILE=ces-revisions
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$ACCOUNT_ID" ]; then
  echo "STOP: no account ID (expired credentials?), so the scan cannot run"
elif grep -rn -e "$ACCOUNT_ID" -e "arn:aws" -e "@" docs/decisions/cloud-gpu-evidence; then
  echo "STOP: the lines above hold an account ID, ARN, or address; remove them first"
else
  echo "evidence scan clean"
fi
```

Expected: `evidence scan clean`. Then commit exactly the files for the regions evaluated:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
git add "$EVIDENCE/README.md" "$EVIDENCE/zone-choice.json"
for REGION in $(jq -r '.evaluated[].region' "$EVIDENCE/zone-choice.json"); do
  git add "$EVIDENCE/pricing-get-products-p5.4xlarge-$REGION.json" \
    "$EVIDENCE/ec2-describe-instance-type-offerings-$REGION.json" \
    "$EVIDENCE/region-$REGION.json"
done
git status --short "$EVIDENCE"
git commit -m "Choose the cloud GPU region and zone from price and offering evidence"
```

Expected: `git status` shows only staged (`A`) files before the commit. An untracked `??` file left from an earlier attempt belongs to no region evaluated, so leave it out of the commit and delete it.

### Task 4: GPU quota requests (Req 2)

**Mode:** the controller runs this task inline, right after Task 3. Step 3 is the human partner's approval.

**Files:**
- Create: `docs/decisions/cloud-gpu-evidence/service-quotas-get-service-quota-<region>.json`, `service-quotas-change-history-<region>.json`, `service-quotas-request-plan-<region>.json`, and `service-quotas-request-increase-<region>.json`
- Modify: `docs/decisions/cloud-gpu-evidence/README.md` (append a section)

**Interfaces:**
- Consumes: Task 3's `zone-choice.json` (`.chosen.region`).
- Produces: the request IDs and submission statuses in `service-quotas-request-increase-<region>.json`. Plan 4's `l4` and `h100` steps wait until `get-requested-service-quota-change` reports `APPROVED` for the matching request.

- [x] **Step 1: Read the prior values and earlier requests**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r '.chosen.region // empty' "$EVIDENCE/zone-choice.json")
if [ -z "$REGION" ]; then
  echo "STOP: zone-choice.json has no chosen region, so Task 4 does not run"
else
  for CODE in L-1216C47A L-DB2E81BA L-417A185B; do
    aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code "$CODE" \
      --query 'Quota.{QuotaName: QuotaName, QuotaCode: QuotaCode, Value: Value, Unit: Unit, Adjustable: Adjustable}' \
      --output json
  done | jq -s . > "$EVIDENCE/service-quotas-get-service-quota-$REGION.json"
  for CODE in L-1216C47A L-DB2E81BA L-417A185B; do
    aws service-quotas list-requested-service-quota-change-history-by-quota --region "$REGION" \
      --service-code ec2 --quota-code "$CODE" \
      --query 'RequestedQuotas[].{Id: Id, QuotaCode: QuotaCode, DesiredValue: DesiredValue, Status: Status, Created: Created}' \
      --output json
  done | jq -s 'add' > "$EVIDENCE/service-quotas-change-history-$REGION.json"
  if ! jq -e 'length == 3 and all(.[]; .Value != null)' \
    "$EVIDENCE/service-quotas-get-service-quota-$REGION.json" > /dev/null; then
    echo "STOP: a get-service-quota call failed (error above)"
  fi
  jq -c '.[] | {QuotaCode, QuotaName, Value}' "$EVIDENCE/service-quotas-get-service-quota-$REGION.json"
fi
```

Expected: three lines, for `L-1216C47A` (Standard, default 5), `L-DB2E81BA` (G and VT, default 0), and `L-417A185B` (P, default 0). Underneath: these quotas count the vCPUs of running On-Demand instances, per instance family and region. `--query` drops `QuotaArn`, which contains the account ID.

- [x] **Step 2: Build the request plan**

```bash
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r '.chosen.region' "$EVIDENCE/zone-choice.json")
jq -n \
  --slurpfile quotas "$EVIDENCE/service-quotas-get-service-quota-$REGION.json" \
  --slurpfile history "$EVIDENCE/service-quotas-change-history-$REGION.json" '
  {"L-1216C47A": 4, "L-DB2E81BA": 4, "L-417A185B": 16} as $targets
  | [$quotas[0][]
     | $targets[.QuotaCode] as $target
     | .QuotaCode as $code
     | ([$history[0][]
         | select(.QuotaCode == $code and (.Status == "PENDING" or .Status == "CASE_OPENED")
                  and .DesiredValue >= $target)] | first) as $open
     | {QuotaName, QuotaCode, Value, DesiredValue: $target, OpenRequestId: $open.Id,
        Action: (if .Value >= $target then "none: already at target"
                 elif $open != null then "none: request already open"
                 else "request" end)}]' \
  > "$EVIDENCE/service-quotas-request-plan-$REGION.json"
jq -r '.[] | "\(.QuotaCode) \(.QuotaName): \(.Value) now, \(.DesiredValue) wanted, \(.Action)"' \
  "$EVIDENCE/service-quotas-request-plan-$REGION.json"
```

Expected: three lines. On a fresh account, G and VT and P each end in `request`, and Standard ends in `none: already at target`. The targets are one g6.xlarge (4 vCPUs), one p5.4xlarge (16), and one m7i.xlarge (4).

- [x] **Step 3: STOP — the human partner approves the requests**

Show your human partner the chosen region and zone, and Step 2's three lines. Then show the command that Step 4 runs once for each line ending in `request`:

```bash
aws service-quotas request-service-quota-increase --region "$REGION" --service-code ec2 \
  --quota-code "$CODE" --desired-value "$DESIRED"
```

Explain that it is outward-facing: AWS may open a support case under the account and email the account's address. Proceed only on a clear yes. If they decline, record nothing further, tell them that Plan 4's GPU steps cannot start without the quotas, and continue with Task 5.

- [x] **Step 4: Submit the approved requests**

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r '.chosen.region' "$EVIDENCE/zone-choice.json")
jq -r '.[] | select(.Action == "request") | "\(.QuotaCode) \(.DesiredValue)"' \
  "$EVIDENCE/service-quotas-request-plan-$REGION.json" |
  while read -r CODE DESIRED; do
    aws service-quotas request-service-quota-increase --region "$REGION" --service-code ec2 \
      --quota-code "$CODE" --desired-value "$DESIRED" --output json \
      --query 'RequestedQuota.{Id: Id, CaseId: CaseId, QuotaName: QuotaName, QuotaCode: QuotaCode, DesiredValue: DesiredValue, Status: Status, Created: Created}' \
      < /dev/null
  done | jq -s . > "$EVIDENCE/service-quotas-request-increase-$REGION.json"
jq -c '.[] | {QuotaName, DesiredValue, Id, CaseId, Status}' \
  "$EVIDENCE/service-quotas-request-increase-$REGION.json"
```

Expected: one line for each approved request, each with an `Id` and a `Status` of `PENDING`, `CASE_OPENED`, or `APPROVED`. If a request fails, stop and show the error. Never rerun a request blindly: rerun Step 1 first, because a request that went through appears in the change history, and a second one would be a duplicate. Underneath: `< /dev/null` keeps the CLI from reading the loop's input, and `--query` drops `Requester` and `QuotaArn`, which name the account and the IAM user.

- [x] **Step 5: Append the quota section to the evidence README**

Append to `docs/decisions/cloud-gpu-evidence/README.md`:

````markdown

## Req 2: GPU quota requests

EC2 On-Demand quotas count the vCPUs of running instances, per instance family and region. Req 2 asks for:

- "Running On-Demand G and VT instances" (`L-DB2E81BA`) of at least 4, enough for one g6.xlarge;
- "Running On-Demand P instances" (`L-417A185B`) of at least 16, enough for one p5.4xlarge.

"Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances" (`L-1216C47A`) covers the m7i.xlarge `dev` size, which needs 4. It was read to confirm that, and would have been requested only if it were below 4.

- `service-quotas-get-service-quota-<region>.json` — `aws service-quotas get-service-quota` for the three codes, before any request: the prior values.
- `service-quotas-change-history-<region>.json` — `aws service-quotas list-requested-service-quota-change-history-by-quota` for the three codes: earlier requests and their statuses.
- `service-quotas-request-plan-<region>.json` — derived from the two files above. It gives each quota's target, any open request that already covers it, and whether a new request was needed.
- `service-quotas-request-increase-<region>.json` — `aws service-quotas request-service-quota-increase` for each quota the plan marked `request`, run after the user approved it. It records the request `Id`, the support `CaseId` once AWS opens a case, and the status at submission.

A request's status moves from `PENDING` or `CASE_OPENED` to `APPROVED`, `DENIED`, `NOT_APPROVED`, or `CASE_CLOSED`. Run from the repo root, this prints each request's current status:

```bash
export AWS_PROFILE=ces-revisions
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r '.chosen.region' "$EVIDENCE/zone-choice.json")
for ID in $(jq -r '.[].Id' "$EVIDENCE/service-quotas-request-increase-$REGION.json"); do
  aws service-quotas get-requested-service-quota-change --region "$REGION" --request-id "$ID" \
    --query 'RequestedQuota.{QuotaName: QuotaName, DesiredValue: DesiredValue, Status: Status}' \
    --output json
done
```
````

- [x] **Step 6: Scan the evidence, then commit**

```bash
export AWS_PROFILE=ces-revisions
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$ACCOUNT_ID" ]; then
  echo "STOP: no account ID (expired credentials?), so the scan cannot run"
elif grep -rn -e "$ACCOUNT_ID" -e "arn:aws" -e "@" docs/decisions/cloud-gpu-evidence; then
  echo "STOP: the lines above hold an account ID, ARN, or address; remove them first"
else
  echo "evidence scan clean"
fi
```

Expected: `evidence scan clean`. Then commit:

```bash
uv run ruff format && uv run ruff check
EVIDENCE=docs/decisions/cloud-gpu-evidence
REGION=$(jq -r '.chosen.region' "$EVIDENCE/zone-choice.json")
git add "$EVIDENCE/README.md" \
  "$EVIDENCE/service-quotas-get-service-quota-$REGION.json" \
  "$EVIDENCE/service-quotas-change-history-$REGION.json" \
  "$EVIDENCE/service-quotas-request-plan-$REGION.json" \
  "$EVIDENCE/service-quotas-request-increase-$REGION.json"
git commit -m "Request the cloud GPU quotas and record prior values and request IDs"
```

- [x] **Step 7: Report the requests to the human partner**

Tell your human partner:
- the request IDs and statuses from Step 4;
- that the `dev` half of Plan 4 can start without them, and its `l4` and `h100` steps wait for `APPROVED`;
- that the status loop in the evidence README checks progress.

If AWS emails the account's address asking for a use case, the reply is theirs to send. They might adapt:

```text
Research computing: float64 JAX and NumPyro state-space model fitting. One On-Demand
instance at a time (a g6.xlarge, and a p5.4xlarge for short benchmark runs), stopped
automatically when idle and capped by a $150/month budget.
```

### Task 5: uv series pin and the `cuda` extra (Req 7)

**Mode:** suitable for a subagent. Before dispatching it, the controller tells the human partner that Step 1 upgrades Homebrew's uv, which every checkout on the Mac shares.

**Files:**
- Modify: `pyproject.toml` (two places)
- Modify: `uv.lock` (regenerated by `uv lock` only)

**Interfaces:**
- Consumes: Homebrew's `uv`.
- Produces:
  - `[project.optional-dependencies] cuda`;
  - `[tool.uv] required-version = "~=0.12.13"`;
  - `uv_build>=0.9.5,<0.13.0`;
  - a lock that resolves the extra.

  From this commit on, a uv outside the 0.12 series refuses to run in this project. Plan 4's `setup.sh` runs `uv sync --locked --extra cuda` against this lock.

- [x] **Step 1: Upgrade uv to the pinned series**

```bash
brew upgrade uv
uv --version
```

Expected: `uv 0.12.13` or a later 0.12 release; Homebrew had 0.12.13 on 2026-09-13. If it prints 0.13 or later, stop and ask your human partner, because the spec pins the series that was newest at plan time. Underneath: the upgrade must come before the pin, because once `required-version` lands, every command from the old uv 0.9.5 fails in this project.

- [x] **Step 2: Record the default environment before the change**

```bash
uv export --frozen --no-hashes --no-header --no-emit-project > /tmp/ces-revisions-export-before.txt
wc -l < /tmp/ces-revisions-export-before.txt
```

Expected: about 209 lines on base `08ed203`. Record the observed number; the correctness check in
Step 5 is equality of the before/after exports, not this historical line count. `--frozen` reads
`uv.lock` as it stands, without resolving again.

- [x] **Step 3: Add the extra, the uv pin, and the wider `uv_build` range**

In `pyproject.toml`, replace:

```toml
[project.scripts]
ces-revisions = "ces_revisions:main"

[build-system]
requires = ["uv_build>=0.9.5,<0.10.0"]
build-backend = "uv_build"
```

with:

```toml
[project.optional-dependencies]
# JAX's CUDA 13 plugin for the cloud GPU sizes. Linux-only, so the Mac's
# `uv sync --extra cuda` installs nothing new.
cuda = ["jax[cuda13]>=0.11.1; sys_platform == 'linux'"]

[project.scripts]
ces-revisions = "ces_revisions:main"

[build-system]
requires = ["uv_build>=0.9.5,<0.13.0"]
build-backend = "uv_build"
```

Then replace:

```toml
[tool.pytest.ini_options]
```

with:

```toml
[tool.uv]
# The Mac and the cloud VM run one uv series, so uv.lock is written one way.
required-version = "~=0.12.13"

[tool.pytest.ini_options]
```

- [x] **Step 4: Regenerate the lock, once**

```bash
uv lock
```

Expected: `Added` lines for `jax-cuda13-plugin`, `jax-cuda13-pjrt`, and the `nvidia-*` CUDA 13 packages, and no `Updated` line for a package that was already locked. Underneath: without `--upgrade`, `uv lock` keeps every existing pin and resolves only what the new extra needs. It writes one universal lock for every platform, and the `sys_platform == 'linux'` marker keeps the CUDA packages off the Mac.

- [x] **Step 5: Verify the lock, the unchanged default environment, and the Mac's empty extra**

```bash
uv lock --check
uv export --frozen --no-hashes --no-header --no-emit-project > /tmp/ces-revisions-export-after.txt
if diff /tmp/ces-revisions-export-before.txt /tmp/ces-revisions-export-after.txt; then
  echo "default environment unchanged"
else
  echo "STOP: the default environment changed"
fi
uv export --frozen --no-hashes --no-header --no-emit-project --extra cuda | grep -c -E "^(jax-cuda13|nvidia)"
uv sync --locked --extra cuda
if uv pip list | grep -i -E "cuda|nvidia"; then
  echo "STOP: CUDA packages were installed on the Mac"
else
  echo "no CUDA packages on the Mac"
fi
uv sync --locked
```

Expected:
- `uv lock --check` exits 0;
- `default environment unchanged`;
- `17`, the count at planning time: `jax-cuda13-plugin`, `jax-cuda13-pjrt`, and 15 `nvidia-*` packages;
- `no CUDA packages on the Mac`.

- [x] **Step 6: Run the fast tier under the new uv**

```bash
uv run pytest -m "not slow and not network" -q
```

Expected on base `08ed203`: `324 passed, 17 deselected`. If the preflight baseline is newer, expect
that recorded fast count unchanged; Task 5 adds no tests.

- [x] **Step 7: Commit**

```bash
uv run ruff format && uv run ruff check
git add pyproject.toml uv.lock
git commit -m "Add a Linux-only cuda extra and pin the uv 0.12 series"
```

### Task 6: Device-aware chain method (Req 7)

**Files:**
- Create: `src/ces_revisions/devices.py`
- Create: `tests/test_devices.py`
- Modify: `tests/test_synthetic_pilot.py` (the first-party import and both slow tests)

**Interfaces:**
- Consumes: the four host devices that `tests/conftest.py` sets on CPU hosts.
- Produces: `ces_revisions.devices.NVIDIA_DEVICE: Path`, `has_nvidia_device() -> bool`, and `chain_method(num_chains: int) -> str`. Task 7 adds `DETERMINISTIC_GPU_FLAG` to the same module.

- [x] **Step 1: Write the failing tests**

Create `tests/test_devices.py`:

```python
"""NumPyro's chain method follows the devices JAX sees on this host."""

import jax

from ces_revisions.devices import chain_method, has_nvidia_device


def test_chains_run_in_parallel_when_every_chain_has_a_device():
    assert chain_method(jax.local_device_count()) == "parallel"


def test_chains_are_vectorized_when_devices_are_fewer_than_chains():
    assert chain_method(jax.local_device_count() + 1) == "vectorized"


def test_four_chains_run_in_parallel_on_cpu_hosts_and_vectorized_on_one_gpu():
    # tests/conftest.py gives CPU hosts four devices; each GPU size has a single GPU.
    expected = "vectorized" if has_nvidia_device() else "parallel"
    assert chain_method(4) == expected
```

The first two tests cover both branches on any host, because they ask relative to the device count JAX reports. The third pins what Req 7's Verification bullets expect: `"parallel"` on the Mac and on `dev`, and `"vectorized"` on `l4` and `h100`.

- [x] **Step 2: Run the tests to see them fail**

```bash
uv run pytest tests/test_devices.py -q
```

Expected: a collection error, `ModuleNotFoundError: No module named 'ces_revisions.devices'`.

- [x] **Step 3: Create the module**

Create `src/ces_revisions/devices.py`:

```python
"""Device facts, so one checkout runs on the Mac's CPU and on a cloud GPU host."""

from pathlib import Path

import jax

NVIDIA_DEVICE = Path("/dev/nvidia0")


def has_nvidia_device() -> bool:
    """Whether the host exposes an NVIDIA GPU, whether or not JAX can use it."""
    return NVIDIA_DEVICE.exists()


def chain_method(num_chains: int) -> str:
    """NumPyro's chain method: parallel with a device per chain, otherwise vectorized."""
    return "parallel" if jax.local_device_count() >= num_chains else "vectorized"
```

Importing `jax` performs no JAX operation, so `tests/conftest.py` can import this module before it sets `XLA_FLAGS`.

- [x] **Step 4: Run the tests to see them pass**

```bash
uv run pytest tests/test_devices.py -q
```

Expected: `3 passed`.

- [x] **Step 5: Switch the pilot's slow tests to `chain_method`**

Edit `tests/test_synthetic_pilot.py` so that `git diff` shows exactly:

```diff
@@ -21,6 +21,7 @@ from synthetic_pilot import (
     simulate_panel,
 )
 
+from ces_revisions.devices import chain_method
 from ces_revisions.kalman import kalman_filter
 
 SEED = sum(map(ord, "ces-revisions-stage-1-synthetic-pilot"))
@@ -83,7 +84,7 @@ def test_pilot_fit_meets_mcmc_thresholds_and_recovers_the_truth(panel):
         num_warmup=NUM_DRAWS,
         num_samples=NUM_DRAWS,
         num_chains=NUM_CHAINS,
-        chain_method="parallel",
+        chain_method=chain_method(NUM_CHAINS),
         progress_bar=False,
     )
     fields = ("diverging", "energy", "num_steps")
@@ -112,13 +113,15 @@ def test_pilot_fit_meets_mcmc_thresholds_and_recovers_the_truth(panel):
 
 @pytest.mark.slow
 def test_pilot_draws_are_identical_under_a_fixed_seed(panel):
+    num_chains = 2
+
     def draws():
         mcmc = MCMC(
             NUTS(pilot_model),
             num_warmup=50,
             num_samples=50,
-            num_chains=2,
-            chain_method="parallel",
+            num_chains=num_chains,
+            chain_method=chain_method(num_chains),
             progress_bar=False,
         )
         mcmc.run(jax.random.PRNGKey(SEED), panel)
```

The keyword `chain_method=` and the function `chain_method` share a name. Python keeps them apart, and the call reads as the value it supplies.

- [x] **Step 6: Run the slow tier**

```bash
uv run pytest -m slow -q
```

Expected on base `08ed203`, after Task 6 adds three fast tests: `13 passed, 331 deselected` in about
30 s. On the Mac, both pilot calls return `"parallel"`, as the literal did; the other 11 selected
cases are Stage 3 build checks and must remain selected.

- [x] **Step 7: Commit**

```bash
uv run ruff format && uv run ruff check
git add src/ces_revisions/devices.py tests/test_devices.py tests/test_synthetic_pilot.py
git commit -m "Choose NumPyro's chain method from the devices JAX sees"
```

### Task 7: Deterministic GPU flag and the device expectation (Req 7)

**Files:**
- Modify: `src/ces_revisions/devices.py` (add `DETERMINISTIC_GPU_FLAG`)
- Modify: `tests/conftest.py` (whole file)
- Modify: `tests/test_stack.py` (whole file)

**Interfaces:**
- Consumes: Task 6's `NVIDIA_DEVICE` and `has_nvidia_device()`.
- Produces: `ces_revisions.devices.DETERMINISTIC_GPU_FLAG = "--xla_gpu_deterministic_ops=true"`, which the test session's `XLA_FLAGS` contains on any host with `/dev/nvidia0`. Plan 4's `l4` and `h100` runs of `uv run pytest` take the GPU branch of `test_session_runs_float64_jax_on_the_expected_devices`.

- [x] **Step 1: Write the failing device-expectation test**

Replace `tests/test_stack.py` with:

```python
"""The pinned stack imports, and the session runs float64 JAX on the host's devices."""

import importlib
import os
import sys

import jax
import jax.numpy as jnp
import pytest

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG, has_nvidia_device

# One importable module per pinned distribution: runtime jax, numpy, numpyro, arviz,
# polars, and fastexcel, the Excel reader behind Stage 3's workbooks; dynamax from the dev
# group, as engine-determination evidence only.
STACK_MODULES = [
    "jax",
    "numpy",
    "numpyro",
    "arviz",
    "polars",
    "fastexcel",
    "dynamax.linear_gaussian_ssm",
]


def test_interpreter_is_python_3_14_or_newer():
    assert sys.version_info >= (3, 14)


@pytest.mark.parametrize("module", STACK_MODULES)
def test_stack_module_imports(module):
    importlib.import_module(module)


def test_session_runs_float64_jax_on_the_expected_devices():
    assert jnp.zeros(1).dtype == jnp.float64
    if has_nvidia_device():  # a GPU size: JAX runs on the GPU, deterministically
        assert jax.default_backend() == "gpu"
        assert DETERMINISTIC_GPU_FLAG in os.environ["XLA_FLAGS"].split()
    else:  # the Mac or dev: the four host devices from tests/conftest.py
        assert jax.local_device_count() == 4
```

- [x] **Step 2: Run the test to see it fail**

```bash
uv run pytest tests/test_stack.py -q
```

Expected: a collection error, `ImportError: cannot import name 'DETERMINISTIC_GPU_FLAG' from 'ces_revisions.devices'`.

- [x] **Step 3: Add the flag to the device module**

Replace `src/ces_revisions/devices.py` with:

```python
"""Device facts, so one checkout runs on the Mac's CPU and on a cloud GPU host."""

from pathlib import Path

import jax

NVIDIA_DEVICE = Path("/dev/nvidia0")
# XLA's switch for run-to-run determinism on GPU, named as in the XLA revision that
# jaxlib 0.11.1 pins (openxla/xla dcf304bc, xla/debug_options_flags.cc).
DETERMINISTIC_GPU_FLAG = "--xla_gpu_deterministic_ops=true"


def has_nvidia_device() -> bool:
    """Whether the host exposes an NVIDIA GPU, whether or not JAX can use it."""
    return NVIDIA_DEVICE.exists()


def chain_method(num_chains: int) -> str:
    """NumPyro's chain method: parallel with a device per chain, otherwise vectorized."""
    return "parallel" if jax.local_device_count() >= num_chains else "vectorized"
```

- [x] **Step 4: Set the flag in the session policy**

Replace `tests/conftest.py` with:

```python
"""Session numerics policy, applied before any test module performs a JAX operation.

Float64 because Req 17 runs the Kalman recursions in 64-bit JAX; four host devices so the
synthetic pilot's four NUTS chains run in parallel on a CPU host; and, on an NVIDIA host,
XLA's deterministic GPU operations, so a fixed seed repeats its draws. All three settings
silently do nothing once a JAX operation has run, which is why they live here rather than
in a test module. The GPU flag goes into XLA_FLAGS first because set_host_device_count
keeps the flags it finds there.
"""

import os

import numpyro

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG, has_nvidia_device

if has_nvidia_device():
    flags = os.environ.get("XLA_FLAGS", "").split()
    os.environ["XLA_FLAGS"] = " ".join([*flags, DETERMINISTIC_GPU_FLAG])
numpyro.set_host_device_count(4)
numpyro.enable_x64()
```

Underneath: XLA reads `XLA_FLAGS` once, when JAX first touches a device. On a GPU host, the host-device-count flag still applies only to CPU devices, so `jax.local_device_count()` counts the one GPU.

- [x] **Step 5: Run the fast tier**

```bash
uv run pytest -m "not slow and not network" -q
```

Expected on base `08ed203`: `327 passed, 17 deselected`. On the Mac, the device test takes its CPU
branch. Its GPU branch runs in Plan 4.

- [x] **Step 6: Check the flag's name against the installed XLA**

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true uv run python -c "import jax.numpy as jnp; print(float(jnp.ones(2).sum()))"
XLA_FLAGS=--xla_no_such_flag=true uv run python -c "import jax.numpy as jnp; print(float(jnp.ones(2).sum()))"
```

Expected: the first prints `2.0`, and the second aborts with `Unknown flag in XLA_FLAGS: --xla_no_such_flag=true`. Together they show that the installed XLA defines the flag, which settles Req 7's open item as far as its name. Whether the flag makes GPU runs deterministic is Plan 4's check.

- [x] **Step 7: Commit**

```bash
uv run ruff format && uv run ruff check
git add src/ces_revisions/devices.py tests/conftest.py tests/test_stack.py
git commit -m "Enable XLA's deterministic GPU ops on NVIDIA hosts and expect the host's devices"
```

### Task 8: Engine timing probe (Req 7)

**Files:**
- Create: `src/ces_revisions/engine_probe.py`
- Create: `tests/test_engine_probe.py`

**Interfaces:**
- Consumes: `ces_revisions.kalman.LinearGaussianSSM` and `kalman_filter`, unchanged.
- Produces:
  - `synthetic_problem(num_steps: int, state_dim: int, obs_dim: int, missing_share: float, seed: int) -> tuple[LinearGaussianSSM, np.ndarray]`;
  - `time_batches(ssm: LinearGaussianSSM, panel: np.ndarray, batch_sizes: list[int], repeats: int, seed: int) -> list[dict[str, float]]`;
  - `main(argv: list[str] | None = None) -> None`, run as `python -m ces_revisions.engine_probe`.

  The JSON record holds:
  - `label`;
  - `dimensions` (`steps`, `states`, `cells`);
  - `missing_share`, `seed`, `repeats`;
  - `batches`, one entry per batch size, each with `batch`, `compile_seconds`, `median_seconds`, and `iqr_seconds`;
  - `backend`;
  - `versions` (`jax`, `jaxlib`, and any installed `jax-cuda*` plugin);
  - `nvidia_driver`, null where `nvidia-smi` is absent;
  - `timestamp_utc`, the run's start in ISO 8601 with its offset.

  Task 9 writes the Mac's record. Plan 4 writes the `dev`, `l4`, and `h100` records and tabulates all four in the decision record.

- [x] **Step 1: Write the failing tests**

Create `tests/test_engine_probe.py`:

```python
"""The engine probe's synthetic model and its JSON record, at tiny dimensions."""

import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import jax
import numpy as np

from ces_revisions.engine_probe import synthetic_problem

NUM_STEPS, STATE_DIM, OBS_DIM = 6, 3, 4
MISSING_SHARE = 0.25


def test_synthetic_problem_is_float64_stable_positive_definite_and_time_varying():
    ssm, panel = synthetic_problem(NUM_STEPS, STATE_DIM, OBS_DIM, MISSING_SHARE, seed=0)

    assert panel.shape == (NUM_STEPS, OBS_DIM)
    assert all(np.asarray(array).dtype == np.float64 for array in (*ssm, panel))
    assert np.abs(np.linalg.eigvals(ssm.transition_matrix)).max() < 1
    for cov in (ssm.initial_cov[None], ssm.transition_cov, ssm.observation_cov):
        assert np.linalg.eigvalsh(cov).min() > 0
    for name in ("transition_matrix", "transition_cov", "observation_matrix"):
        steps = getattr(ssm, name)
        assert not np.allclose(steps[0], steps[1]), name


def test_synthetic_problem_sets_the_requested_share_of_cells_to_nan():
    _, panel = synthetic_problem(NUM_STEPS, STATE_DIM, OBS_DIM, MISSING_SHARE, seed=0)

    assert np.isnan(panel).sum() == round(MISSING_SHARE * NUM_STEPS * OBS_DIM)


def test_probe_module_writes_a_timing_record(tmp_path):
    out = tmp_path / "probe.json"
    command = [sys.executable, "-m", "ces_revisions.engine_probe"]
    command += ["--steps", str(NUM_STEPS), "--states", str(STATE_DIM)]
    command += ["--cells", str(OBS_DIM), "--batch", "1", "2", "--repeats", "3"]
    command += ["--label", "test", "--out", str(out)]
    subprocess.run(command, check=True)
    record = json.loads(out.read_text())

    assert record["label"] == "test"
    assert record["dimensions"] == {
        "steps": NUM_STEPS,
        "states": STATE_DIM,
        "cells": OBS_DIM,
    }
    assert [entry["batch"] for entry in record["batches"]] == [1, 2]
    for entry in record["batches"]:
        assert entry["compile_seconds"] > 0
        assert entry["median_seconds"] > 0
        assert entry["iqr_seconds"] >= 0
    assert record["backend"] == jax.default_backend()
    for name in ("jax", "jaxlib"):
        assert record["versions"][name] == importlib.metadata.version(name)
    assert (record["nvidia_driver"] is None) == (shutil.which("nvidia-smi") is None)
    assert datetime.fromisoformat(record["timestamp_utc"]).utcoffset() == timedelta(0)
```

The record test runs the module in a fresh process for two reasons. It exercises the `python -m` entry that every host uses. And the session's `tests/conftest.py` has already enabled float64 in the pytest process, so only a fresh process shows that the probe enables float64 itself: without that, `kalman_filter` raises `TypeError` and the subprocess fails.

- [x] **Step 2: Run the tests to see them fail**

```bash
uv run pytest tests/test_engine_probe.py -q
```

Expected: a collection error, `ModuleNotFoundError: No module named 'ces_revisions.engine_probe'`.

- [x] **Step 3: Create the probe**

Create `src/ces_revisions/engine_probe.py`:

```python
"""Time the Kalman engine's value and gradient on this host.

Run as ``python -m ces_revisions.engine_probe``. The probe builds a synthetic float64
model at the requested dimensions and times ``jax.value_and_grad`` of the filter's log
likelihood with respect to two log scales, one multiplying every transition covariance
and one every observation covariance, vmapped over a batch of parameter draws. Each run
writes one JSON record, so runs on the Mac and on each cloud size compare field by field.
"""

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from ces_revisions.kalman import LinearGaussianSSM, kalman_filter

# Each transition matrix is this multiple of a random orthogonal matrix, so every
# eigenvalue has exactly this modulus and the state process is stable.
TRANSITION_RADIUS = 0.95
# Standard deviation of the log-scale draws around the scales the model was built at.
LOG_SCALE_SD = 0.1


def synthetic_problem(
    num_steps: int, state_dim: int, obs_dim: int, missing_share: float, seed: int
) -> tuple[LinearGaussianSSM, np.ndarray]:
    """A time-varying float64 model and a panel simulated from it, with NaN cells.

    Each time-indexed array varies by step, every covariance is positive definite, and
    exactly ``round(missing_share * num_steps * obs_dim)`` panel cells are NaN.
    """
    rng = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(num_steps, state_dim, state_dim)))
    ssm = LinearGaussianSSM(
        initial_mean=np.zeros(state_dim),
        initial_cov=np.eye(state_dim),
        transition_matrix=TRANSITION_RADIUS * orthogonal,
        transition_offset=rng.normal(size=(num_steps, state_dim)),
        transition_cov=_random_spd(rng, num_steps, state_dim),
        observation_matrix=rng.normal(size=(num_steps, obs_dim, state_dim))
        / np.sqrt(state_dim),
        observation_offset=rng.normal(size=(num_steps, obs_dim)),
        observation_cov=_random_spd(rng, num_steps, obs_dim),
    )
    panel = _simulate(ssm, rng)
    missing = rng.choice(
        panel.size, size=round(missing_share * panel.size), replace=False
    )
    panel.flat[missing] = np.nan
    return ssm, panel


def time_batches(
    ssm: LinearGaussianSSM,
    panel: np.ndarray,
    batch_sizes: list[int],
    repeats: int,
    seed: int,
) -> list[dict[str, float]]:
    """Seconds per call of the vmapped value and gradient, one entry per batch size.

    A batch size's first call traces and compiles, and is reported as its compile time;
    the median and interquartile range cover the ``repeats`` calls after it. Each call
    pulls its result to the host, so a GPU's asynchronous dispatch cannot hide work.
    """
    ssm = LinearGaussianSSM(*(jnp.asarray(array) for array in ssm))
    panel = jnp.asarray(panel)
    value_and_grad = jax.jit(
        jax.vmap(jax.value_and_grad(_log_likelihood), in_axes=(0, None, None))
    )
    rng = np.random.default_rng(seed)
    timings = []
    for batch in batch_sizes:
        log_scales = jnp.asarray(rng.normal(scale=LOG_SCALE_SD, size=(batch, 2)))
        compile_seconds = _call_seconds(value_and_grad, log_scales, ssm, panel)
        seconds = [
            _call_seconds(value_and_grad, log_scales, ssm, panel)
            for _ in range(repeats)
        ]
        lower, median, upper = np.percentile(seconds, [25, 50, 75])
        timings.append(
            {
                "batch": batch,
                "compile_seconds": compile_seconds,
                "median_seconds": float(median),
                "iqr_seconds": float(upper - lower),
            }
        )
    return timings


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, time the engine, and write the JSON record."""
    args = _parse_args(argv)
    numpyro.enable_x64()  # before the first JAX operation: the engine needs float64
    started = datetime.now(UTC)
    ssm, panel = synthetic_problem(
        args.steps, args.states, args.cells, args.missing_share, args.seed
    )
    record = {
        "label": args.label,
        "dimensions": {"steps": args.steps, "states": args.states, "cells": args.cells},
        "missing_share": args.missing_share,
        "seed": args.seed,
        "repeats": args.repeats,
        "batches": time_batches(ssm, panel, args.batch, args.repeats, args.seed),
        "backend": jax.default_backend(),
        "versions": _jax_versions(),
        "nvidia_driver": _nvidia_driver(),
        "timestamp_utc": started.isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ces_revisions.engine_probe",
        description="Time the Kalman engine's value and gradient on this host.",
    )
    parser.add_argument("--steps", type=int, required=True, help="time steps T")
    parser.add_argument("--states", type=int, required=True, help="state dimension n")
    parser.add_argument("--cells", type=int, required=True, help="cells p per step")
    parser.add_argument(
        "--batch", type=int, nargs="+", required=True, help="batch sizes to time"
    )
    parser.add_argument(
        "--missing-share", type=float, default=0.2, help="share of NaN panel cells"
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="timed calls per batch after compiling"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for model and draws")
    parser.add_argument("--label", required=True, help="host label, e.g. mac or l4")
    parser.add_argument("--out", type=Path, required=True, help="JSON record to write")
    return parser.parse_args(argv)


def _random_spd(rng: np.random.Generator, num_steps: int, dim: int) -> np.ndarray:
    root = rng.normal(size=(num_steps, dim, dim))
    return root @ root.transpose(0, 2, 1) / dim + 0.5 * np.eye(dim)


def _simulate(ssm: LinearGaussianSSM, rng: np.random.Generator) -> np.ndarray:
    state = rng.multivariate_normal(ssm.initial_mean, ssm.initial_cov)
    rows = []
    for a, c, q, z, d, r in zip(
        ssm.transition_matrix,
        ssm.transition_offset,
        ssm.transition_cov,
        ssm.observation_matrix,
        ssm.observation_offset,
        ssm.observation_cov,
        strict=True,
    ):
        state = a @ state + c + np.linalg.cholesky(q) @ rng.normal(size=c.shape)
        rows.append(z @ state + d + np.linalg.cholesky(r) @ rng.normal(size=d.shape))
    return np.stack(rows)


def _log_likelihood(
    log_scales: jax.Array, ssm: LinearGaussianSSM, panel: jax.Array
) -> jax.Array:
    scaled = ssm._replace(
        transition_cov=jnp.exp(log_scales[0]) * ssm.transition_cov,
        observation_cov=jnp.exp(log_scales[1]) * ssm.observation_cov,
    )
    return kalman_filter(scaled, panel).log_likelihood


def _call_seconds(function: Callable[..., object], *args: object) -> float:
    start = time.perf_counter()
    # Pulling the result to the host waits for the device to finish computing it.
    jax.tree.map(np.asarray, function(*args))
    return time.perf_counter() - start


def _jax_versions() -> dict[str, str]:
    """Versions of jax, jaxlib, and any installed JAX CUDA plugin."""
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.name.lower().replace("_", "-")
        if name in {"jax", "jaxlib"} or name.startswith("jax-cuda"):
            versions[name] = dist.version
    return dict(sorted(versions.items()))


def _nvidia_driver() -> str | None:
    """The NVIDIA driver version, or None on a host without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    output = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    return output.splitlines()[0].strip()


if __name__ == "__main__":
    main()
```

Design notes for the reviewer:
- **Stability by construction.** A random orthogonal matrix scaled by 0.95 has spectral radius exactly 0.95, so the test's eigenvalue check cannot fail by chance.
- **Arguments, not constants.** The model and the panel are arguments of the jitted function, not values it closes over. At the record's dimensions, closed-over arrays would be embedded in the compiled program as tens of megabytes of constants, and would slow compilation.
- **One compile per batch size.** `vmap` over the batch adds a leading axis, so each batch size is a new shape and compiles once. That first call is the reported compile time.
- **Synchronization.** Converting each result to NumPy blocks until the device finishes, so the steady-state time is honest on a GPU, where JAX dispatches work asynchronously.

- [x] **Step 4: Run the tests to see them pass**

```bash
uv run pytest tests/test_engine_probe.py -q
uv run pytest -m "not slow and not network" -q
```

Expected: `3 passed`, then, on base `08ed203`, `330 passed, 17 deselected`.

- [x] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check
git add src/ces_revisions/engine_probe.py tests/test_engine_probe.py
git commit -m "Add an engine timing probe that writes one JSON record per host"
```

### Task 9: Mac probe baseline (Verification bullet 9, the Mac run)

**Mode:** suitable for a subagent. Before dispatching it, the controller asks the human partner to pause other heavy work on the Mac.

**Files:**
- Create: `docs/decisions/cloud-gpu-probe/mac.json`

**Interfaces:**
- Consumes: Task 8's `python -m ces_revisions.engine_probe`.
- Produces: the Mac row of the decision record's probe table, at T=280, n=150, p=70, with batch sizes 1, 4, and 16.

- [x] **Step 1: Run the probe at the decision record's dimensions**

```bash
uv run python -m ces_revisions.engine_probe --steps 280 --states 150 --cells 70 --batch 1 4 16 --label mac --out docs/decisions/cloud-gpu-probe/mac.json
```

Expected: exits 0 after about 75 s, printing nothing. Batch 16 peaks near 12 GB of resident memory.

- [x] **Step 2: Check the record**

```bash
jq -e '.label == "mac" and .backend == "cpu" and .nvidia_driver == null
  and .dimensions == {steps: 280, states: 150, cells: 70}
  and [.batches[].batch] == [1, 4, 16]
  and .missing_share == 0.2 and .repeats == 10' docs/decisions/cloud-gpu-probe/mac.json
jq -c '.batches[] | {batch, compile_seconds, median_seconds, iqr_seconds}' docs/decisions/cloud-gpu-probe/mac.json
```

Expected: `true`, then one line per batch size. At planning time, batch 1's median was 0.567 s, and the spec's earlier scratch probe measured 0.54 s. If batch 1's median exceeds about 1.1 s, other work was sharing the CPU: pause it and rerun Step 1.

- [x] **Step 3: Commit**

```bash
uv run ruff format && uv run ruff check
git add docs/decisions/cloud-gpu-probe/mac.json
git commit -m "Record the engine probe's Mac baseline at T=280, n=150, p=70"
```

### Task 10: `.gitignore`, documentation, and final verification (Req 7; Verification bullet 13)

**Files:**
- Modify: `.gitignore` (append a block)
- Modify: `CLAUDE.md` (Layout and tooling; Commands)
- Modify: `README.md` (the Status tree; Getting started)

**Interfaces:**
- Consumes: everything Tasks 5–9 produced.
- Produces: the ignore rules that Plan 4's `infra/` relies on, and documentation of the `cuda`
  extra, `devices.py`, and `engine_probe.py`. Plan 4 adds the `infra/` and runbook references.

- [x] **Step 1: Ignore OpenTofu's local and per-user files**

Append to `.gitignore`:

```gitignore

# OpenTofu local state, provider caches, and per-user backend and variable files
*.tfstate
*.tfstate.*
.terraform/
backend.hcl
terraform.tfvars
```

- [x] **Step 2: Verify the patterns**

```bash
git check-ignore -v infra/state/terraform.tfstate infra/state/terraform.tfstate.backup \
  infra/env/.terraform/providers infra/env/.terraform.tfstate.lock.info \
  infra/env/backend.hcl infra/env/terraform.tfvars
if git check-ignore -q infra/env/backend.hcl.example infra/env/.terraform.lock.hcl; then
  echo "STOP: a file the spec commits is ignored"
else
  echo "backend.hcl.example and .terraform.lock.hcl stay tracked"
fi
```

Expected: six lines, each naming its matching pattern (`*.tfstate`, `*.tfstate.*`, `.terraform/`, `*.tfstate.*`, `backend.hcl`, `terraform.tfvars`), then `backend.hcl.example and .terraform.lock.hcl stay tracked`. Underneath: `git check-ignore` tests paths that need not exist yet. Req 3 commits the provider lock file and the example backend file, and keeps state, provider caches, and personal values out of git.

> Deviation: Git 2.51.0 accepts only one pathname with `git check-ignore -q`, so the two negative checks ran as separate single-path commands; both returned 1 (not ignored).

- [x] **Step 3: Update CLAUDE.md**

Make five edits to `CLAUDE.md`.

First, under **Layout and tooling**, replace the Python bullet:

````markdown
- Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). Runtime dependencies are JAX, NumPy, NumPyro, ArviZ 1.x (`az.from_numpyro` returns an xarray `DataTree`), Polars, and fastexcel (Polars' Excel reader, for Stage 3's workbooks); Dynamax is a dev-only dependency kept as evidence for `docs/decisions/engine.md`.
````

with:

````markdown
- Python 3.14 (`.python-version`; `requires-python = ">=3.14"`) and uv 0.12: `[tool.uv] required-version = "~=0.12.13"` holds the Mac and the cloud VM to one uv series, so `uv.lock` is written one way. Runtime dependencies are JAX, NumPy, NumPyro, ArviZ 1.x (`az.from_numpyro` returns an xarray `DataTree`), Polars, fastexcel (Polars' Excel reader, for Stage 3's workbooks), and python-dotenv (the Stage 4 network-command contact configuration). The `cuda` extra adds `jax[cuda13]` on Linux only: on an NVIDIA host `uv sync --extra cuda` installs JAX's CUDA 13 plugin, and on the Mac it installs nothing. Dynamax is a dev-only dependency kept as evidence for `docs/decisions/engine.md`.
````

Second, in the Kalman engine bullet, replace:

````text
must run before the first JAX operation, which `tests/conftest.py` does for the test session. BlackJAX waits
````

with:

````text
must run before the first JAX operation, which `tests/conftest.py` does for the test session, adding XLA's deterministic GPU flag on hosts with `/dev/nvidia0`. BlackJAX waits
````

Third, insert two bullets directly after the Kalman engine bullet, which ends `confirm Python 3.14 support for any package you add.`:

````markdown
- `src/ces_revisions/devices.py` holds the host's device facts: `has_nvidia_device()`, the `DETERMINISTIC_GPU_FLAG` that `tests/conftest.py` adds to `XLA_FLAGS`, and `chain_method(num_chains)`, which returns NumPyro's `"parallel"` when JAX sees a device per chain and `"vectorized"` otherwise. A CPU host has the four host devices that `tests/conftest.py` sets, and each cloud GPU size has one GPU, so pass `chain_method(n)` to `MCMC` instead of writing `"parallel"`.
- `src/ces_revisions/engine_probe.py` times the engine's value and gradient on the current host (`python -m ces_revisions.engine_probe`) and writes one JSON record. The records behind the cloud GPU decision are in `docs/decisions/cloud-gpu-probe/`, and the AWS account, zone, and quota evidence is in `docs/decisions/cloud-gpu-evidence/`.
````

Fourth, in **Commands**, insert this line directly after the `uv sync` line:

````text
uv sync --extra cuda                            # on a Linux NVIDIA host: also install JAX's CUDA 13 plugin (a no-op on the Mac)
````

Fifth, insert this line directly after the `uv run pytest -m slow` line:

````text
uv run python -m ces_revisions.engine_probe --help   # time the engine's value and gradient on this host; writes a JSON record
````

The `uv sync --extra cuda` line pads its comment to column 49, like its neighbors. The probe command is longer than that column, so its comment follows three spaces, like the single-test line's.

- [x] **Step 4: Update README.md**

Make three edits to `README.md`.

First, in the Status tree, replace:

````text
  plans/                              implementation plans, one per roadmap stage
docs/decisions/                       decision records (engine.md: the Kalman engine)
src/ces_revisions/                    Python package (src layout); kalman.py is the state-space engine, vintages/ the vintage panel
data/raw/                             committed source files for the vintage panel, with a manifest of their hashes
````

with:

````text
  cloud-gpu-environment.md            design spec: the cloud GPU development environment
  plans/                              implementation plans for the roadmap stages and the cloud GPU spec
docs/decisions/                       decision records (engine.md: the Kalman engine)
  cloud-gpu-evidence/                 AWS account, zone, and quota evidence for the cloud GPU environment
  cloud-gpu-probe/                    engine timing records, starting with the Mac baseline
src/ces_revisions/                    Python package (src layout): kalman.py is the state-space engine,
                                      vintages/ and annual/ build the Stage 3/4 inputs,
                                      devices.py picks the chain method, engine_probe.py times the engine
data/raw/                             committed source files for the vintage panel, with a manifest of their hashes
data/annual/raw/                      committed annual benchmark, birth-death, QCEW, and sample sources
````

Second, in **Getting started**, replace:

````markdown
Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.
````

with:

````markdown
Requires [uv](https://docs.astral.sh/uv/) 0.12, the series that `required-version` in `pyproject.toml` pins, and Python 3.14.
````

Third, insert this paragraph, followed by a blank line, directly before the paragraph that begins `The modeling stack is JAX`:

````markdown
On a Linux host with an NVIDIA GPU, `uv sync --extra cuda` also installs JAX's CUDA 13 plugin; the extra is Linux-only, so on the Mac it installs nothing. `src/ces_revisions/devices.py` picks NumPyro's chain method from the devices JAX sees, and `uv run python -m ces_revisions.engine_probe` times the engine's value and gradient on the current host and writes a JSON record like those in [`docs/decisions/cloud-gpu-probe/`](docs/decisions/cloud-gpu-probe/).
````

- [x] **Step 5: Run the final verification**

```bash
uv run pytest -m "not slow and not network" -q
uv run pytest -m slow -q
uv run ruff check
uv run ruff format --check
uv lock --check
```

Expected on base `08ed203`:
- `330 passed, 17 deselected`;
- `13 passed, 334 deselected`;
- `All checks passed!`;
- every file already formatted, including Markdown;
- `uv lock --check` exits 0.

This discharges Verification bullet 13.

- [x] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check
git add .gitignore CLAUDE.md README.md
git commit -m "Document the cuda extra, device module, and engine probe, and ignore OpenTofu state"
```

## Handoff to Plan 4

Plan 4 (Reqs 3–6 and 8–10) consumes this plan's results:

- **Credentials.** The `ces-revisions` profile from Task 2. Whether OpenTofu reads that `aws login` session natively or through `credential_process` is still open (Verification bullet 3).
- **Location.** `.chosen.region` and `.chosen.zone` in `docs/decisions/cloud-gpu-evidence/zone-choice.json`, from Task 3.
- **Quotas.** The request IDs in `service-quotas-request-increase-<region>.json`, from Task 4. Plan
  4's `l4` and `h100` steps wait for `APPROVED`; its `dev` steps do not.
- **Repo.** Plan 4 builds on these pieces:
  - the `cuda` extra and the uv 0.12 pin, since `setup.sh` installs uv from that series and runs `uv sync --locked --extra cuda`;
  - `chain_method` and the device expectation in `tests/test_stack.py`;
  - the probe, with its Mac record at `docs/decisions/cloud-gpu-probe/mac.json`.
- **Evidence for the decision record.** Everything under `docs/decisions/cloud-gpu-evidence/`, plus this plan's planning evidence. That includes the 11.9 GB peak at batch 16, which the `dev` size's 16 GiB must accommodate.
- **Precondition.** The live roadmap makes Stage 6 consume this environment and its decision record;
  Plan 4 checks that semantic text and does not depend on the stale cloud branch's commit hash.
- **Open items Plan 4 discharges.**
  - Req 1: how OpenTofu reads the `aws login` session.
  - Req 2: the first p5.4xlarge On-Demand start.
  - Req 4: the desktop app's SSH connection.
  - Req 6: JAX's CPU fallback on `dev` with the `cuda` extra installed.
  - Req 7: whether the determinism flag makes GPU runs deterministic.
