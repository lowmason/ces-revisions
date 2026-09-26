# Cloud GPU development environment — Design Spec

> Implementation is split between
> [`specs/plans/completed/3-cloud-gpu-environment.md`](plans/completed/3-cloud-gpu-environment.md), which covers Reqs 1,
> 2, and 7 and creates no AWS resource, and
> [`specs/plans/4-cloud-gpu-environment.md`](plans/4-cloud-gpu-environment.md), which covers Reqs 3–6
> and 8–10. Execute Plan 3 before Plan 4, following each plan's required sub-skill. This spec sits
> outside [`specs/ces-revisions-roadmap.md`](ces-revisions-roadmap.md); do not route it through
> derive-roadmap.

A remote development environment for ces-revisions on AWS: one managed EC2 instance generation on
one EBS disk at a time, whose instance type switches between a CPU size for daily work and four GPU
sizes for fits. Within a generation, size switches preserve the instance and root volume. It is
built with OpenTofu, reached through AWS Systems Manager with no inbound port, guarded by an idle
stop, a GPU runtime cap, and a \$150/month budget, and documented by a runbook and a decision record.
The same checkout runs on the Mac and on every size, so the repo gains a Linux-only CUDA extra,
device-aware tests, a chain-method rule, and an engine timing probe.

Design provenance: brainstormed 2026-09-13 in a session that ran alongside roadmap Stage 2. The
user's decisions are marked **(chosen)**: reuse the existing personal AWS account, OpenTofu, a
runbook plus a decision record, a \$150/month ceiling revisited at Stage 6, and one instance with
several sizes. Facts about AWS, NVIDIA hardware, and the tools were checked on 2026-09-13 against
the sources listed at the end; prices are us-east-1 On-Demand on that date. **(open — resolved by
verification)** marks a fact documentation could not settle; a named Verification bullet
discharges each.

Current-main reconciliation: on 2026-09-16 the spec and its two unexecuted plans were transplanted
from `claude/cloud-gpu-ces-revisions-ee5804` onto current `main` at `08ed203`, after roadmap Stages
2–4 had shipped. The plan documents preserve the 2026-09-13 cloud research as dated evidence but
replace the old branch, test-count, dependency, README, and CLAUDE.md assumptions with contracts
against that current-main snapshot. Dynamic AWS offerings, quotas, prices, package versions, and
personal-configuration counts remain execution-time checks rather than rebasing assumptions.

Capacity amendment: on 2026-09-22, the first `l4` start failed with
`InsufficientInstanceCapacity` in the pinned zone. A separate live check found `g6e.xlarge`
offered in that zone at \$1.861/hr, with 4 vCPUs under the already-approved G and VT quota, so the
user chose `l40s` as a permanent fallback tier. Its first start returned the same capacity error.
A second live check found `g5.xlarge` offered in the zone at \$1.006/hr under the same quota, and
the user chose `a10g` as the next permanent fallback. Plan 4 preserves both failures as evidence
and runs the GPU verification on `a10g`; `l4` and `l40s` remain available for later retries.

Same-region H100 recovery amendment: on 2026-09-25, the original managed VM was terminated through
the AWS Management Console. Its delete-on-termination root volume is gone, so instance and root
volume identity cannot continue across that boundary. Four completed, encrypted, project-tagged
DLM snapshots remain. A replacement `p5.4xlarge` launch in `us-east-1a`, from the pinned clean image
with 8 cores and 1 thread per core, exhausted 25 attempts with `InsufficientInstanceCapacity`.
Those CPU options expose 8 active vCPUs, while EC2 still counts the type's 16 default vCPUs against
the P-family quota. AWS named `us-east-1b` through `us-east-1f` as alternatives. The recovery pins
`us-east-1b`, the first alphabetic same-region alternative that offers every configured instance
type, and creates a clean replacement generation. No successful fallback apply is assumed.

Retry amendment: the approved us-east-1b fallback apply moved the empty subnet and its route-table
association, then exhausted another 25 `p5.4xlarge` launch attempts over about 55 minutes with
`InsufficientInstanceCapacity`. It created neither an instance nor a root volume, and the four
snapshots remain. A 2026-09-26 read-only check found both applicable P5 Capacity Block quotas at
zero, so the account cannot yet inspect or purchase a block. Plan 4 permits one separately planned,
reviewed, and approved On-Demand retry against the already-moved us-east-1b network; no successful
replacement is assumed.

## Motivation

- The Mac is not the fit target. On the M4 Max, a scratch probe of the dense engine
  ([`src/ces_revisions/kalman.py`](../src/ces_revisions/kalman.py)) took 0.54 s per
  value-and-gradient at T=280, n=150, p=70, projecting 10–40 h per four-chain fit at Stage 7 scale
  before simulation-based calibration multiplies it.
- Roadmap Stage 6 consumes this environment and its decision record, and records the engine's
  value-and-gradient time on the cloud GPU at Stage 7–9 dimensions. That dependency comes from the
  roadmap amendment made in the Stage 2 session on 2026-09-13.
- Float64 is mandatory (Req 17 of [`specs/ces-revisions.md`](ces-revisions.md)). NVIDIA's Ampere
  (GA102) and Ada (AD102) architecture whitepapers state that those chips carry two FP64 units per
  streaming multiprocessor, run FP64 at 1/64 of their FP32 rate, and include them "to ensure any
  programs with FP64 code operate correctly". NVIDIA rates the A100 at 9.7 and the H100 at 34 FP64
  TFLOPS. By the 1/64 rule, the L4 (g6), A10G (g5), and L40S (g6e) run float64 correctly at about
  0.5, 0.5, and 1.4 TFLOPS.
- A single chain's `lax.scan` launches each operation as its own GPU kernel with no parallelism
  across steps, so a cheap GPU may keep pace there. Batched work (`vmap` over chains or SBC
  replicates) multiplies the arithmetic per kernel, and there the FP64 rating decides. Where that
  crossover falls is measured, not assumed.
- Learning the cloud stack is a project goal, so the runbook explains what happens underneath each
  step.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Provider | AWS, the user's existing personal account **(chosen)** | An account with billing history is the better bet for quick GPU quota approval, and p5.4xlarge (H100, \$6.88/hr) gives the most float64 per dollar in either cloud. Azure's NC24ads A100 v4 (\$3.67/hr) returns only under Req 2's stopping rule. |
| Topology | One managed instance generation, one disk, instance type as a variable **(chosen)** | One environment to keep current within each generation, and the type stays a variable for Stage 6. Rejected: a CPU dev box plus disposable GPU runners (two environments and an S3 job flow before any stage needs them) and the Mac as the dev machine (reverses the move to remote development). |
| Infrastructure as code | OpenTofu **(chosen)** | Plan, apply, and destroy against a state record; in Homebrew core; the same HCL and AWS provider as Terraform. |
| Learning artifacts | Runbook plus decision record **(chosen)** | Explanations persist beside the infrastructure they describe. |
| Budget | \$150/month, revisited at Stage 6 **(chosen)** | Covers the CPU size and disk (about \$40–60/month) plus about 13 H100-hours or 120 L4-hours. |
| Image | Canonical Ubuntu 24.04 with Ubuntu's NVIDIA server driver | AWS's Deep Learning Base AMI lists only GPU instance families as supported, and JAX's CUDA wheels bring their own CUDA libraries. |
| Human credentials | IAM user with MFA, and `aws login` | Temporary credentials and no long-lived keys. IAM Identity Center would require making the account an AWS Organizations management account. |
| Network | Public subnet with no inbound rules; access through Systems Manager | A private subnet needs a NAT gateway (about \$33/month) or Systems Manager VPC endpoints, for no benefit here. |

## Requirements

### Account and infrastructure

**Req 1 — Account, identity, and Mac tooling.** Work happens in the user's existing personal AWS
account. The root user has MFA and is not used after setup. Daily human access is an IAM user with
MFA and administrator permissions, used from the console and from `aws login` (AWS CLI 2.32.0 or
later); that user has no access keys. OpenTofu reads the `aws login` session directly or through a
profile whose `credential_process` runs `aws configure export-credentials --format process`
**(open — resolved by verification: whether the OpenTofu AWS provider reads login sessions
natively)**. The Mac gets AWS CLI v2, the Session Manager plugin, and OpenTofu 1.10 or later from
Homebrew.

**Req 2 — Region and availability zone, chosen by evidence before any apply.** The candidate regions,
in order, are us-east-1, us-east-2, us-west-2, and eu-west-2. A region qualifies when the AWS
Price List API has a Linux On-Demand price for p5.4xlarge there and at least one of its availability
zones offers m7i.xlarge, g6.xlarge, and p5.4xlarge
(`aws ec2 describe-instance-type-offerings --location-type availability-zone`). The environment
uses the first qualifying region in that order and, within it, the alphabetically first zone that
offers all three types. AWS's August 2025 announcement sold
single-GPU P5 On-Demand only outside the US, with Capacity Blocks in US East **(open — resolved by
verification)**. If no candidate qualifies, work stops and the provider choice returns to the user.
The command outputs are kept for the decision record. Immediately after the choice, Service Quotas
increases are requested in that region: "Running On-Demand G and VT instances" to at least 4 vCPUs
and "Running On-Demand P instances" to at least 16, recording the prior values and the request IDs.

**US and Canada H100 fallback amendment (2026-09-24).** The original first-qualified choice and
the deployed environment remain in us-east-1. For regional readiness, evaluate every standard
commercial AWS Region in the United States and Canada: us-east-1, us-east-2, us-west-1, us-west-2,
ca-central-1, and ca-west-1. GovCloud, Local Zones, and Mexico are outside this list. Keep
`p5.4xlarge` as a hard requirement, require a positive Linux Shared On-Demand price and at least
one Availability Zone offering it, and rank eligible Regions by price and measured endpoint
proximity from the current workstation, using the existing environment as a final tie-breaker.
Record opt-in status, the Canonical Ubuntu parameter, P-family quota, and P-quota request history.
Build a request plan for 16 P-family vCPUs only in eligible fallback Regions. A separate approval
is required before submitting those regional requests. The readiness survey does not move or copy
the environment; a cross-region deployment requires a separate reviewed plan and state.

**Same-region H100 zone recovery amendment (2026-09-25).** Preserve the original `us-east-1a`
choice and its evidence as the historical first-qualified result. After the failed 25-attempt
replacement launch there, select `us-east-1b` as the first alphabetic Availability Zone among the
alternatives AWS reported that also offers `m7i.xlarge`, `g5.xlarge`, `g6.xlarge`, `g6e.xlarge`,
and `p5.4xlarge`. Keep `us-east-1`, its price and quota, and the existing VPC, IAM, DLM, budget, and
state architecture. Replace the empty public subnet and its route-table association because a
subnet belongs to one Availability Zone. Create the replacement instance and root volume from the
pinned Canonical image. This offering check does not reserve capacity, so the apply can still fail
with `InsufficientInstanceCapacity`.

**Req 3 — OpenTofu layout and state.**

- `infra/state/` creates the state bucket with versioning on, all public access blocked, default
  encryption, and `prevent_destroy`. Its own state stays local and gitignored.
- `infra/env/` creates everything else and stores its state in that bucket through the S3 backend
  with `use_lockfile = true`, which locks with S3 conditional writes instead of a DynamoDB table.
  Bucket, key, and region come from a gitignored `infra/env/backend.hcl` (partial backend
  configuration); a committed `backend.hcl.example` shows its shape. Personal values, such as the
  budget email address, live in a gitignored `terraform.tfvars`.
- The provider dependency lock file, `.terraform.lock.hcl`, is committed. The AWS provider's
  `default_tags` put `project = ces-revisions` on every resource.
- No committed file contains an account ID, a bucket name, an email address, or a token.

**Req 4 — Network and access.**

- A dedicated VPC has one public subnet in the currently pinned Req 2 zone, an internet gateway,
  and a route table. A same-region zone recovery replaces an empty subnet and its route-table
  association while retaining the VPC, gateway, route table, and security group. The instance gets
  an auto-assigned public IPv4 address, used only for outbound traffic. Its security group has no
  ingress rules and unrestricted egress. No NAT gateway, VPC endpoint, or Elastic IP exists.
- An IAM instance profile grants only `AmazonSSMManagedInstanceCore`. The instance requires IMDSv2.
- Shell access is `aws ssm start-session`. SSH runs through Session Manager's `AWS-StartSSHSession`
  document from a `~/.ssh/config` host entry that uses `ProxyCommand` (the desktop app does not
  support `ProxyJump`), names the AWS CLI by absolute path so it resolves outside a login shell, and
  passes the AWS profile. A dedicated ed25519 key pair authenticates SSH; its private half stays on
  the Mac.
- The Claude Code desktop app connects through that host entry **(open — resolved by
  verification)**. If it cannot, the fallback is `AWS-StartPortForwardingSession` to
  `localhost:2222`, with the port typed into the connection dialog, because the app ignores `Port`
  in `~/.ssh/config`.

**Req 5 — Instance, image, sizes, and disk.** One `aws_instance`, whose type comes from a `size`
variable (default `dev`):

| `size` | Instance type | vCPU / memory | GPU | On-Demand |
|---|---|---|---|---|
| `dev` | m7i.xlarge | 4 / 16 GiB | none | \$0.20/hr |
| `l4` | g6.xlarge | 4 / 16 GiB | L4, 24 GB | \$0.81/hr |
| `l40s` | g6e.xlarge | 4 / 32 GiB | L40S, 48 GB | \$1.861/hr |
| `a10g` | g5.xlarge | 4 / 16 GiB | A10G, 24 GB | \$1.006/hr |
| `h100` | p5.4xlarge | 8 active, 16 default / 256 GiB | H100, 80 GB | \$6.88/hr |

- Within one instance generation, changing `size` updates the instance in place (stop, modify,
  start); the instance ID and root volume do not change. A terminated instance begins a new
  generation with a new instance and root volume.
- On `h100`, CPU options keep all 8 physical cores and expose 1 thread per core, for 8 active vCPUs.
  EC2 still applies the P-family quota requirement based on the instance type's 16 default vCPUs
  and charges the full hourly price.
- The AMI is Canonical Ubuntu 24.04 LTS for amd64. Its ID is read once from Canonical's public
  Systems Manager parameter and pinned as a variable, never looked up at plan time, so no kernel or
  driver change arrives between measurements.
- The root volume is gp3, 100 GB, and is deleted with the instance.
- `instance_initiated_shutdown_behavior = "stop"`, so an operating-system shutdown stops the
  instance rather than terminating it.
- The instance's `lifecycle` block sets `prevent_destroy = true` and ignores changes to `user_data`;
  cloud-init runs once per instance, and later changes arrive through `setup.sh`.

### Machine setup

**Req 6 — First boot, user setup, and carried configuration.**

- At first boot, as root, cloud-init installs Ubuntu's packaged NVIDIA server driver from a branch
  numbered 580 or later (the minimum for `jax[cuda13]`) and holds it with `apt-mark hold`; installs
  `git` and `gh`; installs the SSH public key; and installs the Req 8 guards, which OpenTofu embeds
  in the user data from `infra/vm/guards/`.
- `infra/vm/setup.sh` runs as `ubuntu` and is idempotent. It installs uv from the Req 7 version
  series, runs `uv python install 3.14`, clones `lowmason/ces-revisions` and `lowmason/agent-skills`
  into `~/Projects/`, runs `uv sync --locked --extra cuda`, recreates the Mac's skill, agent, and
  command links (32, 7, and 3 on 2026-09-13) as links into `~/Projects/agent-skills/`, using the
  link names that `sync-config` sent, and reinstalls the guards from the checkout. `sync-config`
  therefore runs before `setup.sh`.
- With the `cuda` extra installed and no NVIDIA device, JAX falls back to the CPU on `dev` **(open —
  resolved by verification)**. If it does not, `setup.sh` gives CPU sizes a separate environment
  through `UV_PROJECT_ENVIRONMENT`, synced without the extra.
- `infra/bin/vm sync-config` copies from the Mac with rsync over the SSH host entry, the Mac being
  the source until cutover: `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, `~/.claude/hooks/`,
  `~/.gitconfig`, and the names of the Mac's skill, agent, and command links. At cutover it copies
  this project's memory folder once from the Mac to the VM's memory folder for
  `~/Projects/ces-revisions`; the VM owns project memory afterward.
- The user creates a fine-grained GitHub token scoped to `lowmason/ces-revisions` (contents and pull
  requests, read and write) and enters it on the VM with `gh auth login --with-token`;
  `gh auth setup-git` makes `git push` use it. The agent never handles the token.
- A replacement generation starts from the pinned clean image and repeats `sync-config` and
  `setup.sh` after the new instance becomes reachable. Its public clone is sufficient through the
  H100 checks. The interactive GitHub token entry waits for final cutover; a token stored only on a
  terminated root volume does not carry forward.
- The desktop app installs Claude Code on the VM at its first SSH connection.

### Repo changes

**Req 7 — The repo on GPU hosts.**

- `pyproject.toml` gains `[project.optional-dependencies]` with
  `cuda = ["jax[cuda13]>=0.11.1; sys_platform == 'linux'"]`, so the Mac's plain `uv sync` installs
  nothing new. `[tool.uv] required-version` pins a compatible-release range (`~=`) of the newest uv
  minor series at plan time, which both the Mac and the VM install; `uv.lock` is regenerated once under it; and
  the `uv_build` range in `[build-system]` widens to include that series.
- [`tests/test_stack.py`](../tests/test_stack.py): the four-host-device test becomes a device
  expectation. Float64 holds everywhere; where `/dev/nvidia0` exists, `jax.default_backend()` is
  `"gpu"`; elsewhere, `jax.local_device_count() == 4`.
- [`tests/conftest.py`](../tests/conftest.py): where `/dev/nvidia0` exists, XLA's deterministic GPU
  operations are enabled through `XLA_FLAGS` before any JAX operation, beside the existing host
  device count **(open — resolved by verification: the flag's name in the installed XLA)**.
- `src/ces_revisions/devices.py` defines `chain_method(num_chains: int) -> str`, returning
  `"parallel"` when `jax.local_device_count() >= num_chains` and `"vectorized"` otherwise. Both slow
  tests in [`tests/test_synthetic_pilot.py`](../tests/test_synthetic_pilot.py) call it instead of
  the `"parallel"` literal, and `tests/test_devices.py` covers both branches.
- `src/ces_revisions/engine_probe.py` runs as `python -m ces_revisions.engine_probe` with
  `--steps`, `--states`, `--cells`, `--batch` (one or more sizes), `--missing-share` (default 0.2),
  `--repeats` (default 10), `--seed` (default 0), `--label`, and `--out`.
  - It builds a synthetic float64 `LinearGaussianSSM` with time-varying arrays, stable transition
    matrices (spectral radius below 1), positive-definite covariances, and the given share of NaN
    cells.
  - It times `jax.value_and_grad` of `kalman_filter(...).log_likelihood` with respect to two
    log-scale parameters, one multiplying every transition covariance and one every observation
    covariance, `vmap`ped over each batch size's parameter draws.
  - It writes JSON holding the dimensions, batch sizes, label, compile time (the first call),
    steady-state seconds per call (median and interquartile range over the repeats, each call
    synchronized by pulling its result to the host), `jax.default_backend()`, the versions of
    `jax`, `jaxlib`, and any CUDA plugin, the NVIDIA driver version when `nvidia-smi` exists, and a
    UTC timestamp.
  - `tests/test_engine_probe.py` runs it in the fast tier at tiny dimensions and checks the JSON
    fields.
- `.gitignore` covers OpenTofu local state, `.terraform/` directories, `backend.hcl`, and
  `terraform.tfvars`.
- CLAUDE.md's Commands and Layout sections and the README's Getting started describe the `cuda`
  extra, `devices.py`, `engine_probe.py`, and `infra/`, and link the runbook.

### Operations

**Req 8 — Cost guards and data safety.**

- Idle stop: a systemd timer runs every 5 minutes and calls `shutdown -h now` once the 1-minute load
  average has stayed below 0.3 for 45 consecutive minutes and, where an NVIDIA device exists, GPU
  utilization has stayed below 5% over the same window. The window and thresholds come from an
  environment file.
- GPU cap: at boot with an NVIDIA device present, a unit schedules `shutdown -h +480`. The runbook
  shows how to inspect, cancel, and reschedule it.
- Budget: an AWS Budgets monthly cost budget of \$150, filtered on the cost allocation tag
  `project = ces-revisions` (activated once in the Billing console), emails at 50% and 80% of forecast spend and at 100%
  of actual spend. At 100% of actual spend, an `aws_budgets_budget_action` of type
  `RUN_SSM_DOCUMENTS` with sub-type `STOP_EC2_INSTANCES` and `approval_model = "AUTOMATIC"` stops
  the instance through an execution role limited to that action. Instance-targeted actions do not
  reset at the next budget period, so the runbook resets it.
- Snapshots: an `aws_dlm_lifecycle_policy` snapshots the project's volume daily, keeps 7 snapshots,
  and copies tags. Restoring within an instance generation uses EC2's replace-root-volume task with
  a snapshot from that generation. Completed snapshots can outlive a terminated source volume; the
  four retained generation-1 snapshots remain recovery evidence while generation 2 starts from the
  pinned clean image.

**Req 9 — Operations wrapper and runbook.**

- `infra/` is operated from the Mac, never from the VM: resizing stops the instance, which would
  kill an apply running on it, and the instance role has no AWS permission beyond Systems Manager.
  `infra/bin/vm` exits with that explanation when run on the VM.
- `infra/bin/vm` provides `start`, `stop`, `status`, `connect` (a Session Manager shell),
  `size <dev|l4|l40s|a10g|h100>` (runs `tofu apply` in `infra/env` with that size), and
  `sync-config` (Req 6). Every subcommand prints the AWS CLI or OpenTofu command it runs before
  running it.
- `docs/cloud-gpu-runbook.md` covers one-time setup, daily use, switching sizes, running a long GPU
  job, checking spend, snapshots and restore, token rotation, updating the held driver or pinned image,
  recovery after a budget stop,
  teardown (removing `prevent_destroy`, then destroying `infra/env` and `infra/state`), and
  troubleshooting (insufficient capacity, quota errors, Session Manager not connecting, JAX not
  seeing the GPU). Each step gives its commands and a short note on what happens underneath. The
  document follows CLAUDE.md's Markdown conventions.

**Req 10 — Decision record.** `docs/decisions/cloud-gpu.md` takes the shape of
[`docs/decisions/engine.md`](../docs/decisions/engine.md). It is written after the Verification
bullets pass, from their recorded evidence, and contains no placeholder.

- Context: the float64 mandate and FP64 facts from Motivation, the Stage 6 dependency, and the Mac
  probe.
- Decision: AWS in the Req 2 zone with its evidence; the five sizes; the image and driver versions;
  the access method that worked; OpenTofu; the guards and budget.
- Evidence: prior and requested quota values; the dated US and Canada `p5.4xlarge` readiness survey,
  its ranked eligible Regions, and the fact that the fallback regional P-quota actions reserved no
  capacity and created no deployment; the failed `l4` and `l40s` capacity attempts and the separately
  qualified `a10g` fallback; the generation-1 console termination and four retained DLM snapshots;
  the 25 failed `p5.4xlarge` launch attempts in `us-east-1a`; the AWS-reported alternatives and
  `us-east-1b` same-region selection; a probe table for the Mac, `dev`, `a10g`, and `h100` at T=280,
  n=150, p=70 with batch sizes 1, 4, and 16 everywhere and 64 on `h100`, with the JSON files under
  `docs/decisions/cloud-gpu-probe/`; and the BLS canary's outcome (allowed or blocked, with its HTTP
  status).
- Alternatives considered: Azure NC24ads A100 v4; the dev-box-plus-runners and Mac-dev topologies;
  the Deep Learning Base AMI; IAM Identity Center; a private subnet with a NAT gateway; SageMaker
  and AWS Batch; and the T4 fallback size.
- Revisit triggers: Stage 6's measurement, which also revisits the budget and the default GPU size;
  a later stage needing parallel fits (the dev-box-plus-runners topology); p5.4xlarge losing
  On-Demand availability in the chosen zone; the probe showing `dev` competitive with the GPU sizes.

Stage 6's measurement at Stage 7–9 dimensions belongs in `docs/decisions/seasonal-state.md`, as the
roadmap assigns it, not in this record.

## Verification — observable outcomes

- [ ] The IAM user has no access keys (`aws iam list-access-keys`), the account summary reports root
      MFA enabled (`aws iam get-account-summary`), and Req 2's zone choice and quota requests are
      recorded with command outputs, prior quota values, and request IDs, the chosen zone meeting
      Req 2's rule.
- [ ] `tofu apply` succeeds from empty state in `infra/state` and then in `infra/env`, and an
      immediate `tofu plan` in each reports no changes.
- [ ] OpenTofu authenticates from an `aws login` session, and the decision record says whether
      `credential_process` was needed (discharges Req 1's open item).
- [ ] The instance's security group has zero ingress rules, checked with the AWS CLI, and a search of
      the committed tree finds no account ID, bucket name, email address, or token.
- [ ] A Session Manager shell and a Claude Code desktop SSH session both open on the VM, and the
      decision record names the connection method (discharges Req 4's open item).
- [ ] On `dev`, `uv sync --locked --extra cuda` succeeds and `uv run pytest` passes in full,
      including `slow`, with the `cpu` backend and four host devices (discharges Req 6's open
      item).
- [ ] On `a10g` and on `h100`, `nvidia-smi` reports the GPU with a driver numbered 580 or later, and
      `uv run pytest` passes in full with the `gpu` backend and `chain_method(4) == "vectorized"`
      (discharges Req 7's open item).
- [ ] Within each generation, successful `size` switches keep the instance ID and root volume ID,
      checked with the AWS CLI. The lineage evidence records that the console termination ended
      generation 1, its four completed DLM snapshots survived, and the failed `us-east-1a` H100
      replacement created neither an instance nor a root volume. The first successful p5.4xlarge
      On-Demand start in a replacement generation discharges Req 2's open item.
- [ ] Probe JSON exists for the Mac, `dev`, `a10g`, and `h100` at T=280, n=150, p=70 with batch sizes
      1, 4, and 16 everywhere and 64 on `h100`, committed under `docs/decisions/cloud-gpu-probe/`.
- [ ] Idle stop stops the instance under a shortened test window, the GPU cap is scheduled after an
      `a10g` boot, the budget and its action exist, and at least one lifecycle snapshot exists.
- [ ] On the VM, `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, and the skill, agent, and command
      links exist, and a Claude Code session there lists the personal skills and reads the project
      memory.
- [ ] One small `download.bls.gov` fetch, run once by hand from a VM shell with a User-Agent naming
      the project and a contact, is recorded in the decision record only as allowed or blocked with
      its HTTP status; no committed file holds the User-Agent string.
- [ ] On the Mac, the fast tier passes, including `tests/test_devices.py` and
      `tests/test_engine_probe.py`, and `uv run ruff check` and `uv run ruff format --check` pass.
- [ ] The runbook and the decision record are committed, and CLAUDE.md and the README describe the
      `cuda` extra, `infra/`, and the runbook.

## Out of scope

- Building anything on Azure; Req 2's stopping rule is its only role.
- A CPU dev box plus disposable GPU runners, and moving run artifacts through S3.
- Spot instances, Capacity Blocks, and Savings Plans.
- Containers, GPU continuous integration, and multi-GPU instances.
- Stage 6's measurement at Stage 7–9 dimensions, and Stage 3's data layout and fetch location.
- Editing `specs/ces-revisions-roadmap.md`.

## Rollout note

This spec sits outside the roadmap and carries no stage stamp. It was designed to run alongside
roadmap Stages 2–5 and gates Stage 6. At the 2026-09-16 current-main reconciliation, Stages 2–4 were
complete and Stage 5 had a committed implementation plan but had not run. Plan 3 should therefore
start before or alongside Stage 5, Plan 4 may overlap Stage 5 after Plan 3 completes, and neither
plan changes the Stage 6 gate recorded in the roadmap.

GPU quota approval is the critical path, so its request goes out first and nothing on the CPU path
waits for it:

1. Req 1: account hygiene and Mac tooling.
2. Req 2: zone choice and quota requests.
3. Req 7 on the Mac, test-first, with the probe's Mac baseline; no AWS resource is needed.
4. Reqs 3–6, 8, and 9 at `size = dev`, and the `dev` Verification bullets.
5. Try `l4`; if it lacks capacity, try `l40s`; if that also lacks capacity, qualify and run
   `a10g`; then run the `h100` bullets once its quota is approved. If a clean H100 replacement
   exhausts capacity attempts in `us-east-1a`, recover in the first qualifying alphabetic
   same-region alternative, currently `us-east-1b`, without treating its offering as reserved
   capacity. If that fallback also exhausts its launch attempts, record the partial network move
   and allow only a separately planned, reviewed, and approved one-shot retry.
6. Req 10, the runbook's final pass, the documentation updates, and the cutover memory copy.

Plan 3 is steps 1–3. It discharges Verification bullets 1 and 13 and the Mac run in bullet 9, and
documents the `cuda` extra, `devices.py`, and `engine_probe.py`. Plan 4 is steps 4–6. It adds the
`infra/` and runbook references to CLAUDE.md and the README and discharges the remaining bullets;
its GPU steps wait on quota approval, and its `dev` steps do not.

Steps that enter credentials or change account security belong to the user: root MFA, creating the
IAM user and registering its MFA device, the `aws login` browser sign-in, activating the cost
allocation tag, and creating and entering the GitHub token. The agent runs read-only checks and
submits quota requests only with the user's explicit approval.

The \$150 ceiling holds until Stage 6's measurement, which revisits it along with the default GPU
size. On plan completion this spec retires to `specs/completed/` under the plan-completion protocol.

## Sources

Checked 2026-09-13, with the G6e fallback checked 2026-09-22 and the same-region H100 recovery
checked 2026-09-25.

- NVIDIA, [Ada Lovelace professional GPU architecture whitepaper v1.1](https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf)
  and [Ampere GA102 GPU architecture whitepaper v2.1](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)
  (FP64 units per streaming multiprocessor and the 1/64 rate).
- NVIDIA, [A100](https://www.nvidia.com/en-us/data-center/a100/) and
  [H100](https://www.nvidia.com/en-us/data-center/h100/) product pages (FP64 TFLOPS).
- Puget Systems, [RTX 4090 scientific computing performance](https://www.pugetsystems.com/labs/hpc/nvidia-rtx4090-ml-ai-and-scientific-computing-performance-preliminary-2382/)
  (measured FP64 HPL on AD102).
- JAX, [discussion #10233](https://github.com/jax-ml/jax/discussions/10233) (`scan` and kernel
  launches on GPU) and [installation](https://docs.jax.dev/en/latest/installation.html) (driver
  minimums).
- AWS, [single-GPU P5 announcement](https://aws.amazon.com/about-aws/whats-new/2025/08/amazon-p5-single-gpu-instances-now-available/),
  [G6e instances](https://aws.amazon.com/ec2/instance-types/g6e/),
  [EC2 instance-type quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html),
  [Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) release notes](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubaseoss-ul2404-2026-09-02.html),
  [login with console credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html),
  and [budget actions](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-controls.html).
- OpenTofu, [S3 backend](https://opentofu.org/docs/language/settings/backends/s3/); Terraform
  Registry, [`aws_budgets_budget_action`](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/budgets_budget_action).
- Claude Code, [desktop documentation](https://code.claude.com/docs/en/desktop) and issues
  [#40967](https://github.com/anthropics/claude-code/issues/40967) (`ProxyCommand` and `ProxyJump`)
  and [#26809](https://github.com/anthropics/claude-code/issues/26809) (`Port` ignored).
- Prices: [instances.vantage.sh](https://instances.vantage.sh/aws/ec2/) (AWS Price List data) and
  the [Azure Retail Prices API](https://prices.azure.com/api/retail/prices).
