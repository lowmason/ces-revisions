# Cloud GPU environment runbook

How to build, use, and take down the ces-revisions cloud GPU development environment: one EC2 instance on one disk, whose instance type switches between a CPU size for daily work and three GPU sizes for fits. The decision behind it, with its evidence, is [`docs/decisions/cloud-gpu.md`](decisions/cloud-gpu.md).

Everything under `infra/` runs on the Mac, from the repository root, never on the VM. Resizing stops the instance, which would kill an apply running on it, and the instance's role has no AWS permission beyond Systems Manager. `infra/bin/vm` refuses to run on the VM.

## Conventions

- Each block that calls `aws`, `tofu`, or `infra/bin/vm` starts with `export AWS_PROFILE=ces-revisions`, the profile that `aws login` signed in. When a command reports expired credentials, run `aws login --profile ces-revisions` again; a session lasts up to 12 hours.
- `infra/bin/vm` prints each AWS CLI, OpenTofu, ssh, and rsync command before running it, so every subcommand also shows the command underneath.
- The sizes:
  - `dev` is m7i.xlarge: 4 vCPU and 16 GiB, \$0.20/hr.
  - `l4` is g6.xlarge: an NVIDIA L4 with 24 GB, \$0.81/hr.
  - `l40s` is g6e.xlarge: an NVIDIA L40S with 48 GB, \$1.861/hr.
  - `h100` is p5.4xlarge: an NVIDIA H100 with 80 GB, \$6.88/hr.

  The `dev`, `l4`, and `h100` prices were checked in us-east-1 on 2026-09-13; `l40s` was checked
  on 2026-09-22.

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

Then create a fine-grained GitHub token for the repository `lowmason/ces-revisions`, with Contents and Pull requests set to read and write. Enter it on the VM without putting its value in shell history:

```bash
read -rsp 'Fine-grained token: ' TOKEN; printf '\n'
printf '%s\n' "$TOKEN" | gh auth login --with-token --insecure-storage
unset TOKEN
gh auth setup-git
```

Finally, run `infra/bin/vm sync-config` from the Mac once more.

Underneath:
- `sync-config` copies `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, and `~/.gitconfig`, and sends the names of the Mac's skill, agent, command, and hook links.
- `setup.sh` clones this repository and `lowmason/agent-skills` into `~/Projects/`, installs uv and Python 3.14, and runs `uv sync --locked --extra cuda`. It then recreates the links into `~/Projects/agent-skills/` and reinstalls the guards.
- The VM has no keyring, so `--insecure-storage` writes the token to the mode-600 file `~/.config/gh/hosts.yml`; the hidden `read` keeps its value out of shell history.
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

Review the plan, which should change only `aws_instance.vm`'s `instance_type`, and answer `yes`. Switch back with `infra/bin/vm size dev`, or switch to `l40s` or `h100` the same way. When `l4` fails with `InsufficientInstanceCapacity`, `l40s` is the qualified fallback in the pinned zone.

Underneath:
- OpenTofu stops the instance, changes its type, and starts it again, even if it was stopped. A switch to a GPU size therefore starts billing at that size's rate.
- The instance ID, its root volume, and everything on it stay.
- `infra/bin/vm` records the size in `infra/env/size.auto.tfvars` after a successful apply, so a later `tofu plan` keeps it.
- A GPU size needs its vCPU quota in the region: 4 in "Running On-Demand G and VT instances" for `l4` or `l40s`, and 16 in "Running On-Demand P instances" for `h100`.
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

Before the GitHub token expires, create a new fine-grained token with the same scope: the repository `lowmason/ces-revisions`, with Contents and Pull requests set to read and write. On the VM, enter it without putting its value in shell history:

```bash
read -rsp 'Fine-grained token: ' TOKEN; printf '\n'
printf '%s\n' "$TOKEN" | gh auth login --with-token --insecure-storage
unset TOKEN
gh auth status
```

Then revoke the old token on github.com, under Settings → Developer settings → Personal access tokens → Fine-grained tokens.

Underneath: the VM has no keyring, so `gh` replaces the stored token for github.com in the mode-600 file `~/.config/gh/hosts.yml`. The hidden `read` keeps its value out of shell history. git keeps using `gh` as its credential helper, so nothing else changes. The token opens only `lowmason/ces-revisions`, so `gh` and `git push` against other repositories fail on the VM by design.

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

A size switch fails with `InsufficientInstanceCapacity`: AWS has no spare instance of that type in the zone at the moment. The instance is left stopped, possibly with the new type already set. Switch back with `infra/bin/vm size dev`, try the GPU size again later, or use the qualified `l40s` fallback when `l4` is unavailable.

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

`l4` and `l40s` each need at least 4 in `L-DB2E81BA` ("Running On-Demand G and VT instances"), and `h100` needs at least 16 in `L-417A185B` ("Running On-Demand P instances"). Request an increase in the Service Quotas console, and switch back to `dev` meanwhile. Underneath: EC2 quotas count the vCPUs of running instances per family, not the instances themselves.

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
