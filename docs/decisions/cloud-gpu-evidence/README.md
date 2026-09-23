# Cloud GPU environment: account and zone evidence

Command outputs recorded by plans 3 and 4, the two plans of [`specs/cloud-gpu-environment.md`](../../../specs/cloud-gpu-environment.md), for its Reqs 1–6 and 8. The decision record `docs/decisions/cloud-gpu.md`, which plan 4 writes, cites them. Every `aws` command ran as the IAM user through the `ces-revisions` profile, and every command on the VM ran through the `ces-revisions-vm` SSH host entry.

No file here holds an account ID, an ARN, an email address, or a token. Outputs that carry one were narrowed with `--query` before they were written. Before each commit, this directory and `../cloud-gpu-probe/` were searched for the account ID, the state bucket's name, ARNs, and at signs.

## Req 1: identity

- `iam-list-access-keys.json` — `aws iam list-access-keys`: the IAM user's access keys. Req 1 requires none.
- `iam-get-account-summary.json` — `aws iam get-account-summary`: account-wide counts and flags. `AccountMFAEnabled` is 1 when the root user has MFA, and `AccountAccessKeysPresent` is 0 when root has no access keys.
- `iam-list-mfa-devices.json` — `aws iam list-mfa-devices`, narrowed to `MFADeviceCount`: how many MFA devices the IAM user has, without their serial-number ARNs.

## Req 2: region and zone

Req 2 evaluates us-east-1, us-east-2, us-west-2, and eu-west-2, in that order, and stops at the first region that qualifies. A region qualifies when the AWS Price List API has a Linux On-Demand price for p5.4xlarge there and at least one of its availability zones offers m7i.xlarge, g6.xlarge, and p5.4xlarge. Within that region, the environment uses the alphabetically first such zone.

For each region evaluated:

- `pricing-get-products-p5.4xlarge-<region>.json` — `aws pricing get-products`, called on the Price List Query API endpoint in us-east-1. It is filtered to p5.4xlarge in the region, with Linux, shared tenancy, no preinstalled software, and used capacity. Each product is decoded from its JSON string and reduced to its SKU, attributes, and On-Demand terms.
- `ec2-describe-instance-type-offerings-<region>.json` — `aws ec2 describe-instance-type-offerings --location-type availability-zone` for the three instance types.
- `region-<region>.json` — derived from the two files above. It holds the positive On-Demand hourly prices of products whose `marketoption` attribute is absent or `OnDemand`, so a Capacity Block price does not count. It also lists the zones that offer all three types, and says whether the region qualifies.

`zone-choice.json` lists the regions evaluated, in order, and `chosen`: the first qualifying region with its first zone, or `null` when none qualified. Each account maps zone names such as `us-east-1a` to physical zones in its own way, so the choice holds for this account only.

## Req 2: GPU quota requests

EC2 On-Demand quotas count the vCPUs of running instances, per instance family and region. Req 2 asks for:

- "Running On-Demand G and VT instances" (`L-DB2E81BA`) of at least 4, enough for one g6.xlarge,
  one g6e.xlarge, or one g5.xlarge;
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

## Reqs 4 and 5: image

- `ssm-get-parameters-ubuntu-24.04-<region>.json` — `aws ssm get-parameters` for Canonical's parameter that names the current Ubuntu 24.04 LTS amd64 gp3 image, narrowed to its name, value, version, and date. The value is the image pinned in `infra/env/pinned.auto.tfvars`.
- `ec2-describe-images-<region>.json` — `aws ec2 describe-images` for that image, narrowed to its ID, name, creation date, architecture, root device type, virtualization type, and boot mode. The owner's account ID is left out.

## Req 3: state bucket

- `s3-state-bucket.json` — `aws s3api get-bucket-versioning`, `get-public-access-block`, and `get-bucket-encryption` for the bucket that `infra/state` created, combined without the bucket's name: versioning enabled, all four public access blocks on, and AES256 default encryption. A destroy plan for `infra/state` failed with `Resource instance cannot be destroyed`, which confirms `prevent_destroy`.

## Reqs 3 to 5: the environment at `dev`

- `opentofu-credentials.json` — the OpenTofu and AWS provider versions, and the AWS profile OpenTofu used: `ces-revisions` when it read the `aws login` session directly, or `ces-revisions-process` when it needed a `credential_process` profile.
- `ec2-instance-dev.json` — `aws ec2 describe-instances`, `describe-instance-attribute`, and `describe-volumes`, with `aws iam list-attached-role-policies` and `list-role-policies`, for the instance at `dev`. It records the type, zone, image, IMDSv2 setting, whether a public address and an instance profile exist, the shutdown behavior, the root volume, and the role's policies. The address and the profile's ARN are left out.
- `ec2-network-dev.json` — `aws ec2 describe-security-groups`, `describe-nat-gateways`, `describe-vpc-endpoints`, and `describe-addresses`. It records the security group's ingress rule count and egress rules, the counts of NAT gateways and VPC endpoints in the VPC, and the count of Elastic IPs tagged `project = ces-revisions`.

## Req 4: access

- `ssm-instance-information.json` — `aws ssm describe-instance-information` for the instance after first boot: the agent's ping status and version, and the platform.
- `access.json` — which access methods opened a session on the VM. It covers a Session Manager shell; SSH through the `ces-revisions-vm` host entry, with the host key checked against the fingerprint read through Session Manager; and how the Claude Code desktop app connected: `host-entry`, `port-forward`, or `neither`.

## Req 6: machine setup

- `vm-first-boot.txt` — on the VM after cloud-init finished: the release and kernels, the held packages, the driver packages' versions, the git and gh versions, the guards' unit states and `systemd-analyze verify`, and `nvidia-smi`'s exit status on `dev`.
- `vm-carried-config.txt` — on the VM after `setup.sh` and the GitHub token: the skill, agent, command, and hook link counts, with broken links counted; the copied `CLAUDE.md` and `settings.json`; and the checkout's branch.

## Reqs 6 and 8 on `dev`

- `vm-dev-checks.txt` — on `dev`: JAX's backend and device count with the `cuda` extra installed and no GPU, the number of stderr lines, and the first message from JAX's CUDA plugin if any; then `uv sync --locked --extra cuda` and the full test run's summary line.
- `bls-canary.json` — one by-hand fetch of `https://download.bls.gov/pub/time.series/ce/ce.datatype` from the VM, recorded as its HTTP status and whether BLS allowed it. The User-Agent is not recorded.
- `vm-idle-stop.txt` — the wait for the idle stop with its window shortened to 10 minutes, its journal line from the boot it ended, and the window restored to 45 minutes.
- `../cloud-gpu-probe/dev.json` — the engine probe on `dev` at T=280, n=150, p=70, with batch sizes 1, 4, and 16.

## Req 8: budget and snapshots

- `ec2-dlm-snapshots.json` — `aws ec2 describe-snapshots` for the root volume's lifecycle snapshots, narrowed to their IDs, start times, states, and copied `project` tags.
- `ce-cost-allocation-tags.json` — `aws ce list-cost-allocation-tags` for `project`: its type and status after activation.
- `budgets-budget.json` — `aws budgets describe-budget`: the name, type, period, limit, and tag filter.
- `budgets-notifications.json` — `aws budgets describe-notifications-for-budget`: each alert's type and threshold, without its subscriber.
- `budgets-actions.json` — `aws budgets describe-budget-actions-for-budget`: the action's type, sub-type, approval model, threshold, status, and instance count, without its role ARN.
- `iam-simulate-budget-action-role.json` — `aws iam simulate-principal-policy` for the action's role, called through Systems Manager. Stopping this VM and starting the stop automation are allowed; stopping another instance and terminating this one are not.

## Req 5: `l4` and `l40s` capacity, and the `a10g` fallback

- `ec2-l4-capacity-failure.json` — the first attempt to start the stopped instance as a
  g6.xlarge, reduced to the date, size, instance type, `InsufficientInstanceCapacity` error code,
  and stopped state afterward.
- `ec2-l40s-fallback.json` — a 2026-09-22 check of the chosen zone's g6e.xlarge offering, public
  instance metadata, the G and VT quota value, and the Linux On-Demand hourly price. It contains no
  account-scoped identifier.
- `ec2-l40s-capacity-failure.json` — the first attempt to start the stopped instance as a
  g6e.xlarge, reduced to the date, size, instance type, `InsufficientInstanceCapacity` error code,
  and stopped state afterward.
- `ec2-a10g-fallback.json` — a 2026-09-22 check of the chosen zone's g5.xlarge offering, public
  instance metadata, the G and VT quota value, and the Linux On-Demand hourly price. It contains no
  account-scoped identifier.

## Sizes: `a10g`

- `size-switches.json` — after each size switch, `aws ec2 describe-instances` narrowed to the
  instance ID, type, and root volume ID, with the size and date. The first entry is the `dev`
  instance from `ec2-instance-dev.json`. Every entry has the same instance ID and the same root
  volume ID.
- `vm-a10g-checks.txt` — on `a10g`: the kernel; the GPU and driver from `nvidia-smi`; the device
  file; the GPU cap's scheduled poweroff; JAX's backend, device count, and `chain_method(4)`; the
  full test run's summary; and four hashes of one batched value and gradient, two with XLA's
  deterministic flag and two without.
- `../cloud-gpu-probe/a10g.json` — the engine probe on `a10g` at T=280, n=150, p=70, with batch
  sizes 1, 4, and 16.
