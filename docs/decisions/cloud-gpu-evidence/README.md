# Cloud GPU environment: account and zone evidence

Command outputs recorded by Plan 3, the first implementation plan for [`specs/cloud-gpu-environment.md`](../../../specs/cloud-gpu-environment.md), for its Reqs 1 and 2. The decision record `docs/decisions/cloud-gpu.md`, which Plan 4 writes, cites them. Every `aws` command ran as the IAM user through the `ces-revisions` profile.

No file here holds an account ID, an ARN, an email address, or a token. Outputs that carry one were narrowed with `--query` before they were written. Before each commit, this directory was searched for the account ID, for ARNs, and for at signs.

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
