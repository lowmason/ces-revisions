# Cloud GPU environment: account and zone evidence

Command outputs recorded by Plan 3, the first implementation plan for [`specs/cloud-gpu-environment.md`](../../../specs/cloud-gpu-environment.md), for its Reqs 1 and 2. The decision record `docs/decisions/cloud-gpu.md`, which Plan 4 writes, cites them. Every `aws` command ran as the IAM user through the `ces-revisions` profile.

No file here holds an account ID, an ARN, an email address, or a token. Outputs that carry one were narrowed with `--query` before they were written. Before each commit, this directory was searched for the account ID, for ARNs, and for at signs.

## Req 1: identity

- `iam-list-access-keys.json` — `aws iam list-access-keys`: the IAM user's access keys. Req 1 requires none.
- `iam-get-account-summary.json` — `aws iam get-account-summary`: account-wide counts and flags. `AccountMFAEnabled` is 1 when the root user has MFA, and `AccountAccessKeysPresent` is 0 when root has no access keys.
- `iam-list-mfa-devices.json` — `aws iam list-mfa-devices`, narrowed to `MFADeviceCount`: how many MFA devices the IAM user has, without their serial-number ARNs.
