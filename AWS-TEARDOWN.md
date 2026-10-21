# AWS and Lambda Cloud teardown

This guide removes every AWS and Lambda Cloud resource left by two abandoned cloud GPU efforts. They are the Plan 4 EC2 environment and the Plans 9-10 Lambda runner. The work stopped on 2026-10-20, because GPUs are now rented by hand from Lambda. Each deletion below is permanent unless its step says otherwise. Read-only steps change nothing. You run every step yourself in your own terminal. An agent must not run them.

The kit at `~/ces-revisions-aws-teardown/` holds copies of both worktrees (state, configuration, runbooks, evidence) and a git bundle of all five cloud branches. It contains private values and must never be committed. The committed code lives in that bundle and, for Plan 4, also on the remote branch `codex/cloud-gpu-environment`. The uncommitted g7e work is only in the kit's `env-worktree/uncommitted.patch` and copied files. The five local cloud branches and their two worktrees were deleted on 2026-10-20, after the kit was saved.

## What may still cost money

Plan 4 used OpenTofu, an open-source fork of Terraform. Plans 9-10 used one AWS CloudFormation stack.

| Resource | Why it may still exist | Rough cost driver |
| --- | --- | --- |
| Lambda Cloud instance | None is recorded. A hand launch cannot be ruled out. | GPU hourly rate, billed until you terminate it |
| Lambda Cloud filesystem | None is expected. | Per GiB-month while it exists |
| EC2 instance `ces-revisions-vm` | An apply after 2026-10-04 cannot be ruled out offline. | Running: \$3.36/hour (g7e.2xlarge) or \$6.88/hour (p5.4xlarge). Stopped: its disk. |
| EBS gp3 volume (EBS is the EC2 disk service, gp3 its general-purpose SSD type) | Only if an instance or a detached volume exists. | About \$0.08/GB-month, so about \$8/month for 100 GiB |
| DLM snapshots (DLM, Data Lifecycle Manager, took daily disk snapshots) | DLM made 4. They are not in OpenTofu state. | \$0.05/GB-month of stored blocks |
| Public IPv4 address | Only while an instance runs, or for an idle Elastic IP. | \$0.005/hour |
| OpenTofu state bucket | OpenTofu needs it until the end. | Negligible |
| `ArtifactBucket` and `LogBucket` | The stack retains them (`DeletionPolicy: Retain`). | S3 storage and request metrics |
| CloudTrail trail | Stack resource. | \$0.10 per 100,000 data events, more if it is a second management-event copy |
| Secret `/ces-revisions/lambda-cloud/api` | Made by hand, outside the stack. | \$0.40/month |
| Reaper log group and two alarms | Stack resources. | Small log storage, about \$0.10/alarm-month |

Budgets, IAM roles, the VPC, the security group, the schedule group, the SQS queue and the SNS topic cost nothing while idle.

## Decide first

Each item gives the default that this guide follows.

- DLM snapshots: delete. They may hold the VM's GitHub token file.
- `ArtifactBucket` and `LogBucket`: delete. This overrides the runner runbook's rule to never empty them. Download anything you want first.
- CloudTrail: the stack deletes trail `ces-revisions-cloud-control-artifacts`. If it is the account's only trail, S3 delivery of management events stops. The 90-day Event history stays.
- Reaper log group and Step Functions execution history: delete them with the stack. Phase 2 step 3 shows how to export them first.
- Lambda SSH key `ces-revisions`, `~/.ssh/lambda.pem` and the firewall rulesets: keep them for hand rentals.
- Secret recovery window: 7 days.
- The IAM user that you sign in with: keep it.
- Cost allocation tag `project`: leave it active. Deactivate it after the November 2026 bill, if you want.
- Open quota support cases: resolve them in the console, if you want.
- `~/.local/share/ces-revisions/` and `~/.local/bin/asm-exec`: keep them until you decide.

## Before you start

1. Revoke the VM's GitHub token now. This is irreversible. It does not depend on anything in AWS, so it can come before the inventory. The main checkout's `.project.env` holds it in plain text on the `ces-revisions-cloud-vm` line, and that file is world-readable.
   1. Run `chmod 600 ~/Projects/ces-revisions/.project.env` at once.
   2. On github.com, open Settings > Developer settings > Personal access tokens > Fine-grained tokens.
   3. Find the token limited to this repository with Contents and Pull requests write access, made in September 2026.
   4. Delete it, even if it shows as expired. If none is listed, it was already deleted or never created.
2. Make a private scratch copy of the OpenTofu roots. Never edit the kit itself. Phase 3 needs OpenTofu 1.13 or later and network access to registry.opentofu.org.
   ```bash
   tofu version                                       # expect v1.13.x
   W=~/ces-revisions-teardown-work
   mkdir -p "$W" && chmod 700 "$W"
   cp -Rp ~/ces-revisions-aws-teardown/env-worktree/infra "$W"/
   cd "$W"
   ```

3. Sign in as the project IAM user. The `ces-revisions` profile is missing from `~/.aws/config`, and `aws login` creates it again. Enter `us-east-1` at the region prompt. Choose the IAM user in the browser, not root. Do not use any other profile or an agent's AWS connector. They reach a different account, so their reads look falsely clean.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html
   aws login --profile ces-revisions
   export AWS_PROFILE=ces-revisions AWS_REGION=us-east-1 AWS_PAGER=''
   REGION=us-east-1
   unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
   ```

4. Pass the account gate. It prints no identifier. In every new terminal, rerun its lines above the first `First run only` comment. From the second run on, they must print `project account`. On the first run, the three existence checks must succeed before you save the account. A 403 from `head-bucket` means the wrong account. Run nothing else until the gate passes.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/s3api/head-bucket.html
   ARN=$(aws sts get-caller-identity --query Arn --output text)
   case "$ARN" in *:user/*) echo 'IAM user session' ;; *) echo 'STOP: not an IAM user session' ;; esac
   unset ARN
   ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)   # never print or paste it
   BUCKET=$(sed -n 's/^bucket *= *"\(.*\)"$/\1/p' "$W/infra/env/backend.hcl")
   if [ -f "$W/account.env" ]; then
     . "$W/account.env"
     [ "$ACCOUNT_ID" = "$EXPECTED_ACCOUNT" ] && echo 'project account' || echo 'STOP: wrong account'
   fi
   # First run only, before Phase 2 and Phase 3:
   aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null && echo 'state bucket found'
   aws iam get-role --role-name ces-revisions-vm --query Role.RoleName --output text
   aws cloudformation describe-stacks --stack-name ces-revisions-cloud-control --query 'Stacks[0].StackStatus' --output text
   # First run only, after all three checks pass:
   printf 'EXPECTED_ACCOUNT=%s\n' "$ACCOUNT_ID" > "$W/account.env" && chmod 600 "$W/account.env"
   ```

   Later phases delete the bucket, the role and the stack. After that, only `account.env` proves that a new session reached the project account. A session in another account returns empty reads that look falsely clean.

5. Run the read-only inventory and keep its output. The file holds identifiers, so it stays in the scratch directory.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/resourcegroupstaggingapi/get-resources.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instances.html
   umask 077
   {
     for r in us-east-1 us-east-2 us-west-2; do
       echo "== $r: tagged resources, then all instances in every state"
       aws resourcegroupstaggingapi get-resources --region "$r" --tag-filters Key=project,Values=ces-revisions \
         --query 'ResourceTagMappingList[].ResourceARN' --output text
       aws ec2 describe-instances --region "$r" --output text \
         --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
     done
     echo "== volumes, snapshots, Recycle Bin rules"
     aws ec2 describe-volumes --filters Name=tag:project,Values=ces-revisions --query 'Volumes[].[VolumeId,State,Size]' --output text
     aws ec2 describe-snapshots --owner-ids self --filters Name=tag:project,Values=ces-revisions \
       --query 'Snapshots[].[SnapshotId,StartTime,State]' --output text
     aws rbin list-rules --resource-type EBS_SNAPSHOT --query 'Rules[].Identifier' --output text
     aws rbin list-rules --resource-type EBS_VOLUME --query 'Rules[].Identifier' --output text
     echo "== stacks, schedules, budgets, secrets, buckets"
     aws cloudformation describe-stacks --stack-name ces-revisions-cloud-control \
       --query 'Stacks[0].{Status:StackStatus,TP:EnableTerminationProtection}'
     aws scheduler list-schedules --name-prefix ces-revisions- --query 'Schedules[].[GroupName,Name,State]' --output text
     aws budgets describe-budgets --account-id "$ACCOUNT_ID" --query 'Budgets[].BudgetName' --output text
     aws secretsmanager list-secrets --include-planned-deletion --filters Key=name,Values=/ces-revisions \
       --query 'SecretList[].[Name,DeletedDate]' --output text
     aws s3api list-buckets --prefix ces-revisions --query 'Buckets[].Name' --output text
   } 2>&1 | tee "$W/inventory-before.txt"
   ```

6. Stop and decide if the inventory shows any of these:
   - An instance that is pending, running, stopping or stopped and belongs to this project. Copy off what you need through Session Manager (the AWS Systems Manager remote shell), then follow Phase 3 step 2.
   - A schedule, which means a lease is armed. Follow Phase 2 step 1.
   - A Recycle Bin rule. Deleted snapshots or volumes then stay in the bin and bill until its retention ends.
   - A project volume in state `available`, or an Elastic IP. Follow Phase 3 step 5.

Most commands read identifiers into shell variables. The step 5 inventory and Phase 2 steps 1 to 3 print identifiers to your terminal. Never paste that output into a chat, an issue or a commit. A new terminal loses the variables, so rerun the line that sets each one:

- `$W` and `$REGION`: steps 2 and 3. `$ACCOUNT_ID` and `$BUCKET` (from `infra/env/backend.hcl`): step 4.
- `$STACK`, `$STACK_ID`, `$ARTIFACT_BUCKET`, `$LOG_BUCKET` and `$REAPER`: Phase 2, step 2, which saves them in `$W/ids.env`.
- In a new terminal, run `W=~/ces-revisions-teardown-work` and `cd "$W"`, the `export` and `REGION=` lines of step 3, and the step 4 lines above `First run only`. Then run `. "$W/ids.env"` and `. "$W/empty-bucket.zsh"` once those files exist.
- `<INSTANCE_ID>`, `<SCHEDULE_NAME>` and `<EXECUTION_ARN>`: the inventory or the preflight output. `<INSTANCE_REGION>`: the `== <region>` header above the instance in the inventory. `<LOCK_ID>`: the OpenTofu lock error.
- `<VOLUME_ID>`, `<VOLUME_REGION>`, `<ALLOCATION_ID>` and `<ADDRESS_REGION>`: the inventory or the Phase 6 sweep.
- `<VPC_ID>`: the `VPC_ID=` line of the Phase 3 fallback. `<POLICY_NAME>`: Phase 2, step 10. `<BRANCH>`: `git branch --list`. `<BUNDLE_PATH>`: where you keep `cloud-branches.bundle`.

## Phase 1: Lambda Cloud

1. Sign in to the Lambda Cloud console. The API returned `403 account-inactive` on 2026-10-07. Console access was not tested since then (unverified: check that the Instances page loads).
2. Open the Instances page. Confirm that it is empty. Termination is irreversible and erases the instance's disk. For any instance, select it, click Terminate, type `erase data on instance` and click Terminate instances. Do not shut it down from inside the OS, because billing continues ([docs](https://docs.lambda.ai/public-cloud/console/), [billing](https://docs.lambda.ai/public-cloud/billing/)).
3. Reload the page and confirm that the instance is gone. Then open the Usage page and note the current charges.
4. Open the Storage page. Confirm that no filesystem exists. Deleting a filesystem is irreversible. Delete any filesystem from its Actions column, after you terminate the instances that use it ([docs](https://docs.lambda.ai/public-cloud/filesystems/)).
5. Do not delete the API key yet. The AWS reaper uses it, so Phase 2 deletes it after the stack.
6. Keep the SSH key `ces-revisions` and the firewall rulesets by default. To delete them, use SSH keys > Actions > Delete and Firewall > Rulesets > menu > Delete. The ruleset IDs per region are in the `LAMBDA_CLOUD_FIREWALL_RULESETS_JSON` key of `~/ces-revisions-aws-teardown/runner-worktree/.project.env`. Read them privately. Review the Global rules tab too (unverified: whether a ruleset attached to a running instance can be deleted, so terminate first). See the [firewall docs](https://docs.lambda.ai/public-cloud/firewalls/).

## Phase 2: Plans 9-10 CloudFormation stack

The stack `ces-revisions-cloud-control` holds 18 resources in `us-east-1`. Delete-stack removes 16 of them. It skips the two buckets.

1. Run the preflight. Expect no schedules, 0 running executions, 0 messages in the DLQ (dead-letter queue) and alarms in `OK` or `INSUFFICIENT_DATA`. A schedule or a message means a lease is live. Read its target privately with `aws scheduler get-schedule --group-name ces-revisions-gpu-leases --name <SCHEDULE_NAME> --query Target.Input`, then terminate that instance in the Lambda console. A running execution is a termination attempt. Let it finish, or stop it with `aws stepfunctions stop-execution --execution-arn <EXECUTION_ARN>`.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/scheduler/list-schedules.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/stepfunctions/stop-execution.html
   STACK=ces-revisions-cloud-control
   aws scheduler list-schedules --name-prefix ces-revisions- --query 'Schedules[].[GroupName,Name,State]' --output text
   REAPER=$(aws cloudformation describe-stacks --stack-name "$STACK" \
     --query "Stacks[0].Outputs[?OutputKey=='ReaperStateMachineArn'].OutputValue | [0]" --output text)
   aws stepfunctions list-executions --state-machine-arn "$REAPER" --status-filter RUNNING --query 'length(executions)'
   DLQ_URL=$(aws cloudformation describe-stack-resource --stack-name "$STACK" \
     --logical-resource-id SchedulerDeadLetterQueue --query StackResourceDetail.PhysicalResourceId --output text)
   aws sqs get-queue-attributes --queue-url "$DLQ_URL" --query Attributes \
     --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible
   aws cloudwatch describe-alarms --alarm-name-prefix "$STACK" --query 'MetricAlarms[].StateValue' --output text
   ```

2. Capture the identifiers. The stack name stops resolving after deletion, and `LogBucket` has no stack output.
   ```bash
   # docs: https://docs.aws.amazon.com/AWSCloudFormation/latest/APIReference/API_DescribeStackResources.html
   STACK_ID=$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackId' --output text)
   ARTIFACT_BUCKET=$(aws cloudformation describe-stack-resource --stack-name "$STACK" \
     --logical-resource-id ArtifactBucket --query StackResourceDetail.PhysicalResourceId --output text)
   LOG_BUCKET=$(aws cloudformation describe-stack-resource --stack-name "$STACK" \
     --logical-resource-id LogBucket --query StackResourceDetail.PhysicalResourceId --output text)
   test -n "$STACK_ID" && test -n "$ARTIFACT_BUCKET" && test -n "$LOG_BUCKET" && test -n "$REAPER" && echo 'captured'
   printf 'STACK=%s\nSTACK_ID=%s\nARTIFACT_BUCKET=%s\nLOG_BUCKET=%s\nREAPER=%s\n' \
     "$STACK" "$STACK_ID" "$ARTIFACT_BUCKET" "$LOG_BUCKET" "$REAPER" > "$W/ids.env"
   chmod 600 "$W/ids.env"
   ```
   If you lose `ids.env`, `aws s3api list-buckets --prefix ces-revisions-cloud-control --query 'Buckets[].Name' --output text` lists both buckets. Their generated names start with the stack name.

3. Optional: copy anything you want to keep to a private directory outside the scratch copy, because Phase 7 deletes the scratch copy. The artifact bucket should hold one 270-byte proof object. The log bucket holds S3 access logs and the trail's CloudTrail logs. These may be the account's only S3 copy of CloudTrail management events. The reaper log group and the state machine's execution history go with the stack. The copies hold identifiers, so keep them private.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/logs/filter-log-events.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/stepfunctions/list-executions.html
   KEEP=~/ces-revisions-teardown-keep
   mkdir -p "$KEEP" && chmod 700 "$KEEP"
   aws s3 sync "s3://$ARTIFACT_BUCKET" "$KEEP/artifact-copy"
   aws s3 sync "s3://$LOG_BUCKET" "$KEEP/log-copy"
   aws logs filter-log-events --log-group-name "/aws/vendedlogs/states/$STACK-reaper" --output json > "$KEEP/reaper-logs.json"
   aws stepfunctions list-executions --state-machine-arn "$REAPER" --output json > "$KEEP/reaper-executions.json"
   ```

4. Delete the stack. This is irreversible, and once `delete-stack` starts it cannot be stopped. The first command turns termination protection off, and it is harmless if protection is already off.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/cloudformation/delete-stack.html
   # docs: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-protect-stacks.html
   aws cloudformation update-termination-protection --stack-name "$STACK" --no-enable-termination-protection
   aws cloudformation delete-stack --stack-name "$STACK"
   aws cloudformation wait stack-delete-complete --stack-name "$STACK_ID"
   aws cloudformation describe-stack-resources --stack-name "$STACK_ID" --output table \
     --query 'StackResources[?ResourceStatus!=`DELETE_COMPLETE`].[LogicalResourceId,ResourceStatus]'
   ```

   Underneath, CloudFormation deletes the trail, both bucket policies and the schedule group with any schedules in it. It also deletes the EventBridge connection, the reaper, the roles, the DLQ, the SNS topic, the alarms, the log group and the budget. Check: only `ArtifactBucket` and `LogBucket` remain, both `DELETE_SKIPPED`. The waiter gives up after 1 hour with exit code 255. Then sign in again if needed and rerun it. On `DELETE_FAILED`, read the reason with `aws cloudformation describe-stack-events --stack-name "$STACK_ID"`, fix it, and run `delete-stack` again.

5. Check that the stack's resources are gone. The connection had a managed secret with the `events!connection/` name prefix. Expect no entry for this stack's connection, or one with a `DeletedDate` (unverified: the docs do not say that DeleteConnection deletes that secret). If one remains without a `DeletedDate`, do not edit it. Ask AWS Support.
   ```bash
   # docs: https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-target-connection.html
   aws secretsmanager list-secrets --include-planned-deletion --filters Key=owning-service,Values=events \
     --query 'SecretList[].[Name,DeletedDate]' --output text
   aws scheduler list-schedule-groups --name-prefix ces-revisions --query 'ScheduleGroups[].[Name,State]' --output text
   aws cloudtrail describe-trails --trail-name-list ces-revisions-cloud-control-artifacts --query 'trailList[].Name'
   aws logs describe-log-groups --log-group-name-prefix /aws/vendedlogs/states/ces-revisions --query 'logGroups[].logGroupName'
   ```

6. Delete the project's API key in the Lambda console. This is irreversible: Cloud API keys > the key's row > Delete > Delete API key. Identify it from your own records. It was made around 2026-10-04. If you need a key later, make a new one ([docs](https://docs.lambda.ai/public-cloud/console/)).
7. Delete the hand-made secret. A 7-day recovery window lets you run `restore-secret` until the deletion date, and AWS does not bill a secret scheduled for deletion. `--force-delete-without-recovery` deletes it at once and cannot be undone. If `describe-secret` shows replicas, remove them first. A customer managed `KmsKeyId` needs its own decision.
   ```bash
   # docs: https://docs.aws.amazon.com/secretsmanager/latest/userguide/manage_delete-secret.html
   aws secretsmanager describe-secret --secret-id /ces-revisions/lambda-cloud/api \
     --query '{Name:Name,Kms:KmsKeyId,Deleted:DeletedDate,Replicas:ReplicationStatus}'
   aws secretsmanager delete-secret --secret-id /ces-revisions/lambda-cloud/api --recovery-window-in-days 7
   ```

8. Empty and delete the artifact bucket. This is irreversible. Both buckets are versioned, so every old version and delete marker must go. `aws s3 rm --recursive` only adds delete markers, so do not use it. The function below deletes at most 1,000 entries per call, the `DeleteObjects` limit. It is saved to `$W/empty-bucket.zsh`, so a new terminal can load it again for step 9. The S3 console's Empty button is an equal alternative.
   ```bash
   # docs: https://docs.aws.amazon.com/AmazonS3/latest/userguide/DeletingObjectVersions.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-objects.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/s3api/put-bucket-logging.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-bucket-metrics-configuration.html
   cat > "$W/empty-bucket.zsh" <<'EOF'
   empty_versioned_bucket() {
     b="$1"; tmp=$(mktemp) || return 1
     while :; do
       objs=$(aws s3api list-object-versions --bucket "$b" --no-paginate --output json \
         --query '[Versions[].{Key: Key, VersionId: VersionId} || `[]`, DeleteMarkers[].{Key: Key, VersionId: VersionId} || `[]`][] | [:1000]') \
         || { rm -f "$tmp"; return 1; }
       [ "$objs" = "[]" ] && break
       printf '{"Objects": %s, "Quiet": true}' "$objs" > "$tmp"
       errs=$(aws s3api delete-objects --bucket "$b" --delete "file://$tmp" --query Errors --output json) \
         || { rm -f "$tmp"; return 1; }
       case "$errs" in null|"[]") ;; *) echo 'delete-objects reported errors' >&2; rm -f "$tmp"; return 1 ;; esac
     done
     rm -f "$tmp"
   }
   EOF
   . "$W/empty-bucket.zsh"
   aws s3api put-bucket-logging --bucket "$ARTIFACT_BUCKET" --bucket-logging-status '{}'
   aws s3api delete-bucket-metrics-configuration --bucket "$ARTIFACT_BUCKET" --id EntireBucket
   empty_versioned_bucket "$ARTIFACT_BUCKET"
   aws s3api list-multipart-uploads --bucket "$ARTIFACT_BUCKET" --query 'Uploads[].[Key,UploadId]'
   aws s3api delete-bucket --bucket "$ARTIFACT_BUCKET"
   aws s3api head-bucket --bucket "$ARTIFACT_BUCKET"   # expect a 404 error
   ```

   Check: abort any listed multipart upload with `aws s3api abort-multipart-upload` before `delete-bucket` ([docs](https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html)). AWS cannot restore a deleted bucket, and another account may later claim its name.

9. Empty and delete the log bucket. This is irreversible. First wait at least 1 hour, ideally a few hours, because S3 delivers access logs late, on a best-effort basis. In a new terminal, first run the new-terminal lines from Before you start, which load the function. Then run `empty_versioned_bucket "$LOG_BUCKET"`, the multipart check and `aws s3api delete-bucket --bucket "$LOG_BUCKET"`. On `BucketNotEmpty`, a late log arrived. Check that `type empty_versioned_bucket` reports a function, then empty the bucket again and retry.
10. Optional: check for an account-level CloudWatch Logs resource policy. The first stack version logged at level ALL and may have made one (unverified). List the policies with `aws logs describe-resource-policies --query 'resourcePolicies[].policyName'`, and read each document privately. Run `aws logs delete-resource-policy --policy-name <POLICY_NAME>` only for a policy that plainly serves `/aws/vendedlogs/states/*` and nothing else ([docs](https://docs.aws.amazon.com/cli/latest/reference/logs/delete-resource-policy.html)).

## Phase 3: Plan 4 OpenTofu environment

The preferred path runs `tofu destroy` from the scratch copy. OpenTofu knows every identifier and the delete order. It cannot delete the DLM snapshots, so step 4 does that by hand. Keep `~/.ssh/ces-revisions-vm.pub`, `~/.config/ces-revisions/infra/` and OpenTofu until this phase is verified. `instance.tf` reads the public key with `file()`, and the backups are your fallback.

1. Initialize both roots and record the state. `init` needs network access to download provider `hashicorp/aws` 6.65.0.
   ```bash
   # docs: https://opentofu.org/docs/cli/commands/init/
   # docs: https://opentofu.org/docs/language/settings/backends/s3/
   cd "$W"
   tofu -chdir=infra/state init -lockfile=readonly
   tofu -chdir=infra/state plan                       # expect: No changes
   tofu -chdir=infra/env init -backend-config=backend.hcl -lockfile=readonly
   tofu -chdir=infra/env state list                   # expect 17 resources and one data source, no aws_instance.vm
   (umask 077; tofu -chdir=infra/env state pull > "$W/env-state-before-teardown.json")
   if n=$(tofu -chdir=infra/env state list | wc -l | tr -d ' ') && [ "$n" -gt 0 ] \
       && [ -s "$W/env-state-before-teardown.json" ]; then
     echo "env state saved: $n addresses"
   else
     echo 'STOP: no env state. Run nothing else in this phase. Read the note below.'
   fi
   ```

   The shell creates the output file before `state pull` runs, so a failed pull leaves an empty file and looks like success. The check above catches that. If it prints `STOP`, delete nothing in this phase, and check the state bucket with `aws s3api head-bucket --bucket "$BUCKET" --region "$REGION"`:
   - A 404 means the state bucket and the env state are gone. Skip steps 3 and 6. Use the manual fallback, then steps 4 and 5.
   - Any other error, such as expired credentials or no network, can be fixed. Fix it and rerun step 1.

2. If `aws_instance.vm` is in state, `tofu destroy` terminates it in step 3. If a project instance exists outside state, in any region, terminate it by hand. Termination is irreversible, and it also deletes the root volume.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/terminate-instances.html
   aws ec2 terminate-instances --region <INSTANCE_REGION> --instance-ids <INSTANCE_ID>
   aws ec2 wait instance-terminated --region <INSTANCE_REGION> --instance-ids <INSTANCE_ID>
   ```

3. Destroy the environment. This is irreversible. Read the plan first. Every line must be a destroy of an expected address.
   ```bash
   # docs: https://opentofu.org/docs/cli/commands/destroy/
   # docs: https://opentofu.org/docs/language/resources/behavior/
   sed -i '' 's/prevent_destroy = true/prevent_destroy = false/' infra/env/instance.tf
   tofu -chdir=infra/env plan -destroy
   tofu -chdir=infra/env destroy                      # answer yes
   tofu -chdir=infra/env state list                   # expect no output
   ```

   Underneath, OpenTofu deletes the budget action, budget `ces-revisions-monthly`, the DLM policy and the instance profile. It deletes the IAM roles `ces-revisions-vm`, `ces-revisions-snapshots` and `ces-revisions-budget-stop`. Then it deletes the security group, route table, subnet, internet gateway and VPC. Failure handling:
   - A lock error: confirm that no other run is active, then run `tofu -chdir=infra/env force-unlock <LOCK_ID>` ([docs](https://opentofu.org/docs/cli/commands/force-unlock/)).
   - `DependencyViolation`: an ENI (network interface) or instance is still in the VPC. Find it with `aws ec2 describe-network-interfaces --filters Name=vpc-id,Values=<VPC_ID>`, remove it and rerun destroy.
   - Expired credentials: run `aws login --profile ces-revisions` again and rerun. Never pass `-refresh=false`.

4. Delete the DLM snapshots. Deletion is permanent. `delete-snapshot` fails if an AMI (machine image) uses the snapshot, and the project made none.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-snapshot.html
   # docs: https://docs.aws.amazon.com/ebs/latest/userguide/recycle-bin.html
   aws ec2 describe-images --region "$REGION" --owners self --query 'Images[].ImageId'   # expect []
   for s in $(aws ec2 describe-snapshots --region "$REGION" --owner-ids self \
       --filters Name=tag:project,Values=ces-revisions Name=tag-key,Values=aws:dlm:lifecycle-policy-id \
       --query 'Snapshots[].SnapshotId' --output text); do
     aws ec2 delete-snapshot --region "$REGION" --snapshot-id "$s"
   done
   aws ec2 describe-snapshots --region "$REGION" --owner-ids self \
     --filters Name=tag:project,Values=ces-revisions --query 'length(Snapshots)'          # expect 0
   ```

5. Delete any detached project volume or idle Elastic IP that the inventory or the Phase 6 sweep shows. This is irreversible. Confirm that each one is this project's first. A volume in state `in-use` belongs to an instance, so terminate the instance first. With a Recycle Bin rule for `EBS_VOLUME`, a deleted volume stays in the bin and bills until its retention ends.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-volume.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/wait/volume-deleted.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/release-address.html
   aws ec2 delete-volume --region <VOLUME_REGION> --volume-id <VOLUME_ID>
   aws ec2 wait volume-deleted --region <VOLUME_REGION> --volume-ids <VOLUME_ID>
   aws ec2 release-address --region <ADDRESS_REGION> --allocation-id <ALLOCATION_ID>
   ```

6. Delete the state bucket last in this phase, and only after step 3's destroy succeeded. This is irreversible. The bucket holds the env state, so deleting it first loses that state for good. Run the guard line first. Continue only if it prints `env destroyed`. `force_destroy` acts only after an apply writes it to state, so the order is edit, apply, then destroy.
   ```bash
   # docs: https://raw.githubusercontent.com/hashicorp/terraform-provider-aws/main/website/docs/r/s3_bucket.html.markdown
   if out=$(tofu -chdir=infra/env state list) && [ -z "$out" ]; then echo 'env destroyed'; else echo 'STOP: env state still lists resources or cannot be read. Keep the bucket.'; fi
   perl -0pi -e 's/prevent_destroy = true/prevent_destroy = false/; s/(bucket_prefix = "ces-revisions-tofu-state-"\n)/$1  force_destroy = true\n/' infra/state/main.tf
   grep -n -E 'prevent_destroy|force_destroy' infra/state/main.tf   # expect false and true
   tofu -chdir=infra/state plan                       # expect 1 to change: force_destroy
   tofu -chdir=infra/state apply
   tofu -chdir=infra/state destroy                    # expect 4 to destroy
   aws s3api head-bucket --bucket "$BUCKET" --region "$REGION"       # expect a 404 error
   ```

   Underneath, the provider deletes every object version and delete marker, then the bucket. The destroy rewrites the scratch copy's local state. The kit keeps the copy from before the teardown.

### Manual fallback

Use it only in one of these cases. For one blocker, fix it by hand and rerun `tofu destroy` instead.

- Step 1 printed `STOP` because the state bucket is gone.
- `init` or the state read fails for another reason.
- Destroy keeps failing.

```bash
# docs: https://docs.aws.amazon.com/cli/latest/reference/budgets/delete-budget-action.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/budgets/delete-budget.html
# docs: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_manage_delete.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/dlm/delete-lifecycle-policy.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/iam/remove-role-from-instance-profile.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/iam/delete-instance-profile.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-security-group.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/disassociate-route-table.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-route-table.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-subnet.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/detach-internet-gateway.html
# docs: https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-internet-gateway.html
# docs: https://docs.aws.amazon.com/vpc/latest/userguide/delete-vpc.html
B=ces-revisions-monthly
ACTION_ID=$(aws budgets describe-budget-actions-for-budget --account-id "$ACCOUNT_ID" --budget-name "$B" --query 'Actions[0].ActionId' --output text)
aws budgets delete-budget-action --account-id "$ACCOUNT_ID" --budget-name "$B" --action-id "$ACTION_ID"
aws budgets delete-budget --account-id "$ACCOUNT_ID" --budget-name "$B"
aws iam delete-role-policy --role-name ces-revisions-budget-stop --policy-name stop-the-vm
aws iam delete-role --role-name ces-revisions-budget-stop
aws dlm delete-lifecycle-policy --policy-id "$(aws dlm get-lifecycle-policies --resource-types VOLUME --target-tags project=ces-revisions --query 'Policies[0].PolicyId' --output text)"
aws iam detach-role-policy --role-name ces-revisions-snapshots --policy-arn arn:aws:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole
aws iam delete-role --role-name ces-revisions-snapshots
aws iam remove-role-from-instance-profile --instance-profile-name ces-revisions-vm --role-name ces-revisions-vm
aws iam delete-instance-profile --instance-profile-name ces-revisions-vm
aws iam detach-role-policy --role-name ces-revisions-vm --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam delete-role --role-name ces-revisions-vm
VPC_ID=$(aws ec2 describe-vpcs --filters Name=tag:Name,Values=ces-revisions Name=cidr,Values=10.42.0.0/16 --query 'Vpcs[].VpcId' --output text)
aws ec2 delete-security-group --group-id "$(aws ec2 describe-security-groups --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values=ces-revisions-vm --query 'SecurityGroups[0].GroupId' --output text)"
RTB_ID=$(aws ec2 describe-route-tables --filters Name=vpc-id,Values="$VPC_ID" Name=tag:Name,Values=ces-revisions-public --query 'RouteTables[0].RouteTableId' --output text)
for a in $(aws ec2 describe-route-tables --route-table-ids "$RTB_ID" --query 'RouteTables[].Associations[?Main==`false`].RouteTableAssociationId' --output text); do aws ec2 disassociate-route-table --association-id "$a"; done
aws ec2 delete-route-table --route-table-id "$RTB_ID"
for s in $(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC_ID" --query 'Subnets[].SubnetId' --output text); do aws ec2 delete-subnet --subnet-id "$s"; done
IGW_ID=$(aws ec2 describe-internet-gateways --filters Name=attachment.vpc-id,Values="$VPC_ID" --query 'InternetGateways[0].InternetGatewayId' --output text)
aws ec2 detach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID"
aws ec2 delete-internet-gateway --internet-gateway-id "$IGW_ID"
aws ec2 delete-vpc --vpc-id "$VPC_ID"
```

After the block:

1. Delete the snapshots as in step 4.
2. Delete the state bucket only after every command in the block succeeded, ending with `delete-vpc`. If any command failed, keep the bucket, fix the failure and rerun from that command. Deleting the bucket is irreversible. Empty it with the S3 console's Empty button, then run `aws s3api delete-bucket --bucket "$BUCKET"`. If `head-bucket` already returns a 404, skip this step.

## Phase 4: Account-level items

1. Check that no project budget remains with `aws budgets describe-budgets --account-id "$ACCOUNT_ID" --query 'Budgets[].BudgetName' --output text` ([docs](https://docs.aws.amazon.com/cli/latest/reference/budgets/describe-budgets.html)). Expect neither `ces-revisions-monthly` nor the stack's budget. If one remains, wait until no action is in `EXECUTION_IN_PROGRESS`. Then delete its actions and the budget with the fallback commands above.
2. Record the quota requests. Service Quotas has no cancel operation, so there is nothing to delete. A raised quota costs nothing.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/service-quotas/list-requested-service-quota-change-history.html
   for r in us-east-1 us-east-2 us-west-2; do
     aws service-quotas list-requested-service-quota-change-history --region "$r" --service-code ec2 \
       --query 'RequestedQuotas[].[QuotaName,DesiredValue,Status]' --output text
   done
   ```

3. Optional: resolve the open quota cases in the Support Center console. Open each case, reply that it is no longer needed, then click Resolve case ([docs](https://docs.aws.amazon.com/awssupport/latest/user/monitoring-your-case.html)). On the Basic support plan, the Support API is not available.
4. Leave the service-linked roles that AWS made. They are free. Keep root MFA.

## Phase 5: The Mac and GitHub

1. Edit the main checkout's `.project.env`. Do not delete it, because `scripts/annual_sources.py fetch` loads `BLS_CONTACT_EMAIL` from it.
   1. Delete the `ces-revisions-cloud-vm` line, but only after the token revoke in Before you start.
   2. Delete the six runner lines: `AWS_PROFILE`, `AWS_REGION` and the four `LAMBDA_CLOUD_*` keys.
   3. The file has no `BLS_CONTACT_EMAIL` line now. Type `BLS_CONTACT_EMAIL=<your address>` on a new line yourself, and end the file with a newline. `scripts/vintage_sources.py` and `scripts/archive_inventory.py` read only the shell environment, so also export it in your shell before you run them.
   4. Check that the mode is still 600 with `ls -l ~/Projects/ces-revisions/.project.env`. Check key names only with `sed -E 's/=.*$/=/' ~/Projects/ces-revisions/.project.env`.
2. Remove the VM's SSH entry and key pair after Phase 3 is verified. Keep `Host lambda`, `~/.ssh/lambda.pem` and every other entry.
   ```bash
   # docs: https://man.openbsd.org/ssh-keygen
   cp -p ~/.ssh/config ~/.ssh/config.bak-teardown
   VM_HOST=$(awk '/^Host ces-revisions-vm$/{f=1;next} /^Host /{f=0} f && $1=="HostName"{print $2}' ~/.ssh/config)
   ssh-keygen -R "$VM_HOST"
   # In an editor, delete the "Host ces-revisions-vm" block and its comment line.
   rm ~/.ssh/ces-revisions-vm ~/.ssh/ces-revisions-vm.pub
   grep -n '^Host ' ~/.ssh/config
   ```

   Also remove any saved Claude Code desktop SSH connection to `ces-revisions-vm`.

3. After Phase 3 is verified, delete the state backups with `rm -rf ~/.config/ces-revisions`. They hold the bucket name and the budget email.
4. Optional: delete `~/.local/share/ces-revisions/` (about 703 MB) and `~/.local/bin/asm-exec`. `plan9-final-audit/` is the only copy of the Plan 9 final materials. Other projects may use `asm-exec`.
5. Optional: remove tools that only this work used, with `brew uninstall --cask session-manager-plugin` and `brew uninstall opentofu cfn-lint cloudformation-guard` ([docs](https://docs.brew.sh/Manpage)). The cask asks for your password. Keep `awscli` and `jq`. Other projects may use `cfn-lint` and `cloudformation-guard`.
6. Optional: delete the remote copy of the Plan 4 branch. The five local cloud branches and their two worktrees are already gone. Verify the bundle first. A deleted remote branch closes any pull request that uses it as its head. After this step the bundle is the only copy of the code.
   ```bash
   # docs: https://git-scm.com/docs/git-bundle
   # docs: https://git-scm.com/docs/git-push
   # docs: https://git-scm.com/docs/git-branch
   cd ~/Projects/ces-revisions
   git bundle verify ~/ces-revisions-aws-teardown/cloud-branches.bundle
   git branch --list '*cloud*' '*lambda*' '*plan-9*' '*plan-10*'   # expect no output
   gh pr list --head codex/cloud-gpu-environment --state all
   git push origin --delete codex/cloud-gpu-environment
   git fetch --prune origin
   # To restore a branch: git fetch <BUNDLE_PATH> refs/heads/<BRANCH>:refs/heads/<BRANCH>
   ```

7. Optional: delete `/tmp/ces-bundle-inspect`, the `/private/tmp/ces-*` files and the empty cloud-named shells in `~/.codex/worktrees/`.

## Phase 6: Final verification sweep

1. Sweep every region that the work touched. First rerun the step 4 gate and confirm that it prints `project account`. Expect 0 everywhere, unless other EC2 work runs in the account. For a nonzero count outside `us-east-1`, run the matching Phase 3 command with that region. The Tagging API returns only resources that still carry the tag, and it can lag briefly. `ca-west-1` is not opted in, so skip it.
   ```bash
   # docs: https://docs.aws.amazon.com/resourcegroupstagging/latest/APIReference/API_GetResources.html
   for r in us-east-1 us-east-2 us-west-2 us-west-1 ca-central-1 eu-west-2; do
     n=$(aws resourcegroupstaggingapi get-resources --region "$r" --tag-filters Key=project,Values=ces-revisions \
       --query 'length(ResourceTagMappingList)' --output text)
     i=$(aws ec2 describe-instances --region "$r" --filters Name=instance-state-name,Values=pending,running,stopping,stopped \
       --query 'length(Reservations[].Instances[])' --output text)
     v=$(aws ec2 describe-volumes --region "$r" --query 'length(Volumes)' --output text)
     s=$(aws ec2 describe-snapshots --region "$r" --owner-ids self --query 'length(Snapshots)' --output text)
     a=$(aws ec2 describe-addresses --region "$r" --query 'length(Addresses)' --output text)
     echo "$r tagged=$n instances=$i volumes=$v snapshots=$s addresses=$a"
   done
   ```

2. Run the direct checks in `us-east-1`. The Tagging API may not list budgets or IAM roles (unverified: these direct checks are the authority). Expect empty output or 0. A secret in its recovery window still appears with a `DeletedDate`, and that is expected.
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/reference/secretsmanager/list-secrets.html
   # docs: https://docs.aws.amazon.com/cli/latest/reference/s3api/list-buckets.html
   aws budgets describe-budgets --account-id "$ACCOUNT_ID" --query 'Budgets[].BudgetName' --output text
   aws iam list-roles --query "Roles[?contains(RoleName,'ces-revisions')].RoleName" --output text
   aws iam list-instance-profiles --query "InstanceProfiles[?contains(InstanceProfileName,'ces-revisions')].InstanceProfileName" --output text
   aws s3api list-buckets --prefix ces-revisions --query 'length(Buckets)'
   aws secretsmanager list-secrets --include-planned-deletion --filters Key=name,Values=/ces-revisions \
     --query 'SecretList[].[Name,DeletedDate]' --output text
   aws cloudformation list-stacks --output text \
     --query "StackSummaries[?starts_with(StackName,'ces-revisions') && StackStatus!='DELETE_COMPLETE'].[StackName,StackStatus]"
   aws scheduler list-schedule-groups --name-prefix ces-revisions --query 'length(ScheduleGroups)'
   aws logs describe-log-groups --log-group-name-prefix /aws/vendedlogs/states/ces-revisions --query 'length(logGroups)'
   aws dlm get-lifecycle-policies --query 'length(Policies)'
   aws ec2 describe-vpcs --filters Name=cidr,Values=10.42.0.0/16 --query 'length(Vpcs)'
   aws ec2 describe-capacity-reservations \
     --query "length(CapacityReservations[?State=='active' || State=='scheduled' || State=='pending'])"
   ```

3. Open EC2 Global View in the console. Confirm that no project instance, volume, VPC or Elastic IP exists in any enabled region.
4. In the Lambda console, confirm that the Instances page is empty and the Usage page shows no new charges.
5. One to three days later, open Billing and Cost Management > Bills > October 2026 > Charges by service. Confirm that the EC2, S3, Secrets Manager, CloudTrail, CloudWatch and Step Functions lines stopped growing. Billing data can lag about a day. Charges in `us-east-2` or `us-west-2` mean an unknown resource. The connection's managed secret may have billed as an untagged Secrets Manager line (unverified: check that line on the bill).
6. Sign out. No CLI command removes a profile section, so edit `~/.aws/config` by hand. `aws logout --all` signs out every login profile. Delete any cache file left whose profile is gone (unverified: whether `--all` reaches those files, so list the directory and check).
   ```bash
   # docs: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html
   aws logout --profile ces-revisions
   # In an editor, delete the [profile ces-revisions] section from ~/.aws/config.
   aws logout --all
   ls -la ~/.aws/login/cache
   ```

## Phase 7: After the November 2026 bill

1. In early December 2026, open the November 2026 bill (status Issued). Confirm that it shows no charge for the services in Phase 6.
2. Optional: deactivate the cost allocation tag `project` in Billing and Cost Management > Cost allocation tags. This affects every resource with that key in the account. Reactivation can take up to 24 hours ([docs](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/activating-tags.html)).
3. Optional: delete the IAM user. The default is to keep it, because it is probably the account's only non-root admin. To delete it, sign in as root, open IAM > Users, select the user and click Delete. The console removes its password, MFA device and policy attachments. A deleted user cannot be restored ([docs](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_remove.html)).
4. Delete the kit and the scratch copy last. Deletion is permanent.
   1. Move anything you want to keep out of `~/ces-revisions-teardown-work` first, such as `env-state-before-teardown.json` and `inventory-before.txt`. They hold identifiers, so keep them private.
   2. For the history, also move `cloud-branches.bundle`, `env-worktree/uncommitted.patch`, `env-worktree/docs/decisions/g7e-trial.md` and `runner-worktree/.sdd/` out of the kit.
   3. Run `rm -rf ~/ces-revisions-aws-teardown ~/ces-revisions-teardown-work`. The Phase 2 copies in `~/ces-revisions-teardown-keep` stay.

## Sources

- AWS CLI sign-in and tagging: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html, https://docs.aws.amazon.com/cli/latest/reference/resourcegroupstaggingapi/get-resources.html, https://docs.aws.amazon.com/resourcegroupstagging/latest/APIReference/API_GetResources.html
- EC2: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/terminate-instances.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/wait/instance-terminated.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instances.html, https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/global-view.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-volume.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/wait/volume-deleted.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/release-address.html
- EBS and DLM: https://docs.aws.amazon.com/ebs/latest/userguide/ebs-deleting-snapshot.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-snapshot.html, https://docs.aws.amazon.com/ebs/latest/userguide/recycle-bin.html, https://docs.aws.amazon.com/cli/latest/reference/dlm/delete-lifecycle-policy.html, https://aws.amazon.com/ebs/pricing/
- VPC: https://docs.aws.amazon.com/vpc/latest/userguide/delete-vpc.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-vpc.html, https://aws.amazon.com/vpc/pricing/, https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-security-group.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/disassociate-route-table.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-route-table.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-subnet.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/detach-internet-gateway.html, https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-internet-gateway.html
- IAM: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_manage_delete.html, https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_remove.html, https://docs.aws.amazon.com/IAM/latest/UserGuide/root-user-best-practices.html, https://docs.aws.amazon.com/cli/latest/reference/iam/remove-role-from-instance-profile.html, https://docs.aws.amazon.com/cli/latest/reference/iam/delete-instance-profile.html
- Budgets and billing: https://docs.aws.amazon.com/cli/latest/reference/budgets/delete-budget.html, https://docs.aws.amazon.com/cli/latest/reference/budgets/delete-budget-action.html, https://aws.amazon.com/aws-cost-management/aws-budgets/pricing/, https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/activating-tags.html, https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/getting-viewing-bill.html, https://docs.aws.amazon.com/cli/latest/reference/budgets/describe-budgets.html
- Service Quotas and Support: https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html, https://docs.aws.amazon.com/awssupport/latest/user/monitoring-your-case.html, https://docs.aws.amazon.com/awssupport/latest/user/about-support-api.html, https://docs.aws.amazon.com/cli/latest/reference/service-quotas/list-requested-service-quota-change-history.html
- S3: https://docs.aws.amazon.com/AmazonS3/latest/userguide/empty-bucket.html, https://docs.aws.amazon.com/AmazonS3/latest/userguide/DeletingObjectVersions.html, https://docs.aws.amazon.com/AmazonS3/latest/API/API_DeleteObjects.html, https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-bucket.html, https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html, https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html, https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-objects.html, https://docs.aws.amazon.com/cli/latest/reference/s3api/head-bucket.html, https://docs.aws.amazon.com/cli/latest/reference/s3api/list-buckets.html, https://docs.aws.amazon.com/cli/latest/reference/s3api/put-bucket-logging.html, https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html, https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-bucket-metrics-configuration.html
- CloudFormation: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-delete-stack.html, https://docs.aws.amazon.com/cli/latest/reference/cloudformation/wait/stack-delete-complete.html, https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-attribute-deletionpolicy.html, https://docs.aws.amazon.com/AWSCloudFormation/latest/APIReference/API_DescribeStackResources.html, https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-protect-stacks.html, https://docs.aws.amazon.com/cli/latest/reference/cloudformation/delete-stack.html
- Secrets Manager and EventBridge: https://docs.aws.amazon.com/cli/latest/reference/secretsmanager/delete-secret.html, https://aws.amazon.com/secrets-manager/pricing/, https://docs.aws.amazon.com/secretsmanager/latest/userguide/integrating_how-services-use-secrets_events.html, https://docs.aws.amazon.com/eventbridge/latest/APIReference/API_DeleteConnection.html, https://docs.aws.amazon.com/scheduler/latest/APIReference/API_DeleteScheduleGroup.html, https://docs.aws.amazon.com/cli/latest/reference/scheduler/list-schedules.html, https://docs.aws.amazon.com/cli/latest/reference/secretsmanager/list-secrets.html, https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-target-connection.html, https://docs.aws.amazon.com/secretsmanager/latest/userguide/manage_delete-secret.html
- Step Functions, CloudTrail and CloudWatch: https://docs.aws.amazon.com/step-functions/latest/dg/cw-logs.html, https://aws.amazon.com/cloudtrail/pricing/, https://aws.amazon.com/cloudwatch/pricing/, https://docs.aws.amazon.com/cli/latest/reference/logs/delete-resource-policy.html, https://docs.aws.amazon.com/cli/latest/reference/stepfunctions/list-executions.html, https://docs.aws.amazon.com/cli/latest/reference/stepfunctions/stop-execution.html, https://docs.aws.amazon.com/cli/latest/reference/logs/filter-log-events.html
- OpenTofu and the AWS provider: https://opentofu.org/docs/cli/commands/init/, https://opentofu.org/docs/cli/commands/destroy/, https://opentofu.org/docs/language/resources/behavior/, https://opentofu.org/docs/language/settings/backends/s3/, https://raw.githubusercontent.com/hashicorp/terraform-provider-aws/main/website/docs/r/s3_bucket.html.markdown, https://opentofu.org/docs/cli/commands/force-unlock/
- Lambda Cloud: https://docs.lambda.ai/public-cloud/console/, https://docs.lambda.ai/public-cloud/billing/, https://docs.lambda.ai/public-cloud/filesystems/, https://docs.lambda.ai/public-cloud/firewalls/
- GitHub and git: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens, https://git-scm.com/docs/git-bundle, https://git-scm.com/docs/git-push, https://git-scm.com/docs/git-branch
- Tools: https://man.openbsd.org/ssh-keygen, https://docs.brew.sh/Manpage
