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
