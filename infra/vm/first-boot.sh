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
