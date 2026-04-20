#!/usr/bin/env bash
# =============================================================================
# Verify NVIDIA driver + container toolkit are present and that Docker can
# reach the GPU. Run once after Docker install / kernel upgrade.
# =============================================================================

set -e

echo "═════════════════════════════════════════════════════"
echo "  GPU sanity check for urban-auto-vox-rs"
echo "═════════════════════════════════════════════════════"
echo

step() { printf '\n[%s] %s\n' "$1" "$2"; }
ok()   { printf '  ✅ %s\n' "$1"; }
fail() { printf '  ❌ %s\n' "$1" >&2; }

step "1/4" "NVIDIA host driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
	fail "nvidia-smi not on PATH — install the NVIDIA driver first."
	exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ok "driver detected"

step "2/4" "NVIDIA Container Toolkit"
if ! command -v nvidia-container-cli >/dev/null 2>&1; then
	fail "nvidia-container-cli missing. Install from:"
	echo "    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
	exit 1
fi
ok "nvidia-container-toolkit installed"

step "3/4" "Docker context"
docker context use default >/dev/null
ok "docker context = default"

step "4/4" "Docker → GPU access"
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 \
		nvidia-smi >/dev/null 2>&1; then
	ok "GPU is reachable from a container"
	docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 \
		nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
	fail "GPU not reachable from Docker. Try:"
	echo "    sudo nvidia-ctk runtime configure --runtime=docker"
	echo "    sudo systemctl restart docker"
	exit 1
fi

echo
echo "═════════════════════════════════════════════════════"
echo "  Ready. Next:  ./start.sh   or   ./gui-modern.sh"
echo "═════════════════════════════════════════════════════"
