#!/bin/bash
# Verify container image: architecture, SSH, and NCCL test binaries
# Usage: ./verify_image.sh [expected_arch]   (default: aarch64)

EXPECTED="${1:-aarch64}"
ACTUAL=$(uname -m)
PASS=0; FAIL=0

check() {
  if [ "$2" = "ok" ]; then
    echo "  PASS  $1"; ((PASS++))
  else
    echo "  FAIL  $1"; ((FAIL++))
  fi
}

echo "=== Image Verification ==="
echo ""

# 1. Architecture
[ "$ACTUAL" = "$EXPECTED" ] && S="ok" || S="fail"
check "arch: expected=$EXPECTED actual=$ACTUAL" "$S"

# 2. SSH
command -v sshd >/dev/null 2>&1 && S="ok" || S="fail"
check "sshd found ($(which sshd 2>/dev/null || echo 'missing'))" "$S"

command -v ssh >/dev/null 2>&1 && S="ok" || S="fail"
check "ssh client found" "$S"

# 3. NCCL tests
command -v all_reduce_perf >/dev/null 2>&1 && S="ok" || S="fail"
check "all_reduce_perf on PATH" "$S"

[ -d /workspace/nccl-tests/build ] && S="ok" || S="fail"
check "nccl-tests/build dir exists" "$S"

# 4. MPI (bonus)
command -v mpirun >/dev/null 2>&1 && S="ok" || S="fail"
check "mpirun found" "$S"

# Summary
echo ""
echo "--- $PASS passed, $FAIL failed ---"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
