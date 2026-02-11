#!/usr/bin/env bash
# Deploy Llama-3.3-70B disaggregated (prefill/decode) single-node with NVIDIA Dynamo + vLLM
#
# Reference: https://github.com/ai-dynamo/dynamo/blob/v0.7.0/recipes/llama-3-70b/vllm/agg/deploy.yaml

set -euo pipefail

PROJECT="dynamo-inference"
WORKLOAD_NAME="llama3-agg"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC_FILE="${SCRIPT_DIR}/llama3-70b-agg-single-node.yaml"

# ── 1. Set the Run:ai project ────────────────────────────────────────────────
echo "Setting Run:ai project to ${PROJECT}..."
runai project set "${PROJECT}"

# ── 2. Deploy the DynamoGraphDeployment ──────────────────────────────────────
echo ""
echo "Deploying Llama-3.3-70B disaggregated single-node (agg) via Dynamo..."
kubectl apply --validate=false -f "${SPEC_FILE}"

# ── 3. Check status ─────────────────────────────────────────────────────────
echo ""
echo "Check deployment status:"
echo "  kubectl get pods -n runai-dynamo-inference"
echo "  kubectl get dynamographdeployment -n runai-dynamo-inference"
echo ""

# ── 4. Port-forward and test ────────────────────────────────────────────────
echo "Once pods are Running, test with:"
echo ""
echo "  # Port-forward the frontend service"
echo "  kubectl port-forward svc/llama3-agg-frontend -n runai-dynamo-inference 8000:8000"
echo ""
echo "  # Chat completion request (in another terminal)"
echo '  curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '"'"'{'
echo '    "model": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",'
echo '    "messages": [{"role": "user", "content": "Tell me about NVIDIA Dynamo"}]'
echo "  }'"'"''
echo ""

# ── 5. View logs ─────────────────────────────────────────────────────────────
echo "To view logs:"
echo "  kubectl logs -l nvidia.com/dynamo-deployment=llama3-70b-disagg-sn -n runai-dynamo-inference --tail=100"
echo ""

# ── 6. Cleanup ───────────────────────────────────────────────────────────────
echo "To delete the deployment:"
echo "  kubectl delete --validate=false -f ${SPEC_FILE}"