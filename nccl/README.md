# NCCL Benchmarking

## 1. Build & push the container

```bash
# from nccl/containers/
source ../../.env
docker login nvcr.io          # user: $oauthtoken  pass: <NGC API key>
docker buildx build --platform linux/arm64 --provenance=false \
  -t $REGISTRY/runai-nccl-pytorch-26.01:latest --push .
```

> `--platform linux/arm64` is required — the DGX nodes are ARM64 (Grace).

### Verify image (optional)

```bash
bash /workspace/verify_image.sh          # expects aarch64
bash /workspace/verify_image.sh x86_64   # if running on x86
```

## 2. Create the registry secret (once)

```bash
kubectl create secret docker-registry nvcr-creds \
  -n runai-nccl-benchmarking \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password='<NGC_API_KEY>'
```

Or create it in the Run:ai UI under **Credentials > New Credential > Docker Registry**.

## 3. Run a job

```bash
runai project set nccl-benchmarking

# single-node (4 GPU) — use `mpi/distributed_within_rack.sh` for multi-node jobs
runai training submit nccl-single-node \
  -p nccl-benchmarking \
  -i $REGISTRY/runai-nccl-pytorch-26.01:latest \
  -g 4 --node-pools default \
  -- bash -c 'sleep 1d'
```

## 4. Exec in and run NCCL tests

```bash
runai training exec nccl-single-node -p nccl-benchmarking -it -- bash

# 4-GPU all-reduce benchmark
all_reduce_perf -b 8 -e 1G -f 2 -g 4 -w 2 --iters 10
```

## 6. Clean up

```bash
runai training delete nccl-single-node -p nccl-benchmarking
```
