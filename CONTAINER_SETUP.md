# Custom container for multi-node (PyTorch + NCCL + MPI + SSH)

This guide walks you through building and using the custom image that includes **sshd** and **nccl-tests** for RunAI multi-node inference and training.

---

## Prerequisites

Set your registry in `.env` (or export it in your shell):

```bash
source .env
# .env contains: REGISTRY=<your-registry>
```

Or export directly:

```bash
export REGISTRY=<your-registry>
```

---

## What's in the image

- **Base:** `nvcr.io/nvidia/pytorch:26.01-py3`
- **OpenMPI:** `openmpi-bin`, `libopenmpi-dev` (for MPI-based NCCL and multi-node)
- **NCCL tests:** built from [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) with `MPI=1` (e.g. `all_reduce_perf` in `PATH`)
- **SSH:** `openssh-server` and `openssh-client`; entrypoint starts **sshd** so MPI jobs can use SSH between workers

---

## Step 1: Log in to Docker and NGC

Docker Desktop may require you to sign in before you can build or pull images. If you see an error like _"Sign in to continue using Docker Desktop"_, log in first:

```bash
# Sign in to Docker Desktop (opens browser or prompts for credentials)
docker login
```

Then log in to the NGC registry so Docker can pull the base image (`nvcr.io/nvidia/pytorch:...`):

```bash
# Username is literally "$oauthtoken"; password is your NGC API key
# Generate an API key at https://ngc.nvidia.com/setup/api-key
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-ngc-api-key>
```

---

## Step 2: Build the image

From the repo root:

```bash
source .env
docker buildx build --platform linux/arm64 -t $REGISTRY/runai-nccl-pytorch-26.01:latest --load .
```

> **Why `--platform linux/arm64`?** The DGX nodes use Grace Hopper (ARM64/aarch64) CPUs.
> Omitting `--platform` may produce an amd64 image (e.g. if Docker Desktop uses Rosetta),
> which will fail at pull time with _"no match for platform in manifest"_.
>
> Use `--load` to import into the local Docker image store, or replace with `--push` to
> push directly to the registry (skipping Step 3).

---

## Step 3: Push to your registry

Push the built image:

```bash
docker push $REGISTRY/runai-nccl-pytorch-26.01:latest
```

Ensure the RunAI cluster can pull from this registry (imagePullSecrets if private).

---

## Step 4: Run multi-node jobs on RunAI

### Option A: MPI job (NCCL tests, MPI-based workloads)

Uses RunAI's **training MPI** job type; good for `all_reduce_perf` and other MPI-launched binaries.

```bash
# Set project
runai project set nccl-benchmarking

# Submit multi-node MPI job using your image
runai training mpi submit nccl-across-racks \
  -p nccl-benchmarking \
  -i $REGISTRY/runai-nccl-pytorch-26.01:latest \
  -g 4 \
  --workers 3 \
  --slots-per-worker 4 \
  --node-pools default \
  -- bash -c 'sleep 1d'
```

Then exec in and run NCCL tests:

```bash
runai training exec nccl-across-racks -p nccl-benchmarking -it -- bash

# 12-GPU NCCL all-reduce (adjust -g to GPUs per node)
/workspace/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 4 -w 2 --iters 10
```

### Option B: PyTorch distributed job (torchrun / inference)

Uses RunAI's **training PyTorch** job type; no MPI/SSH required for PyTorch's built-in NCCL backend.

```bash
runai training pytorch submit my-pytorch-job \
  -p nccl-benchmarking \
  -i $REGISTRY/runai-nccl-pytorch-26.01:latest \
  -g 4 \
  --workers 2 \
  --large-shm \
  -e NCCL_DEBUG=INFO \
  -- bash -c 'torchrun --nproc_per_node=$PET_NPROC_PER_NODE --nnodes=$PET_NNODES --node_rank=$PET_NODE_RANK --master_addr=$PET_MASTER_ADDR --master_port=$PET_MASTER_PORT /workspace/your_script.py'
```

RunAI injects `PET_*` (or equivalent) so PyTorch can form the process group across nodes.

---

## GPU operator, network operator, and Kubeflow MPIJob

You don't install these **inside** your container. They are **cluster-level** components:

| Component | Role | RunAI |
|-----------|------|--------|
| **NVIDIA GPU Operator** | Installs drivers, device plugin, DCGM, etc. on nodes | Usually already present on RunAI clusters; no action in the image. |
| **NVIDIA Network Operator** | Manages RDMA/GPUDirect, Multus, etc. | Cluster-level; admins enable it. Your image just uses NCCL; no extra config in the Dockerfile. |
| **Kubeflow MPIJob** | Custom resource to run MPI jobs on Kubernetes | **Not required** for RunAI. RunAI has its own job types: `runai training mpi` and `runai training pytorch`. Use those instead of installing MPIJob in the cluster or in the image. |

So:

- Use **RunAI's** MPI and PyTorch job types.
- Rely on the cluster for GPU and (if available) network operator; your container only needs PyTorch, OpenMPI, nccl-tests, and sshd as in this image.

---

## Quick reference

```bash
source .env   # exports REGISTRY=<your-registry>
```

| Goal | Command |
|------|---------|
| Build | `docker buildx build --platform linux/arm64 -t $REGISTRY/runai-nccl-pytorch-26.01:latest --load .` |
| Push | `docker push $REGISTRY/runai-nccl-pytorch-26.01:latest` |
| MPI job (multi-node) | `runai training mpi submit ... -i $REGISTRY/runai-nccl-pytorch-26.01:latest ...` |
| PyTorch job (multi-node) | `runai training pytorch submit ... -i $REGISTRY/runai-nccl-pytorch-26.01:latest ...` |
| NCCL binary in image | `all_reduce_perf` (and others) in `/workspace/nccl-tests/build/` and on `PATH` |
