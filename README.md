# Run NCCL Tests with Run:ai to Validate Your AI Factory

Run NCCL `all_reduce_perf_mpi` inside PyTorch containers launched as Run:ai inference workloads. One unified workflow covers single-node, single-rack, and multi-rack validation before any real training job hits the cluster.

> Most NVIDIA reference benchmarks ship as Slurm jobs. This tutorial gives you the **same validation via a single Run:ai workflow** so the cluster serves both researchers and production schedulers without a parallel Slurm path.

## Why run NCCL tests?

A short [`nccl-tests`](https://github.com/NVIDIA/nccl-tests) run is the fastest health check for an AI factory. It validates:

- **Driver + library install.** CUDA, NCCL, and the MPI launcher work end-to-end inside the container.
- **GPU visibility.** All requested GPUs are passed through to the pod.
- **Intra-node fabric.** NVLink / NVSwitch bandwidth on a single node.
- **Inter-node fabric.** InfiniBand / RoCE bandwidth between nodes, including rack-to-rack hops.
- **Scheduling & topology.** Pods land where the fabric expects.

A failing or low-bandwidth NCCL run beats burning GPU-hours on a misconfigured stack.

## Configurations covered

| # | Test | Pods × GPUs | Validates |
| --- | --- | --- | --- |
| 1 | Single node | 1 × 4 | Intra-node NVLink |
| 2 | Single rack | 2 × 4 | Inter-node, same rack |
| 3 | Multi-rack | 3 × 4 | Cross-rack fabric |

**Example cluster — OWL:** GB200, 4 nodes, 4 B200 GPUs per node (**16 GPUs total**), spread across two racks.

## Prerequisites

- Run:ai is deployed and the Run:ai CLI is installed (or installable from the UI; see step 1).
- A PyTorch image with NCCL + `nccl-tests`. The NGC PyTorch image bundles `/usr/local/bin/all_reduce_perf_mpi`.
- For multi-node: the image supports MPI launch between pods (SSH or site PMIx setup).
- Optional: `kubectl` access for rack inspection. *(TODO: confirm with Yang / Paul / Doug which jump host you SSH to in order to run `kubectl` against this cluster.)*

```bash
export PROJECT="nccl-benchmarking"
export NCCL_IMAGE="nvcr.io/nvidia/pytorch:<tag>"
export NCCL_TEST_BIN="/usr/local/bin/all_reduce_perf_mpi"
```

## Node pools, racks, and the scheduler

**Homogeneous cluster? Do not pin to node pools.** Let the Run:ai scheduler place pods. It will:

1. Pack a workload into **one rack** when it fits.
2. **Span racks automatically** when the request is too large for one rack.

Verify placement *after* the workload is running:

```bash
kubectl get nodes -L topology.kubernetes.io/rack          # all nodes + rack label
kubectl get node <node-name> -o yaml | grep -i rack       # one node, full label set
kubectl get pods -n "runai-${PROJECT}" -o wide            # pod → node mapping
```

If your cluster does not expose `topology.kubernetes.io/rack`, ask the cluster admin for the label key or naming convention used to identify racks.

## 1. Log in and select the project

In the Run:ai UI: **Help menu > Researcher Command Line Interface**, choose your cluster + OS, copy the install command, and run it on your jump host.

![Run:ai CLI install dialog](images/img2.png)

```bash
runai login                    # local browser
runai login remote-browser     # headless: paste returned code

runai project set "$PROJECT"
runai node list                # confirm GPU nodes are Ready
runai kubeconfig set           # optional: enable kubectl
```

![Run:ai node list output](images/img6.png)

## 2. Single node — 4 GPUs

Submit a one-pod inference workload requesting all four GPUs on one node:

```bash
runai inference submit nccl-single-node \
  -p "$PROJECT" -i "$NCCL_IMAGE" \
  --gpu-devices-request 4 \
  --min-replicas 1 --max-replicas 1 \
  --serving-port 8080 --large-shm \
  --command -- bash -lc "python3 -m http.server 8080 >/tmp/http.log 2>&1 & sleep infinity"
```

> The `http.server` is just a heartbeat to satisfy the inference workload's serving port. The benchmark itself runs interactively below.

Exec in and run:

```bash
runai inference exec nccl-single-node -p "$PROJECT" --tty --stdin -- bash
nvidia-smi                                                # expect 4 GPUs

mpirun -np 4 --allow-run-as-root -x NCCL_DEBUG=INFO \
  "$NCCL_TEST_BIN" -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
```

![NCCL all-reduce output](images/img10.png)

## 3. Single rack — 2 nodes × 4 GPUs

Submit a distributed inference workload (1 leader + 1 worker = 2 pods × 4 GPUs):

```bash
runai inference distributed submit nccl-single-rack \
  -p "$PROJECT" -i "$NCCL_IMAGE" \
  --replicas 1 --workers 1 \
  --gpu-devices-request 4 \
  --serving-port 8080 --large-shm \
  --command bash \
  --arguments "-lc 'python3 -m http.server 8080 >/tmp/http.log 2>&1 & sleep infinity'"
```

> `--workers N` adds N workers to the leader; `--workers 1` ⇒ 2 pods total.

**Confirm both pods landed on the same rack:**

```bash
kubectl get pods -n "runai-${PROJECT}" -o wide | grep nccl-single-rack
kubectl get nodes -L topology.kubernetes.io/rack
```

Both node names should show the **same** rack label. Exec in, write a hostfile with each pod's hostname/IP, and run an 8-GPU all-reduce:

```bash
runai inference distributed exec nccl-single-rack \
  -p "$PROJECT" --pod <leader-pod-name> --tty --stdin -- bash

cat >/tmp/nccl-hosts <<'EOF'
<pod-0> slots=4
<pod-1> slots=4
EOF

mpirun -np 8 --allow-run-as-root --hostfile /tmp/nccl-hosts \
  -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth0 \
  "$NCCL_TEST_BIN" -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
```

![Run:ai multi-node describe output](images/img11.png)

## 4. Multi-rack — 3 nodes × 4 GPUs

Submit 1 leader + 2 workers (3 pods × 4 GPUs = 12 GPUs). On a 4-node cluster this forces the scheduler to span racks:

```bash
runai inference distributed submit nccl-multi-rack \
  -p "$PROJECT" -i "$NCCL_IMAGE" \
  --replicas 1 --workers 2 \
  --gpu-devices-request 4 \
  --serving-port 8080 --large-shm \
  --command bash \
  --arguments "-lc 'python3 -m http.server 8080 >/tmp/http.log 2>&1 & sleep infinity'"
```

**Verify the placement crosses racks:**

```bash
kubectl get pods -n "runai-${PROJECT}" -o wide | grep nccl-multi-rack
kubectl get nodes -L topology.kubernetes.io/rack
```

At least two of the host nodes should show **different** rack labels. If they don't, the workload fit in one rack — drop a worker, re-submit at a larger size, or temporarily cordon a rack to force the spread.

Exec in, build a 3-entry hostfile, and run a 12-GPU all-reduce (same `mpirun` pattern as step 3 with `-np 12` and three host entries).

![Run:ai multi-node workload status](images/img12.png)

## Pass criteria

For every test:

- `NCCL INFO` lines appear in the output.
- Rank count matches the GPU count (4, 8, 12).
- `Out of bounds values : 0 OK` is printed.
- Bandwidth is non-zero and ordered as expected: **intra-node > single-rack inter-node > multi-rack inter-node**.

If `mpirun` cannot launch ranks on remote pods, the image is missing the site-required MPI launch path (SSH keys / PMIx). Use the site-approved NCCL image.

## Cleanup

```bash
runai inference delete nccl-single-node -p "$PROJECT"
runai inference distributed delete nccl-single-rack -p "$PROJECT"
runai inference distributed delete nccl-multi-rack -p "$PROJECT"
runai workload list                                         # confirm empty
```
