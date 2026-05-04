# Run NCCL Tests with Run:ai to Validate Your AI Factory

Run NCCL `all_reduce_perf_mpi` inside PyTorch containers launched as Run:ai training workloads. One unified workflow covers single-node, single-rack, and multi-rack validation before any real training job hits the cluster.

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

| # | Test | Workers × GPUs | Validates |
| --- | --- | --- | --- |
| 1 | Single node | 1 × 4 = 4 GPUs | Intra-node NVLink |
| 2 | Single rack | 2 × 4 = 8 GPUs | Inter-node, same rack |
| 3 | Multi-rack | 3 × 4 = 12 GPUs | Cross-rack fabric |

**Example cluster — OWL:** GB200, 4 nodes, 4 B200 GPUs per node (**16 GPUs total**), spread across two racks.

## Prerequisites

- Run:ai is deployed and the Run:ai CLI is installed (or installable from the UI; see step 1).
- A PyTorch image with NCCL + `nccl-tests`. The NGC PyTorch image bundles `/usr/local/bin/all_reduce_perf_mpi`.
- For multi-node: the image supports MPI launch between pods (SSH or site PMIx setup).
- Optional: `kubectl` access for rack inspection. *(TODO: confirm with Yang / Paul / Doug which jump host you SSH to in order to run `kubectl` against this cluster.)*

```bash
export PROJECT="nccl-benchmarking"
export NCCL_IMAGE="nvcr.io/nvidia/pytorch:26.01-py3"
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

## 1. Get CLI access

In the Run:ai UI, open the help menu and select **Researcher Command Line Interface**.

![Run:ai help menu showing Researcher Command Line Interface](images/img1.png)

Choose the cluster and operating system, then copy the generated CLI install command.

![Run:ai CLI install dialog](images/img2.png)

Run the copied command on your workstation or jump host.

![Run:ai CLI install output](images/img3.png)

Log in to the control plane:

```bash
runai login                    # local browser
runai login remote-browser     # headless: paste returned code in terminal
```

![Run:ai remote browser login output](images/img4.png)

If your environment requires Kubernetes access through the Run:ai kubeconfig:

```bash
runai kubeconfig set
kubectl get nodes
```

## 2. Select the project and inspect nodes

```bash
runai project list
runai project set "$PROJECT"
```

![Run:ai project list and project set](images/img7.png)

In the UI, **Resources > Nodes** shows GPU nodes, GPU type, free GPUs, and node pool.

![Run:ai nodes page](images/img5.png)

CLI equivalent:

```bash
runai node list
runai nodepool list
```

![Run:ai node list output](images/img6.png)

## Workload pattern (used by all 3 tests)

Every test uses the same `runai training submit` pattern. The container just sleeps; the benchmark is launched interactively from inside the pod with `mpirun`.

| Variable | Meaning |
| --- | --- |
| `--workers N` | Number of GPU worker pods (1 / 2 / 3) |
| `--slots-per-worker 4` | GPUs per worker pod |
| `-g 4` | GPU request used for scheduling |
| `--master-args "-c 'sleep 1d'"` | Keep the launcher pod alive |
| `-- bash -c 'sleep 1d'` | Keep each worker pod alive |

After submission, exec into the launcher and run `mpirun` against a hostfile of worker hostnames.

## 3. Single node — 4 GPUs

```bash
runai training submit nccl-single-node \
  -p "$PROJECT" \
  -i "$NCCL_IMAGE" \
  -g 4 \
  --workers 1 \
  --slots-per-worker 4 \
  --node-pools default \
  --master-command bash \
  --master-args "-c 'sleep 1d'" \
  -- bash -c 'sleep 1d'
```

The workload also appears in **Workload manager > Workloads**.

![Run:ai workloads page](images/img8.png)

Exec in and confirm GPU visibility:

```bash
runai training exec nccl-single-node -p "$PROJECT" -it -- bash
nvidia-smi # expect 4 GPUs in this case
```

![nvidia-smi showing four GPUs](images/img9.png)

Build a hostfile and run the test:

```bash
cat >/tmp/nccl-hosts <<'EOF'
<worker-0>
EOF

# 4-GPU NCCL all-reduce bandwidth/latency benchmark with debug logging
# Parameters: min/max bytes = 8-1G; size multiplier = 2; GPUs per rank = 1;
#             warmup iters = 2; measured iters = 10; validation checks = 10
mpirun -np 4 \
  --hostfile /tmp/nccl-hosts \
  --map-by ppr:4:node \
  --allow-run-as-root \
  -x NCCL_DEBUG=INFO \
  /usr/local/bin/all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
```

> `--map-by ppr:4:node` places 4 ranks per node, so the hostfile only needs the hostname — no `slots=` needed.

![NCCL all-reduce output](images/img10.png)

## 4. Single rack — 8 GPUs

```bash
runai training submit nccl-single-rack \
  -p "$PROJECT" \
  -i "$NCCL_IMAGE" \
  -g 4 \
  --workers 2 \
  --slots-per-worker 4 \
  --node-pools default \
  --master-command bash \
  --master-args "-c 'sleep 1d'" \
  -- bash -c 'sleep 1d'
```

**Confirm both workers landed on the same rack:**

```bash
kubectl get pods -n "runai-${PROJECT}" -o wide | grep nccl-single-rack
kubectl get nodes -L topology.kubernetes.io/rack
```

Both worker host nodes should show the **same** rack label. Exec in and run the 8-GPU all-reduce:

```bash
runai training exec nccl-single-rack -p "$PROJECT" -it -- bash

cat >/tmp/nccl-hosts <<'EOF'
<worker-0>
<worker-1>
EOF

mpirun -np 8 \
  --hostfile /tmp/nccl-hosts \
  --map-by ppr:4:node \
  --allow-run-as-root \
  -x NCCL_DEBUG=INFO \
  /usr/local/bin/all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
```

![Run:ai multi-node describe output](images/img11.png)

## 5. Multi-rack — 12 GPUs

```bash
runai training submit nccl-multi-rack \
  -p "$PROJECT" \
  -i "$NCCL_IMAGE" \
  -g 4 \
  --workers 3 \
  --slots-per-worker 4 \
  --node-pools default \
  --master-command bash \
  --master-args "-c 'sleep 1d'" \
  -- bash -c 'sleep 1d'
```

**Verify the placement crosses racks:**

```bash
kubectl get pods -n "runai-${PROJECT}" -o wide | grep nccl-multi-rack
kubectl get nodes -L topology.kubernetes.io/rack
```

At least two of the host nodes should show **different** rack labels. If they don't, the workload fit in one rack — bump worker count, or temporarily cordon a rack to force the spread.

```bash
runai training exec nccl-multi-rack -p "$PROJECT" -it -- bash

cat >/tmp/nccl-hosts <<'EOF'
<worker-0>
<worker-1>
<worker-2>
EOF

mpirun -np 12 \
  --hostfile /tmp/nccl-hosts \
  --map-by ppr:4:node \
  --allow-run-as-root \
  -x NCCL_DEBUG=INFO \
  /usr/local/bin/all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
```

![Run:ai multi-node workload status](images/img12.png)

## Other NCCL tests

These examples use `all_reduce_perf_mpi`, but [`nccl-tests`](https://github.com/NVIDIA/nccl-tests) ships every standard collective. Swap the binary path — same `mpirun` flags, same hostfile.

| Binary | Collective | What it stresses |
| --- | --- | --- |
| `all_reduce_perf_mpi` | All-reduce | Default; bandwidth + latency for gradient sync |
| `all_gather_perf_mpi` | All-gather | Tensor-parallel forward / activations gather |
| `reduce_scatter_perf_mpi` | Reduce-scatter | ZeRO / sharded-optimizer step |
| `broadcast_perf_mpi` | Broadcast | Weight broadcast from rank 0 |
| `reduce_perf_mpi` | Reduce | One-to-all reduction |
| `alltoall_perf_mpi` | All-to-all | MoE expert dispatch, sequence-parallel |
| `sendrecv_perf_mpi` | Point-to-point | Pipeline-parallel hop |
| `gather_perf_mpi` / `scatter_perf_mpi` | Gather / Scatter | Asymmetric collectives |

Example — run all-to-all on 8 GPUs:

```bash
mpirun -np 8 \
  --hostfile /tmp/nccl-hosts \
  --map-by ppr:4:node \
  --allow-run-as-root \
  -x NCCL_DEBUG=INFO \
  /usr/local/bin/alltoall_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
```

For end-to-end validation of a real workload pattern (e.g. a Megatron training recipe), pick the collectives that workload actually uses and run them at the same message size range.

## Pass criteria

For every test:

- `NCCL INFO` lines appear in the output.
- Rank count matches the GPU count (4, 8, 12).
- `Out of bounds values : 0 OK` is printed.
- Bandwidth is non-zero and ordered as expected: **intra-node > single-rack inter-node > multi-rack inter-node**.

If `mpirun` cannot launch ranks on remote pods, the image is missing the site-required MPI launch path (SSH keys / PMIx). Use the site-approved NCCL image.

## Cleanup

```bash
runai training delete nccl-single-node -p "$PROJECT"
runai training delete nccl-single-rack -p "$PROJECT"
runai training delete nccl-multi-rack  -p "$PROJECT"
runai workload list # confirm the workloads are not present
```