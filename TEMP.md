# Running NCCL Tests Using the Run:ai CLI

This tutorial shows how to validate GPU and network communication on a 4-GPU-per-node GB200 cluster by running NCCL `all_reduce_perf_mpi` from containers launched with the Run:ai CLI.

You will run three checks:

- 1 node x 4 GPUs
- 2 nodes x 4 GPUs per node, for 8 total GPUs
- 3 nodes x 4 GPUs per node, for 12 total GPUs, ideally spanning racks

The screenshots are example chunks from an existing Run:ai environment. Use the command blocks as the generic workflow for a new environment.

## Prerequisites

- Run:ai is deployed and reachable through the Run:ai UI.
- You have a Run:ai user with access to the target project.
- The Run:ai CLI is installed or you can download it from the Run:ai UI.
- You have access to a project with enough quota for the largest test, for example 12 GPUs for three GB200 nodes.
- Your container image includes:
  - NVIDIA drivers exposed by the cluster runtime
  - CUDA and NCCL
  - MPI and NCCL tests, including `/usr/local/bin/all_reduce_perf_mpi`
  - For multi-node tests, MPI launch support between pods, such as SSH or the site-approved MPI/PMIx setup
- Optional but recommended: `kubectl` access to the project namespace so you can inspect pod placement and rack labels.

Set these values before running the tutorial:

```bash
export PROJECT="nccl-benchmarking"
export NODE_POOL="default"
export NCCL_IMAGE="nvcr.io/nvidia/pytorch:26.01-py3"
export NCCL_TEST_BIN="/usr/local/bin/all_reduce_perf_mpi"
export RUNAI_URL="https://<your-runai-url>"
```

If your image uses a different NCCL test path, exec into the container and run:

```bash
which all_reduce_perf_mpi
```

## 1. Open the Run:ai UI and Get CLI Access

In the Run:ai UI, open the help menu and select **Researcher Command Line Interface**.

![Run:ai help menu showing Researcher Command Line Interface](images/img1.png)

Choose the cluster and operating system, then copy the generated CLI install command.

![Run:ai CLI install dialog](images/img2.png)

Run the copied command on your workstation or jump host.

![Run:ai CLI install output](images/img3.png)

Log in to the control plane:

```bash
runai login
```

For a remote shell without browser forwarding, use:

```bash
runai login remote-browser
```

Follow the printed URL in a browser, authenticate, then paste the returned code into the terminal.

![Run:ai remote browser login output](images/img4.png)

If your environment requires Kubernetes access through the Run:ai kubeconfig, refresh the token:

```bash
runai kubeconfig set
kubectl get nodes
```

## 2. Select the Project and Inspect Nodes

List the projects you can access and set the benchmarking project:

```bash
runai project list
runai project set "$PROJECT"
```

![Run:ai project list and project set](images/img7.png)

In the UI, go to **Resources > Nodes** to inspect GPU nodes, GPU type, free GPUs, and node pool.

![Run:ai nodes page](images/img5.png)

The CLI equivalent is:

```bash
runai node list
runai nodepool list
```
