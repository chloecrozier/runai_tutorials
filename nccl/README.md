# NCCL Benchmarking Tests

Simple guide for running NCCL tests with RunAI CLI.

**Custom image (PyTorch + OpenMPI + nccl-tests + sshd):** see [CONTAINER_SETUP.md](CONTAINER_SETUP.md) for building the container and running multi-node MPI/PyTorch jobs.

## Setup

List available projects:
```bash
runai project list
```

Set your project:
```bash
runai project set nccl-benchmarking
```

## Check Running Workloads

List all running workloads in the current project:
```bash
runai workload list
```

## Check Available GPUs

See available GPUs in the cluster:
```bash
runai node list
```

For detailed JSON output:
```bash
runai node list --json
```

Note: Rack info can be inferred from node naming convention (e.g., `s03-*` and `s04-*` are different racks).
