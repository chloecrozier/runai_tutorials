# NCCL Benchmarking Tests

Simple guide for running NCCL tests with RunAI CLI.

## Setup

Set your project:
```bash
runai project set nccl-benchmarking
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
