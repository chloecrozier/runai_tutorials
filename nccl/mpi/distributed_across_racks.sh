# create the workload
runai training mpi submit nccl-across-racks \
  -p nccl-benchmarking \
  -i nvcr.io/nvidia/pytorch:26.01-py3 \
  -g 4 \
  --workers 3 \
  --slots-per-worker 4 \
  --node-pools default \
  -- bash -c 'sleep 1d'

# after creating the workload, you can run
runai training standard describe nccl-across-racks

# exec into the pod
runai training exec nccl-across-racks -p nccl-benchmarking -it -- bash

# 12‑GPU NCCL all‑reduce bandwidth/latency benchmark with debug logging
# parameters: min/max bytes = 8-1G; size multiplier = 2; GPUs per rank = 1; warmup iters = 2; measured iters = 10
/usr/local/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 4 -w 2 --iters 10

# delete the job when done
runai training mpi delete nccl-across-racks -p nccl-benchmarking