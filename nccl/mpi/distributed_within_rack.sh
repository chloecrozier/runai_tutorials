# create the workload
runai training submit nccl-single-node \
  -p nccl-benchmarking \
  -i nvcr.io/nvidia/pytorch:26.01-py3 \
  -g 4 \
  --node-pools default \
  -- bash -c 'sleep 1d'

# after creating the workload, you can run
runai training standard describe nccl-single-node

# exec into the pod
runai training exec nccl-single-node -p nccl-benchmarking -it -- bash

# 4‑GPU NCCL all‑reduce bandwidth/latency benchmark with debug logging
# parameters: min/max bytes = 8-1G; size multiplier = 2; GPUs per rank = 1; warmup iters = 2; measured iters = 10; validation checks = 10 
mpirun -np 4 \
  --allow-run-as-root \
  -x NCCL_DEBUG=INFO \
  /usr/local/bin/all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10

# delete the job when done
runai training delete nccl-single-node -p nccl-benchmarking