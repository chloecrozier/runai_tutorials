## 8 GPUs across 2 nodes in the same rack

# Step 1: Submit the job
runai training mpi submit nccl-within-rack \
  -p nccl-benchmarking \
  -i nvcr.io/nvidia/pytorch:26.01-py3 \
  -g 4 \
  --workers 2 \
  --slots-per-worker 4 \
  --node-pools default \
  --master-command "bash" \
  --master-args "-c 'sleep 1d'" \
  -- bash -c 'sleep 1d'

# Step 2: Exec into the worker pod
runai training mpi exec nccl-within-rack -p nccl-benchmarking --pod nccl-within-rack-worker-0 -it -- bash

# Step 3: Run inside the container
mpirun -np 8 \
  --allow-run-as-root \
  -x NCCL_DEBUG=INFO \
  /usr/local/bin/all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
