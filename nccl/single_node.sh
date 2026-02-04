## 4 GPUs in a single node
## Uses standard training job (simpler resource allocation)

# Step 1: Submit the job
runai training submit nccl-single-node \
  -p nccl-benchmarking \
  -i nvcr.io/nvidia/pytorch:25.01-py3 \
  -g 4 \
  --node-pools default \
  -- bash -c 'sleep 1d'

# Step 2: Exec into the pod
runai training exec nccl-single-node -p nccl-benchmarking -it -- bash

# Step 3: Run inside the container
mpirun -np 4 \
  --allow-run-as-root \
  -x NCCL_DEBUG=INFO \
  /usr/local/bin/all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
