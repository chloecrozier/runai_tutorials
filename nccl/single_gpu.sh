## 1 GPU in a single node
## Uses standard training job (simpler than MPI for single GPU)

# Step 1: Submit the job
runai training submit nccl-single-gpu \
  -p nccl-benchmarking \
  -i nvcr.io/nvidia/pytorch:25.01-py3 \
  -g 1 \
  --node-pools default \
  -- bash -c 'sleep 1d'

# Step 2: Exec into the pod
runai training exec nccl-single-gpu -p nccl-benchmarking -it -- bash

# Step 3: Run inside the container
/usr/local/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10
