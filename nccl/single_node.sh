## 4 GPUs in a single node
runai training mpi submit nccl-single-node \
  -p nccl-benchmarking \
  -i nvcr.io/nvidia/pytorch:25.08-py3 \
  -g 4 \
  --workers 1 \
  --slots-per-worker 4 \
  --node-pools default \
  --master-command "bash" \
  --master-args "-c 'sleep 1d'" \
  -- bash -c 'sleep 1d'

runai training mpi exec nccl-single-node -p nccl-benchmarking -it -- bash

mpirun -np 4 \
  --allow-run-as-root \
  -x NCCL_DEBUG=INFO \
  /usr/local/bin/all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10 -c 10
