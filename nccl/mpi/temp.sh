# quick test with IMEX fix -- within rack, 2 workers, 8 GPUs
runai training mpi submit multi-node \
  -p nccl-benchmarking \
  -i nvcr.io/r2kuatviomfd/runai-nccl-pytorch-26.01:latest \
  -g 4 \
  --workers 3 \
  --slots-per-worker 4 \
  --large-shm \
  --capability IPC_LOCK \
  --host-path path=/dev/nvidia-caps-imex-channels,mount=/dev/nvidia-caps-imex-channels \
  --host-path path=/dev/nvidia-caps,mount=/dev/nvidia-caps \
  --node-pools default \
  -- bash -c 'sleep 1d'

# exec in
runai training mpi exec multi-node -p nccl-benchmarking -it -- bash

# test SSH
ssh $(head -1 /etc/mpi/hostfile | awk '{print $1}') hostname

# run benchmark (no MNNVL/CUMEM disable -- IMEX should work now)
mpirun --allow-run-as-root \
  --hostfile /etc/mpi/hostfile \
  -np 12 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_MNNVL_ENABLE=0 \
  all_reduce_perf_mpi -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10

# cleanup
runai training mpi delete multi-node -p nccl-benchmarking










# quick test with IMEX fix -- within rack, 2 workers, 8 GPUs
runai training mpi submit multi-node \
  -p nccl-benchmarking \
  -i nvcr.io/r2kuatviomfd/runai-nccl-pytorch-26.01:latest \
  -g 4 \
  --workers 2 \
  --slots-per-worker 4 \
  --large-shm \
  --capability IPC_LOCK \
  --host-path path=/dev/nvidia-caps-imex-channels,mount=/dev/nvidia-caps-imex-channels \
  --host-path path=/dev/nvidia-caps,mount=/dev/nvidia-caps \
  --node-pools default \
  --extended-resource rdma/rdma_shared_device_a=1 \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  -- bash -c 'sleep 1d'

# exec in
runai training mpi exec multi-node -p nccl-benchmarking -it -- bash

# test SSH
ssh $(head -1 /etc/mpi/hostfile | awk '{print $1}') hostname

# run benchmark
mpirun --allow-run-as-root \
  --hostfile /etc/mpi/hostfile \
  -np 8 \
  -x NCCL_DEBUG=INFO \
  all_reduce_perf_mpi -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10

# cleanup
runai training mpi delete multi-node -p nccl-benchmarking