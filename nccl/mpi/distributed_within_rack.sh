# create the workload
runai training mpi submit nccl-within-rack \
  -p nccl-benchmarking \
  -i nvcr.io/r2kuatviomfd/runai-nccl-pytorch-26.01:latest \
  -g 4 \
  --workers 2 \
  --slots-per-worker 4 \
  --large-shm \
  --capability IPC_LOCK \
  --node-pools default \
  -- bash -c 'sleep 1d'

# check status
runai training mpi describe nccl-within-rack -p nccl-benchmarking

# exec into the launcher pod
runai training mpi exec nccl-within-rack -p nccl-benchmarking -it -- bash

# test SSH to workers (should print hostnames)
cat /etc/mpi/hostfile
ssh $(sed -n '1p' /etc/mpi/hostfile | awk '{print $1}') hostname
ssh $(sed -n '2p' /etc/mpi/hostfile | awk '{print $1}') hostname

# 8-GPU all-reduce benchmark
mpirun --allow-run-as-root \
  --hostfile /etc/mpi/hostfile \
  -np 8 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_MNNVL_ENABLE=0 \
  -x NCCL_CUMEM_ENABLE=0 \
  all_reduce_perf_mpi -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10

# delete when done
runai training mpi delete nccl-within-rack -p nccl-benchmarking
