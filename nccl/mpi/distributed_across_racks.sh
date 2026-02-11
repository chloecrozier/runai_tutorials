# 3-node, 12-GPU NCCL test (4 GPUs per node, nodes across racks)
# launcher = orchestrator (no GPUs), workers = compute nodes
# --workers 3 = 3 compute nodes x 4 GPUs = 12 GPUs total
runai training mpi submit nccl-across-racks \
  -p nccl-benchmarking \
  -i nvcr.io/r2kuatviomfd/runai-nccl-pytorch-26.01:latest \
  -g 4 \
  --workers 3 \
  --slots-per-worker 4 \
  --large-shm \
  --capability IPC_LOCK \
  --node-pools default \
  -- bash -c 'sleep 1d'

# check status
runai training mpi describe nccl-across-racks -p nccl-benchmarking

# exec into the launcher pod
runai training mpi exec nccl-across-racks -p nccl-benchmarking -it -- bash

# test SSH to workers (should print hostnames)
cat /etc/mpi/hostfile
ssh $(sed -n '1p' /etc/mpi/hostfile | awk '{print $1}') hostname
ssh $(sed -n '2p' /etc/mpi/hostfile | awk '{print $1}') hostname
ssh $(sed -n '3p' /etc/mpi/hostfile | awk '{print $1}') hostname

# 12-GPU all-reduce benchmark via mpirun
mpirun --allow-run-as-root \
  --hostfile /etc/mpi/hostfile \
  -np 12 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_MNNVL_ENABLE=0 \
  -x NCCL_CUMEM_ENABLE=0 \
  all_reduce_perf_mpi -b 8 -e 1G -f 2 -g 1 -w 2 --iters 10

# delete when done
runai training mpi delete nccl-across-racks -p nccl-benchmarking
