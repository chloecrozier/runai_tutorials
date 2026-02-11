# This is what is run in pytorch/distributed_across_racks.sh
import torch
import torch.distributed as dist
import time
import os

dist.init_process_group(backend='nccl')
local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)
rank = dist.get_rank()
world = dist.get_world_size()

if rank == 0:
    nnodes = os.environ.get('PET_NNODES', '?')
    nproc = os.environ.get('PET_NPROC_PER_NODE', '?')
    print('World size: {} | Nodes: {} | GPUs/node: {}'.format(world, nnodes, nproc))
    print('{:>12s}  {:>10s}  {:>12s}'.format('Size', 'Time (ms)', 'BusBW (GB/s)'))
    print('-' * 40)

for nbytes in [8, 256, 8192, 262144, 8388608, 67108864, 268435456, 1073741824]:
    buf = torch.ones(nbytes // 4, device='cuda:{}'.format(local_rank))
    for _ in range(5):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    elapsed = (time.time() - t) / 20
    busbw = nbytes * 2 * (world - 1) / world / elapsed / 1e9
    if rank == 0:
        print('{:>12d}  {:>10.3f}  {:>12.2f}'.format(nbytes, elapsed * 1000, busbw))

dist.destroy_process_group()