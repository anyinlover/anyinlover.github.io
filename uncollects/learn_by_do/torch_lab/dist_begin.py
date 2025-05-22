import os
import sys
import time
import torch
import torch.distributed as dist
import multiprocessing as mp
# import torch.multiprocessing as mp

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # dist.send(tensor=tensor, dst=1)
    else:
        # raise RuntimeError
        sys.exit(1)
        # dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == '__main__':
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    with mp.Pool(processes=world_size) as pool:
        pool.starmap(init_process, [(rank, world_size, run) for rank in range(world_size)])