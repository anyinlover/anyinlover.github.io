import os
import sys
import torch.multiprocessing as mp
import torch.distributed as dist
import time

def info():
    if dist.is_initialized():
        print(f"Process {dist.get_rank()} has PID {os.getpid()}", flush=True)
    else:
        print(f"Process {os.getpid()} is not part of a distributed group", flush=True)

def run(rank, size, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    info()
    try:
        if rank == 0:
            raise RuntimeError("Error from process 0")
    except BaseException as e:
        raise e
    return ""
    
if __name__ == "__main__":
    world_size = 2
    mp.set_start_method('forkserver')
    print(f"Process parent has PID {os.getpid()}")
    
    
    with mp.Pool(processes=world_size) as p:
        async_result = p.starmap_async(run, [(rank, world_size) for rank in range(world_size)])
    
        start_time = time.perf_counter()
        try:
            results = async_result.get(timeout=5)
        except mp.TimeoutError:
            print("Timed out waiting for processes to finish.")
        # except Exception as e:
        #     print(e)
        finally:
            p.starmap(info, [() for _ in range(world_size)])
            p.close()
            p.join()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")