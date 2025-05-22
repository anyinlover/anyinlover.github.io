import os
import sys
import signal
import multiprocessing as mp
import time

def run(rank):
    print(f"Process {rank} has PID {os.getpid()}", flush=True)
    if rank == 0:
        print("hi", flush=True)
    else:
        # signal.signal(signal.SIGTERM, lambda signum, frame: None)
        # time.sleep(60)
        sys.exit(1)

if __name__ == "__main__":
    world_size = 2
    mp.set_start_method('forkserver')
    print(f"Process parent has PID {os.getpid()}")
    
    
    with mp.Pool(processes=world_size) as p:
        async_result = p.map_async(run, [(rank,) for rank in range(world_size)])
    
        start_time = time.perf_counter()
        try:
            results = async_result.get(timeout=5)
        except mp.TimeoutError:
            print("Timed out waiting for processes to finish.")
        except Exception as e:
            print(e)
        finally:
            p.terminate()
            p.join()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")