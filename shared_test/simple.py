import os
import argparse

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def run(backend):
    tensor = torch.zeros(1)
    
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            print("dist send to rank {}".format(rank_recv))
            dist.send(tensor=tensor, dst=rank_recv)
            print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor)
        print('worker_{} has received data from rank {}\n'.format(WORLD_RANK, 0))
    sys.exit(0)

def init_processes(backend):
    print("Waiting for other ranks")
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    print("Rank {} Initialized".format(WORLD_RANK))
    run(backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(backend=args.backend)
