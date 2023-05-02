import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2**30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)
