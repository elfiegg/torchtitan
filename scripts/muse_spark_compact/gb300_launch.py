"""Validate the physical GPU and invoke the pinned compact trainer."""

from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import socket
import sys

import torch
from torchtitan.distributed import activation_offload


OFFLOAD_SHA = '296fecd743925dbf33f5ab97516e5cd21a5efc47120bf61299e8da9d7d0b880f'
TORCH_GIT = 'daf1a40a5de981f7cab47fed4dd9883b1581cd63'
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
precision = os.environ['MUSE_PRECISION']
output = Path(os.environ['REPRO_OUTPUT'])
if (precision not in {'bf16', 'mxfp8'} or os.environ['WORLD_SIZE'] != '64'
        or os.environ['LOCAL_WORLD_SIZE'] != '4' or not 0 <= rank < 64 or local_rank != rank % 4
        or torch.version.git_version != TORCH_GIT or torch.version.cuda != '13.4'
        or sha256(Path(activation_offload.__file__).read_bytes()).hexdigest() != OFFLOAD_SHA):
    raise RuntimeError('Unexpected compact GB300 runtime, sources, or rank mapping')

torch.cuda.set_device(local_rank)
device = torch.cuda.get_device_properties(local_rank)
if device.name != 'NVIDIA GB300' or (device.major, device.minor) != (10, 3) or torch.cuda.device_count() != 4:
    raise RuntimeError(f'Expected a four-GPU GB300 node, found {device}')

if precision == 'mxfp8':
    from exact_host_pool import install

    install()  # BF16 installs the same pool as part of its selective H3 binding.

facts = {'rank': rank, 'host': socket.gethostname(), 'device': device.name,
         'compute_capability': [device.major, device.minor], 'nccl_version': torch.cuda.nccl.version(),
         'cuda_visible_gib': device.total_memory / 2**30, 'torch_git_version': TORCH_GIT}
(output / f'gb300-runtime-rank{rank}.json').write_text(json.dumps(facts) + '\n')
launcher = Path(__file__).with_name("launch.py" if precision == "bf16" else "mxfp8_launch.py")
sys.argv[0] = str(launcher)
runpy.run_path(str(launcher), run_name='__main__')
