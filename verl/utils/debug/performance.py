# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
import logging


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_stats = torch.cuda.mem_get_info(device)
        memory_free = memory_stats[0] / 1024**3
        memory_total = memory_stats[1] / 1024**3
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

        message = (
            f'{head}, memory allocated (GB): {memory_allocated}, '
            f'memory reserved (GB): {memory_reserved}, '
            f'memory free (GB): {memory_free}, '
            f'memory total (GB): {memory_total}, '
            f'max allocated (GB): {max_memory_allocated}, '
            f'max reserved (GB): {max_memory_reserved}'
        )

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)
