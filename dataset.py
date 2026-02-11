import random
import glob
import numpy as np
import torch
import torch.distributed as dist
from typing import Iterator, Tuple, Optional
from pathlib import Path

DEFAULT_DATA_DIR="/teamspace/studios/this_studio/data/TinyStories_all_data/"


class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
        max_seq_len: int,
        split: str = "train",
        data_dir: str = DEFAULT_DATA_DIR,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # worker_info = torch.utils.data.get_worker_info()
        # worker_id = worker_info.id if worker_info else 0
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # seed = 42 + worker_id + 1337 * rank
        bin_dir = Path(self.data_dir)
        shard_filenames = sorted(glob.glob(str(bin_dir / "*.bin")))
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )
        rng = random.Random(self.seed)
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(data) // self.max_seq_len - 1
                idxs = list(range(num_batches))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = idx * self.max_seq_len
                    end = (idx + 1) * self.max_seq_len
                    chunk = torch.from_numpy(data[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


# class PreTokDataset(torch.utils.data.IterableDataset):

#     def __init__(
#         self,
#         max_seq_len,
#         split = "train",
#         seed=None,
#         data_dir=DEFAULT_DATA_DIR,
#     ):
#         super().__init__()
#         self.split = split
#         self.max_seq_len = max_seq_len
#         self.seed = seed
#         self.data_dir = data_dir

#     def __iter__(self):
#         file_shards = glob.glob(f"{self.data_dir}/*.bin")

#         file_shards = file_shards[1:] if self.split=="train" else file_shards[:1]

#         rng = random.Random(self.seed)
#         while True:
#             rng.shuffle(file_shards)
#             for shard in file_shards:
#                 data = np.memmap(shard, dtype=np.uint16, mode="r")
#                 num_batches = (len(data) // self.max_seq_len ) - 1
#                 idxs = list(range(num_batches))
#                 rng.shuffle(idxs)

#                 for idx in idxs:
#                   start = idx * self.max_seq_len
#                   end = (idx + 1) * self.max_seq_len
#                   seq = torch.from_numpy(data[start:end].astype(np.int64))
#                   x, y = seq[:-1], seq[1:]
#                   yield x, y