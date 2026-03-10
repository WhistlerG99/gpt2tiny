import random
import glob
import numpy as np
import torch
import torch.distributed as dist
from typing import Iterator, Tuple, Optional, List
from pathlib import Path

DEFAULT_DATA_DIR = "/teamspace/studios/this_studio/gpt2tiny/data/TinyStories_all_data/"
# print("__file__: ", __file__)


class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        max_seq_len: int,
        split: str = "train",
        data_dir: str = DEFAULT_DATA_DIR,
        weights: Optional[str | List[int]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.weights = weights
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # worker_info = torch.utils.data.get_worker_info()
        # worker_id = worker_info.id if worker_info else 0
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # seed = 42 + worker_id + 1337 * rank
        if isinstance(self.data_dir, list):
            shard_filenames = []
            for data_dir in self.data_dir:

                bin_dir = Path(data_dir)
                fns = sorted(glob.glob(str(bin_dir / "data*.bin")))
                fns = fns[1:] if self.split == "train" else fns[:1]
                shard_filenames.append(fns)

            if self.weights == "Balanced":
                num_shard_files = list(map(len, shard_filenames))
                max_num_shard_files = max(num_shard_files)

                weights = [int(max_num_shard_files / n) for n in num_shard_files]
            elif isinstance(self.weights, list):
                weights = self.weights
            else:
                weights = len(shard_filenames) * [1]

            shard_filenames = [
                fn
                for w, fns in zip(weights, shard_filenames)
                for _ in range(w)
                for fn in fns
            ]
        else:
            bin_dir = Path(self.data_dir)
            shard_filenames = sorted(glob.glob(str(bin_dir / "data*.bin")))
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


class SFTDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_dir: str | List[str],
        split: str = "train",
        weights: Optional[str | List[int]] = None,
        seed: Optional[int] = None,
        device=None,
    ):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.weights = weights
        self.seed = seed
        self.device = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # worker_info = torch.utils.data.get_worker_info()
        # worker_id = worker_info.id if worker_info else 0
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # seed = 42 + worker_id + 1337 * rank
        if isinstance(self.data_dir, list):
            shard_filenames = []
            for data_dir in self.data_dir:

                bin_dir = Path(data_dir)
                shard_fns = sorted(glob.glob(str(bin_dir / "data*.bin")))
                index_fns = sorted(glob.glob(str(bin_dir / "indices*.bin")))
                fns = (
                    list(zip(shard_fns[:-1], index_fns[:-1]))
                    if self.split == "train"
                    else list(zip(shard_fns[-1:], index_fns[-1:]))
                )
                shard_filenames.append(fns)

            if self.weights == "Balanced":
                num_shard_files = list(map(len, shard_filenames))
                max_num_shard_files = max(num_shard_files)

                weights = [int(max_num_shard_files / n) for n in num_shard_files]
            elif isinstance(self.weights, list):
                weights = self.weights
            else:
                weights = len(shard_filenames) * [1]

            shard_filenames = [
                fn
                for w, fns in zip(weights, shard_filenames)
                for _ in range(w)
                for fn in fns
            ]
        else:
            bin_dir = Path(self.data_dir)
            shard_fns = sorted(glob.glob(str(bin_dir / "data*.bin")))
            index_fns = sorted(glob.glob(str(bin_dir / "indices*.bin")))
            shard_filenames = (
                list(zip(shard_fns[:-1], index_fns[:-1]))
                if self.split == "train"
                else list(zip(shard_fns[-1:], index_fns[-1:]))
            )
        rng = random.Random(self.seed)
        while True:
            rng.shuffle(shard_filenames)
            for shard, indices in shard_filenames:
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                pos = np.memmap(indices, dtype=np.uint32, mode="r")

                num_qa = int((len(pos) - 1) / 2)
                idxs = list(range(num_qa))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = pos[2 * idx]
                    mid = pos[2 * idx + 1]
                    end = pos[2 * idx + 2]
                    qa = torch.from_numpy(data[start:end].astype(np.int64))
                    q_len = mid - start
                    yield qa, q_len


def collator(batch, pad_id):
    seq_len = max([x.shape[0] for x, _ in batch])

    seq = torch.full((len(batch), seq_len), fill_value=pad_id, dtype=torch.long)
    msk = torch.zeros_like(seq)
    lns = torch.zeros_like(seq[:, 0])

    for i, (qa, q_len) in enumerate(batch):
        seq[i, : len(qa)] = qa
        msk[i, : len(qa)] = 1
        lns[i] = q_len
    return seq, msk, lns


class RLHFDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_dir: str | List[str],
        split: str = "train",
        weights: Optional[str | List[int]] = None,
        seed: Optional[int] = None,
        device=None,
    ):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.weights = weights
        self.seed = seed
        self.device = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # worker_info = torch.utils.data.get_worker_info()
        # worker_id = worker_info.id if worker_info else 0
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # seed = 42 + worker_id + 1337 * rank
        if isinstance(self.data_dir, list):
            shard_filenames = []
            for data_dir in self.data_dir:

                bin_dir = Path(data_dir)
                shard_fns = sorted(glob.glob(str(bin_dir / "data*.bin")))
                index_fns = sorted(glob.glob(str(bin_dir / "indices*.bin")))
                fns = (
                    list(zip(shard_fns[:-1], index_fns[:-1]))
                    if self.split == "train"
                    else list(zip(shard_fns[-1:], index_fns[-1:]))
                )
                shard_filenames.append(fns)

            if self.weights == "Balanced":
                num_shard_files = list(map(len, shard_filenames))
                max_num_shard_files = max(num_shard_files)

                weights = [int(max_num_shard_files / n) for n in num_shard_files]
            elif isinstance(self.weights, list):
                weights = self.weights
            else:
                weights = len(shard_filenames) * [1]

            shard_filenames = [
                fn
                for w, fns in zip(weights, shard_filenames)
                for _ in range(w)
                for fn in fns
            ]
        else:
            bin_dir = Path(self.data_dir)
            shard_fns = sorted(glob.glob(str(bin_dir / "data*.bin")))
            index_fns = sorted(glob.glob(str(bin_dir / "indices*.bin")))
            shard_filenames = (
                list(zip(shard_fns[:-1], index_fns[:-1]))
                if self.split == "train"
                else list(zip(shard_fns[-1:], index_fns[-1:]))
            )
        rng = random.Random(self.seed)
        while True:
            rng.shuffle(shard_filenames)
            for shard, indices in shard_filenames:
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                pos = np.memmap(indices, dtype=np.uint32, mode="r")

                num_qa = int((len(pos) - 1) / 2)
                idxs = list(range(num_qa))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = pos[2 * idx]
                    mid = pos[2 * idx + 1]
                    end = pos[2 * idx + 2]
                    qa = torch.from_numpy(data[start:end].astype(np.int64)).to(self.device)
                    q_len = mid - start
                    yield qa[: q_len - 1]


class PromptDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_dir: str | List[str],
        split: str = "train",
        weights: Optional[str | List[int]] = None,
        seed: Optional[int] = None,
        device=None,
    ):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.weights = weights
        self.seed = seed
        self.device = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # worker_info = torch.utils.data.get_worker_info()
        # worker_id = worker_info.id if worker_info else 0
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # seed = 42 + worker_id + 1337 * rank
        if isinstance(self.data_dir, list):
            shard_filenames = []
            for data_dir in self.data_dir:

                bin_dir = Path(data_dir)
                shard_fns = sorted(glob.glob(str(bin_dir / "data*.bin")))
                index_fns = sorted(glob.glob(str(bin_dir / "indices*.bin")))
                fns = (
                    list(zip(shard_fns[:-1], index_fns[:-1]))
                    if self.split == "train"
                    else list(zip(shard_fns[-1:], index_fns[-1:]))
                )
                shard_filenames.append(fns)

            if self.weights == "Balanced":
                num_shard_files = list(map(len, shard_filenames))
                max_num_shard_files = max(num_shard_files)

                weights = [int(max_num_shard_files / n) for n in num_shard_files]
            elif isinstance(self.weights, list):
                weights = self.weights
            else:
                weights = len(shard_filenames) * [1]

            shard_filenames = [
                fn
                for w, fns in zip(weights, shard_filenames)
                for _ in range(w)
                for fn in fns
            ]
        else:
            bin_dir = Path(self.data_dir)
            shard_fns = sorted(glob.glob(str(bin_dir / "data*.bin")))
            index_fns = sorted(glob.glob(str(bin_dir / "indices*.bin")))
            shard_filenames = (
                list(zip(shard_fns[:-1], index_fns[:-1]))
                if self.split == "train"
                else list(zip(shard_fns[-1:], index_fns[-1:]))
            )
        rng = random.Random(self.seed)
        while True:
            rng.shuffle(shard_filenames)
            for shard, indices in shard_filenames:
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                pos = np.memmap(indices, dtype=np.uint32, mode="r")

                num_p = int(len(pos) - 1)
                idxs = list(range(num_p))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = pos[idx]
                    end = pos[idx + 1]
                    prompt = torch.from_numpy(data[start:end].astype(np.int64)).to(self.device)
                    yield prompt

def prompt_collator(tokens, max_seq_len=None, pad_id=0, device=None):
    if device is None:
        device = tokens[0].device
    if max_seq_len is None:
        max_seq_len = max([len(tok) for tok in tokens])
    else:
        assert max([len(tok) for tok in tokens]) < max_seq_len, "max_seq_len must be longer than the longest sequence in `tokens`"
    seq = torch.full((len(tokens), max_seq_len), fill_value=pad_id, dtype=torch.long)
    msk = torch.zeros_like(seq)
    lns = torch.zeros_like(seq[:, 0])

    for i, tok in enumerate(tokens):
        seq[i, :len(tok)] = tok
        msk[i, :len(tok)] = 1
        lns[i] = len(tok)
    return seq.to(device), msk.to(device), lns.to(device)