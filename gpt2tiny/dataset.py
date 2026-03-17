import re
import random
import glob
import json
import numpy as np
from itertools import groupby

import torch
import torch.distributed as dist
from typing import Iterator, Tuple, Optional, List
from pathlib import Path

DEFAULT_DATA_DIR = "/teamspace/studios/this_studio/gpt2tiny/data/TinyStories_all_data/"
# print("__file__: ", __file__)


def find_index(t):
    match = re.search(r"(\d+)\b", t)
    if match:
        return match.group()
    else:
        return None


def grab_files(filenames, fn_templates):
    fns = []
    for fnt in fn_templates:
        stem, suffix = fnt.split(".")
        for fn in filenames:
            if re.search(rf"({stem})(\d+).({suffix})", fn):
                fns.append(fn)
    if len(fn_templates)==1:
        return fns[0]
    else:
        return tuple(fns)


def collect_filenames(directory, split):
    bin_dir = Path(directory)
    grps = groupby(
        sorted(
            glob.glob(str(bin_dir / "*.*")),
            key=lambda x: find_index(Path(x).stem)
        ),
        key=lambda x: find_index(Path(x).stem)
    )
    shard_fns = [sorted([x for x in g]) for _, g in grps]
    shard_fns = shard_fns[:-1] if split == "train" else shard_fns[-1:]
    return shard_fns


def locate_shards(
    data_dir: str|Path|List[str|Path],
    split: str,
    fn_templates: Tuple[str],
    weights: Optional[str|List[int]] = None,
) -> List[List[str]]:
    if isinstance(data_dir, list):
        shard_filenames = []
        for _data_dir in data_dir:
            shard_filenames.append(collect_filenames(_data_dir, split))
        if weights == "Balanced":
            num_shard_files = list(map(len, shard_filenames))
            max_num_shard_files = max(num_shard_files)

            weights = [int(max_num_shard_files / n) for n in num_shard_files]
        elif isinstance(weights, list):
            weights = list(weights)
        else:
            weights = len(shard_filenames) * [1]

        shard_filenames = [
            fn
            for w, fns in zip(weights, shard_filenames)
            for _ in range(w)
            for fn in fns
        ]
    else:
        shard_filenames = collect_filenames(data_dir, split)

    shard_filenames = list(map(lambda x: grab_files(x, fn_templates), shard_filenames))
    
    return shard_filenames

    
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
        self.shard_filenames = locate_shards(
            self.data_dir,
            self.split,
            ("data.bin", ),
            self.weights,
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        while True:
            rng.shuffle(self.shard_filenames)
            for shard in self.shard_filenames:
                
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
        self.shard_filenames = locate_shards(
            self.data_dir,
            self.split,
            ("data.bin", "indices.bin", ),
            self.weights,
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        while True:
            rng.shuffle(self.shard_filenames)
            for shard, indices in self.shard_filenames:
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

    @staticmethod
    def collator(batch, pad_id=0):
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
        self.shard_filenames = locate_shards(
            self.data_dir,
            self.split,
            ("data.bin", "indices.bin", ),
            self.weights,
        )
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:

        rng = random.Random(self.seed)
        while True:
            rng.shuffle(self.shard_filenames)
            for shard, indices in self.shard_filenames:
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
        self.shard_filenames = locate_shards(
            self.data_dir,
            self.split,
            ("data.bin", "indices.bin", "data.json",),
            self.weights,
        )
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        while True:
            rng.shuffle(self.shard_filenames)
            for shard_fn, pos_fn, metadata_fn in self.shard_filenames:
                data = np.memmap(shard_fn, dtype=np.uint16, mode="r")
                pos = np.memmap(pos_fn, dtype=np.uint32, mode="r")

                with open(metadata_fn, "r") as f:
                    metadata = json.load(f)
                    
                num_p = int(len(pos) - 1)
                idxs = list(range(num_p))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = pos[idx]
                    end = pos[idx + 1]
                    prompt = torch.from_numpy(data[start:end].astype(np.int64)).to(self.device)
                    yield prompt, metadata[idx]

    @staticmethod
    def collator(tokens, max_seq_len=None, pad_id=0, device=None):
        if device is None:
            device = tokens[0][0].device
        if max_seq_len is None:
            max_seq_len = max([len(tok) for tok, _ in tokens])
        else:
            assert max([len(tok) for tok, _ in tokens]) < max_seq_len, "max_seq_len must be longer than the longest sequence in `tokens`"
        seq = torch.full((len(tokens), max_seq_len), fill_value=pad_id, dtype=torch.long)
        msk = torch.zeros_like(seq)
        lns = torch.zeros_like(seq[:, 0])
    
        metadata = []
        for i, (tok, md) in enumerate(tokens):
            seq[i, :len(tok)] = tok
            msk[i, :len(tok)] = 1
            lns[i] = len(tok)
            metadata.append(md)
        return seq.to(device), msk.to(device), lns.to(device), metadata



