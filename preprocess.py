import os
import re
import argparse
import json
import requests
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sentencepiece as spm
import glob

from gpt2tiny.tokenizer import Tokenizer

DATA_CACHE_DIR = Path("data")
DATA_CACHE_DIR.mkdir(exist_ok=True)


def download_file(url: str, filename: str, chunk_size: int = 1024) -> None:
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)


def download() -> None:
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = DATA_CACHE_DIR / "TinyStories_all_data.tar.gz"

    if not data_filename.exists():
        print("Downloading TinyStories dataset...")
        download_file(data_url, str(data_filename))

    data_dir = DATA_CACHE_DIR / "TinyStories_all_data"
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print("Extracting TinyStories dataset...")
        os.system(f"tar -xvf {data_filename} -C {data_dir}")


def download_math(nshard: int = 4) -> None:

    data_url="https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json"
    data_filename = DATA_CACHE_DIR / "MetaMathQA-395K.json"

    if not data_filename.exists():
        print("Downloading MetaMathQA dataset...")
        download_file(data_url, str(data_filename))

    data_dir = DATA_CACHE_DIR / "MetaMathQA"    
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print("Extracting MetaMathQA dataset...")
        with open(data_filename, "r") as g:
            data = json.load(g)

        nsize = int(np.ceil(len(data)/nshard))
        for idx in range(nshard):
            with open(data_dir / f"data{idx:02d}.json","w") as f:
                json.dump(data[(idx*nsize):((idx+1)*nsize)], f)


def train_vocab(vocab_size: int) -> None:
    prefix = DATA_CACHE_DIR / f"tok{vocab_size}"
    tiny_file = DATA_CACHE_DIR / "tiny.txt"

    # data_dir = DATA_CACHE_DIR / "TinyStories_all_data"
    # shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))

    data_dir = DATA_CACHE_DIR / "MetaMathQA"
    math_filenames = sorted(glob.glob(str(data_dir / "*.json")))

    with open(tiny_file, "w") as f:
        # for shard in shard_filenames[:10]:
        #     with open(shard, "r") as g:
        #         data = json.load(g)
        #     for example in data:
        #         f.write(example["story"].strip() + "\n")

        for shard in math_filenames:
            with open(shard, "r") as g:
                data = json.load(g)
            for example in data:
                f.write(example["query"].strip() + "\n")    
                f.write(example["response"].strip() + "\n")    
    
    spm.SentencePieceTrainer.train(
        input=str(tiny_file),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r"\342\201\207 ",
        normalization_rule_name="identity",
    )


# def process_shard(args: tuple, vocab_size: int) -> None:
#     shard_id, shard = args
#     tokenizer_model = DATA_CACHE_DIR / f"tok{vocab_size}.model"
#     tokenizer = Tokenizer(str(tokenizer_model))

#     with open(shard, "r") as f:
#         data = json.load(f)

#     all_tokens = []
#     for example in tqdm(data, position=shard_id):
#         if "story" in example:
#             text = example["story"].strip()
#         else:
#             text = example["response"].strip()
            
#         tokens = tokenizer.encode(text, bos=True, eos=True)
#         all_tokens.extend(tokens)

#     all_tokens = np.array(all_tokens, dtype=np.uint16)
#     tokenized_filename = str(shard).replace(".json", ".bin")

#     with open(tokenized_filename, "wb") as f:
#         f.write(all_tokens.tobytes())


def process_shard(args: tuple, vocab_size: int) -> None:
    shard_id, shard = args
    tokenizer_model = DATA_CACHE_DIR / f"tok{vocab_size}.model"
    tokenizer = Tokenizer(str(tokenizer_model))

    with open(shard, "r") as f:
        data = json.load(f)

    all_tokens = []
    qa_indices = [0]
    idx = 0
    for example in tqdm(data, position=shard_id):

        text = example["query"].strip()  
        q_tokens = tokenizer.encode(text, bos=True, eos=True)
        q_len = len(q_tokens)
        
        text = example["response"].strip()  
        a_tokens = tokenizer.encode(text, bos=True, eos=True)
        a_len = len(a_tokens)

        if q_len + a_len <= 512:
            all_tokens.extend(q_tokens + a_tokens)
            q_len += idx
            a_len += q_len
            idx = a_len
            qa_indices.extend([q_len, a_len])

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    qa_indices = np.array(qa_indices, dtype=np.uint32)
    
    tokenized_filename = str(shard).replace(".json", ".bin")    
    
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    indices_filename = re.sub(r"data(\d{2})\.json$", r"indices\1.bin", str(shard))

    with open(indices_filename, "wb") as f:
        f.write(qa_indices.tobytes())


def pretokenize(vocab_size: int) -> None:
    # data_dir = DATA_CACHE_DIR / "TinyStories_all_data"
    # shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))

    data_dir = DATA_CACHE_DIR / "MetaMathQA"
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    
    func = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))


def prepare_dataset(vocab_size: int) -> None:
    print("Step 1: Downloading dataset...")
    # download()
    download_math()
    print("\nStep 2: Training vocabulary...")
    train_vocab(vocab_size)
    print("\nStep 3: Pretokenizing dataset...")
    pretokenize(vocab_size)
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MetaMathQA datasets")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    download_parser = subparsers.add_parser(
        "download", help="Download MetaMathQA datasets"
    )

    vocab_parser = subparsers.add_parser("train-vocab", help="Train vocabulary")
    vocab_parser.add_argument(
        "--vocab-size", type=int, required=True, help="Size of vocabulary to train"
    )

    pretok_parser = subparsers.add_parser("pretokenize", help="Pretokenize the dataset")
    pretok_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size to use for tokenization",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-dataset", help="Run all dataset preparation steps sequentially"
    )
    prepare_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size for training and tokenization",
    )

    args = parser.parse_args()

    if args.command == "download":
        download()
        download_math()
    elif args.command == "train-vocab":
        train_vocab(args.vocab_size)
    elif args.command == "pretokenize":
        pretokenize(args.vocab_size)
    elif args.command == "prepare-dataset":
        prepare_dataset(args.vocab_size)
    else:
        parser.print_help()