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
from transformers import AutoTokenizer

from gpt2tiny.tokenizer import Tokenizer


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


# ---------------------------------------------------------------------------
# Pretrain helpers
# ---------------------------------------------------------------------------

def download(data_dir: Path) -> None:
    """Download and extract TinyStories into data_dir/TinyStories_all_data/."""
    data_dir.mkdir(parents=True, exist_ok=True)
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = data_dir / "TinyStories_all_data.tar.gz"

    if not data_filename.exists():
        print("Downloading TinyStories dataset...")
        download_file(data_url, str(data_filename))

    extract_dir = data_dir / "TinyStories_all_data"
    if not extract_dir.exists():
        extract_dir.mkdir(exist_ok=True)
        print("Extracting TinyStories dataset...")
        os.system(f"tar -xvf {data_filename} -C {extract_dir}")


def train_vocab(vocab_size: int, data_dir: Path) -> None:
    """Train a SentencePiece tokenizer on TinyStories shards in data_dir.

    Reads shards from data_dir/TinyStories_all_data/ and saves the tokenizer
    model/vocab to data_dir/tok{vocab_size}.{model,vocab}.
    """
    prefix = data_dir / f"tok{vocab_size}"
    tiny_file = data_dir / "tiny.txt"

    shard_dir = data_dir / "TinyStories_all_data"
    shard_filenames = sorted(glob.glob(str(shard_dir / "*.json")))

    with open(tiny_file, "w") as f:
        for shard in shard_filenames[:10]:
            with open(shard, "r") as g:
                data = json.load(g)
            for example in data:
                f.write(example["story"].strip() + "\n")

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


def process_shard_for_pretrain(args: tuple, vocab_size: int, data_dir: Path) -> None:
    shard_id, shard = args
    tokenizer_model = data_dir / f"tok{vocab_size}.model"
    tokenizer = Tokenizer(str(tokenizer_model))

    with open(shard, "r") as f:
        data = json.load(f)

    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example.get("story", example.get("response", "")).strip()
        tokens = tokenizer.encode(text, bos=True, eos=True)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    tokenized_filename = str(shard).replace(".json", ".bin")

    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())


def pretokenize_pretrain(vocab_size: int, data_dir: Path) -> None:
    shard_dir = data_dir / "TinyStories_all_data"
    shard_filenames = sorted(glob.glob(str(shard_dir / "*.json")))
    print(f"Found {len(shard_filenames)} shards in {shard_dir}")
    func = partial(process_shard_for_pretrain, vocab_size=vocab_size, data_dir=data_dir)
    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))


# ---------------------------------------------------------------------------
# Shared HF tokenizer helper
# ---------------------------------------------------------------------------

def _hf_encode(tokenizer: AutoTokenizer, text: str, bos: bool, eos: bool) -> list:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if bos and tokenizer.bos_token_id is not None:
        tokens = [tokenizer.bos_token_id] + tokens
    if eos and tokenizer.eos_token_id is not None:
        tokens = tokens + [tokenizer.eos_token_id]
    return tokens


# ---------------------------------------------------------------------------
# SFT helpers
# ---------------------------------------------------------------------------

def process_shard_for_sft(args: tuple, tokenizer_name: str) -> None:
    shard_id, shard = args
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(shard, "r") as f:
        data = json.load(f)

    all_tokens = []
    qa_indices = [0]
    idx = 0
    for example in tqdm(data, position=shard_id):
        try:
            text = example["instruction"]["prompt"]
        except KeyError:
            text = example["instruction"]["prompt:"]
        q_tokens = _hf_encode(tokenizer, text, bos=True, eos=False)
        q_len = len(q_tokens)

        text = example["story"].strip()
        a_tokens = _hf_encode(tokenizer, text, bos=False, eos=True)
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


def pretokenize_sft(tokenizer_name: str, data_dir: Path) -> None:
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    print(f"Found {len(shard_filenames)} shards in {data_dir}")
    func = partial(process_shard_for_sft, tokenizer_name=tokenizer_name)
    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))


# ---------------------------------------------------------------------------
# RLHF helpers
# ---------------------------------------------------------------------------

def process_shard_for_rlhf(args: tuple, tokenizer_name: str) -> None:
    shard_id, shard = args
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(shard, "r") as f:
        data = json.load(f)

    all_tokens = []
    all_indices = [0]
    idx = 0
    for example in tqdm(data, position=shard_id):
        text = example["prompt"]
        p_tokens = _hf_encode(tokenizer, text, bos=True, eos=False)
        p_len = len(p_tokens)

        if p_len <= 248:
            all_tokens.extend(p_tokens)
            idx += p_len
            all_indices.append(idx)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    all_indices = np.array(all_indices, dtype=np.uint32)

    tokenized_filename = str(shard).replace(".json", ".bin")
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    indices_filename = re.sub(r"data(\d{2})\.json$", r"indices\1.bin", str(shard))
    with open(indices_filename, "wb") as f:
        f.write(all_indices.tobytes())


def pretokenize_rlhf(tokenizer_name: str, data_dir: Path) -> None:
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    print(f"Found {len(shard_filenames)} shards in {data_dir}")
    func = partial(process_shard_for_rlhf, tokenizer_name=tokenizer_name)
    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preprocessing for pretrain, SFT, and RLHF")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Pretrain: download ---
    download_parser = subparsers.add_parser("download", help="Download TinyStories dataset for pretraining")
    download_parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory where the dataset will be downloaded and extracted (default: data/)",
    )

    # --- Pretrain: train-vocab ---
    vocab_parser = subparsers.add_parser("train-vocab", help="Train a SentencePiece tokenizer on TinyStories")
    vocab_parser.add_argument(
        "--vocab-size", type=int, default=4096,
        help="Vocabulary size (default: 4096)",
    )
    vocab_parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory containing TinyStories_all_data/ shards; tokenizer is saved here too (default: data/)",
    )

    # --- Pretrain: pretokenize ---
    pretok_pretrain_parser = subparsers.add_parser(
        "pretokenize-pretrain",
        help="Pretokenize TinyStories using the trained SentencePiece tokenizer",
    )
    pretok_pretrain_parser.add_argument(
        "--vocab-size", type=int, default=4096,
        help="Vocabulary size matching the trained tokenizer (default: 4096)",
    )
    pretok_pretrain_parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory containing TinyStories_all_data/ shards and the tokenizer model; .bin files are written alongside the shards (default: data/)",
    )

    # --- Pretrain: prepare (download + train-vocab + pretokenize-pretrain) ---
    prepare_parser = subparsers.add_parser(
        "prepare-pretrain",
        help="Run the full pretrain pipeline: download, train-vocab, pretokenize-pretrain",
    )
    prepare_parser.add_argument(
        "--vocab-size", type=int, default=4096,
        help="Vocabulary size (default: 4096)",
    )
    prepare_parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory for downloading, tokenizer training, and pretokenization (default: data/)",
    )

    # --- SFT ---
    pretok_sft_parser = subparsers.add_parser("pretokenize-sft", help="Pretokenize dataset for SFT")
    pretok_sft_parser.add_argument(
        "--tokenizer", type=str, default="gpt2",
        help="HuggingFace model name or local path to tokenizer (default: gpt2)",
    )
    pretok_sft_parser.add_argument(
        "--data-dir", type=Path, default=Path("data/TinyStories_custom_prompts_w_completions_sft_huggingface_gpt2_v1"),
        help="Directory containing the JSON shards to pretokenize (default: data/TinyStories_custom_prompts_w_completions_sft_huggingface_gpt2_v1/)",
    )

    # --- RLHF ---
    pretok_rlhf_parser = subparsers.add_parser("pretokenize-rlhf", help="Pretokenize dataset for RLHF")
    pretok_rlhf_parser.add_argument(
        "--tokenizer", type=str, default="gpt2",
        help="HuggingFace model name or local path to tokenizer (default: gpt2)",
    )
    pretok_rlhf_parser.add_argument(
        "--data-dir", type=Path, default=Path("data/TinyStories_prompts_huggingface_gpt2_v2"),
        help="Directory containing the JSON shards to pretokenize (default: data/TinyStories_prompts_huggingface_gpt2_v2/)",
    )

    args = parser.parse_args()

    if args.command == "download":
        download(args.data_dir)
    elif args.command == "train-vocab":
        train_vocab(args.vocab_size, args.data_dir)
    elif args.command == "pretokenize-pretrain":
        pretokenize_pretrain(args.vocab_size, args.data_dir)
    elif args.command == "pretokenize-sft":
        pretokenize_sft(args.tokenizer, args.data_dir)
    elif args.command == "pretokenize-rlhf":
        pretokenize_rlhf(args.tokenizer, args.data_dir)
    elif args.command == "prepare-pretrain":
        print("Step 1: Downloading dataset...")
        download(args.data_dir)
        print("\nStep 2: Training vocabulary...")
        train_vocab(args.vocab_size, args.data_dir)
        print("\nStep 3: Pretokenizing dataset...")
        pretokenize_pretrain(args.vocab_size, args.data_dir)
        print("\nDone.")
    else:
        parser.print_help()
