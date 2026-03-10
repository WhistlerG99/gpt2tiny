import argparse
import torch
import re
import json
import glob
from typing import List 
from gpt2tiny.tokenizer import Tokenizer
from gpt2tiny.prompter import PromptGenerator, FEATURE_VARIATIONS
from gpt2tiny.dataset import PromptDataset
from pathlib import Path
import numpy as np
from tqdm import tqdm


BASE_DIR = "/teamspace/studios/this_studio/gpt2tiny"
DATA_CACHE_DIR = Path(BASE_DIR) / "data"



def main(
    assets: List[str],
    tokenizer: Tokenizer,
    output_dir: str|Path,
    num_shards: int = 10,
    prompts_per_shard: int = 10_000,
    min_words: int = 0,
    max_words: int = 4,
    min_features: int = 0,
    max_features: int = 3,
    require_subject: bool = False,
    max_prompt_len: int = 128,
):
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    generator = PromptGenerator()

    for i in range(num_shards):
        print(f"i = {i+1:02d}: Generating new batch of prompts")
        all_prompts = generator.generate_many_from_pos_lists(
            n=prompts_per_shard,
            allow_no_subject=(require_subject==False),
            min_words=min_words,
            max_words=max_words,
            min_features=min_features,
            max_features=max_features,
            **assets,
        )

        prompts = []
        tokens = []
        indices = [0]
        idx = 0
        for p in tqdm(all_prompts, position=i):
            p_tokens = tokenizer.encode(p["prompt"], bos=True, eos=False)
        
            p_len = len(p_tokens)
        
            if p_len <= max_prompt_len:
                prompts.append(p)
                tokens.extend(p_tokens)
                p_len += idx
                idx = p_len
                indices.append(p_len)
        
        tokens = np.array(tokens, dtype=np.uint16)
        indices = np.array(indices, dtype=np.uint32)

        with open(output_dir / f"data{i:02d}.json", "w") as f:
            json.dump(prompts, f)
        
        with open(output_dir / f"data{i:02d}.bin", "wb") as f:
            f.write(tokens.tobytes())
    
        with open(output_dir / f"indices{i:02d}.bin", "wb") as f:
            f.write(indices.tobytes())
        print(f"i = {i+1:02d}: Number of prompts {len(indices)-1}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate Random Story Prompts")
    parser.add_argument(
        "--tokenizer", type=str, default=f"{BASE_DIR}/data/tok4096_tinystories.model", help="Path and file name of tokenizer"
    )

    parser.add_argument(
        "--assets", type=str, default="./assets", help="Path to asset files."
    ) 

    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory"
    )     
    
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--prompts-per-shard", type=int, default=10_000)
    parser.add_argument("--min-words", type=int, default=0)
    parser.add_argument("--max-words", type=int, default=4)
    parser.add_argument("--min-features", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=3)
    parser.add_argument("--require-subject", action="store_true")    
    parser.add_argument("--max-prompt-len", type=int, default=128)

    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer)

    asset_files = map(Path, glob.glob((Path(args.assets) / "*.json").as_posix()))    
    assets = {}
    for fn in asset_files:
        with open(fn, "r") as f:
            assets[fn.stem] = json.load(f)
    print(
        f"""
        Reading assets from: {args.assets},
        Using tokenizer: {args.tokenizer},
        Saving generated prompts in: {args.output_dir},
        -----------
        num_shards = {args.num_shards},
        prompts_per_shard = {args.prompts_per_shard},
        min_words = {args.min_words},
        max_words = {args.max_words},
        min_features = {args.min_features},
        max_features = {args.max_features},
        require_subject = {args.require_subject},
        max_prompt_len = {args.max_prompt_len},
    """
    )
    main(
        assets,
        tokenizer,
        args.output_dir,
        num_shards = args.num_shards,
        prompts_per_shard = args.prompts_per_shard,
        min_words = args.min_words,
        max_words = args.max_words,
        min_features = args.min_features,
        max_features = args.max_features,
        require_subject = args.require_subject,
        max_prompt_len = args.max_prompt_len,
    )






