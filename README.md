# tinyGPT2

A self-contained research framework for pretraining, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF via GRPO) on small GPT-2-style language models. The model is trained to generate short stories in the style of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

The custom architecture (8-layer, 512-dim) is pretrained from scratch. SFT and RLHF use the HuggingFace `openai-community/gpt2` (117M) as the base. All training is tracked with MLflow and models are compared with an interactive Dash dashboard.

See [`docs/codebase.md`](docs/codebase.md) for a detailed explanation of every module and the theory behind each training stage.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (tested on NVIDIA T4, 16 GB VRAM)

```bash
pip install -r requirements.txt
pip install -e .
```

For the embedding reward model used in GRPO, also install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

## Project Structure

```
gpt2tiny/
├── gpt2tiny/           # Core package (model, datasets, trainer, rewards)
├── preprocess.py       # Data preprocessing CLI
├── pretrain.py         # Pretraining script
├── train_sft_hf.py     # SFT script (HuggingFace GPT2)
├── train_grpo_hf.py    # GRPO/RLHF script (HuggingFace GPT2)
├── dashboard.py        # Model comparison dashboard
├── run_pretrain.sh
├── run_train_sft_hf.sh
├── run_train_grpo_hf.sh
├── start_mlflow_server.sh
├── generation/         # Prompt and dataset generation tools
└── docs/               # Documentation
```

---

## Dataset

Data for pretraining is the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) (~470 JSON shards of short stories).

Data for SFT and GRPO consists of structured story prompts paired with GPT-4-generated completions. Prompts specify required vocabulary words (with part-of-speech), narrative features (Dialogue, Conflict, Twist, etc.), and a story subject (character, place, action). The prompt generation tooling lives in [`generation/`](generation/).

---

## 1. Preprocessing

`preprocess.py` is a CLI with subcommands for each training stage. All arguments have sensible defaults; override any of them as needed.

### Pretraining (all-in-one)

Download TinyStories, train a SentencePiece tokenizer, and pretokenize the shards in a single step:

```bash
python preprocess.py prepare-pretrain \
    --vocab-size 4096 \
    --data-dir data/
```

Or run each step individually:

```bash
# Step 1 — download and extract TinyStories (~2 GB)
python preprocess.py download --data-dir data/

# Step 2 — train a 4096-token SentencePiece BPE tokenizer
python preprocess.py train-vocab --vocab-size 4096 --data-dir data/

# Step 3 — tokenize all shards (parallelized across CPU cores)
python preprocess.py pretokenize-pretrain --vocab-size 4096 --data-dir data/
```

After this step you will have `data/TinyStories_all_data/*.bin` alongside the source `.json` shards, and `data/tok4096.model` / `data/tok4096.vocab`.

### Supervised Fine-Tuning

```bash
python preprocess.py pretokenize-sft \
    --tokenizer openai-community/gpt2 \
    --data-dir data/TinyStories_custom_prompts_w_completions_sft_huggingface_gpt2_v1/
```

`--tokenizer` accepts a HuggingFace model name or a local path. `--data-dir` is the directory containing the instruction-story JSON shards. The defaults match the paths expected by `train_sft_hf.py`.

### GRPO / RLHF

```bash
python preprocess.py pretokenize-rlhf \
    --tokenizer openai-community/gpt2 \
    --data-dir data/TinyStories_prompts_huggingface_gpt2_v2/
```

Both commands write `.bin` token files and `indices*.bin` boundary files alongside the source `.json` shards.

---

## 2. Training

### MLflow Server

All training scripts log to MLflow. Start the tracking server before training:

```bash
./start_mlflow_server.sh
```

This starts MLflow on `http://0.0.0.0:5000` using the local `mlruns/` directory as the backend store. Open the UI at `http://localhost:5000` to monitor runs, compare metrics, and browse artifacts.

To use a different directory, edit `start_mlflow_server.sh` and update `--backend-store-uri`.

### Pretraining

Edit `run_pretrain.sh` to set your MLflow experiment name, run prefix, and tokenizer path, then run:

```bash
./run_pretrain.sh
```

Or call the script directly with full control over all hyperparameters:

```bash
python pretrain.py \
    --exp-name tinystories-pretrain \
    --run-prefix pretrain \
    --model-name tinystories-pretrain \
    --tokenizer-path data/tok4096.model \
    --block-size 512 \
    --n-layer 8 \
    --n-head 8 \
    --n-embed 512 \
    --dropout 0.2 \
    --max-steps 20000 \
    --lr 3e-5 \
    --warmup-ratio 0.05 \
    --batch-size 64 \
    --grad-accum-step 4 \
    --grad-clip 1.0 \
    --val-interval 25 \
    --val-batches 50 \
    --log-interval 25 \
    --num-workers 4 \
    --gen-max-tokens 128 \
    --gen-temperature 0.8 \
    --gen-top-k 50 \
    --gen-top-p 0.9
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--exp-name` | required | MLflow experiment name |
| `--run-prefix` | required | Prefix for the MLflow run name |
| `--model-name` | required | Name under which to register the model |
| `--tokenizer-path` | required | Path to the trained `.model` file |
| `--block-size` | 512 | Context window length |
| `--n-layer` | 8 | Number of transformer layers |
| `--n-head` | 8 | Number of attention heads |
| `--n-embed` | 512 | Model hidden dimension |
| `--dropout` | 0.2 | Dropout probability |
| `--max-steps` | 20000 | Total optimiser steps |
| `--lr` | 3e-5 | Peak learning rate |
| `--warmup-ratio` | 0.05 | Fraction of steps for linear warmup |
| `--batch-size` | 64 | Per-GPU batch size |
| `--grad-accum-step` | 4 | Gradient accumulation steps |

Sample text generations are logged to MLflow as text artifacts every 500 validation checks.

### Supervised Fine-Tuning

```bash
./run_train_sft_hf.sh
```

Or directly:

```bash
python train_sft_hf.py \
    --exp-name sft-hf \
    --run-prefix sft-hf \
    --model-name SFTGPT2HF \
    --max-steps 20000 \
    --lr 3e-5 \
    --warmup-ratio 0.05 \
    --batch-size 8 \
    --grad-accum-step 8 \
    --grad-clip 1.0 \
    --val-interval 100 \
    --val-batches 50 \
    --log-interval 100 \
    --num-workers 4 \
    --gen-max-tokens 248
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--exp-name` | required | MLflow experiment name |
| `--run-prefix` | required | Run name prefix |
| `--model-name` | required | MLflow registered model name |
| `--max-steps` | 20000 | Total optimiser steps |
| `--lr` | 3e-5 | Peak learning rate |
| `--batch-size` | 8 | Per-GPU batch size |
| `--grad-accum-step` | 8 | Effective batch = batch_size × grad_accum |

The best checkpoint (lowest `val_loss`) is registered in MLflow under `--model-name`. Sample completions for fixed evaluation prompts are logged after every validation.

### GRPO / RLHF

GRPO fine-tunes an existing SFT checkpoint. You need the path to the best checkpoint from your SFT run. Find it in the MLflow artifact directory or from the end-of-training output.

```bash
./run_train_grpo_hf.sh
```

Or directly:

```bash
python train_grpo_hf.py \
    --init-model-path mlruns/<exp_id>/<run_id>/artifacts/best-step=2300-val_loss=2.08.ckpt \
    --exp-name grpo-hf \
    --run-prefix grpo-hf \
    --model-name GRPOGPT2HF \
    --num-gen 8 \
    --max-seq-len 128 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --kl-beta 0.07 \
    --entropy-beta 0.02 \
    --clip-eps 0.2 \
    --lr 4e-5 \
    --warmup-ratio 0.05 \
    --batch-size 4 \
    --grad-accum-step 8 \
    --max-steps 1000 \
    --val-interval 5 \
    --val-batches 16 \
    --num-workers 4 \
    --gen-max-tokens 128 \
    --rw-words 0.1 \
    --rw-pos 0.1 \
    --rw-coherence 0.4 \
    --rw-format 0.0 \
    --rw-repetition-penalty 0.2 \
    --rw-min-chars 750 \
    --rw-max-chars 2000 \
    --rw-min-sentences 5 \
    --rw-max-sentences 10
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--init-model-path` | required | Path to SFT checkpoint to initialise from |
| `--num-gen` | 8 | Completions sampled per prompt (GRPO group size) |
| `--max-seq-len` | 248 | Max tokens to generate per completion |
| `--temperature` | 1.0 | Sampling temperature during rollout |
| `--kl-beta` | 0.02 | Weight of KL divergence penalty from reference policy |
| `--entropy-beta` | 0.02 | Weight of entropy bonus (encourages exploration) |
| `--clip-eps` | 0.2 | PPO clipping range |

**Reward weights** (`--rw-*`): control how each reward component contributes to the final scalar. Set a weight to 0.0 to disable that component entirely.

| Argument | Default | Reward component |
|---|---|---|
| `--rw-words` | 0.25 | Required vocabulary words present |
| `--rw-pos` | 0.15 | Required words used with correct POS |
| `--rw-subject` | 0.20 | Story subject adherence |
| `--rw-features` | 0.25 | Narrative features (Dialogue, Twist, etc.) |
| `--rw-format` | 0.15 | Story length and sentence count |
| `--rw-coherence` | 0.25 | Gibberish detection / coherence |
| `--rw-prompt-copy-penalty` | 0.10 | Penalises copying the prompt verbatim |
| `--rw-meta-penalty` | 0.10 | Penalises meta-commentary ("this story should...") |
| `--rw-repetition-penalty` | 0.10 | Penalises repeated phrases |
| `--rw-stuffing-penalty` | 0.10 | Penalises overuse of required words |

The best checkpoint (highest `val_avg_reward`) is registered under `--model-name`.

---

## 3. MLflow Server

```bash
./start_mlflow_server.sh
```

Opens the tracking UI at `http://localhost:5000`. From there you can:
- Compare loss and reward curves across runs.
- Browse generated text artifacts logged during training.
- View registered model versions.

To configure a different host or port, edit `start_mlflow_server.sh`:

```bash
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri file:/path/to/mlruns \
    --workers 4
```

---

## 4. Dashboard

The dashboard lets you compare two registered MLflow models side-by-side with a shared prompt.

**Setup.** Create a `.env` file in the project root:

```
MLFLOW_TRACKING_URI=http://localhost:5000
```

**Run:**

```bash
python dashboard.py
```

Open `http://localhost:8050` in a browser. Select a model name and version from the dropdowns for Model A (and optionally Model B), enter a prompt, and click **Generate** or press Enter. Both models respond in parallel. Generation parameters (temperature, top-k, top-p, max tokens) are adjustable in the left panel. The last 10 prompt/response pairs are shown in the History section.

---

## Hardware Notes

Training was developed and tested on a rented LightningAI Studio:

- GPU: NVIDIA T4 (16 GB VRAM)
- CPUs: 4
- RAM: 16 GB

Mixed-precision training (`fp16`) is enabled automatically when CUDA is available. For CPUs or MPS, the scripts fall back to `fp32`.
