# Codebase Documentation

This document gives a detailed explanation of every module in the repo, along with the underlying theory for pretraining, supervised fine-tuning, and reinforcement learning from human feedback via GRPO.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Theory Background](#3-theory-background)
   - 3.1 [Language Modelling and Pretraining](#31-language-modelling-and-pretraining)
   - 3.2 [Supervised Fine-Tuning (SFT)](#32-supervised-fine-tuning-sft)
   - 3.3 [Reinforcement Learning from Human Feedback — GRPO](#33-reinforcement-learning-from-human-feedback--grpo)
4. [Data Pipeline — `preprocess.py`](#4-data-pipeline--preprocesspy)
5. [Model Architecture — `gpt2tiny/model.py`](#5-model-architecture--gpt2tinymodelpy)
   - 5.1 [GPTConfig](#51-gptconfig)
   - 5.2 [Embeddings and Positional Encoding](#52-embeddings-and-positional-encoding)
   - 5.3 [RMSNorm](#53-rmsnorm)
   - 5.4 [Causal Self-Attention](#54-causal-self-attention)
   - 5.5 [SwiGLU Feed-Forward Network](#55-swiglu-feed-forward-network)
   - 5.6 [Transformer Block (BlockFFN)](#56-transformer-block-blockffn)
   - 5.7 [Mixture-of-Experts (BlockMOE)](#57-mixture-of-experts-blockmoe)
   - 5.8 [GPT2 — Full Model](#58-gpt2--full-model)
   - 5.9 [Generation Sampling Strategies](#59-generation-sampling-strategies)
6. [Dataset Classes — `gpt2tiny/dataset.py`](#6-dataset-classes--gpt2tinydatasetpy)
   - 6.1 [Shard Discovery Utilities](#61-shard-discovery-utilities)
   - 6.2 [PreTokDataset](#62-pretokdataset)
   - 6.3 [SFTDataset](#63-sftdataset)
   - 6.4 [RLHFDataset and PromptDataset](#64-rlhfdataset-and-promptdataset)
7. [Tokenizer — `gpt2tiny/tokenizer.py`](#7-tokenizer--gpt2tinytokenizerpy)
8. [Training Modules — `gpt2tiny/trainer.py`](#8-training-modules--gpt2tinytrainerpy)
   - 8.1 [PreTrainGPT2Module](#81-pretraingpt2module)
   - 8.2 [SFTGPT2Module (custom model)](#82-sftgpt2module-custom-model)
   - 8.3 [GPT2SFTModule (HuggingFace model)](#83-gpt2sftmodule-huggingface-model)
   - 8.4 [GRPOGPT2Module / GPT2GRPOModule](#84-grpogpt2module--gpt2grpomodule)
   - 8.5 [Learning Rate Schedule](#85-learning-rate-schedule)
9. [Reward Models — `gpt2tiny/rewards/`](#9-reward-models--gpt2tinyrewards)
   - 9.1 [Heuristic Reward — `heuristic.py`](#91-heuristic-reward--heuristicpy)
   - 9.2 [Embedding Reward — `embedding.py`](#92-embedding-reward--embeddingpy)
   - 9.3 [Reward Scoring Pipeline](#93-reward-scoring-pipeline)
   - 9.4 [GRPO Advantage Computation](#94-grpo-advantage-computation)
10. [Callbacks — `gpt2tiny/callbacks.py`](#10-callbacks--gpt2tinycallbackspy)
11. [Prompt Generation — `gpt2tiny/prompter.py`](#11-prompt-generation--gpt2tinypromterpy)
12. [Training Entry Points](#12-training-entry-points)
    - 12.1 [pretrain.py](#121-pretrainpy)
    - 12.2 [train_sft_hf.py](#122-train_sft_hfpy)
    - 12.3 [train_grpo_hf.py](#123-train_grpo_hfpy)
13. [Dashboard — `dashboard.py`](#13-dashboard--dashboardpy)
14. [Generation Pipeline — `generation/`](#14-generation-pipeline--generation)
15. [Hyperparameter Reference](#15-hyperparameter-reference)

---

## 1. Project Overview

tinyGPT2 is a self-contained research framework for training small GPT-2-style language models on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset and then aligning them with RL. The pipeline has three stages:

1. **Pretraining** — a custom 8-layer GPT2 is trained from scratch on TinyStories to learn basic language modelling.
2. **Supervised Fine-Tuning (SFT)** — the HuggingFace `openai-community/gpt2` model (117M params) is fine-tuned on instruction-story pairs to teach it to follow structured story prompts.
3. **GRPO** — the SFT model is further optimised with Group Relative Policy Optimisation, guided by a multi-component embedding-based reward model that scores how well completions follow the prompt constraints.

The repo is designed to run on a single GPU (tested on a T4 with 16 GB VRAM) and uses PyTorch Lightning for training, MLflow for experiment tracking and model registration, and Plotly Dash for a comparison dashboard.

---

## 2. Repository Layout

```
gpt2tiny/
├── gpt2tiny/               # Python package
│   ├── model.py            # GPT2 architecture
│   ├── dataset.py          # IterableDataset implementations
│   ├── tokenizer.py        # SentencePiece tokenizer wrapper
│   ├── trainer.py          # Lightning modules for all three stages
│   ├── callbacks.py        # MLflow / checkpoint callbacks
│   ├── prompter.py         # Story prompt generator
│   └── rewards/
│       ├── heuristic.py    # Rule-based reward
│       └── embedding.py    # NLP + semantic reward
│
├── preprocess.py           # Data download and tokenization CLI
├── pretrain.py             # Pretraining script
├── train_sft_hf.py         # SFT script (HF GPT2)
├── train_grpo_hf.py        # GRPO script (HF GPT2)
│
├── run_pretrain.sh
├── run_train_sft_hf.sh
├── run_train_grpo_hf.sh
├── start_mlflow_server.sh
│
├── dashboard.py            # Dash comparison UI
├── generation/             # Prompt and dataset generation tools
├── assets/                 # Word lists used by prompter
└── data/                   # Training data (gitignored)
```

---

## 3. Theory Background

### 3.1 Language Modelling and Pretraining

A language model learns the probability distribution over sequences of tokens. Given a sequence of tokens $x_1, x_2, \ldots, x_T$, an autoregressive language model factorises this as:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

During pretraining the model is given a sequence $x_{1:T}$ and trained to predict the next token at every position. The loss is cross-entropy averaged over all positions:

$$\mathcal{L}_{\text{LM}} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{1:t-1})$$

The transformer backbone computes a contextualised representation for every token using stacked self-attention and feed-forward layers. Causal (autoregressive) masking ensures that position $t$ can only attend to positions $\leq t$.

### 3.2 Supervised Fine-Tuning (SFT)

After pretraining, the model knows the statistical structure of English but does not follow instructions. SFT provides demonstration examples of the desired behaviour in a prompt → completion format:

```
[prompt] Write a story using the word "jog" as a verb...
[completion] One sunny morning, Alice decided to jog through the park...
```

The loss is computed **only over the completion tokens**, masking out the prompt. Formally, if the prompt occupies positions $1\ldots q$ and the completion occupies positions $q+1\ldots T$:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{T-q} \sum_{t=q+1}^{T} \log P_\theta(x_t \mid x_{1:t-1})$$

This teaches the model to produce outputs conditioned on instruction-style prompts.

### 3.3 Reinforcement Learning from Human Feedback — GRPO

SFT trains the model to *imitate* demonstrations but cannot optimise arbitrary objectives (e.g., "use this word correctly as a verb AND include a twist"). GRPO (Group Relative Policy Optimisation, introduced in DeepSeek-R1) addresses this.

**GRPO overview.** For each prompt, $G$ completions are sampled from the current policy $\pi_\theta$. Each completion receives a scalar reward $r_i$ from a reward model. Advantages are computed by normalising within each group:

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

where $\mu_G$ and $\sigma_G$ are the mean and standard deviation of rewards within the group. This relative normalisation makes the signal stable without a separate value network.

**Policy gradient with clipping.** Borrowing PPO's clipped surrogate objective, for each generated token the probability ratio is:

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

The per-sequence loss is:

$$\mathcal{L}_{\text{PPO}} = -\frac{1}{|gen|}\sum_t \min\left(\rho_t \hat{A}, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}\right)$$

Clipping prevents excessively large policy updates by disregarding gains beyond the $[1-\epsilon, 1+\epsilon]$ band.

**KL divergence penalty.** To prevent the policy from drifting too far from the original SFT model (the reference policy $\pi_\text{ref}$), a KL term is added:

$$\mathcal{L}_{\text{KL}} = \beta_{\text{KL}} \cdot \frac{1}{|gen|}\sum_t \left[\log \pi_\theta(a_t) - \log \pi_{\text{ref}}(a_t)\right]$$

**Entropy bonus.** An entropy term $\beta_H \cdot \mathcal{H}(\pi_\theta)$ is added to encourage exploration and prevent premature collapse to deterministic outputs.

**Full loss:**

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + \mathcal{L}_{\text{KL}} - \beta_H \mathcal{H}(\pi_\theta)$$

The reference policy is a frozen copy of the model taken at the start of GRPO training and is never updated.

---

## 4. Data Pipeline — `preprocess.py`

`preprocess.py` is a CLI with five subcommands covering the full preprocessing pipeline for all three training stages.

### Pretrain pipeline

**`download --data-dir PATH`**

Downloads the TinyStories archive from HuggingFace and extracts it into `PATH/TinyStories_all_data/`. The archive contains ~470 shards (JSON files), each holding a list of `{"story": "..."}` objects. A streaming download with a `tqdm` progress bar is used to handle the large file.

**`train-vocab --vocab-size N --data-dir PATH`**

Trains a [SentencePiece](https://github.com/google/sentencepiece) BPE tokenizer on the first 10 shards of TinyStories. Using only 10 shards keeps training fast while still capturing the vocabulary of the corpus. Key SentencePiece settings:

- `model_type="bpe"` — byte-pair encoding merges the most frequent character pairs iteratively to build sub-word units.
- `split_digits=True` — digits are split into single characters, preventing numbers from consuming rare vocabulary slots.
- `byte_fallback=True` — unseen bytes are encoded as `<0xNN>` hex tokens, guaranteeing lossless encoding of any UTF-8 input.
- `normalization_rule_name="identity"` — no Unicode normalisation is applied so the model sees text exactly as written.

The trained model and vocabulary are saved to `PATH/tok{N}.model` and `PATH/tok{N}.vocab`.

**`pretokenize-pretrain --vocab-size N --data-dir PATH`**

Tokenizes all shards in `PATH/TinyStories_all_data/` in parallel using `ProcessPoolExecutor`. Each worker:
1. Loads the shard's JSON.
2. Iterates over examples, encoding each story with BOS and EOS tokens prepended/appended.
3. Concatenates all token sequences into a single `uint16` array and writes it as a `.bin` file alongside the source `.json`.

Using `uint16` (values 0–65535) is sufficient for vocabularies up to 65535 and halves memory and disk usage compared to `int32`.

### SFT pipeline

**`pretokenize-sft --tokenizer NAME --data-dir PATH`**

Tokenizes instruction-story pairs. Each example has:
- `example["instruction"]["prompt"]` — the story prompt (query).
- `example["story"]` — the completion.

The tokenizer is loaded via `AutoTokenizer.from_pretrained(NAME)`, supporting both a HuggingFace model name (e.g. `"openai-community/gpt2"`) and a local directory. BOS is prepended to the prompt and EOS appended to the completion; nothing extra is inserted at the boundary between them.

Two binary files are written per shard:
- `data{N}.bin` — a flat `uint16` array of all tokens concatenated across all examples that fit within the 512-token limit.
- `indices{N}.bin` — a `uint32` array of boundary pointers. Every pair of consecutive triples `(start, mid, end)` gives the start of an example, the prompt/completion boundary, and the end of the example. The dataset class uses these offsets to retrieve each example without loading the full shard into RAM.

### RLHF pipeline

**`pretokenize-rlhf --tokenizer NAME --data-dir PATH`**

Similar to SFT but stores only the prompt tokens (no completion). Prompts longer than 248 tokens are dropped. The index file records only start/end boundaries (no mid-point), since no prompt–answer split is needed for generation.

---

## 5. Model Architecture — `gpt2tiny/model.py`

### 5.1 GPTConfig

`GPTConfig` is a dataclass holding all architectural hyperparameters:

| Field | Default | Meaning |
|---|---|---|
| `block_size` | 512 | Maximum sequence length (context window) |
| `vocab_size` | 4096 | Vocabulary size |
| `n_layer` | 8 | Number of transformer blocks |
| `n_head` | 8 | Number of attention heads |
| `n_embed` | 512 | Model dimensionality (d_model) |
| `n_expert` | 2 | Number of experts (MOE variant) |
| `k` | 1 | Top-k routing (MOE variant) |
| `dropout` | 0.2 | Dropout probability |
| `bias` | False | Whether linear layers use bias |
| `flash` | True | Use PyTorch Flash Attention if available |
| `noisy_gating` | True | Add learnable noise to MOE router |
| `capacity_factor` | 10 | Expert capacity multiplier |
| `load_loss_coef` | 1e-2 | Weight of the MOE load-balancing loss |

### 5.2 Embeddings and Positional Encoding

The model uses learned token embeddings (`wte`, shape `[vocab_size, n_embed]`) and learned positional embeddings (`wpe`, shape `[block_size, n_embed]`), summed before the first transformer block — the same design as the original GPT-2.

When an attention mask is provided (for padded batches), positions are computed via `cumsum(mask) - 1` rather than a simple `arange`, so padding tokens receive valid (though irrelevant) position indices and genuine tokens keep their correct relative positions.

**Weight tying.** The output projection `lm_head` shares its weight matrix with `wte`:

```python
self.transformer.wte.weight = self.lm_head.weight
```

This is standard practice in language modelling. It forces the input and output spaces to be consistent and reduces the total parameter count by `vocab_size × n_embed` (≈2M for the default config).

### 5.3 RMSNorm

PyTorch's built-in `nn.RMSNorm` is used instead of the original GPT-2's `LayerNorm`. RMSNorm normalises by the root-mean-square of the activations without subtracting the mean:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

It is faster than LayerNorm and has been shown to work equally well for language modelling (it is used in LLaMA, Mistral, etc.).

### 5.4 Causal Self-Attention

`CausalSelfAttention` is a standard multi-head self-attention module. Query, key and value projections are computed in a single fused linear layer `c_attn` of shape `[n_embed, 3 × n_embed]` and then split.

Each head operates on a sub-space of dimension `head_dim = n_embed // n_head = 64` (for the defaults). The attention scores are scaled by `1 / sqrt(head_dim)` to prevent vanishing gradients from softmax saturation.

**Flash Attention.** When `config.flash=True` and `torch.nn.functional.scaled_dot_product_attention` is available (PyTorch ≥ 2.0), the implementation delegates to Flash Attention, which fuses the attention kernel into a single CUDA call, reduces HBM reads/writes quadratically in sequence length, and enables much longer contexts without running out of memory.

**Causal masking.** For the non-Flash path a lower-triangular mask (`torch.tril`) is registered as a buffer (not a parameter) and used to fill attention logits above the diagonal with `-inf` before softmax. For Flash Attention, `is_causal=True` instructs the kernel to apply this mask internally.

**Padding mask.** During SFT and GRPO training, sequences in a batch may have different lengths and are padded to the longest. The attention mask `[B, T]` (1=real, 0=pad) is broadcast into key and query masks so that padding tokens neither attend to nor are attended to by real tokens. After the attention output, padded positions are zeroed out explicitly:

```python
if query_mask is not None:
    y = y.masked_fill(~query_mask, 0.0)
```

### 5.5 SwiGLU Feed-Forward Network

The feed-forward sub-layer uses the SwiGLU activation function (used in LLaMA/PaLM) rather than GPT-2's GELU:

$$\text{SwiGLU}(x) = \text{SiLU}(W_1 x) \odot W_3 x$$
$$\text{FFN}(x) = W_2 \cdot \text{SwiGLU}(x)$$

where $W_1, W_2, W_3$ are linear projections. The hidden dimension follows the LLaMA convention of $\frac{2}{3} \times 4d = \frac{8d}{3}$ rather than $4d$, keeping the parameter count comparable to a standard FFN while using the gating structure.

SiLU (Sigmoid Linear Unit / Swish) is $\text{SiLU}(x) = x \cdot \sigma(x)$, which is smooth, non-monotone, and has been empirically shown to outperform ReLU and GELU in large models.

### 5.6 Transformer Block (BlockFFN)

`BlockFFN` is a standard pre-norm transformer block:

```
x ← x + Attention(RMSNorm(x))
x ← x + FFN(RMSNorm(x))
```

Pre-norm (normalising the input to each sub-layer, rather than the output) stabilises training at scale, as used in most modern LLMs.

After each sub-layer, if a padding mask is present, padded positions are zeroed so they do not accumulate numerical drift through the residual stream.

### 5.7 Mixture-of-Experts (BlockMOE)

`BlockMOE` replaces the feed-forward sub-layer with a Mixture-of-Experts (MOE) layer. MOE allows the model to selectively activate only a subset of its parameters for any given token, increasing effective capacity without proportionally increasing compute.

**TopKRouter.** A linear gate maps each token's hidden state to `n_expert` logit scores. During training, learnable noise (sampled from a Gaussian whose standard deviation is itself a learned function of the input) is added to the logits before softmax (`noisy_gating=True`). This prevents routing collapse — a failure mode where all tokens converge to a single expert — by encouraging exploration during the early stages of training.

The top-k experts (typically k=1) with the highest gate probabilities are selected. Their probabilities are re-normalised to sum to 1.

**Capacity constraint.** Each expert can process at most `capacity = ceil(capacity_factor × N / E)` tokens per forward pass, where N is the total number of tokens in the batch and E is the number of experts. Tokens that are routed to an expert that has already reached capacity are silently dropped (their output is zero). The capacity factor of 10 is generous, meaning drops are rare unless batches are very unbalanced.

**Load-balancing loss.** Without an auxiliary loss, routing can collapse so that one expert always receives far more tokens than others, wasting the capacity of the remaining experts. The load-balancing loss penalises this:

$$\mathcal{L}_{\text{load}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \cdot P_e$$

where $f_e$ is the fraction of tokens routed to expert $e$ and $P_e$ is the average routing probability assigned to expert $e$ across the batch. $\alpha$ is `load_loss_coef`. Minimising this product encourages both uniform routing and uniform gate probabilities.

### 5.8 GPT2 — Full Model

`GPT2` assembles the full transformer:

1. Token + positional embeddings → dropout.
2. Stack of `n_layer` `BlockFFN` blocks.
3. Final `RMSNorm`.
4. Linear head (`lm_head`), weight-tied to `wte`.

**Initialisation.** All linear layers are initialised with $\mathcal{N}(0, 0.02)$ (following the original GPT-2). Residual projections (`c_proj`) use a scaled initialisation $\mathcal{N}(0, 0.02/\sqrt{2L})$ where $L$ is the number of layers — this ensures that residual contributions decrease as the network depth increases, preventing the residual stream from growing unboundedly.

**Forward pass.** When `targets` is provided, logits are computed over the full sequence and cross-entropy loss is returned. When targets is `None` (inference), only the logits for the last token are computed — `x[:, [-1], :]` — which avoids the cost of projecting all `T × n_embed` hidden states to the vocabulary.

### 5.9 Generation Sampling Strategies

`GPT2.generate()` supports four sampling strategies that can be combined:

**Temperature scaling.** The logits are divided by $T > 0$ before sampling. $T < 1$ sharpens the distribution (more deterministic), $T > 1$ flattens it (more random). $T = 1$ is unchanged.

**Top-k sampling.** All tokens except the $k$ highest-probability tokens are set to $-\infty$. This prevents the model from sampling extremely unlikely tokens (incoherent words, rare characters) but retains diversity among the plausible continuations.

**Top-p (nucleus) sampling.** The vocabulary is sorted by probability in descending order. The smallest set of tokens whose cumulative probability exceeds $p$ is kept; all others are zeroed out. This adapts dynamically to the entropy of the distribution — when the model is confident, the nucleus is small; when uncertain, it is large.

**Min-p sampling.** A token is only kept if its log-probability is within `log(min_p)` of the maximum log-probability. This is a relative threshold, so the retained set is always calibrated to the current distribution.

Generation halts early when all sequences in the batch have produced an EOS token.

---

## 6. Dataset Classes — `gpt2tiny/dataset.py`

All dataset classes inherit from `torch.utils.data.IterableDataset`, which is appropriate here because:
- The data is too large to fit in RAM (multiple gigabytes of tokenized text).
- The data is stored in pre-shuffled shards; random access within shards is cheap via `np.memmap`.
- Streaming avoids the overhead of building a full index upfront.

### 6.1 Shard Discovery Utilities

`collect_filenames(directory, split)` groups all files in a directory by their numeric shard index. The last shard is reserved for validation; all others are training shards.

`locate_shards(data_dir, split, fn_templates, weights)` handles multi-directory datasets. When `data_dir` is a list of directories, shards from each are concatenated. The `weights` argument controls mixing:
- `"Balanced"` — each directory contributes equally regardless of its size, by repeating shards from smaller datasets.
- `List[int]` — explicit per-directory repeat counts.
- `None` — equal weight of 1 for each directory.

This is used in pretraining to mix TinyStories with an SFT dataset in a controlled ratio.

### 6.2 PreTokDataset

Loads pre-tokenized `.bin` files (uint16) via `np.memmap` — a memory-mapped view that does not load the full file into RAM. For each shard:

1. The shard is divided into non-overlapping chunks of `max_seq_len` tokens.
2. Chunk indices are shuffled at the start of each pass.
3. For each chunk, `x = chunk[:-1]` and `y = chunk[1:]` — the standard next-token prediction setup.

The dataset loops forever (`while True`), making it a proper infinite stream for PyTorch Lightning's `max_steps`-based training.

### 6.3 SFTDataset

Loads both a `data.bin` (uint16 token array) and an `indices.bin` (uint32 boundary array). The indices file stores triplets `(start, mid, end)` for each example:
- `data[start:mid]` — prompt tokens.
- `data[mid:end]` — completion tokens.

The `__iter__` method yields `(qa_tensor, q_len)` pairs. The collator (`SFTDataset.collator`) pads sequences to the batch maximum and builds:
- `seq` — padded token ids `[B, T]`.
- `msk` — attention mask `[B, T]`.
- `lns` — prompt lengths `[B]`, used by the trainer to mask the loss to completion tokens only.

### 6.4 RLHFDataset and PromptDataset

`RLHFDataset` is identical to `SFTDataset` in structure but yields only the prompt tokens `qa[:q_len-1]` — no completion. These serve as the seed for GRPO's generation phase.

`PromptDataset` additionally loads a `data.json` sidecar file containing per-example metadata:

```json
{
  "prompt": "Write a story ...",
  "words": [{"word": "jog", "pos": "verb"}, ...],
  "features": ["Dialogue", "Twist"],
  "subject": {"character": "pirate", "place": "the dock", ...}
}
```

This metadata is passed through the batch collator and eventually reaches the reward model so it knows exactly what constraints to evaluate. The collator (`PromptDataset.collator`) returns a 4-tuple: `(seq, mask, lengths, metadata_list)`.

---

## 7. Tokenizer — `gpt2tiny/tokenizer.py`

`Tokenizer` is a thin wrapper around `SentencePieceProcessor`. It exposes:

- `encode(s, bos, eos)` — tokenize a string, optionally prepending BOS id and appending EOS id. This explicit control is important because SFT training needs BOS on the prompt but not EOS until the very end of the answer.
- `decode(tokens)` — convert a list of token ids back to a string.
- `n_words`, `bos_id`, `eos_id`, `pad_id` — vocabulary metadata.

The SentencePiece model is loaded from disk at construction time; it is not serialised into checkpoints. The tokenizer path is therefore an explicit argument to training scripts.

---

## 8. Training Modules — `gpt2tiny/trainer.py`

All training modules are PyTorch Lightning `LightningModule` subclasses, which handle the training loop, gradient accumulation, mixed precision, and device placement automatically.

### 8.1 PreTrainGPT2Module

Wraps the custom `GPT2` model for pretraining. The training step is minimal:

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    logits, loss = self.model(x, y)
    return loss
```

At the start of each validation epoch's first batch (on the global-zero process in DDP), `_log_generation_to_mlflow` runs the model on a set of fixed prompts and uploads the resulting text as a `.txt` artifact to MLflow. This gives a qualitative view of model quality at each checkpoint without human intervention.

### 8.2 SFTGPT2Module (custom model)

Subclasses `PreTrainGPT2Module` and overrides `forward` to manually run the transformer layers, computing logits over the full sequence (not just the last token). The loss mask is built from the question lengths:

```python
answer_mask = pos >= first_answer_label_pos
loss = (loss * answer_mask).sum() / answer_mask.sum().clamp_min(1)
```

This ensures that only completion tokens contribute to the gradient, exactly implementing the SFT objective.

### 8.3 GPT2SFTModule (HuggingFace model)

A cleaner implementation built on top of `GPT2LMHeadModel` from HuggingFace. `_build_labels` constructs a labels tensor where prompt positions and padding positions are set to `-100` (the `ignore_index` for `F.cross_entropy`), so PyTorch's built-in cross-entropy function handles the masking automatically.

`_loss_and_entropy` also computes the mean token entropy over completion tokens:

$$\mathcal{H}(t) = -\sum_{v} p_v \log p_v$$

This is logged as `train_entropy` / `val_entropy` — a low entropy (model is very confident in one token) can indicate memorisation or mode collapse.

`generate_from_prompts` uses HuggingFace's built-in `.generate()` method, which supports beam search, sampling, and various decoding strategies out of the box.

### 8.4 GRPOGPT2Module / GPT2GRPOModule

Both implement the same GRPO algorithm — `GRPOGPT2Module` wraps the custom model, while `GPT2GRPOModule` wraps the HuggingFace model. The active version is `GPT2GRPOModule`.

**Reference policy.** `_init_ref_policy` creates a deep copy of the current model, sets all parameters to `requires_grad=False`, and permanently sets it to `eval()` mode with all dropout disabled. This frozen copy represents $\pi_\text{ref}$ — the SFT model before any RL updates.

To keep checkpoint sizes small, the reference policy weights are excluded from `state_dict()` and rebuilt from the main model weights when a checkpoint is loaded (via `load_state_dict` and `on_load_checkpoint`).

**`generate_w_logp`** generates $G \times B$ completions (all prompts replicated $G$ times) in parallel using a custom autoregressive loop rather than `model.generate()`. It returns:
- `all_ids` — full sequences (prompt + generated tokens), `[B*G, max_seq_len]`.
- `all_masks` — attention mask for the full sequences.
- `gen_ids`, `gen_masks` — the generated portion only.
- `gen_logp` — per-token log-probabilities under the *sampling* distribution ($\pi_{\theta_{\text{old}}}$). These are stored as `logp_old` for the PPO ratio.

**Important detail.** In `GPT2GRPOModule.generate_w_logp`, log-probs are recorded from the *unscaled* (untempered) policy distribution (`logprobs_policy = log_softmax(logits / 1.0)`) while sampling uses the temperature-scaled distribution. This is because the PPO loss computes probability ratios, and those ratios should be with respect to the unmodified policy to keep the KL penalty well-calibrated.

**`gen_token_logprobs`** re-runs the model in a single forward pass over the full `all_ids` sequences to get differentiable log-probabilities `logp_new` for the policy gradient. It also computes the mean token entropy for monitoring purposes.

**`grpo_loss`** implements:

```
ratio = exp(logp_new - logp_old)
expected_adv = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
ppo_loss = -(expected_adv * gen_masks).sum() / gen_masks.sum()
kl_loss = beta_kl * (logp_new - logp_ref) * gen_masks).sum() / gen_masks.sum()
total = ppo_loss + kl_loss + beta_H * entropy
```

Metrics logged per step: `train_loss`, `train_ppo`, `train_kldiv`, `train_avg_reward`, `train_avg_adv`, `train_std_adv`, `train_avg_ent`, `train_avg_length`, and per-component reward breakdowns.

### 8.5 Learning Rate Schedule

All modules use the same two-phase schedule:

1. **Linear warmup** — learning rate ramps from `≈0` to `lr` over `warmup_ratio × total_steps` steps. This prevents large gradient magnitudes at the start of training when the model is randomly initialised.
2. **Cosine annealing** — learning rate decays following a cosine curve from `lr` down to `0` over the remaining steps. Cosine annealing tends to outperform linear decay because it reduces the learning rate slowly at first (allowing continued progress) and more aggressively near the end (enabling fine convergence).

The schedule is implemented with `SequentialLR` combining `LinearLR` and `CosineAnnealingLR` from `torch.optim.lr_scheduler`, stepped at every gradient update (`interval: "step"`).

---

## 9. Reward Models — `gpt2tiny/rewards/`

### 9.1 Heuristic Reward — `heuristic.py`

An older, fully rule-based reward model. It scores completions on dimensions like prompt adherence (word overlap), coherence (sentence length), style (diversity of vocabulary), length, safety (absence of disallowed phrases), and fluency (average word length as a proxy). A degeneracy penalty fires on repetitive text. Scores are combined with configurable weights and clipped to `[-1, 1]`.

This is included for reference and experimentation. The active reward model used in `train_grpo_hf.py` is the embedding-based `StoryReward`.

### 9.2 Embedding Reward — `embedding.py`

`StoryReward` is the primary reward model. It combines multiple NLP signals to produce a scalar reward in `[-1, 1]` for each generated story.

**NLP tools loaded at construction:**
- `spacy` (`en_core_web_sm`) — for part-of-speech tagging, lemmatisation, and sentence boundary detection.
- `SentenceTransformer` (`all-MiniLM-L6-v2`) — a 22M-parameter sentence embedding model. It maps sentences to a 384-dimensional embedding space where semantically similar sentences are close (by cosine similarity).
- `AutoModelForSequenceClassification` (`madhurjindal/autonlp-Gibberish-Detector-492513457`) — a binary classifier that distinguishes coherent text from gibberish.

### 9.3 Reward Scoring Pipeline

`score(generated_text, metadata)` computes six positive components and four penalties.

**Positive components:**

| Component | Method | Description |
|---|---|---|
| `words` | `score_required_words` | Binary: 1.0 if the required word (exact or lemma match) appears in the text, 0.0 otherwise. Averaged over all required words. |
| `pos` | `score_pos_usage` | Binary: 1.0 if the required word appears **and** is used with the correct part-of-speech tag (as determined by spaCy). Rewards grammatical correctness, not just presence. |
| `subject` | `score_subject` | Weighted average of character, place, action, goal, and adjective adherence. Character and place use a 65%/35% blend of lexical overlap and semantic cosine similarity. Action and goal use a 65%/35% blend of sentence-level semantic similarity and content-word overlap. |
| `features` | `score_features` | Per-feature scores (see below), combined with feature-specific weights. |
| `format` | `score_format` | `band_score` on character count and sentence count (both have configurable min/max), minus repetition penalty and junk-text penalty. The band score is 1.0 inside the target range and decays exponentially outside it. |
| `coherence` | `score_gibberish_sentence_level` | Averages the gibberish detector's `not_gibberish` probability across all sentences. This catches locally incoherent sentences that might not be detected by full-text scoring. |

**Feature scoring:**

| Feature | Method |
|---|---|
| `BadEnding` | Keyword fraction of `bad_ending_keywords` in the last 2 sentences + semantic similarity to prototype "bad ending" sentences + a negative-vs-positive sentiment proxy using marker word counts. |
| `Conflict` | Keyword fraction of `conflict_keywords` + semantic similarity to "conflict" prototypes + fraction of opposition markers (but, however, yet). |
| `Dialogue` | Presence of quoted text (double or single quotes) + presence of reporting verbs (say, ask, whisper, etc.). |
| `Foreshadowing` | Overlap between content words in the first half and second half of the story + payoff markers in the second half + recurrence of early nouns in the final sentence. |
| `MoralValue` | Regex match for moral-cue patterns (e.g., "learned that", "the lesson") in the last 2 sentences + semantic similarity to "moral" prototypes. |
| `Twist` | Twist marker keywords in the final 2 sentences + semantic similarity to "twist" prototypes + dissimilarity between the opening and the ending (a twist should be a departure). |

**Penalties:**

| Penalty | Description |
|---|---|
| `prompt_copy` | Token overlap between the generated text and the original prompt (excluding required words). A generation that mostly parrots the prompt has not produced new content. |
| `meta_language` | Regex hits for patterns indicating the model is talking *about* the task rather than *doing* it (e.g., "this story should...", "required words:"). |
| `repetition` | Unigram repetition (tokens occurring more than twice) plus bigram repetition (bigrams occurring more than once), combined 50/50. |
| `keyword_stuffing` | Penalises required words appearing excessively (more than 3 times each), which indicates the model has learned to mention words repeatedly rather than integrate them naturally. |

**Score aggregation:**

Active weights are summed based on which constraints exist in the metadata. The positive score is normalised by the sum of active weights:

```
positive_score = Σ(weight_i × score_i) / Σ(active_weight_i)
penalty_total  = Σ(weight_j × penalty_j)
raw_total      = positive_score - penalty_total
total          = clip(raw_total, -1.0, 1.0)
```

### 9.4 GRPO Advantage Computation

`StoryReward.compute_grpo_advantages` implements group-relative normalisation:

```python
grouped = rewards.view(B // G, G)          # [num_prompts, G]
means = grouped.mean(dim=1, keepdim=True)   # per-prompt mean reward
stds  = grouped.std(dim=1, keepdim=True)    # per-prompt std reward
advantages = (grouped - means) / (stds + eps)
```

This is the core of GRPO: advantages are computed relative to the group, not relative to an absolute baseline. A completion that scores 0.3 when the group average is 0.1 gets a positive advantage; the same score with a group average of 0.5 gets a negative advantage. This makes the signal robust to the absolute scale of the reward function.

---

## 10. Callbacks — `gpt2tiny/callbacks.py`

**`LogBestCkptAndPyfuncToMLflow`** fires `on_train_end`. It retrieves the best checkpoint path from `ModelCheckpoint`, wraps the trained Lightning module in a `LightningPyfunc` MLflow model, and registers it in the MLflow Model Registry under `register_name`. This makes the model callable from the dashboard or any downstream code via `mlflow.pyfunc.load_model("models:/name/version")`.

**`LightningPyfunc`** is the MLflow model wrapper. Its `predict` method accepts a list of prompt strings (or dicts for GRPO) and calls `generate_from_prompts(...)` on the underlying Lightning module, returning a list of `{"prompt": ..., "completion": ...}` dicts. This is the interface the dashboard uses.

**`MLflowGenerationCallback`** fires `on_validation_epoch_end`. It calls `generate_from_prompts` on the module for the fixed evaluation prompts and logs the results as a text artifact. For the GRPO module the output also includes the reward breakdown for each generated story, making it easy to track qualitative progress in the MLflow UI.

**`UploadLastAndBestToMLflow`** uploads checkpoint files as MLflow artifacts after each validation.

---

## 11. Prompt Generation — `gpt2tiny/prompter.py`

`PromptGenerator` builds story prompts with controlled vocabulary and narrative constraints. It reads word lists from `assets/` (nouns, verbs, adjectives, characters, actions, places, etc.) and randomly selects:

- 1–3 **required words** with associated POS tags (e.g., "use 'cane' as a noun").
- 0–3 **narrative features** chosen from: BadEnding, Conflict, Dialogue, Foreshadowing, MoralValue, Twist.
- 0–1 **subject** (character + optional adjective + action + place + goal).

Feature names are varied across six pre-written phrasings per feature to reduce prompt monotony:

```python
FEATURE_VARIATIONS = {
    "Dialogue": [
        "include at least one moment of direct conversation between characters",
        "the narrative should have at least one spoken conversation",
        ...
    ],
    ...
}
```

The final prompt string assembles these components into a paragraph using one of 10 template openings ("Write a short story.", "Compose a narrative.", etc.). The full metadata dict (including all word specs, features, and subject fields) is saved alongside the tokenized prompt so the reward model can evaluate the generated story against the exact constraints.

---

## 12. Training Entry Points

### 12.1 pretrain.py

Instantiates `PreTrainDataModule` pointing at one or more data directories (supporting balanced mixing), builds a `GPTConfig` from CLI args, wraps it in `PreTrainGPT2Module`, and calls `torch.compile()` to fuse kernels with TorchInductor.

Key design choices:
- `val_check_interval = val_interval × grad_accum_step` ensures validation happens every `val_interval` **optimizer steps**, not gradient accumulation steps.
- Mixed precision is enabled on CUDA (`"16-mixed"`), halving memory usage and increasing throughput.
- The `ModelCheckpoint` callback saves directly into the MLflow artifact directory so that checkpoints are automatically tracked as MLflow artifacts.

### 12.2 train_sft_hf.py

Loads the HuggingFace `gpt2` tokenizer and sets `pad_token = eos_token` (GPT-2 has no dedicated padding token). Instantiates `GPT2SFTModule` which loads `GPT2LMHeadModel.from_pretrained("gpt2")` and fine-tunes it. The `MLflowGenerationCallback` logs completions for fixed evaluation prompts after every validation step.

### 12.3 train_grpo_hf.py

Loads a checkpoint from a completed SFT run (via `--init-model-path`) using `GPT2GRPOModule.load_from_checkpoint(...)`. The checkpoint provides the pre-trained weights; the GRPO hyperparameters (num_gen, kl_beta, etc.) and reward weights are supplied as CLI arguments so they can be tuned without retraining. The reward config (`min_chars`, `max_chars`, `min_sentences`, `max_sentences`) controls the format constraint in the reward model. Dropout is explicitly disabled in all layers before training starts.

GRPO monitors `val_avg_reward` (rather than `val_loss`) for checkpoint selection — the best model is the one that achieves the highest average reward on the validation set.

---

## 13. Dashboard — `dashboard.py`

A [Plotly Dash](https://dash.plotly.com/) web application for interactively comparing two registered MLflow models side-by-side.

**Layout.** A left control panel contains:
- Dropdowns for Model A and Model B name (populated from the MLflow registry).
- Version selectors (automatically updated when a model name is chosen, listing versions in reverse chronological order).
- Sampling parameters: temperature, top-k, top-p, max tokens.
- A "Generate" button.

A right panel contains a shared prompt input, two output panels (Model A / Model B), and a scrollable history of the last 10 prompt–response pairs.

**Model caching.** Loaded models are cached in `MODEL_CACHE` (a plain dict keyed by model URI). The first call for a given URI loads the model via `mlflow.pyfunc.load_model`, unwraps the `LightningPyfunc` wrapper to get the underlying Lightning module, and moves it to GPU. Subsequent calls return the cached model, avoiding repeated loading.

**Generation.** The `generate` callback fires on button click or Enter key. It calls `model.generate_from_prompts([prompt], **params)` for each selected model and returns the completion text. Results are also appended to a `dcc.Store` history component, rendered in the history panel.

**Environment.** `MLFLOW_TRACKING_URI` is loaded from a `.env` file, allowing the dashboard to connect to a remote MLflow server or a local one.

---

## 14. Generation Pipeline — `generation/`

The `generation/` directory contains scripts for building the SFT and RLHF datasets programmatically.

- **`generate_prompts.py`** — uses `PromptGenerator` to create a large batch of structured story prompts and saves them as JSON.
- **`build_prompts.py`** — formats prompts for OpenAI Batch API calls (each prompt becomes a `{"custom_id": ..., "messages": [...]}` entry).
- **`parse_results.py`** — reads the completed batch results from OpenAI and pairs each prompt with its generated story, writing the combined dataset.
- **`build_validation.py`** — creates a smaller validation split.
- Shell scripts (`run_prompt_build.sh`, etc.) wrap each step for easy execution.

The resulting dataset files (saved in `generation/batch_data/`) are then processed by `pretokenize-sft` or `pretokenize-rlhf` to produce the `.bin` files used during training.

---

## 15. Hyperparameter Reference

### Pretraining (`run_pretrain.sh`)

| Parameter | Value | Notes |
|---|---|---|
| `block_size` | 512 | Context window |
| `n_layer` | 8 | Depth |
| `n_head` | 8 | Attention heads |
| `n_embed` | 512 | Hidden dimension |
| `dropout` | 0.2 | Applied to attention and FFN |
| `vocab_size` | 4096 | Custom SentencePiece tokenizer |
| `max_steps` | 20 000 | Total optimizer steps |
| `lr` | 3e-5 | Peak learning rate |
| `warmup_ratio` | 0.05 | 5% of steps for warmup |
| `batch_size` | 64 | Examples per gradient step |
| `grad_accum_step` | 4 | Effective batch = 256 |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `val_interval` | 25 | Steps between validations |

### SFT (`run_train_sft_hf.sh`)

| Parameter | Value | Notes |
|---|---|---|
| `model` | `openai-community/gpt2` | 117M parameters |
| `batch_size` | 8 | Small due to longer sequences |
| `grad_accum_step` | 8 | Effective batch = 64 |
| `max_steps` | 20 000 | |
| `lr` | 3e-5 | |
| `warmup_ratio` | 0.05 | |

### GRPO (`run_train_grpo_hf.sh`)

| Parameter | Value | Notes |
|---|---|---|
| `num_gen` | 8 | Completions sampled per prompt |
| `max_seq_len` | 128 | Max total length for sampling |
| `temperature` | 0.8 | Sampling temperature |
| `kl_beta` | 0.07 | KL penalty strength |
| `entropy_beta` | 0.02 | Entropy bonus strength |
| `clip_eps` | 0.2 | PPO clipping range |
| `batch_size` | 4 | Prompts per batch (×8 gen = 32 sequences) |
| `grad_accum_step` | 8 | Effective batch = 32 prompts |
| `rw_words` | 0.1 | Required word reward weight |
| `rw_coherence` | 0.4 | Coherence reward weight |
| `rw_repetition_penalty` | 0.2 | Repetition penalty weight |
| `rw_min_chars` | 750 | Minimum story length in characters |
| `rw_max_sentences` | 10 | Maximum sentences |
