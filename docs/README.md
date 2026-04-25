# GPT-2 Fine-Tuning: Supervised Fine-Tuning & GRPO

A complete codebase for fine-tuning GPT-2 on creative story generation using two
sequential stages:

1. **Supervised Fine-Tuning (SFT)** — teach GPT-2 to complete story prompts correctly.
2. **Group Relative Policy Optimization (GRPO)** — refine the SFT model with reinforcement
   learning so that generated stories score higher on a multi-dimensional reward function.

The project contains two model families that are kept in sync:

| Family | Where defined | Notes |
|---|---|---|
| Custom (`gpt2tiny`) | `gpt2tiny/model.py` | Built from scratch; optionally uses Mixture-of-Experts |
| HuggingFace | `gpt2tiny/trainer.py` (`GPT2SFTModule`, `GPT2GRPOModule`) | Wraps `GPT2LMHeadModel` from `transformers` |

Training for both families uses [PyTorch Lightning](https://lightning.ai/) and
[MLflow](https://mlflow.org/) for experiment tracking.

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [Theoretical Background](#2-theoretical-background)
   - 2.1 [Transformer Architecture](#21-transformer-architecture)
   - 2.2 [Mixture of Experts](#22-mixture-of-experts)
   - 2.3 [Causal Language Modelling](#23-causal-language-modelling)
   - 2.4 [Supervised Fine-Tuning](#24-supervised-fine-tuning)
   - 2.5 [Reinforcement Learning from Human Feedback (RLHF)](#25-reinforcement-learning-from-human-feedback-rlhf)
   - 2.6 [GRPO: Group Relative Policy Optimization](#26-grpo-group-relative-policy-optimization)
3. [Data Pipeline](#3-data-pipeline)
   - 3.1 [Datasets](#31-datasets)
   - 3.2 [Tokenisation](#32-tokenisation)
   - 3.3 [Preprocessing](#33-preprocessing)
   - 3.4 [Prompt Generation](#34-prompt-generation)
4. [Model Architecture](#4-model-architecture)
   - 4.1 [GPTConfig](#41-gptconfig)
   - 4.2 [CausalSelfAttention](#42-causalselfAttention)
   - 4.3 [FeedForward (SwiGLU)](#43-feedforward-swiglu)
   - 4.4 [BlockFFN and BlockMOE](#44-blockffn-and-blockmoe)
   - 4.5 [Mixture of Experts Internals](#45-mixture-of-experts-internals)
   - 4.6 [GPT2 and GPT2MOE](#46-gpt2-and-gpt2moe)
5. [Dataset Classes](#5-dataset-classes)
6. [Training Modules](#6-training-modules)
   - 6.1 [GPT2Module (base)](#61-gpt2module-base)
   - 6.2 [SFTGPT2Module / GPT2SFTModule](#62-sftgpt2module--gpt2sftmodule)
   - 6.3 [GRPOGPT2Module / GPT2GRPOModule](#63-grpogpt2module--gpt2grpomodule)
7. [Reward Functions](#7-reward-functions)
   - 7.1 [Heuristic Reward (`heuristic.py`)](#71-heuristic-reward-heuristicpy)
   - 7.2 [Embedding-Based Reward (`embedding.py`)](#72-embedding-based-reward-embeddingpy)
8. [Callbacks and Infrastructure](#8-callbacks-and-infrastructure)
9. [Training Scripts](#9-training-scripts)
10. [End-to-End Workflow](#10-end-to-end-workflow)
11. [Key Design Decisions & Trade-offs](#11-key-design-decisions--trade-offs)

---

## 1. Repository Layout

```
.
├── assets/                          # Vocabulary pools used by the prompt generator
│   ├── adjectives.json              # ~230 common adjectives
│   ├── nouns.json                   # Common nouns
│   ├── verbs.json                   # Common verbs
│   ├── features.json                # Story-feature keys (e.g. "bad_ending_prompt")
│   ├── subject_characters.json      # ~200 character archetypes (e.g. "pirate", "chef")
│   ├── subject_actions.json         # Actions characters can perform
│   ├── subject_adjectives.json      # Adjectives that describe characters
│   ├── subject_goals.json           # Goals characters can have
│   └── subject_places.json          # Locations stories can be set in
│
├── generation/                      # Offline LLM-assisted data generation
│   ├── build_prompts.py             # Convert structured prompts → Batch API request JSONL
│   ├── build_validation.py          # Build validation batch requests
│   ├── generate_prompts.py          # Use PromptGenerator to create prompt shards
│   ├── parse_results.py             # Parse LLM batch-API completions into SFT data
│   ├── run_parser.sh                # Shell wrapper for parse_results.py
│   ├── run_prompt_build.sh          # Shell wrapper for build_prompts.py
│   ├── run_validation_build.sh      # Shell wrapper for build_validation.py
│   └── batch_data/                  # Raw batch results (00/, 01/, 02/ shards)
│       ├── completions*.jsonl       # LLM responses
│       ├── prompts*.jsonl           # The original batch request JSONL
│       ├── data*.json               # Structured prompt metadata
│       ├── validation_prompts*.jsonl
│       ├── validations*.jsonl
│       ├── failed*.json             # Failed requests
│       └── summary*.json           # Batch statistics
│
├── gpt2tiny/                        # Core Python package
│   ├── __init__.py
│   ├── model.py                     # GPT-2 architecture (FFN & MoE variants)
│   ├── dataset.py                   # Dataset / DataLoader classes
│   ├── tokenizer.py                 # SentencePiece tokenizer wrapper
│   ├── prompter.py                  # Randomised story-prompt generator
│   ├── trainer.py                   # PyTorch Lightning LightningModules (SFT & GRPO)
│   ├── callbacks.py                 # MLflow integration callbacks
│   ├── utils.py                     # MLflow checkpoint download helper
│   └── rewards/
│       ├── __init__.py
│       ├── heuristic.py             # Lexical / statistical reward (Reward class)
│       └── embedding.py             # NLP + semantic-similarity reward (StoryReward class)
│
├── preprocess.py                    # Dataset download, vocab training, tokenisation
├── train.py                         # Low-level manual training loop (custom GPT)
├── train_sft_hf.py                  # SFT training entry point (HF GPT-2)
├── train_grpo_hf.py                 # GRPO training entry point (HF GPT-2)
│
├── run_train_sft_hf.sh              # Convenience launcher for train_sft_hf.py
├── run_train_grpo_hf.sh             # Convenience launcher for train_grpo_hf.py
├── start_mlflow_server.sh           # Start MLflow tracking server
│
├── gpt2_supervised_finetuning.ipynb # Notebook: SFT walkthrough
├── gpt2_rl_grpo.ipynb               # Notebook: GRPO walkthrough
│
├── pyproject.toml
└── requirements.txt
```

---

## 2. Theoretical Background

### 2.1 Transformer Architecture

The fundamental building block is the **Transformer** introduced by Vaswani et al. (2017).
A Transformer is a sequence-to-sequence architecture that uses **attention** instead of
recurrence to model dependencies between positions.

#### Token and Position Embeddings

Every input token is first mapped to a dense vector (the *token embedding*), then a
*positional embedding* is added so that the model can distinguish token positions:

```
x = wte(token_ids) + wpe(position_ids)
```

In the standard GPT-2 case positions are absolute integers `0, 1, …, T-1`. When an
attention mask is present the code uses **cumulative-sum positional encoding** — positions
are assigned by counting how many real (non-padding) tokens precede each position:

```python
pos = torch.cumsum(attention_mask, dim=-1) - 1
```

This means padding tokens keep the same positional index as the real token immediately
before them, which is a clean way to handle left-padded or variable-length batches.

#### Scaled Dot-Product Attention

For a sequence of length `T` with embedding dimension `C` and `h` heads, each head has
dimension `d = C / h`. Queries, keys, and values are obtained from a single linear
projection followed by a reshape:

```
Q, K, V = split(W_qkv · x)     shape each: (B, h, T, d)
Attention(Q,K,V) = softmax( QKᵀ / √d ) V
```

A **causal mask** (lower-triangular) prevents position `i` from attending to positions
`j > i`, which is required for autoregressive language generation.

The implementation supports two paths:
- **Flash Attention** (`torch.nn.functional.scaled_dot_product_attention`) when
  `config.flash=True` and PyTorch ≥ 2.0 is available. Flash Attention reorders the
  softmax computation to avoid materialising the full `T×T` attention matrix, yielding
  large memory and speed savings.
- **Manual attention** that explicitly builds the `T×T` matrix, suitable for debugging or
  older PyTorch.

**Padding mask handling**: when an attention mask is provided, the key mask zeros out
attention to padding positions (so they do not influence other positions) and the query
mask zeros out the output for padding positions themselves (so padding embeddings remain
clean throughout the forward pass).

#### Residual Connections and Normalisation

Each sub-layer (attention, FFN) is wrapped with a pre-norm residual connection:

```
x = x + SubLayer(RMSNorm(x))
```

**RMSNorm** (Root Mean Square Layer Normalisation) is used instead of the standard
LayerNorm. RMSNorm omits the mean-centering step:

```
RMSNorm(x) = x / RMS(x) * γ,   RMS(x) = sqrt(mean(x²) + ε)
```

This reduces compute and has been found empirically to work as well as full LayerNorm in
most settings. GPT-2 originally used LayerNorm; Llama and many modern LLMs use RMSNorm.

---

### 2.2 Mixture of Experts

A standard transformer applies the same FFN sub-layer to every token. A **Mixture of
Experts (MoE)** layer instead routes each token to one or more specialised sub-networks
called *experts*, allowing the total number of parameters to scale independently of the
compute per token.

#### Routing

A small linear *gate* projects each token into a score over all `E` experts:

```
logits = W_gate · x           shape (N, E)
probs  = softmax(logits)
top-k  indices, top-k probs   ← topk(probs, k=K)
```

**Noisy gating** (used during training when `config.noisy_gating=True`) adds Gaussian
noise scaled by a learned per-token, per-expert standard deviation:

```
noise_std = softplus(W_noise · x) + ε
logits += randn_like(logits) * noise_std
```

This encourages exploration and prevents routing collapse (all tokens going to the same
expert).

#### Capacity

Without a limit, one expert could receive all tokens in a batch. A *capacity factor*
controls the maximum fraction of tokens any expert can process:

```
capacity = ceil(capacity_factor * N / E)
```

Tokens that would exceed an expert's capacity are **dropped** (their contribution to the
output is zero). `n_dropped` is tracked and logged. This is the "token dropping" approach
from the Switch Transformer (Fedus et al., 2021).

#### Load-Balancing Loss

To discourage routing collapse the model is trained with an auxiliary **load-balancing
loss** (Shazeer et al., 2017):

```
load_frac[e]  = fraction of tokens routed to expert e (hard assignment)
importance[e] = sum of routing probabilities to expert e (soft, differentiable)

load_loss = coef * E * sum(load_frac[e] * importance[e])
```

This product is minimised when both quantities are uniform (each expert gets `1/E` of
tokens), but `importance` is differentiable so gradients can flow back through the router.

---

### 2.3 Causal Language Modelling

GPT-2 is a **decoder-only** Transformer trained with the **causal language modelling
(CLM)** objective: predict the next token from all previous tokens. The loss is
cross-entropy summed over all positions:

```
L_CLM = - Σ_t log P(x_t | x_{<t})
```

The model is trained to minimise this over a large text corpus. Once trained it can
generate text autoregressively by sampling one token at a time from `P(x_t | x_{<t})`.

#### Weight Tying

The output projection (`lm_head`) shares its weight matrix with the token embedding
(`wte`):

```python
self.transformer.wte.weight = self.lm_head.weight
```

This is a standard practice in language models (Press & Wolf, 2017). It reduces
parameters and acts as a regulariser by forcing the model to use the same representation
for "what token am I?" (embedding) and "which token should come next?" (lm_head).

#### Weight Initialisation

- All linear and embedding layers use N(0, 0.02).
- Residual projection layers (`c_proj.weight`) use a scaled-down initialisation:
  `std = 0.2 / sqrt(2 * n_layer)`. The factor `1/sqrt(2*n_layer)` comes from the GPT-2
  paper; because each token's representation passes through `n_layer` residual additions,
  the variance grows as O(n_layer) without this correction.

---

### 2.4 Supervised Fine-Tuning

Pre-trained language models learn general language statistics. To make a model *follow
instructions* (e.g. "Write a story using the word 'cane' as a noun") we use **supervised
fine-tuning (SFT)**:

1. Collect pairs of `(prompt, completion)`.
2. Concatenate them: `[BOS] prompt [completion] [EOS]`.
3. Fine-tune the model with CLM loss, but **only on completion tokens** — prompt tokens
   are masked out (label = −100 / −1 in the loss).

The key insight is that masking the prompt means the model is never rewarded for
predicting the prompt text (which it could trivially do by memorising it). It only learns
to produce good continuations given a prefix.

In code (`SFTGPT2Module.forward` and `GPT2SFTModule._build_labels`):

```python
# Build labels: ignore padding and prompt positions
labels = input_ids.clone()
labels = labels.masked_fill(attention_mask == 0, -100)       # ignore padding
labels = labels.masked_fill(positions < question_lengths, -100)  # ignore prompt
```

The resulting loss only accumulates over answer/completion tokens.

**Token entropy** is also tracked (but not used in the loss). It measures how peaked or
spread out the model's token distribution is. A healthy model should not become
overconfident (low entropy → repetitive/degenerate output) but also should not become
completely random (high entropy → incoherent output).

---

### 2.5 Reinforcement Learning from Human Feedback (RLHF)

SFT teaches the model to complete prompts in the correct format, but it does not optimise
for *quality*. RLHF is a family of algorithms that use a reward signal to push the model
towards higher-quality outputs.

The canonical RLHF pipeline (Ouyang et al., "InstructGPT", 2022):

1. SFT on human demonstrations.
2. Collect human preference data (which of two completions is better?).
3. Train a **reward model** on these preferences.
4. Fine-tune the SFT model using **Proximal Policy Optimization (PPO)**, rewarding
   completions that the reward model scores highly while penalising large divergence from
   the SFT model via a KL penalty.

This project approximates step 2–3 with a deterministic **heuristic / embedding-based
reward function** (no human preference data needed) and replaces PPO with **GRPO**.

---

### 2.6 GRPO: Group Relative Policy Optimization

GRPO (DeepSeekMath, 2024) is a variant of PPO that **eliminates the need for a separate
value / critic network** by instead using relative rewards within a group of completions.

#### Sampling Phase

For each prompt `p`, sample `G` completions from the current policy `π_θ`:

```
c_{p,1}, c_{p,2}, ..., c_{p,G}  ~  π_θ(· | p)
```

Score each completion with the reward function `r`:

```
R_{p,g} = r(p, c_{p,g})
```

#### Advantage Normalisation

Within each group of `G` completions for the same prompt, normalise the rewards to
produce **advantages** (zero mean, unit std within the group):

```
μ_p = mean(R_{p,1:G})
σ_p = std(R_{p,1:G})
A_{p,g} = (R_{p,g} - μ_p) / (σ_p + ε)
```

This is the "baseline subtraction" that reduces variance in policy gradient estimates.
Because we normalise *within a group* from the *same prompt*, the baseline already
accounts for prompt difficulty — a prompt that consistently elicits high-reward
completions will not produce an inflated advantage.

In code (`StoryReward.compute_grpo_advantages`):

```python
grouped = rewards.view(B // group_size, group_size)
means   = grouped.mean(dim=1, keepdim=True)
stds    = grouped.std(dim=1, keepdim=True, unbiased=False)
advantages = (grouped - means) / (stds + eps)
```

#### PPO Surrogate Loss

GRPO uses the PPO clipped surrogate objective applied **at the token level** (each
generated token is treated as one "action"):

```
ratio_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
        = exp(log π_θ(a_t) - log π_θ_old(a_t))

L_PPO = -mean_t[ min(ratio_t * A,
                     clip(ratio_t, 1-ε, 1+ε) * A) ]
```

Clipping the ratio to `[1-ε, 1+ε]` (typically ε = 0.2) prevents the policy from moving
too far from the old policy in a single step, which would otherwise destabilise training.

#### KL Penalty

An additional **KL divergence penalty** discourages the learned policy from drifting too
far from the *frozen reference policy* (a copy of the SFT model):

```
L_KL = β * mean_t[ log π_θ(a_t) - log π_ref(a_t) ]
```

This is a soft constraint. The hyperparameter `β` (typically 0.01–0.05) controls the
trade-off between reward maximisation and staying close to the SFT distribution.

#### Total Loss

```
L = L_PPO + β * L_KL
```

The reference policy's weights are **frozen** (no gradients) and are saved separately
from the model checkpoint (using `state_dict` filtering) so checkpoints stay small.

#### Why GRPO over PPO?

Standard PPO requires a *value function* (critic) to estimate the expected return from
each state. The critic network adds parameters, compute, and training complexity. GRPO
avoids this entirely by using multiple samples from the same prompt as a natural
contrastive signal — the baseline is implicitly computed across the group.

---

## 3. Data Pipeline

### 3.1 Datasets

The project is designed around the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset: short stories written at a child's reading level, ideal for small model training.
It also supports [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) for
vocabulary training on mathematical text.

For GRPO a **custom prompt dataset** is generated (see §3.4) consisting of structured
story-writing instructions with metadata (required words, story features, subject).

### 3.2 Tokenisation

Two tokeniser choices are used in different parts of the project:

**SentencePiece BPE** (`gpt2tiny/tokenizer.py`):
- Wraps `sentencepiece.SentencePieceProcessor`.
- Trained on TinyStories or MetaMathQA text via `preprocess.py:train_vocab()`.
- The vocabulary size is configurable; 4096 is the default.
- The `Tokenizer` class exposes `encode(text, bos, eos)` (prepend/append special tokens
  on demand) and `decode(tokens)`.

**HuggingFace GPT-2 tokeniser** (`transformers.AutoTokenizer`):
- Used by the HF-based training scripts (`train_sft_hf.py`, `train_grpo_hf.py`).
- 50257-token BPE vocabulary.
- Since GPT-2 has no official pad token, `pad_token` is set to `eos_token`.

### 3.3 Preprocessing (`preprocess.py`)

`preprocess.py` orchestrates the full data-preparation pipeline through a set of
subcommands:

| Subcommand | What it does |
|---|---|
| `download` | Downloads TinyStories and MetaMathQA from HuggingFace |
| `train-vocab` | Trains a SentencePiece BPE model on story/math text |
| `pretokenize-sft` | Tokenises prompt+completion pairs into binary shards |
| `pretokenize-rlhf` | Tokenises prompts-only into binary shards |
| `prepare-dataset` | Runs all the above in sequence |

#### Binary Shard Format

Data is stored in flat binary files for memory-mapped efficient streaming:

**CLM data** (`data.bin`):
- `uint16` array of token IDs, one story after another.
- At training time, a contiguous chunk of `max_seq_len` tokens is sliced; `x = chunk[:-1]`,
  `y = chunk[1:]` — the standard next-token prediction setup.

**SFT data** (`data.bin` + `indices.bin`):
- `data.bin`: `uint16` token IDs of concatenated `[prompt][answer]` sequences.
- `indices.bin`: `uint32` array where every two consecutive entries `[start, mid, end]`
  describe one example: tokens `[start:mid]` are the prompt, tokens `[mid:end]` are the
  answer. The file stores `2*N + 1` entries for `N` examples.

**Prompt data** (`data.bin` + `indices.bin` + `data.json`):
- Same binary layout but with only prompts (no answers).
- `data.json` stores structured metadata per prompt (required words, features, subject).

### 3.4 Prompt Generation (`gpt2tiny/prompter.py`, `generation/generate_prompts.py`)

Story prompts are programmatically generated using a **template-based system**:

#### Assets

The `assets/` directory provides word pools:
- `adjectives.json`, `nouns.json`, `verbs.json`: common English words at various
  parts of speech.
- `subject_characters.json`: ~200 character archetypes (pirate, chef, librarian, …).
- `subject_actions.json`, `subject_places.json`, `subject_adjectives.json`,
  `subject_goals.json`: building blocks for the story subject.
- `features.json`: the six supported story features.

#### PromptGenerator

`PromptGenerator` assembles prompts by:

1. **Sampling words**: picks `k ∈ [min_words, max_words]` word/POS pairs at random.
2. **Sampling features**: picks `k ∈ [min_features, max_features]` story features.
3. **Sampling a story subject**: with probability 0.75, constructs a `StorySubject`
   (character + action + place + optional adjective/goal).
4. **Picking a template** (10 templates: "Write a short story.", "Compose a narrative.",
   etc.) and a phrasing style for each word requirement (7 styles: "use the noun 'X'",
   "include 'X' as a noun", etc.).
5. **Building clauses**:
   - Subject clause: "Write about a lonely pirate who searches for treasure in Paris."
   - Word clause: "The story should use 'cane' as a noun and include 'jog' as a verb."
   - Feature clause: "Also ensure that the story has a bad ending."
6. **Returning a structured dictionary** with the full prompt text and all structured
   metadata (needed by the reward function).

The returned structure for each prompt looks like:
```json
{
  "prompt": "Write a short story. Make it about a lonely pirate who searches for...",
  "words": [{"word": "cane", "pos": "noun"}],
  "features": ["BadEnding"],
  "subject": {"character": "pirate", "adjective": "lonely", ...},
  "feature_phrases": ["the story has a bad ending"],
  "word_clause": "...",
  "feature_clause": "...",
  "subject_clause": "..."
}
```

#### Offline LLM Completion (`generation/`)

For SFT training, prompts need corresponding high-quality completions. These are obtained
by sending prompts to an LLM API (batch mode) and parsing the results:

- `build_prompts.py`: converts structured prompt metadata into Batch API request JSONL.
- `parse_results.py`: parses the JSONL responses and creates SFT-ready binary shards.
- `build_validation.py`: same pipeline but for a validation split.

---

## 4. Model Architecture

### 4.1 GPTConfig

```python
@dataclass
class GPTConfig:
    block_size:      int   = 512    # Maximum context length
    vocab_size:      int   = 4096   # Vocabulary size
    n_layer:         int   = 8      # Number of transformer blocks
    n_head:          int   = 8      # Number of attention heads
    n_embed:         int   = 512    # Embedding / residual dimension
    n_expert:        int   = 2      # Number of MoE experts (MoE only)
    k:               int   = 1      # Top-K routing (MoE only)
    dropout:         float = 0.2    # Dropout probability
    bias:            bool  = False  # Use bias in linear layers
    flash:           bool  = True   # Use Flash Attention if available
    noisy_gating:    bool  = True   # Add noise to router logits during training
    capacity_factor: int   = 10     # Max tokens per expert (as fraction of N/E)
    load_loss_coef:  float = 1e-2   # Weight of MoE load-balancing loss
```

With `n_embed=512`, `n_head=8`, `n_layer=8`, `vocab_size=4096` the model has roughly
**~25M parameters** (similar in scale to the original GPT-2 small but with a smaller
vocabulary).

---

### 4.2 CausalSelfAttention

```
Input:  x  [B, T, C]
Output: y  [B, T, C]
```

1. **Single projection**: `W_qkv ∈ ℝ^{C × 3C}` produces Q, K, V in one matmul.
2. **Reshape to multi-head**: `(B, T, C)` → `(B, h, T, d)` where `d = C/h`.
3. **Attention**: Flash or manual causal attention.
4. **Re-merge heads**: `(B, h, T, d)` → `(B, T, C)`.
5. **Output projection**: `W_proj ∈ ℝ^{C × C}`, followed by residual dropout.

The class stores `config.bias` — if False, all linear layers have no bias term. This
is consistent with modern GPT variants (LLaMA, Mistral) that find bias-free models
train more cleanly.

---

### 4.3 FeedForward (SwiGLU)

Standard GPT-2 uses a two-layer MLP with GeLU. This codebase uses the **SwiGLU**
activation (Noam Shazeer, 2020) which uses *gated* linear units:

```
FFN(x) = dropout( W2 · (SiLU(W1·x) ⊙ W3·x) )
```

- `W1, W3`: two separate up-projection matrices (`n_embed → hidden_dim`).
- `W2`: down-projection (`hidden_dim → n_embed`).
- `⊙`: element-wise multiplication.
- `SiLU(z) = z · σ(z)`: Sigmoid Linear Unit (also called Swish).

The hidden dimension is `hidden_dim = int(2/3 * 4 * n_embed)`. This fractional scaling
ensures the parameter count matches a standard `4×n_embed` two-layer MLP while enabling
the gated structure.

**Why SwiGLU?** Empirically it achieves better perplexity than GeLU at the same parameter
count. It is used in LLaMA, PaLM, and many other modern LLMs.

---

### 4.4 BlockFFN and BlockMOE

Both block types follow the same pre-norm residual pattern:

**BlockFFN** (standard):
```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

**BlockMOE** (mixture of experts):
```
x = x + Attention(RMSNorm(x))
x = x + MOE(RMSNorm(x)).y         # also collects load_loss
```

Note that `BlockFFN` applies the padding mask after both sub-layers to prevent padding
positions from leaking into the residual stream:

```python
if attention_mask is not None:
    x = x * attention_mask[:, :, None].to(x.dtype)
```

---

### 4.5 Mixture of Experts Internals

The `MOE` module handles the full dispatch–compute–combine logic:

```
Input:  x  [B, T, C]
Output: MOEOutput(y=[B,T,C], load_loss, load, importance, n_dropped)
```

**Step-by-step:**

1. **Flatten**: `x_flat = x.reshape(B*T, C)` — all `N = B*T` tokens treated equally.

2. **Route**: `topk_idx, topk_probs, probs = router(x_flat)` — shapes `(N, K)`.

3. **Flatten K assignments**: each token appears `K` times (once per chosen expert):
   ```
   token_ids   = arange(N).expand(K, N).T.reshape(-1)   # shape (N*K,)
   expert_ids  = topk_idx.reshape(-1)                    # shape (N*K,)
   expert_probs = topk_probs.reshape(-1)                  # shape (N*K,)
   ```

4. **Sort by expert**: sort all `(token, expert, prob)` triples by expert ID. This
   groups all tokens going to the same expert contiguously, enabling efficient sequential
   processing.

5. **Capacity enforcement**: for each expert `e`, compute how many tokens are assigned
   (`expert_alloc`) and mask out those beyond `capacity`:
   ```python
   capacity_mask = expert_alloc < capacity
   n_dropped = (~capacity_mask).sum()
   ```

6. **Dispatch**: iterate over experts; for each expert `e`:
   ```python
   mask = (expert_ids_s == e) & capacity_mask
   y[token_ids_s[mask]] += expert_e(x_flat[token_ids_s[mask]]) * probs_s[mask]
   ```
   The output is the weighted sum of each token's chosen experts' outputs.

7. **Reshape**: `y.reshape(B, T, C)`.

8. **Load loss** (auxiliary): computed from `topk_idx` and the full softmax `probs`
   (see §2.2).

---

### 4.6 GPT2 and GPT2MOE

Both the standard (`GPT2`) and MoE (`GPT2MOE`) models share the same overall structure:

```
Input token IDs  [B, T]
     ↓
Token embedding (wte) + Positional embedding (wpe)
     ↓
Dropout
     ↓
n_layer × Block (BlockFFN or BlockMOE)
     ↓
Final RMSNorm (ln_f)
     ↓
LM Head (linear, tied to wte)
     ↓
Logits  [B, T, vocab_size]
```

**Training mode** (`targets` provided): returns logits over all positions and
cross-entropy loss.

**Inference mode** (`targets=None`): only computes logits for the **last token**
(`x[:, [-1], :]`) — a micro-optimisation that avoids computing the LM head for positions
that are never used during autoregressive generation.

#### `generate()` Method

The model's `generate` method supports:
- String input (text → tokens via tokenizer).
- **Temperature scaling**: `logits /= temperature` — higher temperature = more random.
- **Top-P (nucleus) sampling**: keep the smallest set of tokens whose cumulative
  probability ≥ p; mask out the rest. This cuts off the long tail of unlikely tokens.
- **Top-K sampling**: keep only the K highest-probability tokens.
- **Min-P sampling**: keep tokens whose log-probability ≥ max_logprob + log(min_p).
  Relative threshold that adapts to the entropy of the distribution.
- **Early stopping** on EOS token.

---

## 5. Dataset Classes

All datasets inherit from `torch.utils.data.IterableDataset`. Iterable datasets are
appropriate here because:
- The data is stored in large binary files that are memory-mapped.
- We want infinite streaming (training never explicitly "finishes an epoch").
- Data sharding and interleaving is managed manually.

### `PreTokDataset`

Used for pre-training / CLM fine-tuning. Streams fixed-length chunks of tokens from
pre-tokenised `.bin` shards.

```
shard → memmap uint16 array
chunk = data[i*max_seq_len : (i+1)*max_seq_len]
x = chunk[:-1],  y = chunk[1:]    # next-token prediction
```

### `SFTDataset`

Used for supervised fine-tuning. Each example spans variable lengths.

- `data.bin`: uint16 token IDs of all examples concatenated.
- `indices.bin`: uint32 array; positions `[2i, 2i+1, 2i+2]` give `(start, mid, end)`.
  - `data[start:mid]` = prompt tokens.
  - `data[mid:end]` = answer tokens.
- `yield qa, q_len`: yields the full sequence and the prompt length so the training
  loop knows where the answer starts.

The `collator` static method pads variable-length sequences into a rectangular batch
tensor, creates a binary attention mask, and records each example's question length.

### `RLHFDataset`

Identical structure to `SFTDataset` but yields **only the prompt tokens** (up to but
not including the EOS of the prompt). Used when training an RL policy that will generate
completions on-the-fly.

### `PromptDataset`

For GRPO training. Each shard has three files:
- `data.bin`: uint16 tokenised prompts.
- `indices.bin`: uint32 cumulative token offsets.
- `data.json`: list of structured metadata dicts (one per prompt).

`yield prompt_tensor, metadata_dict` — the training loop uses the metadata to score
the model's generated completions.

The `collator` static method pads prompts, builds masks, records lengths, and collates
metadata lists. The return signature is `(seq, mask, lengths, metadata_list)`.

---

## 6. Training Modules

All training modules are PyTorch Lightning `LightningModule` subclasses. This gives:
- Automatic device placement (CPU / GPU / multi-GPU).
- Gradient accumulation, clipping, precision (fp16/bf16/fp32).
- Checkpointing.
- Logging to MLflow via `MLFlowLogger`.

### 6.1 GPT2Module (base)

The simplest module: wraps the custom `GPT2` model, runs CLM loss, logs to MLflow,
and calls `model.generate()` after each validation epoch (logging results as MLflow
artefacts).

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    logits, loss = self.model(x, y)
    self.log("train_loss", loss)
    return loss
```

The optimizer is AdamW. No scheduler is configured in this base class.

### 6.2 SFTGPT2Module / GPT2SFTModule

Two parallel implementations of SFT — one for the custom model, one for HF GPT-2.

#### Answer-only loss masking

The core difference from CLM is masking the prompt out of the loss:

```python
# (custom model, SFTGPT2Module.forward)
pos = torch.arange(T-1).unsqueeze(0).expand(B, -1)  # (B, T-1)
first_answer_pos = (question_length - 1).unsqueeze(1)  # (B, 1)
answer_mask = pos >= first_answer_pos                  # (B, T-1)

loss = (ce_loss * answer_mask).sum() / answer_mask.sum()
```

```python
# (HF model, GPT2SFTModule._build_labels)
labels = input_ids.clone()
labels = labels.masked_fill(attention_mask == 0, -100)          # padding
labels = labels.masked_fill(positions < question_lengths, -100) # prompt
# HF cross_entropy ignores index=-100 by convention
```

#### Learning rate schedule (GPT2SFTModule)

Uses a **linear warmup + cosine decay** schedule:

```
step 0 → warmup_steps:  lr grows from ~0 → base_lr
step warmup_steps → total_steps: lr decays via cosine → 0
```

This is a standard schedule for fine-tuning LLMs. The warmup prevents large early
gradient updates that can destabilise training; the cosine decay smoothly reduces the
learning rate as the model converges.

### 6.3 GRPOGPT2Module / GPT2GRPOModule

The GRPO module adds the following components on top of SFT:

#### Reference Policy

On `on_train_start`, the current model is deep-copied into `self.ref_policy`:
```python
self.ref_policy = copy.deepcopy(self).eval()
for p in self.ref_policy.parameters():
    p.requires_grad = False
```

The reference policy is frozen throughout training. Its weights are excluded from
checkpoints via `state_dict` filtering to keep checkpoint size reasonable.

#### `generate_w_logp()`

A custom autoregressive generation function that, unlike `GPT2.generate()`, also
**records the log-probabilities of the sampled tokens** at the time of generation:

```python
gen_logp[:, t] = logprobs_policy[batch_idxs, next_tok]
```

This is `log π_θ_old(a_t | s_t)` — the "old" policy log-probs needed for the PPO ratio.

**Important subtlety**: for HF GPT-2 (`GPT2GRPOModule`), the log-prob stored during
generation is computed from the **untempered** logits (`logprobs_policy`), while
sampling uses tempered + nucleus-filtered logits. This is the correct approach: the
temperature and top-p filtering change *what* is sampled but the probability ratio
should be evaluated against the true model distribution, not the filtered/tempered one.

#### `gen_token_logprobs()`

After generation, computes log-probabilities for all generated tokens in a **single
forward pass** (more efficient than token-by-token):

```python
logits = model(all_ids, attention_mask=all_masks).logits   # full sequence
log_probs = log_softmax(logits[:, :-1, :])                 # shifted

# Index into log_probs at the generated positions
token_log_probs = log_probs.gather(-1, all_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
token_log_probs = token_log_probs.gather(-1, gen_pos)      # only generation positions
```

This gives `logp_new` (the **current** policy's log-probs on the same sequences).

Token entropy is also computed here (for monitoring, not the loss):
```python
token_entropy = -(probs * log_probs).sum(dim=-1)
```

#### `grpo_loss()`

Implements the combined PPO + KL loss:

```python
ratio = exp(logp_new - logp_old)                # π_θ / π_θ_old, per token

# PPO clipped surrogate
adv1 = ratio * advantages                       # unclipped
adv2 = clip(ratio, 1-ε, 1+ε) * advantages      # clipped
ppo_term = -mean( min(adv1, adv2) * gen_mask )  # per sequence, then averaged

# KL penalty (forward KL approximation)
kl_term = mean( (logp_new - logp_ref) * gen_mask )

loss = ppo_term + β * kl_term
```

**Why `min(unclipped, clipped) * adv`?**
- When advantage > 0 (good completion): we want to increase the probability. The clip
  prevents over-increasing by capping the ratio at `1+ε`.
- When advantage < 0 (bad completion): we want to decrease the probability. The clip
  prevents over-decreasing by flooring the ratio at `1-ε`.
Taking the `min` of the two picks the more conservative (pessimistic) update.

#### `_loss()` method

The full training step logic:

```python
# 1. Expand batch: B prompts → B*G prompts (G completions per prompt)
prompt_ids = prompt_ids.expand(G, B, ...).contiguous().view(B*G, ...)

# 2. Generate G completions from current policy
all_ids, all_masks, all_lens, _, gen_masks, logp_old = generate_w_logp(...)

# 3. Compute new-policy log-probs (differentiable)
logp_new, avg_entropy = gen_token_logprobs(...)

# 4. Compute reference-policy log-probs (no grad)
with torch.no_grad():
    logp_ref, _ = ref_policy.gen_token_logprobs(...)

# 5. Score completions and normalise within groups
judgements = judge.score_grouped_from_token_ids(...)

# 6. PPO + KL loss
loss = grpo_loss(logp_new, logp_old, logp_ref, gen_masks, judgements["advantages"])
```

---

## 7. Reward Functions

Two reward implementations co-exist. `StoryReward` (embedding-based) is the primary
one used in the current training scripts; `Reward` (heuristic) is the earlier version.

### 7.1 Heuristic Reward (`heuristic.py`)

`Reward` scores a completion using pure text statistics — no neural networks required.

| Component | Weight | What it measures |
|---|---|---|
| `prompt_adherence` | 0.35 | Keyword overlap between prompt and completion |
| `coherence` | 0.25 | Sentence flow, temporal connectives, story structure |
| `style` | 0.15 | Sentence length, word length, positive tone, dialogue |
| `length` | 0.10 | Word count in preferred range (60–180 words) |
| `safety` | 0.05 | Absence of unsafe/violent/explicit words |
| `fluency` | 0.10 | Capitalisation, terminal punctuation, weird chars |
| `degeneracy_penalty` | 0.25 | Repeated n-grams, repeated sentences, trailing fragments |

The raw weighted sum is clamped to `[0, 1]` then scaled to `[-1, 1]`:
```
reward = 2.0 * raw - 1.0
```

The `score_from_concat_ids()` method takes token IDs directly (no pre-decoded text),
decodes prompt and completion separately (stripping BOS/EOS/PAD), then calls
`score_texts()`.

`compute_grpo_advantages()` accepts a flat rewards tensor and group indices to compute
within-group normalised advantages (same algorithm as `StoryReward.compute_grpo_advantages`).

### 7.2 Embedding-Based Reward (`embedding.py`)

`StoryReward` is a richer reward function that uses:
- **[spaCy](https://spacy.io/)** (`en_core_web_sm`) for NLP: tokenisation, lemmatisation,
  POS tagging, sentence segmentation.
- **[SentenceTransformers](https://www.sbert.net/)** (`all-MiniLM-L6-v2`) for semantic
  sentence embeddings.

#### Scoring Breakdown

| Component | What it evaluates |
|---|---|
| `words` | Fraction of required words present (exact/lemma match) |
| `pos` | Fraction of required words used in the correct part of speech |
| `subject` | How well the story incorporates the specified character, action, place, etc. |
| `features` | How well each required story feature is present |
| `format` | Text length, sentence count, and basic cleanliness |
| `coherence` | Semantic similarity between adjacent sentences |

#### Penalty Terms

| Penalty | What it penalises |
|---|---|
| `prompt_copy` | Copying prompt text verbatim into the completion |
| `meta_language` | Writing *about* the task ("this story", "required words", …) |
| `repetition` | Repeated unigrams and bigrams |
| `keyword_stuffing` | Using the same required word too many times |
| `gibberish` | Low vocabulary diversity, repeated runs, suspicious tokens |

#### Feature Scoring Details

Each of the six story features is scored semi-independently:

- **BadEnding**: keyword score (sad/dead/failed…) + semantic similarity to prototype
  sentences + sentiment proxy in the final 2 sentences.
- **Conflict**: keyword fraction (fight/danger/struggle…) + semantic similarity +
  opposition markers (but/however/against…).
- **Dialogue**: presence of quoted strings + reporting verbs (say/ask/reply…).
- **Foreshadowing**: vocabulary overlap between early and late story halves + payoff
  markers (finally/later/turned out…) + recurrence of early nouns in the final sentence.
- **MoralValue**: moral cue patterns (learned that / the lesson / from then on…) +
  semantic similarity to moral prototype sentences.
- **Twist**: twist markers (suddenly/however/revealed…) + semantic similarity +
  dissimilarity between opening and ending sentences.

#### Subject Scoring

Scored per sub-field with different weights:
- **character** (weight 1.0): lexical mention + semantic similarity.
- **adjective** (weight 0.6): checks if the adjective appears near the character token.
- **place** (weight 0.9): lexical mention + semantic similarity.
- **action** (weight 1.1): sentence-level semantic match + content-word overlap.
- **goal** (weight 1.0): same as action.

#### Composite Score Calculation

The positive score is a **weighted average over active components only** — components
that are not relevant to the current prompt (e.g. `subject` when no subject was
specified) are excluded from both numerator and denominator.

The final score:
```python
raw_total = positive_score - penalty_total
total = clip(raw_total, -1.0, 1.0)
```

#### `score_grouped_from_token_ids()`

Convenience method for GRPO:

```python
# Expects B = num_prompts * group_size rows
# metadata_list has num_prompts entries — internally expanded to B
_metadata_list = [md for md in metadata_list for _ in range(group_size)]
out = score_from_token_ids(input_ids, attention_mask, prompt_lengths, _metadata_list)
advantages = compute_grpo_advantages(out["rewards"], group_size)
```

---

## 8. Callbacks and Infrastructure

### `UploadLastAndBestToMLflow`

Triggered at the end of every `n`-th validation epoch. Uploads the latest and best
checkpoints to the active MLflow run as artefacts under `checkpoints/last.ckpt` and
`checkpoints/best.ckpt`. Uses a stable file name so successive uploads *overwrite*
the previous version (no accumulation of hundreds of checkpoint files).

### `SetCheckpointDirCallback`

Ensures that Lightning's `ModelCheckpoint` saves `.ckpt` files inside the MLflow
run's local artefact directory, so they are automatically tracked.

### `LightningPyfunc`

Wraps a Lightning module as an **MLflow PyFunc model**. This makes the model
deployable via the standard MLflow serving stack: `mlflow models serve -m …`.

### `LogBestCkptAndPyfuncToMLflow`

Triggered at `on_fit_end`. Uploads the best checkpoint and the PyFunc wrapper to MLflow.
Optionally registers the model in the MLflow Model Registry.

### `MLflowGenerationCallback`

Triggered at the end of every validation epoch. Calls `pl_module.generate_from_prompts()`
on a list of fixed evaluation prompts, formats the results, and uploads them to MLflow
as a text artefact. This gives a human-readable view of how generation quality evolves
during training.

### `utils.py`

`_download_resume_ckpt()`: locates a named MLflow run in a given experiment and
downloads either `last.ckpt` or `best.ckpt` to a local directory. Used to resume
training from a previous run.

---

## 9. Training Scripts

### `preprocess.py`

Run first to prepare data:

```bash
# Download datasets
python preprocess.py download

# Train a 4096-token SentencePiece vocabulary
python preprocess.py train-vocab --vocab-size 4096

# Tokenise SFT dataset
python preprocess.py pretokenize-sft

# Tokenise RLHF prompts dataset
python preprocess.py pretokenize-rlhf
```

### `train_sft_hf.py` (HuggingFace GPT-2 SFT)

Supervised fine-tuning on the custom prompt+completion dataset.

Key hyperparameters:
```
MODEL_NAME     = "gpt2"          # HF model hub name
BATCH_SIZE     = 8
GRAD_ACC_STEPS = 8               # Effective batch = 64
LR             = 3e-5
WARMUP_RATIO   = 0.05
MAX_STEPS      = 20000
```

Shell invocation (via `run_train_sft_hf.sh`):
```bash
python train_sft_hf.py \
  --exp-name my-sft-experiment \
  --run-prefix sft-run \
  --model-name my-model \
  --max-steps 20000 \
  --lr 3e-5 \
  --batch-size 8 \
  --grad-accum-step 8
```

### `train_grpo_hf.py` (HuggingFace GPT-2 GRPO)

GRPO fine-tuning on top of an SFT checkpoint.

Key hyperparameters:
```
MODEL_NAME      = "gpt2"
NUM_GEN         = 8              # G completions per prompt
MAX_SEQ_LEN     = 248            # Maximum generation length (tokens)
TEMPERATURE     = 1.0            # Sampling temperature during rollout
TOP_K / TOP_P   = 50 / 0.95     # Nucleus + top-K sampling
KL_BETA         = 0.02           # KL penalty coefficient
CLIP_EPS        = 0.2            # PPO clipping ε
LR              = 3e-5
WARMUP_RATIO    = 0.05
BATCH_SIZE      = 2
GRAD_ACC_STEPS  = 16             # Effective batch = 32
MAX_STEPS       = 1000
```

Note: `BATCH_SIZE` is small because each "batch" is expanded by `NUM_GEN` (×8) before
generation, making the true computational batch size `2 × 8 = 16` sequences, each up to
248 tokens. GRPO is memory-intensive.

Shell invocation (via `run_train_grpo_hf.sh`):
```bash
python train_grpo_hf.py \
  --init-model-path checkpoints/sft-best.ckpt \
  --exp-name my-grpo-experiment \
  --run-prefix grpo-run \
  --model-name my-grpo-model \
  --num-gen 8 \
  --max-seq-len 248 \
  --kl-beta 0.02 \
  --clip-eps 0.2 \
  --lr 3e-5
```

GRPO **loads from an existing SFT checkpoint** via
`GPT2GRPOModule.load_from_checkpoint(args.init_model_path, ...)`. This is important:
GRPO is not trained from scratch — it starts from a policy that already knows how to
write coherent stories, and refines it to score higher on the reward function.

### `train.py` (Custom model, manual training loop)

A lower-level training loop for the custom `GPT2` / `GPT2MOE` model without Lightning.
Useful for rapid experimentation or when you want full control over the training loop.

### `start_mlflow_server.sh`

```bash
mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Starts a local MLflow tracking server. Both training scripts point to
`file:{BASE_DIR}/mlruns` which also works without a running server (file-based tracking).

---

## 10. End-to-End Workflow

```
                         ┌─────────────────────────────────────────┐
                         │            DATA PREPARATION              │
                         │                                          │
                         │  1. Download TinyStories / MetaMathQA   │
                         │  2. Train SentencePiece vocabulary       │
                         │  3. Generate story prompts (prompter.py) │
                         │  4. Send prompts to LLM → get stories    │
                         │     (build_prompts.py + batch API)       │
                         │  5. Parse LLM responses → binary shards  │
                         │     (parse_results.py)                   │
                         │  6. Tokenise prompt-only shards          │
                         │     for GRPO (pretokenize-rlhf)          │
                         └──────────────────┬──────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────┐
                         │         SUPERVISED FINE-TUNING           │
                         │                                          │
                         │  Input: (prompt, story) pairs            │
                         │  Loss:  CE on story tokens only          │
                         │  Model: GPT-2 (HF) or custom GPT2        │
                         │  Script: train_sft_hf.py                 │
                         │  Output: SFT checkpoint (best.ckpt)      │
                         └──────────────────┬──────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────┐
                         │      GROUP RELATIVE POLICY OPTIM.        │
                         │                                          │
                         │  Input: prompts with metadata            │
                         │  Rollout: generate G completions each    │
                         │  Reward: StoryReward (multi-dimensional) │
                         │  Loss:  PPO clip + KL(θ ‖ θ_ref)        │
                         │  Script: train_grpo_hf.py                │
                         │  Output: GRPO-refined checkpoint         │
                         └──────────────────┬──────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────┐
                         │              EVALUATION                  │
                         │                                          │
                         │  - MLflow logs: loss, reward, KL, adv   │
                         │  - Generated text artefacts per epoch    │
                         │  - Reward component breakdowns           │
                         └─────────────────────────────────────────┘
```

---

## 11. Key Design Decisions & Trade-offs

### Weight Tying (lm_head ↔ wte)
Reduces parameters and regularises training. The downside is that it creates a coupling
between the embedding space and the output space, which can theoretically limit
expressiveness — but in practice is a net positive for small models.

### RMSNorm over LayerNorm
Faster (no mean subtraction), empirically equivalent or better. Pre-norm placement
(before the sub-layer, not after) is now standard; it provides better gradient flow than
post-norm.

### SwiGLU over GeLU
The gated linear unit introduces a multiplicative interaction that gives each neuron more
expressive power. The cost is an extra weight matrix (`W3`) and slightly more compute.

### Flash Attention
Dramatically reduces memory from O(T²) to O(T) by tiling the computation. No accuracy
loss — it computes exactly the same attention as the manual path.

### Noisy Gating in MoE
Without noise, the router quickly converges to always choosing the same expert for each
token type, defeating the purpose of having multiple experts. Noise encourages all experts
to be useful early in training, after which they can specialise.

### Capacity Dropping
Dropping tokens that overflow an expert's capacity is a pragmatic choice that keeps
batch computation rectangular (no variable-length expert inputs). The alternative —
dynamic routing with no capacity — requires more complex scatter/gather operations. Token
dropping is acceptable because a small fraction of tokens being silently ignored has
minimal effect on language modelling.

### GRPO vs PPO
GRPO's group-normalised advantage estimate has higher variance than PPO's value-function
baseline for prompts with very different difficulties — the group std can be near zero when
all G completions have the same reward. The `eps` term prevents division-by-zero, but
near-zero std means the advantage signal is very noisy. A practical remedy is to ensure
`G` is large enough (≥ 8) and that the reward function has sufficient spread.

### Generation Log-probs
A subtle correctness issue: the log-probs stored during generation (`logp_old`) and the
log-probs recomputed from the frozen forward pass (`logp_new`) must agree when the policy
hasn't changed. The codebase includes sanity-check prints:
```python
print("max difference between new and old", (exp(logp_new - logp_old) - 1).abs().max())
```
This should be ~0 at the start of training. A large value indicates a bug (e.g. different
sampling temperatures applied to the log-prob computation vs. the generation step).

### Checkpoint Serialisation of the Reference Policy
The reference policy is a deep copy of the model at GRPO start. If it were saved to
every checkpoint, checkpoint files would double in size. The `state_dict` override
filters out `ref_policy.*` keys. On load, the reference policy is reconstructed from
the loaded weights via `_init_ref_policy()`.

### Structured Reward Metadata
The prompt generator stores structured metadata (required words with POS, feature names,
subject fields) alongside the raw prompt text. The reward function uses this metadata
to perform targeted scoring (e.g. it checks that "cane" was used as a *noun*, not just
that the word appeared). This is substantially more informative than a simple
surface-form match.

### Batch-Mode LLM Story Generation
The SFT dataset is not sourced from human writers but from an LLM (referenced as
"gpt-5.4" in `build_prompts.py`, likely a placeholder name). The batch API workflow
allows generating tens of thousands of (prompt, story) pairs cheaply. This is knowledge
distillation / model distillation applied at the data level: the small GPT-2 model learns
to imitate the output distribution of a much larger teacher model on a constrained task.
