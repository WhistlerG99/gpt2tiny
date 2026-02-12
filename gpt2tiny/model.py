import math
import torch
import random
from torch import nn
from dataclasses import dataclass
import logging

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 512
    dropout: float = 0.2
    bias: bool = False
    use_rotary: bool = False
    flash: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = config.n_embed // config.n_head
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = config.flash and hasattr(nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0),
            )

    def forward(self, x):
          config = self.config
          B, T, C = x.shape
          q, k, v = self.c_attn(x).split(self.config.n_embed, dim=2)
          
          q = q.view(B, T, config.n_head, C // config.n_head).transpose(1,2)
          k = k.view(B, T, config.n_head, C // config.n_head).transpose(1,2)
          v = v.view(B, T, config.n_head, C // config.n_head).transpose(1,2)


          if self.flash:
              y = nn.functional.scaled_dot_product_attention(
                  q,
                  k,
                  v,
                  attn_mask=None,
                  dropout_p = self.config.dropout if self.training else 0,
                  is_causal=True,
              )
          else:
              attn_pattern = (q @ k.transpose(-2,-1)) / math.sqrt(k.shape[-1])

              attn_pattern = attn_pattern.masked_fill(self.bias[: ,:, :T, :T] == 0, float("-inf"))

              attn = nn.functional.softmax(attn_pattern, dim=-1)
              y = attn @ v

          y = y.transpose(1,2).contiguous().view(B, T, C)

          y = self.resid_dropout(self.c_proj(y))

          return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = 4 * config.n_embed
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embed, bias=False)
        self.w3 = nn.Linear(config.n_embed, hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embed)
        self.ffd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffd(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer_dict = {
            "wte": nn.Embedding(config.vocab_size, config.n_embed),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.RMSNorm(config.n_embed),
        }

        self.transformer_dict["wpe"] = nn.Embedding(config.block_size, config.n_embed)

        self.transformer = nn.ModuleDict(self.transformer_dict)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0, std = 0.2 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        x = self.transformer.wte(idx)

        pos_emb = self.transformer.wpe(
            torch.arange(T, device = idx.device, dtype=torch.long)
        )

        x = x + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(
                
                logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


    @torch.no_grad()
    def generate(
        self,
        x,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        min_p=None,
        tokenizer=None,
    ):
        if isinstance(x,str):
            if tokenizer is None:
                raise ValueError("tokenizer must be specified if 'x' is string")
            idx = tokenizer.encode(x, bos=True, eos=True)
            idx = torch.tensor(idx, dtype=torch.long).unsqueeze(0)
        else:
            idx = x            

        device = next(self.parameters()).device
        idx = idx.to(device)
        
        for _ in range(max_new_tokens):
            context = (
                idx
                if idx.size(1) < self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(context)

            logits = logits[:, -1, :] / temperature

            if top_p is not None and top_p > 0.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                mask = cumulative_probs >= top_p
                mask[..., 0] = True

                cutoff_indices = mask.int().argmax(dim=-1, keepdim=True)

                top_p_mask = torch.zeros_like(logits, dtype=torch.bool)
                for b in range(logits.size(0)):
                    cut = cutoff_indices[b].item()
                    kept_indices = sorted_indices[b, : cut + 1]
                    top_p_mask[b, kept_indices] = True
                logits[~top_p_mask] = float("-inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if min_p is not None and min_p > 0.0:
                logit_max = logits.max(dim=-1, keepdim=True).values
                threshold = logit_max + torch.log(
                    torch.tensor(min_p, device=logits.device, dtype=logits.dtype)
                )
                logits[logits < threshold] = float("-inf")

            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == 2:
                break
            idx = torch.cat([idx, idx_next], dim=-1)
        
        if tokenizer:
            result = tokenizer.decode(idx[0].tolist())
        else:
            result = idx

        return result

  
      
