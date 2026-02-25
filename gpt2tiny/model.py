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
    n_expert: int = 2
    k: int = 1
    dropout: float = 0.2
    bias: bool = False
    use_rotary: bool = False
    flash: bool = True
    noisy_gating: bool = True
    capacity_factor: int = 10
    load_loss_coef: float = 1e-2
    


@dataclass
class MOEOutput:
    y: torch.Tensor                    # [B, T, D]
    load_loss: torch.Tensor             # scalar
    load: torch.Tensor                 # [E] fraction of tokens routed to each expert
    importance: torch.Tensor           # [E] sum of routing probs to each expert
    n_dropped: torch.Tensor            # scalar (tokens dropped due to capacity)


class ExpertNN(nn.Module):
    def __init__(self, n_embed, n_hidden, dropout=0.1):
        super().__init__()
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.fc1 = nn.Linear(n_embed, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.silu(x)
        x = self.fc2(self.dropout(x))
        return x


class TopKRouter(nn.Module):
    def __init__(self, n_embed, n_expert, k=1, noisy_gating=True):
        super().__init__()
        self.n_expert = n_expert
        self.k = k
        self.n_embed = n_embed
        self.noisy_gating = noisy_gating
        self.gate = nn.Linear(n_embed, n_expert, bias=False)

        if noisy_gating:
            self.noisy_gate = nn.Linear(n_embed, n_expert, bias=False)

    
    def forward(self, x):
        logits = self.gate(x) # [N, C] -> [N, E]

        if self.noisy_gating and self.training:
            noise_std = nn.functional.softplus(self.noisy_gate(x)) + 1e-9
            logits = logits + torch.randn_like(logits) * noise_std
        
        probs = nn.functional.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=self.k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        return topk_idx, topk_probs, probs


class MOE(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        
        n_hidden = 4 * config.n_embed
        n_hidden = int(2 * n_hidden / 3)
        
        self.router = TopKRouter(
            config.n_embed,
            config.n_expert,
            k=config.k,
            noisy_gating=config.noisy_gating
        )
        self.experts = nn.ModuleList([ExpertNN(config.n_embed, n_hidden, config.dropout) for _ in range(config.n_expert)])


    def _load_balancing_loss(self, topk_idx, probs):
        E = self.config.n_expert
        
        load_cnts = torch.bincount(topk_idx[:,0], minlength=E)
        load_frac = load_cnts / (load_cnts.sum() + 1e-9)
        
        importance = probs.sum(dim=0) / (probs.sum() + 1e-9)
        load_loss = self.config.load_loss_coef * E * (load_frac * importance).sum()
        return load_loss, load_frac, importance

    
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)
        N = x_flat.size(0)

        E = self.config.n_expert
        k = self.config.k
        
        capacity = int(math.ceil(self.config.capacity_factor * N / E))
        capacity = max(capacity, 1)
        
        topk_idx, topk_probs, probs = self.router(x_flat)

        token_ids = torch.arange(N, device = x.device).expand(k, N).transpose(1,0).reshape(-1)
        expert_ids = topk_idx.reshape(-1)
        expert_probs = topk_probs.reshape(-1)

        
        expert_idx_s = torch.argsort(expert_ids)
        token_ids_s = token_ids[expert_idx_s]
        expert_ids_s = expert_ids[expert_idx_s]
        expert_probs_s = expert_probs[expert_idx_s]


        is_same = torch.ones_like(expert_ids_s, dtype=torch.bool)
        is_same[1:] = expert_ids_s[1:] != expert_ids_s[:-1]
        
        start_idx = torch.where(is_same)[0]
        end_idx = torch.cat((start_idx[1:],torch.tensor([expert_ids_s.numel()], device=x.device)))
        
        expert_alloc = end_idx - start_idx
        expert_alloc = torch.cat([torch.arange(n, device=x.device) for n in expert_alloc])
        capacity_mask = expert_alloc < capacity

        n_dropped = (~capacity_mask).sum()

        y = torch.zeros_like(x_flat)
        
        for e, expert in enumerate(self.experts):
            mask = (expert_ids_s==e) & capacity_mask
            y[token_ids_s[mask]] += expert(x_flat[token_ids_s[mask]]) * expert_probs_s[mask].unsqueeze(-1)
        y = y.reshape(B, T, C)
        
        load_loss, load_frac, importance = self._load_balancing_loss(topk_idx, probs)
        return MOEOutput(y, load_loss, load_frac, importance, n_dropped)


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

    def forward(self, x, attention_mask=None):
        config = self.config
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.config.n_embed, dim=2)
        
        q = q.view(B, T, config.n_head, C // config.n_head).transpose(1,2)
        k = k.view(B, T, config.n_head, C // config.n_head).transpose(1,2)
        v = v.view(B, T, config.n_head, C // config.n_head).transpose(1,2)
        
        key_mask = None
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool) # (B, 1, 1, T)
        
        if self.flash:
            attn_mask = None
            if key_mask is not None:
                attn_mask = key_mask.expand(B, 1, T, T) 
                
            y = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p = self.config.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            attn_pattern = (q @ k.transpose(-2,-1)) / math.sqrt(k.shape[-1])
            
            attn_pattern = attn_pattern.masked_fill(self.bias[: ,:, :T, :T] == 0, float("-inf"))

            if key_mask is not None:
                attn_pattern = attn_pattern.mak_fill(~key_mask, float("-inf"))                
            
            attn = nn.functional.softmax(attn_pattern, dim=-1)
            y = attn @ v
        
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))

        if attention_mask is not None:
            y = y * attention_mask[:, :, None].to(dtype=y.dtype)
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



class BlockFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embed)
        self.ffd = FeedForward(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.ffd(self.ln_2(x))
        return x


class BlockMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embed)
        self.moe = MOE(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)

        x_moe = self.moe(self.ln_2(x))
        x = x + x_moe.y
        load_loss = x_moe.load_loss
        return x, load_loss


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer_dict = {
            "wte": nn.Embedding(config.vocab_size, config.n_embed),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([BlockFFN(config) for _ in range(config.n_layer)]),
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


    def forward(self, idx, targets=None, attention_mask=None):
        B, T = idx.shape

        x = self.transformer.wte(idx)

        pos_emb = self.transformer.wpe(
            torch.arange(T, device = idx.device, dtype=torch.long)
        )

        x = x + pos_emb

        for block in self.transformer.h:
            x  = block(x, attention_mask=attention_mask)
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

        prompt_len = idx.shape[-1]
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
            result = tokenizer.decode(idx[0,prompt_len:].tolist())
        else:
            result = idx[:,prompt_len:]

        return result


class GPT2MOE(GPT2):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer_dict = {
            "wte": nn.Embedding(config.vocab_size, config.n_embed),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([BlockMOE(config) for _ in range(config.n_layer)]),
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

    def forward(self, idx, targets=None, attention_mask=None):
        B, T = idx.shape

        x = self.transformer.wte(idx)

        pos_emb = self.transformer.wpe(
            torch.arange(T, device = idx.device, dtype=torch.long)
        )

        x = x + pos_emb

        total_load_loss = torch.tensor(0, dtype=torch.float32, device=idx.device)
        for block in self.transformer.h:
            x, load_loss  = block(x, attention_mask=attention_mask)
            total_load_loss += load_loss
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(
                
                logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1
            )
            loss += total_load_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
  
      
