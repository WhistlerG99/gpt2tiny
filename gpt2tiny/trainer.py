import os
import torch
from typing import List, Optional
from dataclasses import dataclass
from .model import GPT2
import pytorch_lightning as pl
import torch.nn as nn


@dataclass
class TrainingConfig:
    learning_rate: float = 6e-4
    max_iters: int = 30000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5

    eval_interval: int = 100
    log_interval: int = 10
    eval_iters: int = 200
    gradient_accumulation_steps: int = 4
    batch_size: int = 64
    num_workers: int = 1

    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "bfloat16"
    compile: bool = True


class GPT2Module(pl.LightningModule):
    def __init__(
        self,
        config,
        tokenizer,
        prompts: Optional[List[str]] = ["A dragon in a cave"],
        max_seq_len: int = 128,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.8,
        gen_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT2(config)
        self.tokenizer = tokenizer

        self.prompts = prompts
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.gen_every_n_epochs = gen_every_n_epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.model(x, y, attention_mask=None)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.model(x, y, attention_mask=None)
        self.log("val_loss", loss, prog_bar=True)

        # if getattr(self.trainer, "sanity_checking", False):
        #     return  # don't log artifacts during sanity check
        
        if (
            batch_idx == 0
            and self.trainer.is_global_zero
            and (self.current_epoch % self.gen_every_n_epochs == 0)
        ):
            self._log_generation_to_mlflow()

        return loss

    @torch.no_grad()
    def _log_generation_to_mlflow(self):
        if self.prompts is None:
            return 
        
        # --- log to MLflow ---
        logger = getattr(self, "logger", None)
        mlf = getattr(logger, "experiment", None) if logger is not None else None
        run_id = getattr(logger, "run_id", None) if logger is not None else None
        if mlf is None or run_id is None:
            # Not using MLFlowLogger (or no active run)
            self.print(f"[epoch={self.current_epoch}] sample generation:\n{text}\n")
            return

        os.makedirs("mlflow_artifacts", exist_ok=True)
        fname = f"gen_epoch_{self.current_epoch:04d}_step_{self.global_step:09d}.txt"
        fpath = os.path.join("mlflow_artifacts", fname)

        self.model.eval()

        with open(fpath, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(self.prompts):
                text = self.model.generate(
                    prompt,
                    self.max_seq_len,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    tokenizer=self.tokenizer,
                    temperature=self.temperature,
                )
                
                f.write(f"prompt: {prompt}\n")
                f.write(f"max_seq_len: {self.max_seq_len}\n")
                f.write(f"top_k: {self.top_k}\n")
                f.write(f"top_p: {self.top_p}\n")
                f.write(f"temperature: {self.temperature}\n")
                f.write("\n---\n")
                f.write(text)
                if i+1<len(self.prompts):
                    f.write("\n\n+++++++++++++\n\n")

        mlf.log_artifact(run_id, fpath, artifact_path="generations")

        # Optional: short preview as a tag
        preview = str(text).replace("\n", " ")[:200]
        mlf.set_tag(run_id, f"gen_preview_epoch_{self.current_epoch}", preview)

    def configure_optimizers(self, lr=0.001):
        return torch.optim.AdamW(self.parameters(), lr=lr)



class SFTGPT2Module(GPT2Module):

    def forward(self, idx, attention_mask, question_length=None):
        B, T = idx.shape
        device = idx.device
        
        x = self.model.transformer.wte(idx)

        if attention_mask is not None:
            pos = torch.cumsum(attention_mask, dim=-1) - 1
        else:
            pos = torch.arange(T, device = idx.device, dtype=torch.long)

        pos_emb = self.model.transformer.wpe(pos)
    
        x = x + pos_emb
    
        for block in self.model.transformer.h:
            x  = block(x, attention_mask=attention_mask)
        x = self.model.transformer.ln_f(x)
    
        logits = self.model.lm_head(x)

        if question_length is not None:
            logits_shifted = logits[:,:-1,:].contiguous()
            targets = idx[:,1:].contiguous()
        
            loss = nn.functional.cross_entropy(
                logits_shifted.view(-1, logits_shifted.shape[-1]), targets.view(-1), reduction="none",
            ).view(B,-1)
        
            pos = torch.arange(T - 1, device=device).unsqueeze(0).expand(B, -1)  # (B, T-1)
            first_answer_label_pos = (question_length - 1).unsqueeze(1)             # (B, 1)
            answer_mask = pos >= first_answer_label_pos  
        
            if attention_mask is not None:
                answer_mask = answer_mask * attention_mask[:,1:]
            
            loss = (loss * answer_mask).sum() / answer_mask.sum().clamp_min(1)
        else:
            loss = None
            
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self(*batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self(*batch)
        self.log("val_loss", loss, prog_bar=True)

        if (
            batch_idx == 0
            and self.trainer.is_global_zero
            and (self.current_epoch % self.gen_every_n_epochs == 0)
        ):
            self._log_generation_to_mlflow()

        return loss    

    def configure_optimizers(self, lr=1e-5):
        return torch.optim.AdamW(self.parameters(), lr=lr)


class RLHFGPT2Module(SFTGPT2Module):

    @torch.no_grad()
    def generate_w_logp(
        self,
        prompt_ids: torch.Tensor,
        prompt_masks: torch.Tensor,
        prompt_lens: torch.Tensor,
        max_seq_len: int,
        temperature: float = 1.0,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = None,
        pad_id: int = 0,
        eos_id: int = 2,
    ):
        
        device = self.device
        BG,T = prompt_ids.shape
    
        assert max_seq_len > T, "`max_seq_len` must be larger than second dimension of `prompt_ids`"
        
        all_ids = torch.full((BG, max_seq_len), fill_value=pad_id, dtype=torch.long, device=device)
        all_masks = torch.zeros((BG, max_seq_len), dtype=torch.long, device=device)
    
        finished = torch.zeros_like(prompt_lens).to(torch.bool)
        
        all_ids[:, :T] = prompt_ids
        all_masks[:, :T] = prompt_masks
        all_lens = prompt_lens.clone()
    
        gen_ids = torch.full((BG, max_seq_len - T), fill_value=pad_id, dtype=torch.long, device=device)    
        gen_masks = torch.zeros((BG, max_seq_len - T), dtype=torch.long, device=device)    
        gen_logp = torch.zeros((BG, max_seq_len - T), dtype=torch.float32, device=device)    
    
        for t in range(max_seq_len-T):
            max_len = int(all_lens.max().item())
            batch = all_ids[:, :max_len], all_masks[:,:max_len]
            logits, _ = self(*batch)
            
            batch_idxs = torch.arange(0, BG, device=device)
            logits_next = logits[batch_idxs, all_lens-1, :]
            logits_next /= temperature
            
            
            if top_k is not None:
                topk_logits, _ = torch.topk(logits_next, k=top_k, dim=-1)
                cutoffs = topk_logits[:,[-1]]
                logits_next[logits_next < cutoffs] = -float('inf')
                
            
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits_next, descending=True, dim=-1)
                sorted_probs = nn.functional.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                
                # Mask tokens with cumulative prob > top_p (keep at least 1 token)
                sorted_mask = cumprobs > top_p
                sorted_mask[:, 0] = False
                
                filtered_sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
                # Unsort back to vocab order
                logits_next = torch.full_like(logits_next, -float("inf"))
                logits_next = logits_next.scatter_(dim=-1, index=sorted_idx, src=filtered_sorted_logits)
            
            logprobs = nn.functional.log_softmax(logits_next, dim=-1)
            probs = nn.functional.softmax(logits_next, dim=-1)
            
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tok = torch.where(finished, pad_id, next_tok)
            
            gen_logp[:, t] = torch.where(finished, 0, logprobs[batch_idxs, next_tok])
            gen_ids[:, t] = next_tok
            gen_masks[:, t] = ~finished
            
            all_ids[batch_idxs, all_lens] = next_tok
            all_masks[batch_idxs, all_lens] = torch.where(finished, 0, 1)
            all_lens += torch.where(finished, 0, 1)
            finished = finished | (next_tok == eos_id)
    
            if finished.all():
                break
                            
        return all_ids, all_masks, all_lens, gen_ids, gen_masks, gen_logp

    def gen_token_logprobs(
        self,
        all_ids,
        all_masks,
        prompt_lengths,
        gen_masks,
    ):
        B, T = gen_masks.shape
        logits, _ = self(all_ids, attention_mask=all_masks)
        
        log_probs= nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        
        log_probs = log_probs.gather(dim=-1, index=all_ids[:, 1:].unsqueeze(-1)).squeeze()
    
        gen_pos = torch.arange(T, device=all_ids.device, dtype=torch.long).expand(B,-1) 
        gen_pos = gen_pos + (prompt_lengths - 1).view(B,-1)
        
        log_probs = log_probs.gather(dim=-1, index=gen_pos)
    
        log_probs = log_probs * gen_masks.to(log_probs.dtype)
        return log_probs

    def grpo_loss(
        self,
        logp_new,
        logp_old,
        gen_masks,
        advantages,
        eps = 0.2,
    ):
        ratio = torch.exp(logp_new - logp_old)
        
        expected_adv1 = ratio * advantages.unsqueeze(-1)
        expected_adv2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages.unsqueeze(-1)
        expected_adv = torch.minimum(expected_adv1, expected_adv2)
        
        loss = -(expected_adv * gen_masks).sum(dim=1) / gen_masks.sum(dim=1).clamp_min(1)
        return loss.mean()

    
    def configure_optimizers(self, lr=1e-5):
        return torch.optim.AdamW(self.parameters(), lr=lr)
