import os
import torch
import copy
import yaml
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .model import GPT2
from .reward import Reward
from .reward_v2 import (
    StoryReward,
    StoryRewardConfig,
    RewardWeights,
)
import pytorch_lightning as pl
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
import torch.nn as nn
import textwrap
from transformers import AutoTokenizer, GPT2LMHeadModel


def round_floats(obj, n=2):
    """Recursively rounds floats in a nested dictionary/list."""
    if isinstance(obj, float):
        return round(obj, n)
    elif isinstance(obj, dict):
        return {k: round_floats(v, n=n) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, n=n) for i in obj]
    return obj


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
        lr: float = 1e-5,
        prompts: Optional[List[str]] = ["A dragon in a cave"],
        max_seq_len: int = 128,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.8,
        gen_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
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
            f.write(f"max_seq_len: {self.max_seq_len}\n")
            f.write(f"top_k: {self.top_k}\n")
            f.write(f"top_p: {self.top_p}\n")
            f.write(f"temperature: {self.temperature}\n")
            f.write("\n\n+++++++++++++\n\n")

            for i, prompt in enumerate(self.prompts):
                text = self.model.generate(
                    prompt,
                    self.max_seq_len,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    tokenizer=self.tokenizer,
                    temperature=self.temperature,
                )
                
                prompt = '\n'.join([textwrap.fill(line, width=100) for line in prompt.splitlines()])
                text = '\n'.join([textwrap.fill(line, width=100) for line in text.splitlines()])
                
                f.write(f"prompt:\n{prompt}\n")
                f.write("\n---\n\n")
                f.write(f"completion:\n{text}")
                if i+1<len(self.prompts):
                    f.write("\n\n+++++++++++++\n\n")

        mlf.log_artifact(run_id, fpath, artifact_path="generations")

        # Optional: short preview as a tag
        preview = str(text).replace("\n", " ")[:200]
        mlf.set_tag(run_id, f"gen_preview_epoch_{self.current_epoch}", preview)

    def configure_optimizers(self):
        print(f"lr = {self.lr}")
        return torch.optim.AdamW(self.parameters(), lr=self.lr)



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


class GRPOGPT2Module(SFTGPT2Module):

    def __init__(
        self,
        config,
        tokenizer,
        lr: float = 1e-5,
        prompts: Optional[List[str]] = ["A dragon in a cave"],
        max_seq_len: int = 128,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.8,
        gen_every_n_epochs: int = 1,
        num_gen: int = 10,
        clip_eps: float = 0.2,
        kl_beta: float = 0.02,
        reward_config: Optional[StoryRewardConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
    ):
        super().__init__(
            config,
            tokenizer,
            lr=lr,
            prompts = prompts,
            max_seq_len = max_seq_len,
            top_k = top_k,
            top_p = top_p,
            temperature = temperature,
            gen_every_n_epochs = gen_every_n_epochs,
        )
        self.save_hyperparameters()
        self.ref_policy = None
        # self.judge = Reward(
        #     tokenizer=tokenizer,
        #     eos_token_id=tokenizer.eos_id,
        #     bos_token_id=tokenizer.bos_id,
        #     pad_token_id=0,
        #     reward_weights=reward_weights,
        # )
        self.judge = StoryReward(
            tokenizer=tokenizer,
            config=reward_config,
            weights=reward_weights,
        )
        self.num_gen = num_gen
        # self.lr = lr
    
    def _init_ref_policy(self):
        self.ref_policy = copy.deepcopy(self).eval()

        self.ref_policy.ref_policy = None

        for p in self.ref_policy.parameters():
            p.requires_grad = False
    
    # --------------------------------------------------
    # remove reference model from checkpoint
    # --------------------------------------------------
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
    
        # remove reference policy weights
        return {k: v for k, v in state.items() if not k.startswith("ref_policy.")}


    def load_state_dict(self, state_dict, strict=True):
        # temporarily remove reference policy so torch doesn't expect its weights
        ref = self.ref_policy
        self.ref_policy = None
    
        result = super().load_state_dict(state_dict, strict=strict)
    
        # rebuild reference policy after loading
        self._init_ref_policy()
    
        return result
    # --------------------------------------------------
    # rebuild reference policy after loading checkpoint
    # --------------------------------------------------
    def on_load_checkpoint(self, checkpoint):
        self._init_ref_policy()

    def on_train_start(self):
        if self.ref_policy is None:
            self._init_ref_policy()
    
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
        was_training = self.training
        self.eval()
        
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
        
        if was_training:
            self.train()
        
        return all_ids, all_masks, all_lens, gen_ids, gen_masks, gen_logp
    
    def gen_token_logprobs(
        self,
        all_ids,
        all_masks,
        prompt_lengths,
        gen_masks,
    ):
        B, T = gen_masks.shape

        gen_pos = torch.arange(T, device=all_ids.device, dtype=torch.long).expand(B,-1) 
        gen_pos = gen_pos + (prompt_lengths - 1).view(B,-1)        
        
        logits, _ = self(all_ids, attention_mask=all_masks)
        
        log_probs= nn.functional.log_softmax(logits[:, :-1, :], dim=-1)

        # Entropy for logging only: detach first so autograd does not keep extra graph
        with torch.no_grad():
            log_probs_detached = log_probs.detach()
            probs = log_probs_detached.exp()
            token_entropy = -(probs * log_probs_detached).sum(dim=-1)
            token_entropy = token_entropy.gather(dim=-1, index=gen_pos)
            token_entropy = token_entropy * gen_masks.to(token_entropy.dtype)
            avg_entropy = token_entropy.sum() / gen_masks.sum().clamp_min(1)
        
        token_log_probs = log_probs.gather(dim=-1, index=all_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        token_log_probs = token_log_probs.gather(dim=-1, index=gen_pos)
        token_log_probs = token_log_probs * gen_masks.to(log_probs.dtype)

        return token_log_probs, avg_entropy

    def grpo_loss(
        self,
        logp_new,
        logp_old,
        logp_ref,
        gen_masks,
        advantages,
    ):
        ratio = torch.exp(logp_new - logp_old)
        
        expected_adv1 = ratio * advantages.unsqueeze(-1)
        expected_adv2 = torch.clamp(ratio, 1 - self.hparams.clip_eps, 1 + self.hparams.clip_eps) * advantages.unsqueeze(-1)
        expected_adv = torch.minimum(expected_adv1, expected_adv2)
        
        ppo_loss = -(expected_adv * gen_masks).sum(dim=1) / gen_masks.sum(dim=1).clamp_min(1)
        ppo_loss = ppo_loss.mean()
        
        kl_loss = ((logp_new - logp_ref) * gen_masks).sum() / gen_masks.sum()

        return ppo_loss + self.hparams.kl_beta * kl_loss, ppo_loss.detach(), self.hparams.kl_beta * kl_loss.detach()


    
    def _loss(self, batch):
        self.judge.device = self.device
        prompt_ids, prompt_masks, prompt_lengths, prompt_metadata = batch
        B, T = prompt_ids.shape
        G = self.num_gen
        
        prompt_ids = prompt_ids.expand(G, B, -1).transpose(1,0).contiguous().view(B*G, -1)
        prompt_masks = prompt_masks.expand(G, B, -1).transpose(1,0).contiguous().view(B*G, -1)
        prompt_lengths = prompt_lengths.expand(G, B).transpose(1,0).contiguous().view(B*G)

        all_ids, all_masks, _, _, gen_masks, logp_old = self.generate_w_logp(
            prompt_ids,
            prompt_masks,
            prompt_lengths,
            self.max_seq_len,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )

        logp_new, avg_entropy = self.gen_token_logprobs(all_ids, all_masks, prompt_lengths, gen_masks)

        with torch.no_grad():
            logp_ref, _ = self.ref_policy.gen_token_logprobs(all_ids, all_masks, prompt_lengths, gen_masks)
        
        # rewards = self.judge.score_from_concat_ids(
        #     sequences=all_ids,
        #     attention_mask=all_masks,
        #     prompt_lens=prompt_lengths
        # )
        # advantages = self.judge.compute_grpo_advantages(rewards.rewards, group_size=self.num_gen)
        
        # loss, ppo_loss, kl_loss = self.grpo_loss(logp_new, logp_old, logp_ref, gen_masks, advantages.group_advantages)
        # avg_reward = rewards.rewards.mean()
        # avg_adv = advantages.group_advantages.mean()
        # std_adv = advantages.group_advantages.std()

        judgements = self.judge.score_grouped_from_token_ids(
            input_ids=all_ids,
            attention_mask=all_masks,
            prompt_lengths=prompt_lengths,
            metadata_list=prompt_metadata,
            group_size=self.num_gen,
            return_breakdown=False,
        )
        
        loss, ppo_loss, kl_loss = self.grpo_loss(logp_new, logp_old, logp_ref, gen_masks, judgements["advantages"])
        avg_reward = judgements["rewards"].mean()
        avg_adv = judgements["advantages"].mean()
        std_adv = judgements["advantages"].std()
                
        return loss, ppo_loss, kl_loss, avg_reward, avg_adv, std_adv, avg_entropy
    
    def training_step(self, batch, batch_idx):
        if self.ref_policy is None:
            self._init_ref_policy()
            
        loss, ppo_loss, kl_loss, avg_reward, avg_adv, std_adv, avg_entropy = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ppo", ppo_loss, prog_bar=False)
        self.log("train_kldiv", kl_loss, prog_bar=False)
        self.log("train_avg_reward", avg_reward, prog_bar=False)
        self.log("train_avg_adv", avg_adv, prog_bar=False)
        self.log("train_std_adv", std_adv, prog_bar=False)
        self.log("train_avg_ent", avg_entropy, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.ref_policy is None:
            self._init_ref_policy()

        loss, ppo_loss, kl_loss, avg_reward, avg_adv, std_adv, avg_entropy = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ppo", ppo_loss, prog_bar=False)
        self.log("val_kldiv", kl_loss, prog_bar=False)
        self.log("val_avg_reward", avg_reward, prog_bar=False)
        self.log("val_avg_adv", avg_adv, prog_bar=False)
        self.log("val_std_adv", std_adv, prog_bar=False)        
        self.log("val_avg_ent", avg_entropy, prog_bar=False)
        if (
            batch_idx == 0
            and self.trainer.is_global_zero
            and (self.current_epoch % self.gen_every_n_epochs == 0)
        ):
            self._log_generation_to_mlflow()

        return loss
        
    # def configure_optimizers(self):#, lr=1e-5):
    #     print(f"lr = {self.lr}")
    #     return torch.optim.AdamW(self.parameters(), lr=self.lr)


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
            f.write(f"max_seq_len: {self.max_seq_len}\n")
            f.write(f"top_k: {self.top_k}\n")
            f.write(f"top_p: {self.top_p}\n")
            f.write(f"temperature: {self.temperature}\n\n")
            
            for i, prompt in enumerate(self.prompts):

                text = self.model.generate(
                    prompt["prompt"],
                    self.max_seq_len,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    tokenizer=self.tokenizer,
                    temperature=self.temperature,
                )
                prompt_ = copy.deepcopy(prompt)
                prompt_.pop("word_clause", None)
                prompt_yaml = yaml.dump(prompt_, sort_keys=False) # sort_keys=False preserves key order
                
                reward = self.judge.score(text, prompt)
                reward_yaml = yaml.dump(round_floats(reward, n=4), sort_keys=False) # sort_keys=False preserves key order
                # reward = reward.__dict__.copy()

                # _ = reward.pop("prompt_texts")
                # _ = reward.pop("completion_texts")
                # _ = reward.pop("raw_rewards_0_1")

                # f.write(f"prompt: {prompt}\n")
                f.write(f"{prompt_yaml}\n")
                f.write("\n---\n")
                f.write(f"{reward_yaml}\n")

                # for k,v in reward.items():
                #     f.write(f"{k[:-1]}: {np.round(v[0].item(),4)}\n")
                
                # f.write(text)
                if i+1<len(self.prompts):
                    f.write("\n\n+++++++++++++\n\n")

        mlf.log_artifact(run_id, fpath, artifact_path="generations")

        # Optional: short preview as a tag
        preview = str(text).replace("\n", " ")[:200]
        mlf.set_tag(run_id, f"gen_preview_epoch_{self.current_epoch}", preview)







# ============================================================
# LightningModule
# ============================================================
class GPT2SFTModule(pl.LightningModule):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model_name: str = "gpt2",
        lr: float = 3e-5,
        warmup_ratio: float = 0.0,
        weight_decay: float = 0.01,
        generation_prompts: Optional[List[str]] = None,
        generation_max_new_tokens: int = 248,
        generation_temperature: float = 0.8,
        generation_top_k: int = 50,
        generation_top_p: float = 0.95,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generation_prompts"])
        self.tokenizer = tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_prompts = generation_prompts or []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        # --- total steps (Lightning-safe) ---
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)

        # --- Warmup: near 0 → base LR ---
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # --- Cosine decay ---
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=0.0,
        )

        # --- Combine ---
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # critical for warmup
                "frequency": 1,
            },
        }

    def forward(self, input_ids, attention_mask=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Labels:
          - ignore prompt tokens
          - ignore padding
          - keep answer/completion tokens only
        """
        B, T = input_ids.shape
        labels = input_ids.clone()

        # ignore padding
        labels = labels.masked_fill(attention_mask == 0, -100)

        # ignore prompt tokens
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        prompt_mask = positions < question_lengths.unsqueeze(1)            # (B, T)
        labels = labels.masked_fill(prompt_mask, -100)

        return labels

    def _loss_and_entropy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: scalar CE loss over completion tokens only
            entropy: scalar mean token entropy over completion tokens only
        """
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (B, T, V)

        labels = self._build_labels(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_lengths=question_lengths,
        )

        shift_logits = logits[:, :-1, :].contiguous()   # predict token t+1 from token t
        shift_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Entropy on supervised positions only
        with torch.no_grad():
            probs = torch.softmax(shift_logits, dim=-1)
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, T-1)

            valid_mask = (shift_labels != -100).float()
            entropy = (token_entropy * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

        return loss, entropy

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, question_lengths = batch

        loss, entropy = self._loss_and_entropy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_lengths=question_lengths,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]        
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_entropy", entropy, prog_bar=True)
        self.log("lr", lr, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, question_lengths = batch

        loss, entropy = self._loss_and_entropy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_lengths=question_lengths,
        )

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_entropy", entropy, prog_bar=True)
        
        return {"val_loss": loss.detach(), "val_entropy": entropy.detach()}

    @torch.no_grad()
    def generate_from_prompts(self, prompts: List[str]) -> List[dict]:
        self.eval()

        results = []
        device = self.device

        for prompt in prompts:
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.hparams.generation_max_new_tokens,
                do_sample=True,
                temperature=self.hparams.generation_temperature,
                top_k=self.hparams.generation_top_k,
                top_p=self.hparams.generation_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # completion-only text
            prompt_len = input_ids.shape[1]
            completion_ids = generated_ids[0, prompt_len:]
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            results.append(
                {
                    "prompt": prompt,
                    "completion": completion_text,
                    "full_text": full_text,
                }
            )

        return results
        

class GPT2GRPOModule(GPT2SFTModule):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model_name: str = "gpt2",
        num_gen: int = 8,
        max_seq_len: int = 248,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        lr: float = 3e-5,
        warmup_ratio: float = 0.0,
        weight_decay: float = 0.01,
        generation_prompts: Optional[List[str]] = None,
        generation_max_new_tokens: int = 248,
        generation_temperature: float = 0.8,
        generation_top_k: int = 50,
        generation_top_p: float = 0.95,
        clip_eps: float = 0.2,
        kl_beta: float = 0.02,
        reward_config: Optional[StoryRewardConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
    ):
        super().__init__(
            tokenizer = tokenizer,
            model_name = model_name,
            lr = lr,
            warmup_ratio = warmup_ratio,
            weight_decay = weight_decay,
            generation_prompts = generation_prompts,
            generation_max_new_tokens = generation_max_new_tokens,
            generation_temperature = generation_temperature,
            generation_top_k = generation_top_k,
            generation_top_p = generation_top_p,
        )
        self.save_hyperparameters(ignore=["generation_prompts"])
        self.ref_policy = None

        self.judge = StoryReward(
            tokenizer=self.tokenizer,
            config=reward_config,
            weights=reward_weights,
        )
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

   
    def _init_ref_policy(self):
        self.ref_policy = copy.deepcopy(self).eval()

        self.ref_policy.ref_policy = None

        for p in self.ref_policy.parameters():
            p.requires_grad = False
    
    # --------------------------------------------------
    # remove reference model from checkpoint
    # --------------------------------------------------
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
    
        # remove reference policy weights
        return {k: v for k, v in state.items() if not k.startswith("ref_policy.")}


    def load_state_dict(self, state_dict, strict=True):
        # temporarily remove reference policy so torch doesn't expect its weights
        ref = self.ref_policy
        self.ref_policy = None
    
        result = super().load_state_dict(state_dict, strict=strict)
    
        # rebuild reference policy after loading
        self._init_ref_policy()
    
        return result
    # --------------------------------------------------
    # rebuild reference policy after loading checkpoint
    # --------------------------------------------------
    def on_load_checkpoint(self, checkpoint):
        self._init_ref_policy()

    def on_train_start(self):
        if self.ref_policy is None:
            self._init_ref_policy()
    
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
    ):
        was_training = self.training
        self.eval()
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        
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
            logits = self(*batch).logits
            
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
        
        if was_training:
            self.train()
        
        return all_ids, all_masks, all_lens, gen_ids, gen_masks, gen_logp
    
    def gen_token_logprobs(
        self,
        all_ids,
        all_masks,
        prompt_lengths,
        gen_masks,
    ):
        B, T = gen_masks.shape

        gen_pos = torch.arange(T, device=all_ids.device, dtype=torch.long).expand(B,-1) 
        gen_pos = gen_pos + (prompt_lengths - 1).view(B,-1)        
        
        # logits, _ = self(all_ids, attention_mask=all_masks)
        logits = self(all_ids, attention_mask=all_masks).logits
        
        log_probs= nn.functional.log_softmax(logits[:, :-1, :], dim=-1)

        # Entropy for logging only: detach first so autograd does not keep extra graph
        with torch.no_grad():
            log_probs_detached = log_probs.detach()
            probs = log_probs_detached.exp()
            token_entropy = -(probs * log_probs_detached).sum(dim=-1)
            token_entropy = token_entropy.gather(dim=-1, index=gen_pos)
            token_entropy = token_entropy * gen_masks.to(token_entropy.dtype)
            avg_entropy = token_entropy.sum() / gen_masks.sum().clamp_min(1)
        
        token_log_probs = log_probs.gather(dim=-1, index=all_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        token_log_probs = token_log_probs.gather(dim=-1, index=gen_pos)
        token_log_probs = token_log_probs * gen_masks.to(log_probs.dtype)

        return token_log_probs, avg_entropy

    def grpo_loss(
        self,
        logp_new,
        logp_old,
        logp_ref,
        gen_masks,
        advantages,
    ):
        ratio = torch.exp(logp_new - logp_old)
        
        expected_adv1 = ratio * advantages.unsqueeze(-1)
        expected_adv2 = torch.clamp(ratio, 1 - self.hparams.clip_eps, 1 + self.hparams.clip_eps) * advantages.unsqueeze(-1)
        expected_adv = torch.minimum(expected_adv1, expected_adv2)
        
        ppo_loss = -(expected_adv * gen_masks).sum(dim=1) / gen_masks.sum(dim=1).clamp_min(1)
        ppo_loss = ppo_loss.mean()
        
        kl_loss = ((logp_new - logp_ref) * gen_masks).sum() / gen_masks.sum()

        return ppo_loss + self.hparams.kl_beta * kl_loss, ppo_loss.detach(), self.hparams.kl_beta * kl_loss.detach()


    
    def _loss(self, batch):
        self.judge.device = self.device
        prompt_ids, prompt_masks, prompt_lengths, prompt_metadata = batch
        B, T = prompt_ids.shape
        G = self.hparams.num_gen
        
        prompt_ids = prompt_ids.expand(G, B, -1).transpose(1,0).contiguous().view(B*G, -1)
        prompt_masks = prompt_masks.expand(G, B, -1).transpose(1,0).contiguous().view(B*G, -1)
        prompt_lengths = prompt_lengths.expand(G, B).transpose(1,0).contiguous().view(B*G)

        all_ids, all_masks, _, _, gen_masks, logp_old = self.generate_w_logp(
            prompt_ids,
            prompt_masks,
            prompt_lengths,
            self.hparams.max_seq_len,
            temperature=self.hparams.temperature,
            top_p=self.hparams.top_p,
            top_k=self.hparams.top_k
        )

        logp_new, avg_entropy = self.gen_token_logprobs(all_ids, all_masks, prompt_lengths, gen_masks)

        with torch.no_grad():
            logp_ref, _ = self.ref_policy.gen_token_logprobs(all_ids, all_masks, prompt_lengths, gen_masks)
        
        # rewards = self.judge.score_from_concat_ids(
        #     sequences=all_ids,
        #     attention_mask=all_masks,
        #     prompt_lens=prompt_lengths
        # )
        # advantages = self.judge.compute_grpo_advantages(rewards.rewards, group_size=self.num_gen)
        
        # loss, ppo_loss, kl_loss = self.grpo_loss(logp_new, logp_old, logp_ref, gen_masks, advantages.group_advantages)
        # avg_reward = rewards.rewards.mean()
        # avg_adv = advantages.group_advantages.mean()
        # std_adv = advantages.group_advantages.std()

        judgements = self.judge.score_grouped_from_token_ids(
            input_ids=all_ids,
            attention_mask=all_masks,
            prompt_lengths=prompt_lengths,
            metadata_list=prompt_metadata,
            group_size=self.hparams.num_gen,
            return_breakdown=False,
        )
        
        loss, ppo_loss, kl_loss = self.grpo_loss(logp_new, logp_old, logp_ref, gen_masks, judgements["advantages"])
        avg_reward = judgements["rewards"].mean()
        avg_adv = judgements["advantages"].mean()
        std_adv = judgements["advantages"].std()
                
        return loss, ppo_loss, kl_loss, avg_reward, avg_adv, std_adv, avg_entropy
    
    def training_step(self, batch, batch_idx):
        if self.ref_policy is None:
            self._init_ref_policy()
            
        loss, ppo_loss, kl_loss, avg_reward, avg_adv, std_adv, avg_entropy = self._loss(batch)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]        
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ppo", ppo_loss, prog_bar=False)
        self.log("train_kldiv", kl_loss, prog_bar=False)
        self.log("train_avg_reward", avg_reward, prog_bar=False)
        self.log("train_avg_adv", avg_adv, prog_bar=False)
        self.log("train_std_adv", std_adv, prog_bar=False)
        self.log("train_avg_ent", avg_entropy, prog_bar=False)
        self.log("lr", lr, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.ref_policy is None:
            self._init_ref_policy()

        loss, ppo_loss, kl_loss, avg_reward, avg_adv, std_adv, avg_entropy = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ppo", ppo_loss, prog_bar=False)
        self.log("val_kldiv", kl_loss, prog_bar=False)
        self.log("val_avg_reward", avg_reward, prog_bar=False)
        self.log("val_avg_adv", avg_adv, prog_bar=False)
        self.log("val_std_adv", std_adv, prog_bar=False)        
        self.log("val_avg_ent", avg_entropy, prog_bar=False)

        return {
            "val_loss": loss.detach(),
            "val_ppo": ppo_loss.detach(),
            "val_kldiv": kl_loss.detach(),
            "val_avg_reward": avg_reward.detach(),
            "val_avg_adv": avg_adv.detach(),
            "val_std_adv": std_adv.detach(),
            "val_avg_ent": avg_entropy.detach(),
        }
    

    @torch.no_grad()
    def generate_from_prompts(self, prompts: List[str]) -> List[dict]:
        self.eval()

        results = []
        device = self.device

        for prompt in prompts:
            encoded = self.tokenizer(
                prompt["prompt"],
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.hparams.generation_max_new_tokens,
                do_sample=True,
                temperature=self.hparams.generation_temperature,
                top_k=self.hparams.generation_top_k,
                top_p=self.hparams.generation_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            reward = self.judge.score(full_text, prompt)
            _ = reward["debug"].pop("text", None)
            if not reward.get("feature_breakdown", {}):
                _ = reward.pop("feature_breakdown", {})
            reward_yaml = yaml.dump(round_floats(reward, n=4), sort_keys=False) # sort_keys=False preserves key order
            
            # completion-only text
            prompt_len = input_ids.shape[1]
            completion_ids = generated_ids[0, prompt_len:]
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            results.append(
                {
                    "prompt": prompt["prompt"],
                    "completion": completion_text,
                    "full_text": full_text,
                    "reward": reward_yaml,
                }
            )

        return results