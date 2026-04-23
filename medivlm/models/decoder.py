"""GPT-2 report decoder with 4 fine-tuned transformer blocks.

We wrap the HF GPT-2 language model and fuse the cross-attention
context by prepending visual prefix tokens (a classic "soft prompt"
approach which integrates cleanly with the causal LM head).

Only the last ``trainable_blocks`` transformer layers + the LM head
are trained; earlier layers are kept frozen.

Notes on generation:
  * ``beam_size=4`` follows common practice for medical report
    generation; override via decoder config.
  * Max length = 77 tokens (~4 sentences, Table 7 best setting).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:  # pragma: no cover
    GPT2LMHeadModel = None  # type: ignore
    GPT2Tokenizer = None    # type: ignore


class GPT2ReportDecoder(nn.Module):
    """Projects fused cross-modal features into the GPT-2 embedding space
    and concatenates them as a prefix to the token embeddings before the
    causal stack.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        trainable_blocks: int = 4,
        fusion_dim: Optional[int] = None,
        max_length: int = 77,
        beam_size: int = 4,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        if GPT2LMHeadModel is None:
            raise ImportError("transformers is required.")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2 has no pad token by default; use eos as pad.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.hidden_size = self.model.config.n_embd
        fusion_dim = fusion_dim or self.hidden_size
        self.fusion_proj = nn.Linear(fusion_dim, self.hidden_size)

        self.max_length = max_length
        self.beam_size = beam_size
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty

        self._freeze_except_last_blocks(trainable_blocks)

    # ------------------------------------------------------------------
    def _freeze_except_last_blocks(self, n: int) -> None:
        blocks = self.model.transformer.h  # nn.ModuleList
        num_blocks = len(blocks)
        n = max(0, min(n, num_blocks))
        for i, block in enumerate(blocks):
            trainable = i >= num_blocks - n
            for p in block.parameters():
                p.requires_grad_(trainable)
        # embeddings frozen, final LN + LM head trainable
        for p in self.model.transformer.wte.parameters():
            p.requires_grad_(False)
        for p in self.model.transformer.wpe.parameters():
            p.requires_grad_(False)
        for p in self.model.transformer.ln_f.parameters():
            p.requires_grad_(True)
        # lm_head shares weights with wte by default, but still enable grad
        for p in self.model.lm_head.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    def _build_inputs_embeds(
        self,
        fusion: torch.Tensor,          # (B, P, D_fusion)
        input_ids: Optional[torch.Tensor],  # (B, L)
    ) -> Dict[str, torch.Tensor]:
        B = fusion.size(0)
        prefix = self.fusion_proj(fusion)  # (B, P, D_gpt2)

        if input_ids is None:
            token_embeds = prefix.new_zeros((B, 0, self.hidden_size))
            token_mask = torch.ones(B, 0, device=prefix.device, dtype=torch.long)
        else:
            token_embeds = self.model.transformer.wte(input_ids)    # (B, L, D)
            token_mask = (input_ids != self.tokenizer.pad_token_id).long()

        prefix_mask = torch.ones(B, prefix.size(1), device=prefix.device, dtype=torch.long)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, token_mask], dim=1)
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "prefix_len": prefix.size(1),
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        fusion: torch.Tensor,
        labels_input_ids: torch.Tensor,   # (B, L) with <bos> ... <eos>
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced training pass. Returns CE loss over report tokens only.

        We mask the prefix positions with label = -100 so that only real
        report tokens contribute to the cross-entropy (Eq. 6).
        """
        built = self._build_inputs_embeds(fusion, labels_input_ids)
        inputs_embeds = built["inputs_embeds"]
        attention_mask = built["attention_mask"]
        prefix_len = built["prefix_len"]

        labels = labels_input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        # The HF CausalLM shifts labels internally, so we prepend a -100
        # column per prefix token so positions line up.
        pad_labels = labels.new_full((labels.size(0), prefix_len), -100)
        full_labels = torch.cat([pad_labels, labels], dim=1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
            return_dict=True,
        )
        return {"loss": out.loss, "logits": out.logits}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        fusion: torch.Tensor,
        prompt: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        do_sample: bool = False,
    ) -> List[str]:
        """Beam / greedy generation conditioned on the fused prefix."""
        device = fusion.device
        B = fusion.size(0)
        if prompt is None:
            prompt = [""] * B
        enc = self.tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = enc["input_ids"] if enc["input_ids"].numel() > 0 else None
        built = self._build_inputs_embeds(fusion, input_ids)
        inputs_embeds = built["inputs_embeds"]
        attention_mask = built["attention_mask"]

        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_length or self.max_length,
            num_beams=num_beams or self.beam_size,
            do_sample=do_sample,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            length_penalty=self.length_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)
