from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass
class GeneratedCandidate:
    text: str
    hidden: torch.Tensor
    token_ids: torch.Tensor
    logprob: float
    schema_penalty: float

    def as_cache_item(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "hidden": self.hidden.detach().to(device="cpu", dtype=torch.float32),
            "logprob": self.logprob,
            "schema_penalty": self.schema_penalty,
        }


def _schema_penalty(text: str) -> float:
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, flags=re.S)
    payload = match.group(1) if match else text[text.find("{") : text.rfind("}") + 1]
    try:
        plan = json.loads(payload)
        attribute = plan["attribute"]
        valid = (
            bool(str(plan["entity"]).strip())
            and bool(str(plan["action"]).strip())
            and -1.0 <= float(attribute["valence"]) <= 1.0
            and 0.0 <= float(attribute["arousal"]) <= 1.0
            and 0.0 <= float(attribute["dominance"]) <= 1.0
            and attribute["intensity"] in {"weak", "moderate", "strong"}
        )
        return 0.0 if valid else 1.0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 1.0


class QwenPlannerBackend:
    """Frozen Qwen2.5-Omni Thinker used for plan generation and hidden states."""

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        flash_attention: bool = False,
    ) -> None:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, dtype),
            "device_map": device_map,
        }
        if flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.model.disable_talker()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    def _conversation(
        self,
        raw: dict[str, Any],
        available: list[str],
        missing: list[str],
        feedback: str | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        content: list[dict[str, Any]] = []
        use_audio_in_video = False
        if "vision" in available and raw.get("video"):
            content.append({"type": "video", "video": raw["video"]})
            use_audio_in_video = "audio" in available
        elif "vision" in available and raw.get("image"):
            content.append({"type": "image", "image": raw["image"]})
        if "audio" in available and not use_audio_in_video and raw.get("audio"):
            content.append({"type": "audio", "audio": raw["audio"]})
        if "text" in available and raw.get("text"):
            content.append({"type": "text", "text": str(raw["text"])})
        content.append(
            {
                "type": "text",
                "text": build_user_prompt(available, missing, feedback),
            }
        )
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return conversation, use_audio_in_video

    @staticmethod
    def _generation_hidden(output: Any) -> torch.Tensor:
        if not getattr(output, "hidden_states", None):
            raise RuntimeError("Transformers generation did not return hidden states")
        step_states = []
        for step in output.hidden_states:
            last_layer = step[-1] if isinstance(step, (tuple, list)) else step
            step_states.append(last_layer[:, -1, :])
        return torch.stack(step_states, dim=1)

    @torch.inference_mode()
    def generate_candidates(
        self,
        raw: dict[str, Any],
        available: list[str],
        missing: list[str],
        count: int,
        temperature: float,
        feedback: str | None = None,
        max_new_tokens: int = 192,
    ) -> list[GeneratedCandidate]:
        from qwen_omni_utils import process_mm_info

        conversation, use_audio_in_video = self._conversation(
            raw, available, missing, feedback
        )
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video
        )
        inputs = self.processor(
            text=prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        output = self.model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=count,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated = output.sequences[:, prompt_length:]
        texts = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        hidden = self._generation_hidden(output)
        transition = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        logprob = transition.sum(dim=1)
        return [
            GeneratedCandidate(
                text=text,
                hidden=hidden[index].float().cpu(),
                token_ids=generated[index].long().cpu(),
                logprob=float(logprob[index].cpu()),
                schema_penalty=_schema_penalty(text),
            )
            for index, text in enumerate(texts)
        ]

    def field_hidden(self, candidate: GeneratedCandidate) -> torch.Tensor:
        """Pool generated token states into Entity/Action/Attribute constraints."""
        lower = candidate.text.lower()
        starts = [
            lower.find('"entity"'),
            lower.find('"action"'),
            lower.find('"attribute"'),
        ]
        if any(position < 0 for position in starts) or starts != sorted(starts):
            pooled = candidate.hidden.mean(dim=0)
            return pooled.unsqueeze(0).expand(3, -1).clone()
        token_count = min(candidate.hidden.shape[0], candidate.token_ids.shape[0])
        if token_count < 1:
            raise ValueError("Generated plan has no token states")
        character_ends = []
        for end in range(1, token_count + 1):
            prefix = self.processor.tokenizer.decode(
                candidate.token_ids[:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            character_ends.append(len(prefix))
        field_ends = [starts[1], starts[2], max(len(candidate.text), starts[2] + 1)]
        fields = []
        previous_end = 0
        for start, end in zip(starts, field_ends):
            indices = [
                index
                for index, character_end in enumerate(character_ends)
                if character_end > start and previous_end < end
                and (index == 0 or character_ends[index - 1] < end)
            ]
            if indices:
                fields.append(candidate.hidden[indices].mean(dim=0))
                previous_end = character_ends[indices[-1]]
            else:
                fields.append(candidate.hidden[:token_count].mean(dim=0))
        return torch.stack(fields)

    def cache_candidate(
        self,
        candidate: GeneratedCandidate,
        round_index: int,
    ) -> dict[str, Any]:
        item = candidate.as_cache_item()
        item["hidden"] = self.field_hidden(candidate).to(torch.float16)
        item["round"] = int(round_index)
        return item

    @torch.inference_mode()
    def encode_plan_text(self, text: str) -> torch.Tensor:
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        thinker = getattr(self.model, "thinker", self.model)
        output = thinker(**inputs, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states[-1][0]
        mask = inputs.get("attention_mask")
        if mask is not None:
            hidden = hidden[mask[0].bool()]
        return hidden.float().cpu()


def controlled_corruptions(plan_text: str) -> list[str]:
    """Create entity, action-conflict, and attribute-inversion negatives."""
    match = re.search(r"\{.*\}", plan_text, flags=re.S)
    if not match:
        return [
            plan_text + '\n{"entity":"unrelated person"}',
            plan_text + "\nThe described action did not occur.",
            plan_text + '\n{"attribute":{"valence":-1.0}}',
        ]
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [plan_text + suffix for suffix in ("\nENTITY_CONFLICT", "\nACTION_CONFLICT", "\nATTRIBUTE_INVERSION")]
    corruptions = []
    entity = json.loads(json.dumps(payload))
    entity["entity"] = _wordnet_entity_replacement(str(entity.get("entity", "")))
    corruptions.append(json.dumps(entity, sort_keys=True))
    action = json.loads(json.dumps(payload))
    action["action"] = "the semantic opposite of the observed action"
    corruptions.append(json.dumps(action, sort_keys=True))
    attribute = json.loads(json.dumps(payload))
    if isinstance(attribute.get("attribute"), dict):
        current = float(attribute["attribute"].get("valence", 0.0))
        attribute["attribute"]["valence"] = -current if current else 1.0
    corruptions.append(json.dumps(attribute, sort_keys=True))
    return corruptions


def _wordnet_entity_replacement(value: str) -> str:
    from nltk.corpus import wordnet

    words = value.split()
    try:
        for index, word in enumerate(words):
            synsets = wordnet.synsets(word, pos=wordnet.NOUN)
            for synset in synsets:
                relatives = sorted(
                    {
                        lemma.replace("_", " ")
                        for hypernym in synset.hypernyms()
                        for sibling in hypernym.hyponyms()
                        for lemma in sibling.lemma_names()
                        if lemma.lower() != word.lower()
                    }
                )
                if relatives:
                    replacement = list(words)
                    replacement[index] = relatives[0]
                    return " ".join(replacement)
    except LookupError as exc:
        raise RuntimeError(
            "WordNet data is required for verifier negatives. Run "
            "`python3 -m nltk.downloader wordnet` in the reproduction environment."
        ) from exc
    return "an unrelated entity"
