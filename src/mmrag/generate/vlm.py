"""Vision-language model backends behind one small interface.

A prompt is a list of content parts, each either {"type": "text", "text": str}
or {"type": "image", "image": PIL.Image}. Text-only prompts go through the
same model, which lets the text-RAG and image-RAG conditions share a generator
and differ only in the evidence modality.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from mmrag.config import VLMCfg
from mmrag.runtime import model_size, pick_device

Content = list[dict[str, Any]]


@dataclass
class GenOutput:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    # Mean log-probability of the generated tokens under greedy decoding.
    mean_logprob: float | None = None


class VLM(Protocol):
    name: str

    def generate(
        self, content: Content, max_new_tokens: int | None = None, image_max_side: int = 1024
    ) -> GenOutput: ...

    def info(self) -> dict[str, Any]: ...


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(image) -> dict[str, Any]:
    return {"type": "image", "image": image}


class TransformersVLM:
    """Any chat-template image-text-to-text model (SmolVLM, Qwen-VL, ...)."""

    def __init__(self, cfg: VLMCfg):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        self.name = cfg.model.split("/")[-1]
        self.device = pick_device(cfg.device)
        self._torch = torch

        if cfg.dtype == "auto":
            dtype = torch.float32 if self.device == "cpu" else torch.float16
        else:
            dtype = getattr(torch, cfg.dtype)
        kwargs: dict[str, Any] = {"dtype": dtype}
        if cfg.load_in_4bit:
            if self.device != "cuda":
                raise ValueError("load_in_4bit needs CUDA and bitsandbytes (use it on the T4)")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = {"": 0}
        self.processor = AutoProcessor.from_pretrained(cfg.model)
        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model, **kwargs)
        if not cfg.load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        self._dtype = dtype
        ip = self.processor.image_processor
        # Idefics3/SmolVLM tile the image into fixed-size crops; Qwen-VL uses a pixel budget.
        self._tile = (
            getattr(ip, "max_image_size", {}).get("longest_edge")
            if hasattr(ip, "max_image_size")
            else None
        )

    def info(self) -> dict[str, Any]:
        return {
            "model": self.cfg.model,
            "device": self.device,
            "dtype": str(self._dtype).replace("torch.", ""),
            "load_in_4bit": self.cfg.load_in_4bit,
            **model_size(self.model),
        }

    def _image_kwargs(self, image_max_side: int) -> dict[str, Any]:
        if self._tile:
            # Longest edge is rounded up to a whole number of tiles.
            n = max(1, -(-image_max_side // self._tile))
            return {"size": {"longest_edge": n * self._tile}}
        return {}

    def generate(
        self, content: Content, max_new_tokens: int | None = None, image_max_side: int = 1024
    ) -> GenOutput:
        torch = self._torch
        images = []
        for part in content:
            if part["type"] == "image":
                img = part["image"].convert("RGB")
                if max(img.size) > image_max_side:
                    img = img.copy()
                    img.thumbnail((image_max_side, image_max_side))
                images.append(img)
        template = [
            {"type": p["type"], **({"text": p["text"]} if p["type"] == "text" else {})}
            for p in content
        ]
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": template}], tokenize=False, add_generation_prompt=True
        )
        proc_kwargs = self._image_kwargs(image_max_side) if images else {}
        inputs = self.processor(
            text=[prompt], images=images or None, return_tensors="pt", **proc_kwargs
        )
        inputs = {
            k: (
                v.to(self.device, dtype=self._dtype) if v.is_floating_point() else v.to(self.device)
            )
            for k, v in inputs.items()
            if hasattr(v, "to")
        }
        n_in = int(inputs["input_ids"].shape[1])

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        if self.device == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0

        new_tokens = out.sequences[0, n_in:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        mean_lp = None
        if out.scores:
            trans = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )[0].float()
            finite = trans[torch.isfinite(trans)]
            if finite.numel():
                mean_lp = float(finite.mean().cpu())
        return GenOutput(text, n_in, int(new_tokens.shape[0]), dt, mean_lp)


class MockVLM:
    """Deterministic stand-in used by tests, CI and UI smoke runs.

    It quotes the first sentence of the first evidence block and cites it, and
    abstains when the prompt carries no evidence. It performs no recognition.
    """

    name = "mock"

    def __init__(self, cfg: VLMCfg | None = None):
        self.cfg = cfg or VLMCfg(backend="mock", model="mock")

    def info(self) -> dict[str, Any]:
        return {"model": "mock", "device": "cpu", "params_m": 0.0, "weights_mb": 0.0}

    def generate(
        self, content: Content, max_new_tokens: int | None = None, image_max_side: int = 1024
    ) -> GenOutput:
        text = "\n".join(p["text"] for p in content if p["type"] == "text")
        n_img = sum(1 for p in content if p["type"] == "image")
        # Evidence text arrives as its own part: "[P1]\n<text>" or "Extracted text of P1:\n<text>".
        blocks = []
        for p in content:
            if p["type"] != "text":
                continue
            m = re.match(r"(?:\[(P\d+)\]|Extracted text of (P\d+):)\n(.+)", p["text"], flags=re.S)
            if m and m.group(3).strip():
                blocks.append((m.group(1) or m.group(2), m.group(3).strip()))
        if blocks:
            label, body = blocks[0]
            sentence = re.split(r"(?<=[.!?])\s", body)[0][:200]
            reply = f"ANSWER: {sentence}\nCITATIONS: {label}"
        elif n_img and "Describe" in text:
            reply = "A document page."
        else:
            reply = "INSUFFICIENT_EVIDENCE"
        return GenOutput(reply, len(text.split()) + 64 * n_img, len(reply.split()), 0.0, -0.1)


def load_vlm(cfg: VLMCfg) -> VLM:
    return MockVLM(cfg) if cfg.backend == "mock" else TransformersVLM(cfg)
