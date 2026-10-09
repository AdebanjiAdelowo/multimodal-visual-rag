"""Single-vector visual index: CLIP / SigLIP image embeddings queried with text."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmrag.config import VisualCfg
from mmrag.data.corpus import Corpus
from mmrag.index import store
from mmrag.runtime import pick_device


def _as_tensor(out):
    """get_*_features returns a tensor or a pooled model output depending on the version."""
    return out if hasattr(out, "shape") else out.pooler_output


class VisualIndex:
    def __init__(
        self,
        corpus: Corpus,
        cfg: VisualCfg,
        root: Path = store.DEFAULT_ROOT,
        device: str = "auto",
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.cfg = cfg
        self.name = f"visual[{cfg.model.split('/')[-1]}]"
        self.device = pick_device(device)
        self._torch = torch
        self.model = AutoModel.from_pretrained(cfg.model).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model)
        # SigLIP text towers are trained with fixed-length padding; CLIP is not.
        self._is_siglip = "siglip" in self.model.config.model_type

        params = {"model": cfg.model, "corpus": corpus.fingerprint()}
        self.dir = store.index_dir(root, corpus.name, "visual", params)
        if store.is_built(self.dir):
            self.emb = np.load(self.dir / "embeddings.npy")
        else:
            embs = []
            ids = corpus.page_ids
            for start in range(0, len(ids), batch_size):
                images = [corpus.image(pid) for pid in ids[start : start + batch_size]]
                embs.append(self.embed_images(images))
                if (start // batch_size) % 10 == 0:
                    print(f"  visual embed {start}/{len(ids)}", flush=True)
            self.emb = np.concatenate(embs).astype(np.float32)
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(self.dir / "embeddings.npy", self.emb)
            store.write_meta(self.dir, params, {"n_pages": len(ids), "dim": self.emb.shape[1]})

    def embed_images(self, images: list) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            feats = _as_tensor(self.model.get_image_features(**inputs))
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()

    def embed_text(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        kwargs = (
            {"padding": "max_length", "max_length": 64, "truncation": True}
            if self._is_siglip
            else {"padding": True, "truncation": True}
        )
        inputs = self.processor(text=texts, return_tensors="pt", **kwargs).to(self.device)
        with torch.inference_mode():
            feats = _as_tensor(self.model.get_text_features(**inputs))
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()

    def score_all(self, query: str) -> np.ndarray:
        return (self.emb @ self.embed_text([query])[0]).astype(np.float32)

    def score_image(self, image) -> np.ndarray:
        """Image-to-image similarity, used for query-by-image search in the app."""
        return (self.emb @ self.embed_images([image])[0]).astype(np.float32)
