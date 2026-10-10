"""Multi-vector visual index (ColPali family) scored with late interaction (MaxSim).

Requires the optional `late` extra (colpali-engine). Each page is stored as a
matrix of patch embeddings; a query is scored by summing, over query tokens,
the maximum similarity to any page patch.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mmrag.config import LateInteractionCfg
from mmrag.data.corpus import Corpus
from mmrag.index import store
from mmrag.runtime import pick_device


def maxsim(query: np.ndarray, pages: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Late-interaction scores.

    query: (Tq, D); pages: (N, Tp, D) zero-padded; mask: (N, Tp) True for real patches.
    """
    sim = np.einsum("qd,npd->nqp", query, pages)
    sim = np.where(mask[:, None, :], sim, -np.inf)
    return sim.max(axis=2).sum(axis=1).astype(np.float32)


def embed_resumable(
    ids: list[str],
    embed: Callable[[list[str]], list[np.ndarray]],
    partial_dir: Path,
    batch_size: int = 4,
    shard_size: int = 64,
) -> list[np.ndarray]:
    """Embed pages shard by shard, keeping finished shards on disk.

    A full build takes tens of minutes on a laptop; an interrupted build resumes
    from the last complete shard. Shard files are named by position and checked
    against the page ids they hold, so a shard from a different page order is
    recomputed rather than reused.
    """
    partial_dir.mkdir(parents=True, exist_ok=True)
    mats: list[np.ndarray] = []
    for start in range(0, len(ids), shard_size):
        shard_ids = ids[start : start + shard_size]
        path = partial_dir / f"shard_{start:06d}.npz"
        if path.exists():
            z = np.load(path)
            if z["ids"].tolist() == shard_ids:
                mats.extend(z[f"m{i}"] for i in range(len(shard_ids)))
                continue
        shard: list[np.ndarray] = []
        for b in range(0, len(shard_ids), batch_size):
            shard.extend(embed(shard_ids[b : b + batch_size]))
        tmp = partial_dir / f"shard_{start:06d}.tmp.npz"
        np.savez(tmp, ids=np.array(shard_ids), **{f"m{i}": m for i, m in enumerate(shard)})
        tmp.replace(path)  # atomic: a shard file is always complete
        mats.extend(shard)
        print(f"  late embed {min(start + shard_size, len(ids))}/{len(ids)}", flush=True)
    return mats


def _load_model(model_id: str, device: str):
    import torch

    try:
        import colpali_engine.models as cm
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'late' extra: pip install -e '.[late]'") from e

    lowered = model_id.lower()
    if "colsmol" in lowered or "colidefics3" in lowered:
        model_cls, proc_cls = cm.ColIdefics3, cm.ColIdefics3Processor
    elif "colqwen2.5" in lowered:
        model_cls, proc_cls = cm.ColQwen2_5, cm.ColQwen2_5_Processor
    elif "colqwen" in lowered:
        model_cls, proc_cls = cm.ColQwen2, cm.ColQwen2Processor
    else:
        model_cls, proc_cls = cm.ColPali, cm.ColPaliProcessor
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = model_cls.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, proc_cls.from_pretrained(model_id)


class LateInteractionIndex:
    def __init__(
        self,
        corpus: Corpus,
        cfg: LateInteractionCfg,
        root: Path = store.DEFAULT_ROOT,
        device: str = "auto",
        batch_size: int = 4,
    ):
        import torch

        self.cfg = cfg
        self.name = f"late[{cfg.model.split('/')[-1]}]"
        self.device = pick_device(device)
        self._torch = torch
        self.model, self.processor = _load_model(cfg.model, self.device)

        params = {"model": cfg.model, "corpus": corpus.fingerprint()}
        self.dir = store.index_dir(root, corpus.name, "late", params)
        if store.is_built(self.dir):
            z = np.load(self.dir / "embeddings.npz")
            self.pages, self.mask = z["pages"], z["mask"]
        else:
            ids = corpus.page_ids

            def embed(page_ids: list[str]) -> list[np.ndarray]:
                images = [corpus.image(pid) for pid in page_ids]
                batch = self.processor.process_images(images).to(self.device)
                with torch.inference_mode():
                    out = self.model(**batch).float().cpu().numpy()
                attn = batch["attention_mask"].cpu().numpy().astype(bool)
                return [o[m].astype(np.float16) for o, m in zip(out, attn, strict=True)]

            mats = embed_resumable(ids, embed, self.dir / "partial", batch_size)
            tmax = max(m.shape[0] for m in mats)
            dim = mats[0].shape[1]
            self.pages = np.zeros((len(mats), tmax, dim), dtype=np.float16)
            self.mask = np.zeros((len(mats), tmax), dtype=bool)
            for i, m in enumerate(mats):
                self.pages[i, : m.shape[0]] = m
                self.mask[i, : m.shape[0]] = True
            self.dir.mkdir(parents=True, exist_ok=True)
            np.savez(self.dir / "embeddings.npz", pages=self.pages, mask=self.mask)
            store.write_meta(
                self.dir, params, {"n_pages": len(ids), "max_patches": tmax, "dim": dim}
            )
            shutil.rmtree(self.dir / "partial", ignore_errors=True)
        self._pages_t = torch.from_numpy(self.pages).to(self.device)
        self._mask_t = torch.from_numpy(self.mask).to(self.device)

    def score_all(self, query: str) -> np.ndarray:
        torch = self._torch
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.inference_mode():
            q = self.model(**batch)[0][batch["attention_mask"][0].bool()]
            sim = torch.einsum("qd,npd->nqp", q.to(self._pages_t.dtype), self._pages_t)
            sim = sim.masked_fill(~self._mask_t[:, None, :], float("-inf"))
            scores = sim.max(dim=2).values.float().sum(dim=1)
        return scores.cpu().numpy().astype(np.float32)
