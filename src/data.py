"""
data.py — data download, tokenization, caching, and FL dataset partitioning.

Key design: download + tokenize ONCE, cache to disk (cache/ dir).
All 120 experiments share the same cached tensors; only the client splitting
(create_client_datasets) runs per-experiment, which is cheap.

Cache layout:
  cache/tokenizer.json            — trained BPE tokenizer
  cache/lang_train_{code}.pt      — per-language train token tensor
  cache/lang_val_{code}.pt        — per-language val token tensor
"""

from __future__ import annotations
import os, gc, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import DataConfig


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def train_tokenizer(train_texts: List[str], vocab_size: int):
    """Train a BPE tokenizer on train_texts and return the HF Tokenizer object."""
    from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders

    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|bos|>"],
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(train_texts, trainer=trainer)
    return tok


def load_tokenizer(path: str):
    from tokenizers import Tokenizer as HFTokenizer
    return HFTokenizer.from_file(path)


def tokenize_texts(tok_model, texts: List[str], desc: str = "Tokenizing") -> torch.Tensor:
    """Tokenize a list of texts (prepend BOS to each doc), return a 1-D long tensor."""
    bos_id = tok_model.token_to_id("<|bos|>")
    all_ids: List[int] = []
    batch_size = 1000
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = texts[i : i + batch_size]
        encoded = tok_model.encode_batch(batch)
        for enc in encoded:
            all_ids.append(bos_id)
            all_ids.extend(enc.ids)
    return torch.tensor(all_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class DataManager:
    """
    Manages download, tokenization, caching, and retrieval of language corpora.

    Usage
    -----
    dm = DataManager(data_config, cache_dir="cache", hf_token="hf_...")
    dm.prepare()                         # no-op if cache is fresh
    lang_train, lang_val = dm.get_tensors()
    tok = dm.get_tokenizer()
    """

    def __init__(
        self,
        config: DataConfig,
        cache_dir: str = "cache",
        hf_token: Optional[str] = None,
    ):
        self.config    = config
        self.cache_dir = cache_dir
        self.hf_token  = hf_token
        os.makedirs(cache_dir, exist_ok=True)

    # ---- public interface ----

    def prepare(self) -> None:
        """Download, tokenize, and cache everything. Idempotent."""
        if self._cache_is_complete():
            print("[data] Cache is complete — skipping download/tokenisation.")
            return
        self._download_and_tokenize()

    def get_tokenizer(self):
        """Load and return the cached tokenizer."""
        path = os.path.join(self.cache_dir, "tokenizer.json")
        if not os.path.exists(path):
            raise RuntimeError("Tokenizer not cached yet — call prepare() first.")
        return load_tokenizer(path)

    def get_tensors(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Return (lang_train_tensors, lang_val_tensors) dicts, loading from cache.
        Keys are language codes like 'fra_Latn'.
        """
        lang_train, lang_val = {}, {}
        for lc, _, _ in self.config.languages:
            train_path = os.path.join(self.cache_dir, f"lang_train_{lc}.pt")
            val_path   = os.path.join(self.cache_dir, f"lang_val_{lc}.pt")
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                raise RuntimeError(f"Missing cache for {lc} — call prepare() first.")
            lang_train[lc] = torch.load(train_path, weights_only=True)
            lang_val[lc]   = torch.load(val_path,   weights_only=True)
        return lang_train, lang_val

    # ---- internals ----

    def _cache_is_complete(self) -> bool:
        tok_path = os.path.join(self.cache_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            return False
        for lc, _, _ in self.config.languages:
            if not os.path.exists(os.path.join(self.cache_dir, f"lang_train_{lc}.pt")):
                return False
            if not os.path.exists(os.path.join(self.cache_dir, f"lang_val_{lc}.pt")):
                return False
        return True

    def _download_and_tokenize(self) -> None:
        from huggingface_hub import login
        from datasets import load_dataset

        if self.hf_token:
            login(token=self.hf_token, add_to_git_credential=False)
            print("[data] Logged in to HuggingFace Hub.")

        lang_train_texts: Dict[str, List[str]] = {}
        lang_val_texts:   Dict[str, List[str]] = {}

        print("[data] Downloading FineWeb-2 language subsets...")
        for lc, num_train, num_val in self.config.languages:
            ds = load_dataset(
                "HuggingFaceFW/fineweb-2",
                name=lc,
                split="train",
                streaming=True,
            )
            texts: List[str] = []
            for ex in tqdm(ds, total=num_train + num_val, desc=f"  {lc}"):
                texts.append(ex["text"])
                if len(texts) >= num_train + num_val:
                    break
            lang_train_texts[lc] = texts[:num_train]
            lang_val_texts[lc]   = texts[num_train : num_train + num_val]
            print(f"    → {len(lang_train_texts[lc])} train, {len(lang_val_texts[lc])} val docs")

        # Train tokenizer on ALL training texts
        all_train_texts = [t for ts in lang_train_texts.values() for t in ts]
        print(f"\n[data] Training BPE tokenizer (vocab_size={self.config.vocab_size})...")
        t0 = time.time()
        tok = train_tokenizer(all_train_texts, self.config.vocab_size)
        print(f"[data] Tokenizer trained in {time.time() - t0:.1f}s  "
              f"(vocab={tok.get_vocab_size()})")
        tok_path = os.path.join(self.cache_dir, "tokenizer.json")
        tok.save(tok_path)
        print(f"[data] Tokenizer saved → {tok_path}")

        # Tokenize per language and save tensors
        print("[data] Tokenizing per language...")
        for lc in self.config.language_codes():
            tr_tensor = tokenize_texts(tok, lang_train_texts[lc], desc=f"train {lc}")
            va_tensor = tokenize_texts(tok, lang_val_texts[lc],   desc=f"val   {lc}")
            train_path = os.path.join(self.cache_dir, f"lang_train_{lc}.pt")
            val_path   = os.path.join(self.cache_dir, f"lang_val_{lc}.pt")
            torch.save(tr_tensor, train_path)
            torch.save(va_tensor, val_path)
            print(f"  {lc}: {len(tr_tensor):>10,} train tokens, "
                  f"{len(va_tensor):>8,} val tokens  → cached")

        del lang_train_texts, lang_val_texts, all_train_texts
        gc.collect()
        print("[data] Download + tokenisation complete.")


# ---------------------------------------------------------------------------
# Client dataset partitioning (cheap — runs per experiment)
# ---------------------------------------------------------------------------

def create_client_datasets(
    lang_train_tensors: Dict[str, torch.Tensor],
    lang_val_tensors:   Dict[str, torch.Tensor],
    num_clients: int,
    data_distribution: str,
    dirichlet_alpha: float = 0.5,
    tokens_per_client: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, np.ndarray]:
    """
    Partition language tensors into per-client shards.

    Returns
    -------
    client_train   : list[Tensor]    one training tensor per client
    client_val     : list[Tensor]    local val tensor per client (dominant language)
    global_val     : Tensor          all-language val concatenated
    lang_fracs     : ndarray[C, L]   language fraction per client
    """
    rng       = np.random.default_rng(seed)
    lc_list   = list(lang_train_tensors.keys())
    num_langs = len(lc_list)
    arrays    = {lc: lang_train_tensors[lc].numpy() for lc in lc_list}

    if data_distribution == "multi_language":
        chunks = [[] for _ in range(num_clients)]
        fracs  = np.full((num_clients, num_langs), 1.0 / num_langs)
        for li, lc in enumerate(lc_list):
            arr    = arrays[lc]
            n_each = len(arr) // num_clients
            for ci in range(num_clients):
                chunks[ci].append(arr[ci * n_each : (ci + 1) * n_each])
        client_train = [torch.from_numpy(np.concatenate(c).copy()) for c in chunks]

    elif data_distribution == "single_language":
        client_train = []
        fracs = np.zeros((num_clients, num_langs))
        for ci in range(num_clients):
            li       = ci % num_langs
            lc       = lc_list[li]
            arr      = arrays[lc]
            n_per    = len(arr) // max(num_clients // num_langs, 1)
            offset   = (ci // num_langs) * n_per
            client_train.append(torch.from_numpy(arr[offset : offset + n_per].copy()))
            fracs[ci, li] = 1.0

    elif data_distribution == "dirichlet":
        props      = rng.dirichlet(np.ones(num_langs) * dirichlet_alpha, size=num_clients)
        col_sums   = props.sum(axis=0, keepdims=True).clip(min=1e-8)
        props_norm = props / col_sums

        chunks = [[] for _ in range(num_clients)]
        for li, lc in enumerate(lc_list):
            arr   = arrays[lc]
            start = 0
            for ci in range(num_clients):
                if ci < num_clients - 1:
                    n = int(len(arr) * props_norm[ci, li])
                else:
                    n = len(arr) - start
                if n > 0:
                    chunks[ci].append(arr[start : start + n])
                start += n

        client_train = [
            torch.from_numpy(np.concatenate(c).copy()) if c
            else torch.zeros(1, dtype=torch.long)
            for c in chunks
        ]
        fracs = props

    else:
        raise ValueError(f"Unknown data_distribution: {data_distribution!r}")

    if tokens_per_client is not None:
        client_train = [t[:tokens_per_client] for t in client_train]

    dominant   = [int(fracs[ci].argmax()) for ci in range(num_clients)]
    client_val = [lang_val_tensors[lc_list[dominant[ci]]] for ci in range(num_clients)]
    global_val = torch.cat(list(lang_val_tensors.values()))

    return client_train, client_val, global_val, fracs


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def make_dataloader(data_tensor: torch.Tensor, batch_size: int, seq_len: int, device):
    """
    Infinite generator yielding random (x, y) batches from a flat token tensor.
    x = data[pos:pos+seq_len], y = data[pos+1:pos+seq_len+1].
    """
    n = len(data_tensor) - seq_len - 1
    assert n > 0, (
        f"Data tensor too short ({len(data_tensor)} tokens) for seq_len={seq_len}"
    )
    while True:
        ix = torch.randint(0, n, (batch_size,))
        x  = torch.stack([data_tensor[i     : i + seq_len    ] for i in ix]).to(device)
        y  = torch.stack([data_tensor[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        yield x, y
