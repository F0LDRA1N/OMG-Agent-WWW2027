from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.manifest import MODALITIES, ManifestRecord


@dataclass
class RetrievedEvidence:
    values: torch.Tensor
    similarities: torch.Tensor
    indices: torch.Tensor


def _mean_feature(path: Any) -> torch.Tensor:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return torch.from_numpy(
        np.asarray(array, dtype=np.float32).mean(axis=0).copy()
    )


def _source_pattern(target_index: int) -> str:
    bits = ["1", "1", "1"]
    bits[target_index] = "0"
    return "".join(bits)


class TrainingMemory(nn.Module):
    """One N_train-member, train-only IVF-Flat memory per target modality."""

    def __init__(
        self,
        records: list[ManifestRecord],
        plan_cache: dict[str, Any],
        feature_dims: dict[str, int],
        plan_hidden_dim: int,
        shared_dim: int,
        key_dim: int,
        top_k: int,
        temperature: float,
        nlist: int,
        nprobe: int,
        index_seed: int,
    ) -> None:
        super().__init__()
        del feature_dims
        if not records or any(record.split != "train" for record in records):
            raise ValueError("TrainingMemory accepts non-empty training records only")
        ids = [record.sample_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate training IDs in target memory")
        self.modalities = tuple(MODALITIES)
        self.sample_ids = ids
        self.id_to_index = {
            sample_id: index for index, sample_id in enumerate(ids)
        }
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.index_seed = int(index_seed)
        self.source_entries: dict[str, list[dict[str, Any]]] = {}
        if self.top_k >= len(records):
            raise ValueError(
                f"top_k={self.top_k} must be smaller than {len(records)} "
                "training samples so a training query can exclude itself"
            )

        for modality in self.modalities:
            values = torch.stack(
                [_mean_feature(record.features[modality]) for record in records]
            )
            self.register_buffer(
                f"source_{modality}", values, persistent=False
            )

        for target_index, target in enumerate(self.modalities):
            pattern = _source_pattern(target_index)
            entries = [plan_cache[sample_id][pattern] for sample_id in ids]
            for entry in entries:
                for candidate in entry.get("candidates", []):
                    if candidate["hidden"].shape[-1] != plan_hidden_dim:
                        raise ValueError(
                            f"{target} memory plan width is "
                            f"{candidate['hidden'].shape[-1]}, "
                            f"expected {plan_hidden_dim}"
                        )
            self.source_entries[target] = entries
            availability = torch.ones(
                len(records), len(self.modalities), dtype=torch.bool
            )
            availability[:, target_index] = False
            self.register_buffer(
                f"source_availability_{target}",
                availability,
                persistent=False,
            )
            self.register_buffer(
                f"cached_keys_{target}",
                torch.empty(len(records), key_dim),
                persistent=False,
            )

        self.query_projection = nn.ModuleDict(
            {
                target: nn.Linear(shared_dim * 2, key_dim)
                for target in self.modalities
            }
        )
        self.key_projection = nn.ModuleDict(
            {
                target: nn.Linear(shared_dim * 2, key_dim)
                for target in self.modalities
            }
        )
        self._indices: dict[str, Any] = {}

    def _pooled_sources(
        self, indices: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        values = {
            modality: getattr(self, f"source_{modality}")
            for modality in self.modalities
        }
        if indices is not None:
            values = {
                modality: value.index_select(0, indices.to(value.device))
                for modality, value in values.items()
            }
        return values

    def _source_keys(
        self,
        target: str,
        plan_selector: nn.Module,
        context_encoder: nn.Module,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        availability = getattr(self, f"source_availability_{target}")
        if indices is not None:
            device_indices = indices.to(availability.device)
            availability = availability.index_select(0, device_indices)
            entries = [
                self.source_entries[target][int(index)]
                for index in indices.detach().cpu().tolist()
            ]
        else:
            entries = self.source_entries[target]
        context = context_encoder.from_pooled(
            self._pooled_sources(indices), availability
        )
        plan = plan_selector(
            entries,
            context,
            compute_verifier_loss=False,
        ).pooled
        return F.normalize(
            self.key_projection[target](torch.cat([plan, context], dim=-1)),
            dim=-1,
        )

    @torch.no_grad()
    def refresh(
        self,
        plan_selector: nn.Module,
        context_encoder: nn.Module,
        refresh_batch_size: int = 512,
    ) -> None:
        was_plan_training = plan_selector.training
        was_context_training = context_encoder.training
        plan_selector.eval()
        context_encoder.eval()
        for target in self.modalities:
            key_chunks = []
            availability = getattr(self, f"source_availability_{target}")
            for start in range(0, len(self.sample_ids), refresh_batch_size):
                indices = torch.arange(
                    start,
                    min(start + refresh_batch_size, len(self.sample_ids)),
                    device=availability.device,
                )
                key_chunks.append(
                    self._source_keys(
                        target,
                        plan_selector,
                        context_encoder,
                        indices,
                    )
                )
            keys = torch.cat(key_chunks, dim=0)
            cache = getattr(self, f"cached_keys_{target}")
            cache.copy_(keys.to(cache.device))
            self._indices[target] = self._build_index(
                keys.float().cpu().numpy()
            )
        plan_selector.train(was_plan_training)
        context_encoder.train(was_context_training)

    def _build_index(self, keys: np.ndarray) -> Any:
        try:
            import faiss
        except ImportError:
            return None
        faiss.omp_set_num_threads(1)
        dimension = keys.shape[1]
        clusters = min(self.nlist, max(1, len(keys) // 16))
        if clusters <= 1:
            index = faiss.IndexFlatIP(dimension)
        else:
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(
                quantizer,
                dimension,
                clusters,
                faiss.METRIC_INNER_PRODUCT,
            )
            if hasattr(index, "cp"):
                index.cp.seed = self.index_seed
            index.train(np.ascontiguousarray(keys, dtype=np.float32))
            index.nprobe = min(self.nprobe, clusters)
        index.add(np.ascontiguousarray(keys, dtype=np.float32))
        return index

    def _search(
        self,
        target: str,
        query: torch.Tensor,
        query_ids: list[str],
    ) -> torch.Tensor:
        requested = min(len(self.sample_ids), self.top_k + 16)
        index = self._indices.get(target)
        if index is None:
            cached = getattr(self, f"cached_keys_{target}").to(query.device)
            candidates = (
                query.detach() @ cached.T
            ).topk(requested, dim=1).indices.cpu()
        else:
            _, raw = index.search(
                np.ascontiguousarray(query.detach().float().cpu().numpy()),
                requested,
            )
            candidates = torch.from_numpy(raw)
        selected = []
        for row, sample_id in enumerate(query_ids):
            self_index = self.id_to_index.get(sample_id)
            values = [
                int(index_value)
                for index_value in candidates[row].tolist()
                if index_value >= 0 and index_value != self_index
            ][: self.top_k]
            if len(values) != self.top_k:
                raise RuntimeError(
                    f"Could not retrieve {self.top_k} non-self neighbors "
                    f"for {sample_id}"
                )
            selected.append(values)
        return torch.tensor(
            selected, dtype=torch.long, device=query.device
        )

    def retrieve(
        self,
        target: str,
        query_plan: torch.Tensor,
        query_context: torch.Tensor,
        query_availability: torch.Tensor,
        query_ids: list[str],
        plan_selector: nn.Module,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
    ) -> RetrievedEvidence:
        del query_availability
        if target not in self.modalities:
            raise ValueError(f"Unknown target modality: {target}")
        if target not in self._indices:
            raise RuntimeError("Memory index has not been refreshed")
        query = F.normalize(
            self.query_projection[target](
                torch.cat([query_plan, query_context], dim=-1)
            ),
            dim=-1,
        )
        indices = self._search(target, query, query_ids)
        flat = indices.flatten()
        source_keys = self._source_keys(
            target, plan_selector, context_encoder, flat
        ).view(indices.shape[0], indices.shape[1], -1)
        similarities = torch.einsum("bd,bkd->bk", query, source_keys)
        weights = (similarities / self.temperature).softmax(dim=1)
        source_values = getattr(self, f"source_{target}")
        target_values = source_values.index_select(
            0, flat.to(source_values.device)
        )
        target_values = target_encoder(target_values).view(
            indices.shape[0], indices.shape[1], -1
        )
        evidence = (
            target_values * weights.unsqueeze(-1)
        ).sum(dim=1, keepdim=True)
        return RetrievedEvidence(
            values=evidence,
            similarities=similarities,
            indices=indices,
        )
