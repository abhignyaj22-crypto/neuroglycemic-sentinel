"""Self-supervised masked-reconstruction pretraining for the neural model.

Glucose-referenced windows are scarce (a same-patient CGM anchor is required),
while unlabeled EEG/wearable windows accumulate continuously from LSL replay.
This module lets the encoders and the cross-modal context layer learn the
joint statistics of the sensor streams *before* any glucose label is touched:

1. a random subset of observed feature entries is hidden from the encoders;
2. the corrupted embeddings pass through the cross-modal context layer;
3. per-modality decoders reconstruct the hidden standardized values.

The procedure never reads glucose targets, so it cannot leak label
information.  Fine-tuning then starts from the pretrained weights and learns
only the mapping from sensor state to glucose.

Only the training split may be pretrained on; validation and test patients
remain untouched for honest model selection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .neural_training import Batch, _move_batch, _resolve_device, _seed_everything

PRETRAIN_SCHEMA = "neuroglycemic-pretrain-v1"


def masked_reconstruction_loss(
    model: nn.Module,
    batch: Batch,
    *,
    hide_probability: float,
    generator: torch.Generator,
) -> tuple[Tensor, int]:
    """Hide observed entries, reconstruct them, and score only hidden entries."""

    if not 0 < hide_probability < 1:
        raise ValueError("hide_probability must be in (0, 1).")
    if not getattr(model, "has_reconstruction_heads", False):
        raise ValueError(
            "Pretraining requires a model built with build_reconstruction_heads=True."
        )
    features: dict[str, Tensor] = {}
    masks: dict[str, Tensor] = {}
    hidden_targets: dict[str, Tensor] = {}
    hidden_masks: dict[str, Tensor] = {}
    for name in model.modalities:
        observed = batch["feature_masks"][name].to(dtype=torch.bool)
        values = batch["features"][name]
        hide = (
            torch.rand(
                observed.shape,
                generator=generator,
                device=observed.device,
            )
            < hide_probability
        ) & observed
        hidden_masks[name] = hide
        hidden_targets[name] = torch.where(hide, values, torch.zeros_like(values))
        masks[name] = observed & ~hide
        features[name] = values
    embeddings, _ = model.encode_contextualized(
        features, masks, batch["availability"].to(dtype=torch.bool)
    )
    total = embeddings.new_zeros(())
    terms = 0
    for index, name in enumerate(model.modalities):
        hide = hidden_masks[name]
        if not bool(hide.any()):
            continue
        reconstruction = model.reconstruction_heads[name](embeddings[:, index])
        total = total + F.mse_loss(reconstruction[hide], hidden_targets[name][hide])
        terms += 1
    if terms == 0:
        raise RuntimeError("No observed feature entries were available to hide.")
    return total / terms, terms


def pretrain_masked_reconstruction(
    model: nn.Module,
    batches: Iterable[Batch],
    *,
    epochs: int,
    learning_rate: float,
    hide_probability: float = 0.25,
    gradient_clip_norm: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
) -> list[dict[str, float | int]]:
    """Optimize the reconstruction objective on unlabeled training batches.

    Returns one history row per epoch.  The caller is responsible for passing
    batches built from the *training* patient partition only.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    _seed_everything(seed)
    resolved = _resolve_device(device)
    model.to(resolved)
    batch_values = [_move_batch(batch, resolved) for batch in batches]
    if not batch_values:
        raise ValueError("Pretraining batches must not be empty.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=resolved)
    generator.manual_seed(seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train(True)
        order = list(range(len(batch_values)))
        random.Random(seed + epoch).shuffle(order)
        loss_sum = 0.0
        for index in order:
            batch = batch_values[index]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = masked_reconstruction_loss(
                model, batch, hide_probability=hide_probability, generator=generator
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Pretraining loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None],
                max_norm=gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
        history.append({"epoch": epoch, "pretrain_loss": loss_sum / len(order)})
    return history


def save_pretrain_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save encoder/context weights for later fine-tuning."""

    payload = {
        "schema_version": PRETRAIN_SCHEMA,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pretrain_weights(
    model: nn.Module,
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load pretrained weights, skipping reconstruction decoders.

    Shape mismatches raise immediately; absent reconstruction-head keys are
    expected because forecasting checkpoints never contain them.
    """

    resolved = torch.device(device)
    try:
        payload = torch.load(path, map_location=resolved, weights_only=True)
    except TypeError:  # older torch without weights_only
        payload = torch.load(path, map_location=resolved)
    if payload.get("schema_version") != PRETRAIN_SCHEMA:
        raise ValueError("Unsupported pretraining checkpoint schema.")
    state = payload["model_state_dict"]
    own = model.state_dict()
    incompatible = [
        name
        for name, value in state.items()
        if name in own and tuple(own[name].shape) != tuple(value.shape)
    ]
    if incompatible:
        raise ValueError(
            f"Pretrained weights do not match this architecture: {incompatible}"
        )
    filtered = {
        name: value
        for name, value in state.items()
        if name in own and not name.startswith("reconstruction_heads.")
    }
    model.load_state_dict(filtered, strict=False)
    return payload


def pretrain_history_metadata(history: list[dict[str, float | int]]) -> dict[str, Any]:
    """Compact provenance for the fine-tuning checkpoint metadata."""

    if not history:
        return {}
    return {
        "pretrain_epochs": int(history[-1]["epoch"]),
        "pretrain_initial_loss": float(history[0]["pretrain_loss"]),
        "pretrain_final_loss": float(history[-1]["pretrain_loss"]),
    }


def history_to_json(history: list[dict[str, float | int]]) -> str:
    return json.dumps(history, indent=2)