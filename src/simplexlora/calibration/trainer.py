"""Trainer subclass that captures the exact per-optimizer-step training loss."""

from __future__ import annotations

from typing import Any, cast

import torch
from transformers import Seq2SeqTrainer, Trainer


class SimplexCalibrationMixin:
    """Feed micro-batch losses to ``SimplexAdamW`` for any Trainer subclass.

    The contract of ``Trainer.training_step`` is preserved: the return value is
    exactly the base class's ``loss.detach()``, and optimization, backward and
    gradient accumulation are untouched. The only side effect is
    ``optimizer.add_microbatch_loss(loss.detach())`` so that ``SimplexAdamW``
    can aggregate the very loss the Trainer uses for its training loss.
    """

    def __init__(self, *args: Any, calibration_optimizer: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calibration_optimizer = calibration_optimizer

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        parent = cast(Any, super())
        loss = parent.training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.calibration_optimizer is not None and loss is not None:
            self.calibration_optimizer.add_microbatch_loss(loss.detach())
        return loss


class SimplexCalibrationTrainer(SimplexCalibrationMixin, Trainer):
    """Causal-LM Trainer with Simplex calibration loss capture."""


class SimplexCalibrationSeq2SeqTrainer(SimplexCalibrationMixin, Seq2SeqTrainer):
    """Seq2Seq Trainer with Simplex calibration loss capture."""
