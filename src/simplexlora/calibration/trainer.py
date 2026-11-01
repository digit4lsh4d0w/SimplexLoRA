"""Trainer subclass that captures the exact per-optimizer-step training loss."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer


class SimplexCalibrationTrainer(Trainer):
    """``transformers.Trainer`` that feeds micro-batch losses to ``SimplexAdamW``.

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
        model: Any,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.calibration_optimizer is not None and loss is not None:
            detached = loss.detach() if not loss.is_leaf else loss.detach()
            self.calibration_optimizer.add_microbatch_loss(detached)
        return loss
