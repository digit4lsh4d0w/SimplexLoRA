"""Typed entry points for Transformers dataclass argument parsing."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, cast

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import HfArgumentParser, PreTrainedTokenizerBase
from transformers.hf_argparser import DataClassType

from simplexlora.config import DataTrainingArguments, ModelArguments, TrainingArguments


class MetricProtocol(Protocol):
    def compute(self, *args: object, **kwargs: object) -> dict[str, object]: ...


def parse_training_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    """Parse the three project argument dataclasses with concrete return types."""
    dataclasses = cast(
        Iterable[DataClassType],
        (ModelArguments, DataTrainingArguments, TrainingArguments),
    )
    parser = HfArgumentParser(dataclasses)
    parsed = parser.parse_args_into_dataclasses(list(argv) if argv is not None else None)
    return cast(tuple[ModelArguments, DataTrainingArguments, TrainingArguments], parsed)


def as_tokenizer(tokenizer: object) -> PreTrainedTokenizerBase:
    """Narrow Transformers' dynamic tokenizer factory to its runtime base type."""
    return cast(PreTrainedTokenizerBase, tokenizer)


def as_metric(metric: object) -> MetricProtocol:
    """Narrow evaluate.load's optional dynamic module to its compute interface."""
    if metric is None:
        raise TypeError("evaluate.load returned no metric")
    return cast(MetricProtocol, metric)


def as_trainer_optimizers(
    optimizer: Optimizer,
    scheduler: object,
) -> tuple[Optimizer | None, LambdaLR | None]:
    """Match Transformers' overly narrow scheduler annotation at the boundary."""
    return cast(tuple[Optimizer | None, LambdaLR | None], (optimizer, scheduler))


def uses_wandb(report_to: None | str | list[str]) -> bool:
    """Normalize the three valid TrainingArguments report_to shapes."""
    if report_to == "wandb":
        return True
    return isinstance(report_to, list) and "wandb" in report_to
