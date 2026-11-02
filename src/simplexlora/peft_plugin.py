"""Simplex LoRA support layered on top of the public PEFT package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from peft.utils.other import transpose


_SIMPLEX_LORA_ATTRIBUTE = "_simplex_lora_plugin"


class SimplexLoraLinear(Linear):
    """PEFT LoRA linear layer with one learned simplex weight per adapter."""

    adapter_layer_names = (*Linear.adapter_layer_names, "lora_weight")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        config: LoraConfig,
        **kwargs: Any,
    ) -> None:
        if not hasattr(self, "lora_weight"):
            self.lora_weight = nn.ParameterDict()
        if not hasattr(self, "_simplex_lora_configs"):
            self._simplex_lora_configs: dict[str, LoraConfig] = {}

        existing_weight = None
        if adapter_name in self.lora_weight:
            existing_weight = cast(torch.Tensor, self.lora_weight[adapter_name])

        super().update_layer(adapter_name, r, lora_alpha, config, **kwargs)

        if existing_weight is None:
            adapter_weight = cast(nn.Linear, self.lora_A[adapter_name]).weight
            self.lora_weight[adapter_name] = nn.Parameter(adapter_weight.new_ones(1))
        else:
            # SimplexAdamW keeps references to these parameters across rank updates.
            self.lora_weight[adapter_name] = existing_weight
        self._simplex_lora_configs[adapter_name] = config

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """Return the weighted LoRA delta used by merge and unmerge."""
        lora_a_weight = cast(nn.Linear, self.lora_A[adapter]).weight
        lora_b_weight = cast(nn.Linear, self.lora_B[adapter]).weight
        device = lora_b_weight.device
        dtype = lora_b_weight.dtype
        cast_to_fp32 = device.type == "cpu" and dtype in (torch.float16, torch.bfloat16)

        weight_a = lora_a_weight
        weight_b = lora_b_weight
        weight_lora = cast(torch.Tensor, self.lora_weight[adapter])
        if cast_to_fp32:
            weight_a = weight_a.float()
            weight_b = weight_b.float()
            weight_lora = weight_lora.float()

        output = transpose(weight_b @ weight_a, self.fan_in_fan_out) * self.scaling[adapter]
        output = output * weight_lora
        return output.to(dtype=dtype) if cast_to_fp32 else output

    def _mixed_batch_forward(
        self,
        x: torch.Tensor,
        *args: Any,
        adapter_names: list[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype
        unique_adapters = set(adapter_names)
        sub_batch_indices = {
            adapter: [index for index, name in enumerate(adapter_names) if name == adapter]
            for adapter in unique_adapters
        }

        for active_adapter in unique_adapters:
            if active_adapter == "__base__" or active_adapter not in self.lora_A:
                continue

            indices = sub_batch_indices[active_adapter]
            lora_a = cast(nn.Linear, self.lora_A[active_adapter])
            lora_b = cast(nn.Linear, self.lora_B[active_adapter])
            dropout = self.lora_dropout[active_adapter]
            sub_batch = x[indices].to(lora_a.weight.dtype)
            lora_output = lora_b(lora_a(dropout(sub_batch))) * self.scaling[active_adapter]
            lora_output = lora_output * cast(torch.Tensor, self.lora_weight[active_adapter])
            result[indices] += lora_output.to(result_dtype)

        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue

            lora_a = cast(nn.Linear, self.lora_A[active_adapter])
            lora_b = cast(nn.Linear, self.lora_B[active_adapter])
            dropout = self.lora_dropout[active_adapter]
            adapter_input = x.to(lora_a.weight.dtype)
            lora_output = lora_b(lora_a(dropout(adapter_input))) * self.scaling[active_adapter]
            result = result + lora_output * cast(torch.Tensor, self.lora_weight[active_adapter])

        return result.to(result_dtype)

    @property
    def lora_variants(self) -> Mapping[tuple[str, ...], None]:
        """Simplex LoRA currently composes only with vanilla LoRA."""
        return {(): None}

    def _update_lora_rank_QR(self, new_rank: int, adapter_name: str) -> None:
        """Resize an adapter at an intermediate Simplex projection.

        The historical method name is retained because ``SimplexAdamW`` calls
        it. The latest vendored implementation reinitialized the adapter at a
        new rank instead of applying QR, so this method preserves that contract.
        Calling it with the current rank is intentionally a no-op to avoid
        discarding learned A/B matrices on every projection.
        """
        if new_rank < 0:
            raise ValueError(f"new_rank must be non-negative, got {new_rank}")

        current_rank = int(self.r[adapter_name])
        if new_rank == current_rank:
            return

        lora_weight = cast(torch.Tensor, self.lora_weight[adapter_name])
        lora_a = cast(nn.Linear, self.lora_A[adapter_name]).weight
        lora_b = cast(nn.Linear, self.lora_B[adapter_name]).weight

        if new_rank == 0:
            lora_weight.data.zero_()
            lora_weight.requires_grad_(False)
            lora_a.requires_grad_(False)
            lora_b.requires_grad_(False)
            self.r[adapter_name] = 0
            self.scaling[adapter_name] = 0.0
            return

        config = self._simplex_lora_configs[adapter_name]
        self.update_layer(
            adapter_name,
            new_rank,
            self.lora_alpha[adapter_name],
            config=config,
        )
        lora_weight.requires_grad_(True)

    def _final_lora_rank_update(self, new_rank: int, adapter_name: str) -> None:
        """Set the final rank and fix the learned layer weight at one."""
        self._update_lora_rank_QR(new_rank, adapter_name)
        final_weight = cast(torch.Tensor, self.lora_weight[adapter_name])
        final_weight.data.fill_(1 if new_rank > 0 else 0)
        final_weight.requires_grad_(False)


def register_simplex_lora(config: LoraConfig) -> LoraConfig:
    """Register the out-of-tree Simplex LoRA layer on a PEFT config."""
    if getattr(config, _SIMPLEX_LORA_ATTRIBUTE, False):
        return config

    config._register_custom_module({nn.Linear: SimplexLoraLinear})
    setattr(config, _SIMPLEX_LORA_ATTRIBUTE, True)
    return config


def make_simplex_lora_config(**kwargs: Any) -> LoraConfig:
    """Create and register a standard PEFT LoRA config for Simplex LoRA."""
    return register_simplex_lora(LoraConfig(**kwargs))


def get_simplex_peft_model(
    model: nn.Module,
    config: Any,
    adapter_name: str = "default",
) -> Any:
    """Create a PEFT model, registering Simplex LoRA only when requested."""
    if isinstance(config, LoraConfig) and getattr(config, _SIMPLEX_LORA_ATTRIBUTE, False):
        register_simplex_lora(config)
    return get_peft_model(model, config, adapter_name=adapter_name)


def load_simplex_lora_config(
    pretrained_model_name_or_path: str,
    **kwargs: Any,
) -> LoraConfig:
    """Load an adapter config and restore its runtime custom-module mapping."""
    config = LoraConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    return register_simplex_lora(config)


def load_simplex_peft_model(
    model: nn.Module,
    pretrained_model_name_or_path: str,
    **kwargs: Any,
) -> Any:
    """Load a Simplex LoRA adapter after re-registering its custom layer."""
    config = kwargs.pop("config", None)
    if config is None:
        config = load_simplex_lora_config(pretrained_model_name_or_path, **kwargs)
    elif isinstance(config, LoraConfig):
        config = register_simplex_lora(config)
    return PeftModel.from_pretrained(model, pretrained_model_name_or_path, config=config, **kwargs)
