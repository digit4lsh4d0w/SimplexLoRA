"""Stable identification of all calibrated ``SimplexLoraLinear`` modules."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, cast

from torch import nn

from simplexlora.calibration.models import (
    CalibrationRegistryError,
    LayerIdentity,
    LayerRegistry,
)
from simplexlora.peft_plugin import SimplexLoraLinear

_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)")


def _parse_layer_index(module_name: str) -> int:
    match = _LAYER_INDEX_PATTERN.search(module_name)
    if match is None:
        return -1
    return int(match.group(1))


def _component_of(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def _module_in_features(layer: SimplexLoraLinear) -> int:
    features = getattr(layer, "in_features", None)
    if features is None:
        features = getattr(layer.get_base_layer(), "in_features")
    return int(features)


def _module_out_features(layer: SimplexLoraLinear) -> int:
    features = getattr(layer, "out_features", None)
    if features is None:
        features = getattr(layer.get_base_layer(), "out_features")
    return int(features)


def _wrapped_base(model: Any) -> Any:
    base_model = getattr(model, "base_model", None)
    wrapped = getattr(base_model, "model", None) if base_model is not None else None
    if wrapped is None:
        # Not a PEFT-wrapped model: fall back to the model itself.
        return model
    return wrapped


def _matches_target(module_name: str, component: str, target: str) -> bool:
    if module_name == target or component == target:
        return True
    return module_name.endswith("." + target)


def build_layer_registry(
    model: Any,
    adapter_name: str,
    *,
    target_modules: Sequence[str] | None = None,
) -> LayerRegistry:
    """Discover all ``SimplexLoraLinear`` modules once and identify them stably.

    The registry links optimizer parameters to human-readable module names and
    removes any dependency on the unstable ordering of ``optimizer.param_groups``.
    It never mutates the model.

    Ordering is deterministic: ``layer_index``, then ``component``, then ``module_name``.
    """
    wrapped = _wrapped_base(model)

    full_names: dict[int, str] = {}
    for name, module in model.named_modules():
        full_names.setdefault(id(module), name)

    peft_configs = getattr(model, "peft_config", None)
    if peft_configs is not None and len(peft_configs) > 1:
        raise CalibrationRegistryError(
            "more than one PEFT adapter is loaded; calibration of multiple adapters is not supported in schema 1.0"
        )

    targets = tuple(target_modules) if target_modules is not None else None

    identities: list[LayerIdentity] = []
    seen_module_names: dict[str, str] = {}
    seen_peft_names: dict[str, str] = {}
    for canonical_name, module in wrapped.named_modules():
        if not isinstance(module, SimplexLoraLinear):
            continue
        peft_module_name = full_names.get(id(module))
        if peft_module_name is None:
            raise CalibrationRegistryError(
                f"module {canonical_name!r} is not reachable from the PEFT model; cannot build a registry"
            )

        adapters = set(getattr(module, "lora_A", {}).keys())
        if adapters and adapters != {adapter_name}:
            raise CalibrationRegistryError(
                f"module {canonical_name!r} has adapters {sorted(adapters)}, expected only {adapter_name!r}"
            )
        if adapter_name not in adapters:
            raise CalibrationRegistryError(f"adapter {adapter_name!r} is not registered on module {canonical_name!r}")

        if canonical_name in seen_module_names:
            raise CalibrationRegistryError(f"duplicate canonical module name: {canonical_name!r}")
        if peft_module_name in seen_peft_names:
            raise CalibrationRegistryError(f"duplicate PEFT module name: {peft_module_name!r}")
        seen_module_names[canonical_name] = peft_module_name
        seen_peft_names[peft_module_name] = canonical_name

        # The canonical name must resolve back to the very same base module.
        resolved = wrapped.get_submodule(canonical_name)
        if resolved is not module:
            raise CalibrationRegistryError(f"canonical name {canonical_name!r} does not resolve to its own module")

        component = _component_of(canonical_name)
        if targets is not None and not any(_matches_target(canonical_name, component, target) for target in targets):
            continue

        rank = int(module.r[adapter_name])
        in_features = _module_in_features(module)
        out_features = _module_out_features(module)
        lora_a_weight = cast(nn.Linear, module.lora_A[adapter_name]).weight
        lora_b_weight = cast(nn.Linear, module.lora_B[adapter_name]).weight
        expected_a = (rank, in_features)
        expected_b = (out_features, rank)
        if tuple(lora_a_weight.shape) != expected_a:
            raise CalibrationRegistryError(
                f"{canonical_name!r}: lora_A weight shape {tuple(lora_a_weight.shape)} != {expected_a}"
            )
        if tuple(lora_b_weight.shape) != expected_b:
            raise CalibrationRegistryError(
                f"{canonical_name!r}: lora_B weight shape {tuple(lora_b_weight.shape)} != {expected_b}"
            )

        identities.append(
            LayerIdentity(
                module_name=canonical_name,
                peft_module_name=peft_module_name,
                module_index=0,
                layer_index=_parse_layer_index(canonical_name),
                component=component,
                in_features=in_features,
                out_features=out_features,
                initial_rank=rank,
                lora_alpha=int(module.lora_alpha[adapter_name]),
                scaling=float(module.scaling[adapter_name]),
            )
        )

    if not identities:
        raise CalibrationRegistryError("no SimplexLoraLinear modules found; the model has no calibrated layers")

    identities.sort(key=lambda identity: (identity.layer_index, identity.component, identity.module_name))
    ordered = tuple(
        LayerIdentity(
            module_name=identity.module_name,
            peft_module_name=identity.peft_module_name,
            module_index=index,
            layer_index=identity.layer_index,
            component=identity.component,
            in_features=identity.in_features,
            out_features=identity.out_features,
            initial_rank=identity.initial_rank,
            lora_alpha=identity.lora_alpha,
            scaling=identity.scaling,
        )
        for index, identity in enumerate(identities)
    )

    name_to_index = {identity.module_name: identity.module_index for identity in ordered}
    if len(name_to_index) != len(ordered):
        raise CalibrationRegistryError("conflicting canonical module names after ordering")

    return LayerRegistry(
        adapter_name=adapter_name,
        identities=ordered,
        single_adapter=True,
        _module_name_to_index=name_to_index,
    )
