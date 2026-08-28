"""Exponential-moving-average utilities for trainable parameters."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class EMAModuleWrapper:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ) -> None:
        parameters = list(parameters)
        self.ema_parameters = [parameter.detach().clone().to(device) for parameter in parameters]
        self.temp_stored_parameters: list[torch.Tensor] | None = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step: int) -> float:
        return min((1 + optimization_step) / (10 + optimization_step), self.decay)

    @torch.no_grad()
    def step(
        self, parameters: Iterable[torch.nn.Parameter], optimization_step: int
    ) -> None:
        if (optimization_step + 1) % self.update_step_interval:
            return
        one_minus_decay = 1 - self.get_current_decay(optimization_step)
        for ema_parameter, parameter in zip(
            self.ema_parameters, list(parameters), strict=True
        ):
            if not parameter.requires_grad:
                continue
            update = parameter.detach().to(ema_parameter.device) - ema_parameter
            ema_parameter.add_(update, alpha=one_minus_decay)

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.device = device
        self.ema_parameters = [
            parameter.to(device=device, dtype=dtype)
            if parameter.is_floating_point()
            else parameter.to(device=device)
            for parameter in self.ema_parameters
        ]

    def copy_ema_to(
        self,
        parameters: Iterable[torch.nn.Parameter],
        store_temp: bool = True,
    ) -> None:
        parameters = list(parameters)
        if store_temp:
            self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters]
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        if self.temp_stored_parameters is None:
            raise RuntimeError("No temporary parameters have been stored")
        for saved, parameter in zip(
            self.temp_stored_parameters, list(parameters), strict=True
        ):
            parameter.data.copy_(saved.to(parameter.device).data)
        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict["ema_parameters"]
        self.to(self.device)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "ema_parameters": self.ema_parameters}
