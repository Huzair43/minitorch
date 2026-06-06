from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SGD:
    lr: float = 0.01
    name: str = "sgd"


@dataclass(frozen=True)
class Adam:
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    name: str = "adam"
