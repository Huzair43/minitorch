from __future__ import annotations

import ctypes
from dataclasses import dataclass

from ._lib import (
    MT_LOSS_BCE,
    MT_LOSS_CROSS_ENTROPY_FROM_LOGITS,
    MtTrainHistory,
    lib,
)
from .data import Dataset
from .models import Model
from .optim import Adam, SGD


@dataclass
class FitResult:
    history: dict
    train: dict
    validation: dict | None
    test: dict


class Trainer:
    def __init__(self, model: Model, optimizer: Adam | SGD | str = "adam", lr: float | None = None, loss: str = "auto"):
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.last_result: FitResult | None = None

    def fit(
        self,
        dataset: Dataset,
        epochs: int = 100,
        batch_size: int = 4,
        split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        shuffle: bool = True,
    ) -> FitResult:
        train_ratio, val_ratio, _ = split
        train_set, val_set, test_set = dataset.split(train_ratio, val_ratio)
        optim_ptr = self._create_optimizer()

        try:
            if not lib().mt_optimizer_add_model(optim_ptr, self.model._ptr):
                raise RuntimeError("Impossible d'ajouter le modèle à l'optimiseur.")

            self.model._checkpoint = lib().ag_checkpoint(self.model._tape)
            loss = lib().mt_loss_create(self._loss_kind())
            config = lib().mt_train_config_default()
            config.epochs = int(epochs)
            config.batch_size = int(batch_size)
            config.shuffle = 1 if shuffle else 0
            config.print_every = 0
            history = MtTrainHistory()

            if self.model.is_binary:
                ok = lib().mt_trainer_train_binary(
                    self.model._tape,
                    self.model._ptr,
                    optim_ptr,
                    ctypes.byref(loss),
                    self.model._checkpoint,
                    train_set._ptr,
                    ctypes.byref(config),
                    ctypes.byref(history),
                )
            else:
                ok = lib().mt_trainer_train_multiclass(
                    self.model._tape,
                    self.model._ptr,
                    optim_ptr,
                    ctypes.byref(loss),
                    self.model._checkpoint,
                    train_set._ptr,
                    ctypes.byref(config),
                    ctypes.byref(history),
                )

            if not ok:
                raise RuntimeError("Entraînement échoué côté C.")

            self.model._checkpoint = lib().ag_checkpoint(self.model._tape)
            result = FitResult(
                history={
                    "last_loss": float(history.last_loss),
                    "epochs": int(history.epochs_ran),
                    "batches": int(history.batches_seen),
                    "samples": int(history.samples_seen),
                },
                train=self.model.evaluate(train_set),
                validation=self.model.evaluate(val_set) if val_set else None,
                test=self.model.evaluate(test_set),
            )
            self.last_result = result
            return result
        finally:
            lib().mt_optimizer_free(optim_ptr)
            train_set.close()
            if val_set:
                val_set.close()
            test_set.close()

    def evaluate(self, dataset: Dataset) -> dict:
        return self.model.evaluate(dataset)

    def _loss_kind(self) -> int:
        if self.loss == "auto":
            return MT_LOSS_BCE if self.model.is_binary else MT_LOSS_CROSS_ENTROPY_FROM_LOGITS
        normalized = self.loss.lower()
        if normalized in {"bce", "binary_cross_entropy"}:
            return MT_LOSS_BCE
        if normalized in {"ce", "cross_entropy", "cross_entropy_from_logits"}:
            return MT_LOSS_CROSS_ENTROPY_FROM_LOGITS
        raise ValueError(f"Loss inconnue: {self.loss}")

    def _create_optimizer(self):
        opt = self.optimizer
        if isinstance(opt, str):
            opt = Adam(lr=self.lr if self.lr is not None else 0.001) if opt.lower() == "adam" else SGD(
                lr=self.lr if self.lr is not None else 0.01
            )
        elif self.lr is not None:
            if isinstance(opt, Adam):
                opt = Adam(lr=self.lr, beta1=opt.beta1, beta2=opt.beta2, eps=opt.eps)
            elif isinstance(opt, SGD):
                opt = SGD(lr=self.lr)

        if isinstance(opt, Adam):
            ptr = lib().mt_adam_create(
                ctypes.c_float(opt.lr),
                ctypes.c_float(opt.beta1),
                ctypes.c_float(opt.beta2),
                ctypes.c_float(opt.eps),
            )
        elif isinstance(opt, SGD):
            ptr = lib().mt_sgd_create(ctypes.c_float(opt.lr))
        else:
            raise ValueError("Optimiseur inconnu. Utilise 'adam', 'sgd', Adam(...) ou SGD(...).")

        if not ptr:
            raise RuntimeError("Création de l'optimiseur C échouée.")
        return ptr
