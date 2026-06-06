from __future__ import annotations

import ctypes
from pathlib import Path

from ._lib import MtEvalResult, as_bytes, lib
from .data import Dataset


class Model:
    def __init__(self):
        raise TypeError("Utilise mt.Linear(...) ou mt.MLP(...) pour créer un modèle.")

    @classmethod
    def _from_factory(cls, input_size: int, output_size: int, hidden_size: int | None, kind: str) -> "Model":
        c_lib = lib()
        tape = c_lib.ag_tape_create()
        if not tape:
            raise RuntimeError("Création du tape autograd échouée.")

        if output_size == 1:
            task = "binary"
            if kind == "linear":
                ptr = c_lib.mt_model_create_linear_binary(tape, input_size)
            else:
                ptr = c_lib.mt_model_create_mlp_binary(tape, input_size, hidden_size or 4)
        else:
            task = "multiclass"
            if kind == "linear":
                ptr = c_lib.mt_model_create_linear_multiclass(tape, input_size, output_size)
            else:
                ptr = c_lib.mt_model_create_mlp_multiclass(tape, input_size, hidden_size or 8, output_size)

        model = cls.__new__(cls)
        model._tape = tape
        model._ptr = ptr
        model.input_size = input_size
        model.output_size = output_size
        model.task = task
        if not ptr:
            c_lib.ag_tape_free(tape)
            raise RuntimeError("Création du modèle C échouée.")
        model._checkpoint = c_lib.ag_checkpoint(tape)
        return model

    @property
    def is_binary(self) -> bool:
        return self.task == "binary"

    def predict(self, features: list[float] | tuple[float, ...]) -> list[float] | float:
        if len(features) != self.input_size:
            raise ValueError(f"Le modèle attend {self.input_size} features.")

        features_array = (ctypes.c_float * self.input_size)(*features)
        output_array = (ctypes.c_float * self.output_size)()
        ok = lib().mt_model_predict(
            self._tape,
            self._ptr,
            self._checkpoint,
            features_array,
            output_array,
            self.output_size,
        )
        if not ok:
            raise RuntimeError("Prédiction échouée.")

        values = [float(output_array[i]) for i in range(self.output_size)]
        return values[0] if self.is_binary else values

    def evaluate(self, dataset: Dataset) -> dict:
        result = MtEvalResult()
        c_lib = lib()
        if self.is_binary:
            ok = c_lib.mt_model_eval_binary(
                self._tape,
                self._ptr,
                self._checkpoint,
                dataset._ptr,
                ctypes.c_float(0.5),
                ctypes.byref(result),
            )
            confusion = None
        else:
            confusion_array = (ctypes.c_int * (self.output_size * self.output_size))()
            ok = c_lib.mt_model_eval_multiclass(
                self._tape,
                self._ptr,
                self._checkpoint,
                dataset._ptr,
                ctypes.byref(result),
                confusion_array,
            )
            confusion = [
                [int(confusion_array[row * self.output_size + col]) for col in range(self.output_size)]
                for row in range(self.output_size)
            ]

        if not ok:
            raise RuntimeError("Évaluation échouée.")

        return {
            "loss": float(result.loss_mean),
            "accuracy": float(result.accuracy),
            "correct": int(result.correct),
            "total": int(result.total),
            "confusion_matrix": confusion,
        }

    def save(self, path: str | Path) -> None:
        ok = lib().mt_model_save(self._tape, self._ptr, as_bytes(path))
        if not ok:
            raise RuntimeError("Sauvegarde du modèle échouée.")

    def load(self, path: str | Path) -> None:
        ok = lib().mt_model_load(self._tape, self._ptr, as_bytes(path))
        if not ok:
            raise RuntimeError("Chargement du modèle échoué.")
        self._checkpoint = lib().ag_checkpoint(self._tape)

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            lib().mt_model_free(self._ptr)
            self._ptr = None
        if getattr(self, "_tape", None):
            lib().ag_tape_free(self._tape)
            self._tape = None

    def __del__(self) -> None:
        self.close()


class Linear(Model):
    def __init__(self, input_size: int, output_size: int = 1):
        model = self._from_factory(input_size, output_size, None, "linear")
        self.__dict__.update(model.__dict__)
        model._ptr = None
        model._tape = None


class MLP(Model):
    def __init__(self, input_size: int, hidden_size: int = 8, output_size: int = 1):
        model = self._from_factory(input_size, output_size, hidden_size, "mlp")
        self.__dict__.update(model.__dict__)
        model._ptr = None
        model._tape = None
