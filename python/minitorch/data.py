from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from ._lib import MtCsvInfo, MtDataset, as_bytes, lib


@dataclass(frozen=True)
class CsvInfo:
    n_rows: int
    n_columns: int
    n_features: int
    has_header: bool
    n_classes: int
    is_classification: bool
    is_binary: bool
    label_min: float
    label_max: float

    @classmethod
    def from_c(cls, info: MtCsvInfo) -> "CsvInfo":
        return cls(
            n_rows=info.n_rows,
            n_columns=info.n_columns,
            n_features=info.n_features,
            has_header=bool(info.has_header),
            n_classes=info.n_classes,
            is_classification=bool(info.is_classification),
            is_binary=bool(info.is_binary),
            label_min=float(info.label_min),
            label_max=float(info.label_max),
        )


class Dataset:
    def __init__(self, ptr: ctypes.POINTER(MtDataset), info: CsvInfo | None = None, owns: bool = True):
        if not ptr:
            raise RuntimeError("Dataset C invalide.")
        self._ptr = ptr
        self._owns = owns
        self.info = info

    @classmethod
    def from_csv(cls, path: str | Path, has_header: bool = True) -> "Dataset":
        c_info = MtCsvInfo()
        c_lib = lib()
        ptr = c_lib.mt_dataset_load_csv_auto(as_bytes(path), int(has_header), ctypes.byref(c_info))
        if not ptr:
            raise RuntimeError(
                "Chargement CSV échoué. Le fichier doit être numérique, avec le label en dernière colonne."
            )
        return cls(ptr, CsvInfo.from_c(c_info), owns=True)

    @property
    def n_samples(self) -> int:
        return int(self._ptr.contents.n_samples)

    @property
    def n_features(self) -> int:
        return int(self._ptr.contents.n_features)

    @property
    def n_classes(self) -> int:
        if self.info and self.info.is_classification:
            return self.info.n_classes
        labels = {int(round(self.label(i))) for i in range(self.n_samples)}
        return len(labels)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    def feature(self, sample: int, feature: int) -> float:
        return float(lib().mt_dataset_feature(self._ptr, sample, feature))

    def label(self, sample: int) -> float:
        return float(lib().mt_dataset_label(self._ptr, sample))

    def split(self, train: float = 0.7, validation: float = 0.15) -> tuple["Dataset", "Dataset | None", "Dataset"]:
        train_ptr = ctypes.POINTER(MtDataset)()
        val_ptr = ctypes.POINTER(MtDataset)()
        test_ptr = ctypes.POINTER(MtDataset)()

        ok = lib().mt_dataset_split(
            self._ptr,
            ctypes.c_float(train),
            ctypes.c_float(validation),
            ctypes.byref(train_ptr),
            ctypes.byref(val_ptr),
            ctypes.byref(test_ptr),
        )
        if not ok:
            raise RuntimeError("Split du dataset échoué.")

        train_set = Dataset(train_ptr, self.info, owns=True)
        val_set = Dataset(val_ptr, self.info, owns=True) if bool(val_ptr) else None
        test_set = Dataset(test_ptr, self.info, owns=True)
        return train_set, val_set, test_set

    def head(self, n: int = 5) -> list[tuple[list[float], float]]:
        limit = min(n, self.n_samples)
        rows = []
        for sample in range(limit):
            features = [self.feature(sample, feature) for feature in range(self.n_features)]
            rows.append((features, self.label(sample)))
        return rows

    def close(self) -> None:
        if getattr(self, "_ptr", None) and self._owns:
            lib().mt_dataset_free(self._ptr)
        self._ptr = None
        self._owns = False

    def __del__(self) -> None:
        self.close()
