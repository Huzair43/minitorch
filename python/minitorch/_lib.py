from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


class MtCsvInfo(ctypes.Structure):
    _fields_ = [
        ("n_rows", ctypes.c_int),
        ("n_columns", ctypes.c_int),
        ("n_features", ctypes.c_int),
        ("has_header", ctypes.c_int),
        ("n_classes", ctypes.c_int),
        ("is_classification", ctypes.c_int),
        ("is_binary", ctypes.c_int),
        ("label_min", ctypes.c_float),
        ("label_max", ctypes.c_float),
    ]


class MtDataset(ctypes.Structure):
    _fields_ = [
        ("n_samples", ctypes.c_int),
        ("n_features", ctypes.c_int),
        ("x", ctypes.POINTER(ctypes.c_float)),
        ("y", ctypes.POINTER(ctypes.c_float)),
        ("indices", ctypes.POINTER(ctypes.c_int)),
    ]


class MtEvalResult(ctypes.Structure):
    _fields_ = [
        ("loss_mean", ctypes.c_float),
        ("accuracy", ctypes.c_float),
        ("correct", ctypes.c_int),
        ("total", ctypes.c_int),
    ]


class MtLoss(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int)]


class MtTrainConfig(ctypes.Structure):
    _fields_ = [
        ("epochs", ctypes.c_int),
        ("batch_size", ctypes.c_int),
        ("shuffle", ctypes.c_int),
        ("print_every", ctypes.c_int),
    ]


class MtTrainHistory(ctypes.Structure):
    _fields_ = [
        ("last_loss", ctypes.c_float),
        ("epochs_ran", ctypes.c_int),
        ("batches_seen", ctypes.c_int),
        ("samples_seen", ctypes.c_int),
    ]


MT_LOSS_MSE = 0
MT_LOSS_BCE = 1
MT_LOSS_CROSS_ENTROPY = 2
MT_LOSS_CROSS_ENTROPY_FROM_LOGITS = 3


_LIB = None


def _candidate_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["minitorch_c.dll", "libminitorch_c.dll"]
    if sys.platform == "darwin":
        return ["libminitorch_c.dylib", "minitorch_c.dylib"]
    return ["libminitorch_c.so", "minitorch_c.so"]


def _candidate_paths() -> list[Path]:
    env_path = os.environ.get("MINITORCH_LIB")
    paths: list[Path] = []
    if env_path:
        paths.append(Path(env_path))

    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parents[1]
    cwd = Path.cwd()
    names = _candidate_names()
    search_dirs = [
        repo_root / "build",
        repo_root / "build" / "Debug",
        repo_root / "build" / "Release",
        repo_root / "build" / "RelWithDebInfo",
        repo_root,
        cwd / "build",
        cwd / "build" / "Debug",
        cwd / "build" / "Release",
        cwd,
    ]
    for directory in search_dirs:
        for name in names:
            paths.append(directory / name)
    return paths


def _load_cdll() -> ctypes.CDLL:
    attempted = []
    for candidate in _candidate_paths():
        attempted.append(str(candidate))
        if candidate.exists():
            return ctypes.CDLL(str(candidate))

    names = ", ".join(_candidate_names())
    attempted_text = "\n  ".join(attempted)
    raise RuntimeError(
        "Impossible de charger la librairie C MiniTorch.\n"
        f"Noms attendus: {names}\n"
        "Compile d'abord la cible partagée, par exemple:\n"
        "  cmake -S . -B build\n"
        "  cmake --build build\n"
        "Ou définis MINITORCH_LIB vers le fichier .so, .dll ou .dylib.\n"
        f"Chemins testés:\n  {attempted_text}"
    )


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_cdll()
        _configure(_LIB)
    return _LIB


def library_candidates() -> list[Path]:
    return _candidate_paths()


def find_library_path() -> Path | None:
    for candidate in _candidate_paths():
        if candidate.exists():
            return candidate
    return None


def _configure(c_lib: ctypes.CDLL) -> None:
    dataset_p = ctypes.POINTER(MtDataset)
    dataset_pp = ctypes.POINTER(dataset_p)

    c_lib.ag_tape_create.argtypes = []
    c_lib.ag_tape_create.restype = ctypes.c_void_p
    c_lib.ag_tape_free.argtypes = [ctypes.c_void_p]
    c_lib.ag_tape_free.restype = None
    c_lib.ag_checkpoint.argtypes = [ctypes.c_void_p]
    c_lib.ag_checkpoint.restype = ctypes.c_int

    c_lib.mt_dataset_analyze_csv.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(MtCsvInfo),
    ]
    c_lib.mt_dataset_analyze_csv.restype = ctypes.c_int
    c_lib.mt_dataset_load_csv_auto.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(MtCsvInfo),
    ]
    c_lib.mt_dataset_load_csv_auto.restype = dataset_p
    c_lib.mt_dataset_split.argtypes = [
        dataset_p,
        ctypes.c_float,
        ctypes.c_float,
        dataset_pp,
        dataset_pp,
        dataset_pp,
    ]
    c_lib.mt_dataset_split.restype = ctypes.c_int
    c_lib.mt_dataset_free.argtypes = [dataset_p]
    c_lib.mt_dataset_free.restype = None
    c_lib.mt_dataset_feature.argtypes = [dataset_p, ctypes.c_int, ctypes.c_int]
    c_lib.mt_dataset_feature.restype = ctypes.c_float
    c_lib.mt_dataset_label.argtypes = [dataset_p, ctypes.c_int]
    c_lib.mt_dataset_label.restype = ctypes.c_float

    c_lib.mt_model_create_linear_binary.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mt_model_create_linear_binary.restype = ctypes.c_void_p
    c_lib.mt_model_create_mlp_binary.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mt_model_create_mlp_binary.restype = ctypes.c_void_p
    c_lib.mt_model_create_linear_multiclass.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mt_model_create_linear_multiclass.restype = ctypes.c_void_p
    c_lib.mt_model_create_mlp_multiclass.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    c_lib.mt_model_create_mlp_multiclass.restype = ctypes.c_void_p
    c_lib.mt_model_free.argtypes = [ctypes.c_void_p]
    c_lib.mt_model_free.restype = None
    c_lib.mt_model_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    c_lib.mt_model_predict.restype = ctypes.c_int
    c_lib.mt_model_eval_binary.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        dataset_p,
        ctypes.c_float,
        ctypes.POINTER(MtEvalResult),
    ]
    c_lib.mt_model_eval_binary.restype = ctypes.c_int
    c_lib.mt_model_eval_multiclass.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        dataset_p,
        ctypes.POINTER(MtEvalResult),
        ctypes.POINTER(ctypes.c_int),
    ]
    c_lib.mt_model_eval_multiclass.restype = ctypes.c_int
    c_lib.mt_model_save.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    c_lib.mt_model_save.restype = ctypes.c_int
    c_lib.mt_model_load.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    c_lib.mt_model_load.restype = ctypes.c_int

    c_lib.mt_sgd_create.argtypes = [ctypes.c_float]
    c_lib.mt_sgd_create.restype = ctypes.c_void_p
    c_lib.mt_adam_create.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    c_lib.mt_adam_create.restype = ctypes.c_void_p
    c_lib.mt_optimizer_free.argtypes = [ctypes.c_void_p]
    c_lib.mt_optimizer_free.restype = None
    c_lib.mt_optimizer_add_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    c_lib.mt_optimizer_add_model.restype = ctypes.c_int

    c_lib.mt_loss_create.argtypes = [ctypes.c_int]
    c_lib.mt_loss_create.restype = MtLoss
    c_lib.mt_train_config_default.argtypes = []
    c_lib.mt_train_config_default.restype = MtTrainConfig
    c_lib.mt_trainer_train_binary.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(MtLoss),
        ctypes.c_int,
        dataset_p,
        ctypes.POINTER(MtTrainConfig),
        ctypes.POINTER(MtTrainHistory),
    ]
    c_lib.mt_trainer_train_binary.restype = ctypes.c_int
    c_lib.mt_trainer_train_multiclass.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(MtLoss),
        ctypes.c_int,
        dataset_p,
        ctypes.POINTER(MtTrainConfig),
        ctypes.POINTER(MtTrainHistory),
    ]
    c_lib.mt_trainer_train_multiclass.restype = ctypes.c_int


def as_bytes(path: str | os.PathLike[str]) -> bytes:
    return os.fsencode(Path(path))
