"""
火球 K/C 双单任务 GP、数据与配置（见 document/fireball_gp_mogp_module_design.md）。

``gpytorch`` / ``torch`` 仅在子模块 ``gp_model.train_infer`` 中必需；
若仅需数据解析，可只 ``from gp_model import data_input``。
"""

from gp_model.data_input import Dataset, SampleMeta, load_training_dir
from gp_model.config import (
    DEFAULT_TRAINED_FILENAME,
    default_hyperparams,
    load_hyperparams_json,
    merge_hyperparams,
    save_trained_artifact,
    load_trained_artifact,
)

__all__ = [
    "Dataset",
    "SampleMeta",
    "load_training_dir",
    "DEFAULT_TRAINED_FILENAME",
    "default_hyperparams",
    "load_hyperparams_json",
    "merge_hyperparams",
    "save_trained_artifact",
    "load_trained_artifact",
]
