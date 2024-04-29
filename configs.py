
"""
This module contains the configurations needed to allow for every modality
"""

from dataclasses import dataclass, field
from enum import Enum, auto

class Mode(Enum):
    CLUSTER = auto()
    VISUALIZE = auto()

@dataclass
class Config:
    log_path: str
    model_dir: str=""

@dataclass
class ConfigVisualize(Config):
    view: bool=True
    variant_coverage: float=1.0
    compute_models_performance: bool=False

@dataclass
class ConfigCluster(Config):
    elbow_curve_first_level: bool=False
    num_points_elbow_curve_first_level: int=0
    elbow_curve_second_level: bool=False
    num_points_elbow_curve_second_level: int=0
    cluster_first_level: bool=False
    cluster_second_level: bool=False
    n_clusters_first_level: int=0
    n_clusters_second_level: list=field(default_factory=lambda: [0])

@dataclass
class ConfigMain:
    dataset: str
    mode: Mode
    elbow_curve_first_level: bool=False
    elbow_curve_second_level: bool=False
    cluster_first_level: bool=False
    cluster_second_level: bool=False
