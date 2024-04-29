"""
This module contains classes responsible for loading data
"""

import os
import json
from abc import ABC, abstractmethod
import pandas as pd
from configs import Config
from utils.multilevel_clustering import get_clusters_trace_attrs_variants
from utils.loading import load_xes
from utils.logs import LogUtilities

class ClusterLoader(ABC):
    @abstractmethod
    def get_clusters(self) -> dict:
        """ Returns a dict containing clusters """

class MultilevelTraceAttributesVariants(ClusterLoader):
    def __init__(self, configs: Config, log_utils: LogUtilities) -> None:
        super().__init__()
        dir_attributes = os.path.join(configs.model_dir, 'attributes', 'clusters')
        dir_cluster_seq = os.path.join(configs.model_dir, 'sequences', 'clusters')
        log_path = configs.log_path
        self.clusters_attrs = pd.read_csv(os.path.join(dir_attributes, 'attributes.csv'))
        log = load_xes(log_path, only_lifecycle_start=log_utils.only_lifecycle_start)
        self.clusters = get_clusters_trace_attrs_variants(self.clusters_attrs, log, dir_cluster_seq)
        with open(os.path.join(configs.model_dir, 'config.json'), 'rb') as file:
            self.configs = json.load(file)

    def get_clusters(self):
        return self.clusters
