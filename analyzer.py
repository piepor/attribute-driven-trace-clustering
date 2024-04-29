from typing import Tuple, List
import os
import numpy as np
import pandas as pd
from loaders import ClusterLoader
from distances import DistanceFunction
from utils.logs import LogUtilities
from utils.counting import count_activities, count_start_end_activities
from utils.distribution import distribution_end_activities, distribution_start_activities, get_activities_distribution_divergence
from utils.distribution import get_activities_perc_clusters, get_activities_perc_subclusters, get_variants_distribution 
from utils.generals import get_sizes, get_percentage_shared_variants_subclusters, get_percentage_shared_variants_clusters
from utils.generals import compute_models_performance


class Analyzer:
    def __init__(self, loader: ClusterLoader, distance: DistanceFunction, log_utilities: LogUtilities, save_dir: str):
        self.loader = loader
        self.distance = distance
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.start_end_activities = count_start_end_activities(self.loader.get_clusters())
        self.performance_function = log_utilities.performance
        self.get_traces_attrs_cluster = log_utilities.trace_attributes_distribution
        self.log_name = log_utilities.log_name

    def get_continuous_attrs(self) -> List[str]:
        return self.distance.get_continuous_attrs()

    def get_similarities(self, compute: bool=False) -> pd.DataFrame:
        return self.distance.get_similarities(self.save_dir, compute) 

    def get_sizes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return get_sizes(self.loader.get_clusters()) 

    def get_activities_distribution(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        acts_count = count_activities(self.loader.get_clusters())
        perc_subcluster = get_activities_perc_subclusters(acts_count)
        perc_clusters = get_activities_perc_clusters(acts_count)
        return perc_subcluster, perc_clusters
    
    def get_activites_divergence(self) -> pd.DataFrame:
        cluster_div = get_activities_distribution_divergence(self.loader.get_clusters(), 'cluster') 
        cluster_div.insert(len(cluster_div.keys()), 'cluster_type', np.array(len(cluster_div)*['cluster']))
        subcluster_div = get_activities_distribution_divergence(self.loader.get_clusters(), 'subcluster') 
        subcluster_div.insert(len(subcluster_div.keys()), 'cluster_type', np.array(len(subcluster_div)*['subcluster']))
        return pd.concat([cluster_div, subcluster_div], ignore_index=True)

    def get_traces_attributes(self) -> pd.DataFrame:
        clusters = self.loader.get_clusters()
        trace_attrs_distr = pd.DataFrame()
        for cluster in clusters:
            cluster_trace_attrs = self.get_traces_attrs_cluster(clusters[cluster])
            cluster_trace_attrs['cluster'] = cluster.split('-')[0] 
            cluster_trace_attrs['subcluster'] = cluster 
            trace_attrs_distr = pd.concat([trace_attrs_distr, cluster_trace_attrs], ignore_index=True)
        return trace_attrs_distr
    
    def get_variants_distribution(self, coverage: float=1.0) -> dict:
        return get_variants_distribution(self.loader.get_clusters(), coverage)

    def get_variants_sharing_subclusters(self) -> pd.DataFrame:
        return get_percentage_shared_variants_subclusters(self.loader.get_clusters())

    def get_variants_sharing_clusters(self) -> pd.DataFrame:
        return get_percentage_shared_variants_clusters(self.loader.get_clusters())

    def distribution_start_activities_cluster(self) -> pd.DataFrame:
        start_end_activities = self.start_end_activities.copy()
        return distribution_start_activities(start_end_activities, 'cluster')

    def distribution_start_activities_subcluster(self) -> pd.DataFrame:
        start_end_activities = self.start_end_activities.copy()
        return distribution_start_activities(start_end_activities, 'subcluster')

    def distribution_end_activities_cluster(self) -> pd.DataFrame:
        start_end_activities = self.start_end_activities.copy()
        return distribution_end_activities(start_end_activities, 'cluster')

    def distribution_end_activities_subcluster(self) -> pd.DataFrame:
        start_end_activities = self.start_end_activities.copy()
        return distribution_end_activities(start_end_activities, 'subcluster')

    def get_performance(self) -> pd.DataFrame:
        return self.performance_function(self.loader.get_clusters())

    def get_models_performance(self, coverages: list=[1], version_filter: str='v2', compute: bool=False) -> pd.DataFrame:
        if compute:
            results = compute_models_performance(self.loader.get_clusters(), coverages, version_filter)
            results.to_csv(os.path.join(self.save_dir, 'models-performance.csv'), index=False)
        else:
            results = pd.read_csv(os.path.join(self.save_dir, 'models-performance.csv'))
        return results
