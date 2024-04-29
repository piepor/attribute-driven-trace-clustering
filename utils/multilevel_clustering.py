"""
This module contains all the functions related to the multilevel clustering 
"""

import os
import pandas as pd
import pickle as pkl
import numpy as np
from pm4py.objects.log.obj import EventLog
from utils.filtering import filter_dataframe_on_cluster, filter_log_on_variants
from utils.filtering import filter_log_on_cluster, filter_variants

def get_clusters_trace_attrs_variants(clusters_attributes: pd.DataFrame, log: EventLog|pd.DataFrame, dir_attributes: str) -> dict:
    """ retrieves the clusters extracted considering trace attributes in the first place and then clustering variants """
    clusters = {}
    for cluster in clusters_attributes['cluster'].unique():
        cluster_trace_idx = filter_dataframe_on_cluster(clusters_attributes, cluster)['CASE_ID'].tolist()
        log_cluster = filter_log_on_cluster(log, cluster_trace_idx)
        with open(os.path.join(dir_attributes, f'cluster{cluster}-medoids.pkl'), 'rb') as file:
            km = pkl.load(file)
        with open(os.path.join(dir_attributes, f'cluster{cluster}-variants.pkl'), 'rb') as file:
            variants = pkl.load(file)
        number_of_medoids = np.unique(km.labels).shape[0]
        clusterized_variants_dict = {variant: cluster_id for variant, cluster_id in zip(variants, list(km.labels))}

        for i in range(number_of_medoids):
            cluster_variants = filter_variants(clusterized_variants_dict, i)
            cluster_variants_log = filter_log_on_variants(log_cluster, cluster_variants)
            clusters[f'cluster{cluster}-medoid{i}'] = cluster_variants_log 
            # clusters.append((cluster_variants_log, f'cluster{cluster}-medoid{i}'))
    return clusters
