import os
import numpy as np
import pandas as pd
import pickle
from abc import ABC, abstractmethod
from tqdm import tqdm
from pm4py.objects.log.obj import EventLog, Trace
from typing import Tuple, List
from loaders import ClusterLoader
from utils.distance_computing import compute_distances_on_gpu, compute_sequence_distances, get_centroid_id, get_total_distance
from utils.distance_computing import get_distances_matrix, continuous_distance, categorical_distance, sequence_distance
from utils.generals import extract_case_ids, extract_max_continuous_values_from_clusters, extract_variant, get_map_cat_attrs_from_clusters
from utils.logs import LogUtilities


class DistanceFunction(ABC):

    @abstractmethod
    def compute_distance_between_two_traces(self, trace_x: Trace, trace_y: Trace):
        pass

    @abstractmethod
    def compute_distances_clusters(self):
        pass

    @abstractmethod
    def get_similarities(self, save_dir: str, compute: bool) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_continuous_attrs(self) -> List[str]:
        pass


class MultiviewDistanceFunction(DistanceFunction):
    def __init__(self, loader: ClusterLoader, log_utils: LogUtilities):
        self.clusters = loader.get_clusters()
        self.extract_attrs = log_utils.trace_attributes
        self.map_cat_attrs = get_map_cat_attrs_from_clusters(
                loader.get_clusters(), log_utils.map_categorical_attributes)
        self.maxs_cont = extract_max_continuous_values_from_clusters(
                loader.get_clusters(), log_utils.max_continuous_values)
        self.continuous_attrs = list(self.maxs_cont.keys())
        self.centroids = {}
        self.max_distances = {'sequence': 0, 'categorical': 0, 'continuous': 0}
        self.max_dist_intraclusters = {}
        self.clusters_case_ids = {}
        self.dist_intraclusters_centroid = {}
        self.overall_dist_intraclusters = {}
        self.avg_dist_intracluster = {}
    
    def get_continuous_attrs(self):
        return self.continuous_attrs

    def get_similarities(self, save_dir: str, compute: bool=False):
        # distances_df = {'cluster_relation': [], 'cluster_x': [], 'cluster_y': [], 'distance_considered': [], 'value': []}
        if not "similarities.pickle" in os.listdir(save_dir) or compute:
            distances = self.compute_distances_clusters()
            with open(os.path.join(save_dir, 'similarities.pickle'), 'wb') as file:
                pickle.dump(distances, file)
        else:
            with open(os.path.join(save_dir, 'similarities.pickle'), 'rb') as file:
                distances = pickle.load(file)
        # for cluster in distances['intracluster']:
        #     for distance_considered in distances['intracluster'][cluster]:
        #         distances_df['cluster_relation'].append('intracluster')
        #         distances_df['cluster_x'].append(cluster)
        #         distances_df['cluster_y'].append(cluster)
        #         distances_df['distance_considered'].append(distance_considered)
        #         distances_df['value'].append(distances['intracluster'][cluster][distance_considered])
        # for cluster_x in distances['intercluster']:
        #     for cluster_y in distances['intercluster'][cluster_x]:
        #         for distance_considered in distances['intercluster'][cluster_x][cluster_y]:
        #             distances_df['cluster_relation'].append('intercluster')
        #             distances_df['cluster_x'].append(cluster_x)
        #             distances_df['cluster_y'].append(cluster_y)
        #             distances_df['distance_considered'].append(distance_considered)
        #             distances_df['value'].append(distances['intercluster'][cluster_x][cluster_y][distance_considered])
        # return pd.DataFrame.from_dict(distances_df) 
        return distances

    def compute_distance_between_two_traces(self, trace_x: Trace, trace_y: Trace) -> int:
        trace_x_variant = extract_variant(trace_x)
        trace_x_attrs, num_cont = self.extract_attrs(trace_x, self.maxs_cont, self.map_cat_attrs)
        trace_y_variant = extract_variant(trace_y)
        trace_y_attrs, _ = self.extract_attrs(trace_y, self.maxs_cont, self.map_cat_attrs)
        cont_dist = continuous_distance(
                trace_x_attrs[:num_cont], trace_y_attrs[:num_cont])
        cat_dist = categorical_distance(
                trace_x_attrs[num_cont:], trace_y_attrs[num_cont:])
        seq_dist = sequence_distance(trace_x_variant, trace_y_variant)
        if not self.max_distances['continuous'] == 0:
            cont_dist = cont_dist/self.max_distances['continuous'] 
        if not self.max_distances['categorical'] == 0:
            cat_dist = cat_dist/self.max_distances['categorical'] 
        if not self.max_distances['sequence'] == 0:
            seq_dist = seq_dist/self.max_distances['sequence'] 
        distance = cont_dist + cat_dist + seq_dist
        return distance 

    def compute_distances_clusters(self) -> pd.DataFrame:
        overall_distances = {'cluster_x': [], 'cluster_y': [], 'distance_considered': [], 'value': []}
        distances_matrix = self.compute_distances_matrix()
        self.get_max_distances(distances_matrix)
        self.get_centroid(distances_matrix['intracluster'])
        overall_distances_intracluster = self.compute_intraclusters_distances(distances_matrix['intracluster'])
        for cluster in overall_distances_intracluster:
            for dist_considered in overall_distances_intracluster[cluster]:
                overall_distances['cluster_x'].append(cluster)
                overall_distances['cluster_y'].append(cluster)
                overall_distances['distance_considered'].append(dist_considered)
                overall_distances['value'].append(overall_distances_intracluster[cluster][dist_considered])
        overall_distances_intercluster = self.compute_interclusters_distances(distances_matrix['intercluster'])
        for cluster_x in overall_distances_intercluster:
            for cluster_y in overall_distances_intercluster[cluster_x]:
                for dist_considered in overall_distances_intercluster[cluster_x][cluster_y]:
                    overall_distances['cluster_x'].append(cluster_x)
                    overall_distances['cluster_y'].append(cluster_y)
                    overall_distances['distance_considered'].append(dist_considered)
                    overall_distances['value'].append(
                            overall_distances_intercluster[cluster_x][cluster_y][dist_considered]) # overall_distances['intracluster'] = overall_distances_intracluster
        # overall_distances['intercluster'] = overall_distances_intercluster
        return pd.DataFrame.from_dict(overall_distances)

    def compute_intraclusters_distances(self, distance_matrix: dict) -> dict:
        print("Compute intracluster distances")
        self.compute_intracluster_avg_distances_from_centroids()
        overall_dist_intraclusters = {}
        for cluster in distance_matrix:
            overall_dist_intraclusters[cluster] = {}
            overall_dist_intraclusters[cluster]['average_intracluster'] = np.round(self.avg_dist_intracluster[cluster], 4)
            overall_dist_intraclusters[cluster]['max_intracluster'] = np.round(self.max_dist_intraclusters[cluster], 4)
            overall_dist_intraclusters[cluster]['average_from_centroid_intracluster'] = np.round(self.dist_intraclusters_centroid[cluster], 4)
        return overall_dist_intraclusters
     
    def compute_interclusters_distances(self, distance_matrix_interclusters: dict) -> dict:
        print("Compute intercluster distances")
        overall_dist_interclusters = {}
        for cluster_x in distance_matrix_interclusters:
            overall_dist_interclusters[cluster_x] = {}
            for cluster_y in distance_matrix_interclusters[cluster_x]:
                overall_dist_interclusters[cluster_x][cluster_y] = {}
                distances_matrix = get_distances_matrix(
                        distance_matrix_interclusters[cluster_x][cluster_y]['categorical'],
                        distance_matrix_interclusters[cluster_x][cluster_y]['continuous'], 
                        distance_matrix_interclusters[cluster_x][cluster_y]['sequence'])
                overall_distances = get_total_distance(distances_matrix, self.max_distances) 
                centroid_x = self.centroids[cluster_x]
                centroid_y = self.centroids[cluster_y]
                distance_from_centroid_y = []
                for trace in self.clusters[cluster_x]:
                    distance_from_centroid_y.append(
                            self.compute_distance_between_two_traces(trace, centroid_y))
                overall_dist_interclusters[cluster_x][cluster_y]['average_intercluster'] = np.round(np.mean(overall_distances), 4)
                overall_dist_interclusters[cluster_x][cluster_y]['average_from_centroid_intercluster'] = np.round(
                        np.mean(distance_from_centroid_y), 4)
                overall_dist_interclusters[cluster_x][cluster_y]['between_centroids_intercluster'] = np.round(
                        self.compute_distance_between_two_traces(centroid_x, centroid_y), 4)
        return overall_dist_interclusters

    def extract_attrs_matrix(self, cluster: EventLog) -> Tuple[np.ndarray, int]:
        matrix = []
        for trace in cluster:
            attrs, _ = self.extract_attrs(trace, self.maxs_cont, self.map_cat_attrs)
            matrix.append(attrs)
        return np.asarray(matrix), len(self.maxs_cont)

    def compute_intraclusters_distances_matrix(self) -> dict:
        print("Compute intracluster distances matrix")
        distances = {}
        for cluster in tqdm(self.clusters):
            self.clusters_case_ids[cluster] = extract_case_ids(self.clusters[cluster])
            set_x, continuous_num = self.extract_attrs_matrix(self.clusters[cluster])
            categorical_distances, continuous_distances = compute_distances_on_gpu(
                    set_x, set_x, continuous_num, intercluster=False)
            sequence_distances = compute_sequence_distances(
                    self.clusters[cluster], self.clusters[cluster], intercluster=False)
            distances[cluster] = {
                    'categorical': categorical_distances, 
                    'continuous': continuous_distances,
                    'sequence': sequence_distances}
        return distances

    def get_max_distances(self, distance_matrix: dict):
        for cluster in distance_matrix['intracluster']:
            for dist_type in distance_matrix['intracluster'][cluster]:
                for dist in distance_matrix['intracluster'][cluster][dist_type]:
                    if dist[1] > self.max_distances[dist_type]:
                        # the max distance is store at position 1 in the distances array
                        self.max_distances[dist_type] = dist[1]
        for cluster_x in distance_matrix['intercluster']:
            for cluster_y in distance_matrix['intercluster'][cluster_x]:
                for dist_type in distance_matrix['intercluster'][cluster_x][cluster_y]:
                    for dist in distance_matrix['intercluster'][cluster_x][cluster_y][dist_type]:
                        if dist[1] > self.max_distances[dist_type]:
                            # the max distance is store at position 1 in the distances array
                            self.max_distances[dist_type] = dist[1]

    def get_centroid(self, distance_matrix_intraclusters: dict):
        centroids_ids = {}
        for cluster in distance_matrix_intraclusters:
            distances_matrix = get_distances_matrix(
                    distance_matrix_intraclusters[cluster]['categorical'],
                    distance_matrix_intraclusters[cluster]['continuous'], 
                    distance_matrix_intraclusters[cluster]['sequence'])
            overall_distances = get_total_distance(distances_matrix, self.max_distances) 
            self.avg_dist_intracluster[cluster] = np.mean(overall_distances)
            self.max_dist_intraclusters[cluster] = np.max(overall_distances)
            centroids_ids[cluster] = get_centroid_id(
                    overall_distances, self.clusters_case_ids[cluster])
        for cluster in self.clusters:
            for trace in self.clusters[cluster]:
                if trace.attributes['concept:name'] == centroids_ids[cluster]:
                    self.centroids[cluster] = trace

    def compute_intracluster_avg_distances_from_centroids(self):
        print("Computing average intracluster distance from centroid")
        for cluster in tqdm(self.clusters):
            centroid = self.centroids[cluster]
            total_distance = 0
            for trace in self.clusters[cluster]:
                if trace.attributes['concept:name'] != centroid.attributes['concept:name']:
                    distance = self.compute_distance_between_two_traces(trace, centroid)
                    total_distance += distance
            self.dist_intraclusters_centroid[cluster] = total_distance/(len(self.clusters[cluster])-1) 

    def compute_interclusters_distances_matrix(self) -> dict:
        print("Compute intercluster distances matrix")
        distances = {}
        for cluster_x in tqdm(self.clusters):
            distances[cluster_x] = {}
            for cluster_y in self.clusters:
                if not cluster_x == cluster_y:
                    self.clusters_case_ids[cluster_x] = extract_case_ids(self.clusters[cluster_x])
                    set_x, continuous_num = self.extract_attrs_matrix(self.clusters[cluster_x])
                    self.clusters_case_ids[cluster_y] = extract_case_ids(self.clusters[cluster_y])
                    set_y, continuous_num = self.extract_attrs_matrix(self.clusters[cluster_y])
                    categorical_distances, continuous_distances = compute_distances_on_gpu(
                            set_x, set_y, continuous_num)
                    sequence_distances = compute_sequence_distances(self.clusters[cluster_x], self.clusters[cluster_y])
                    distances[cluster_x][cluster_y] = {
                            'categorical': categorical_distances, 
                            'continuous': continuous_distances,
                            'sequence': sequence_distances}
        return distances

    def compute_distances_matrix(self) -> dict:
        distances_matrix = {'intracluster': {}, 'intercluster': {}}
        distances_matrix['intracluster'] = self.compute_intraclusters_distances_matrix()
        distances_matrix['intercluster'] = self.compute_interclusters_distances_matrix()
        return distances_matrix
