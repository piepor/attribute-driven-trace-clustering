"""
This module contains classes for the clustering algorithms implemented
"""

import os
import pm4py
import pandas as pd
import numpy as np
import pickle as pkl
import kmedoids
import json
from pm4py.objects.log.obj import EventLog
from datetime import datetime
from abc import ABC, abstractmethod
from kmodes.kprototypes import KPrototypes
from tqdm import tqdm
import plotly.graph_objects as go
from loaders import MultilevelTraceAttributesVariants
from configs import Config
from utils.clustering import scale_data, create_map_activities, custom_distance
from utils.clustering import from_act_to_num, create_distance_matrix
from utils.filtering import filter_log_by_case_id, filter_dataframe_on_cluster
from utils.filtering import filter_df_by_time, filter_log_by_time
from utils.generals import attributes_names_bpic2019, attributes_names_bpic2017, attributes_names_bpic2012 
from utils.generals import attributes_names_traffic_fine
from utils.logs import LogUtilities


class Clustering(ABC):
    @abstractmethod
    def clusterize(self):
        pass


class MultiLevelAttributesThenVariants(Clustering):
    def __init__(self, log_path_sequences: str, log_path_attributes: str, save_dir: str, model_dir: str="",
                 attributes_dates=False, dates_col=None, filter_lifecycle=True) -> None:
        if not 'multilevel-attributes-sequences' in os.listdir(save_dir):
            os.mkdir(os.path.join(save_dir, 'multilevel-attributes-sequences'))
        if not model_dir:
            model_dir = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            os.mkdir(os.path.join(save_dir, 'multilevel-attributes-sequences', model_dir))
        self.save_dir = os.path.join(save_dir, 'multilevel-attributes-sequences', model_dir)
        self.log_sequences = log_path_sequences
        self.log_attributes = log_path_attributes
        if 'BPI_Challenge_2019' in log_path_sequences:
            num_clmns, cat_clmns = attributes_names_bpic2019()
        elif 'road_traffic_fine_management_process' in log_path_sequences: 
            num_clmns, cat_clmns = attributes_names_traffic_fine()
        elif 'BPI_Challenge_2017' in log_path_sequences:
            num_clmns, cat_clmns = attributes_names_bpic2017()
        elif 'BPI_Challenge_2012' in log_path_sequences:
            num_clmns, cat_clmns = attributes_names_bpic2012()
        else:
            raise NotImplementedError('Log not implemented')
        if attributes_dates:
            self.first_level = AttributesClustering(log_path_attributes, num_clmns, cat_clmns, self.save_dir, time_stamp=False, parse_dates=True, dates_col=dates_col)
        else:
            self.first_level = AttributesClustering(log_path_attributes, num_clmns, cat_clmns, self.save_dir, time_stamp=False, parse_dates=False)
        self.second_level = SequencesClustering(log_path_sequences, self.save_dir, time_stamp=False, filter_lifecycle=filter_lifecycle)
        self.n_clusters_first_level = 0
        self.n_clusters_second_level = []
        config = {}
        if "config.json" in os.listdir(os.path.join(self.save_dir)):
            with open(os.path.join(self.save_dir, "config.json"), 'r') as file:
                config = json.load(file)
        config["log-name"] = log_path_sequences.split("/")[-1]
        with open(os.path.join(self.save_dir, "config.json"), 'w') as file:
            json.dump(config, file)

    def clusterize(self):
        if not self.n_clusters_first_level or not self.n_clusters_second_level:
            raise ValueError('You must set the number of clusters for both levels.')
        self.clusterize_first_level()
        self.clusterize_second_level()

    def elbow_curve_first_level(self, n_clusters: int):
        self.first_level.elbow_curve(n_clusters)

    def elbow_curve_second_level(self, n_first_level: int, first_level_cluster_path: str, n_clusters: int):
        clusters = pd.read_csv(first_level_cluster_path)
        for i in range(n_first_level):
            cluster_trace_idx = filter_dataframe_on_cluster(clusters, i)['CASE_ID'].tolist()
            self.second_level.case_ids = cluster_trace_idx
            self.second_level.elbow_curve(n_clusters, name=f'cluster{i}')

    def filter_first_level(self, date_col: str, start_date: str, end_date: str):
        self.first_level.filter_data(date_col, start_date, end_date)

    def filter_second_level(self, date_col: str, start_date: str, end_date: str):
        self.second_level.filter_data(date_col, start_date, end_date)

    def set_n_clusters_first_level(self, n_clusters: int):
        self.n_clusters_first_level = n_clusters

    def set_n_clusters_second_level(self, n_clusters: list[int]):
        self.n_clusters_second_level = n_clusters
    
    def set_first_level_cluster_path(self, first_level_cluster_path: str):
        self.first_level_cluster_path = first_level_cluster_path

    def clusterize_first_level(self):
        self.first_level.set_clusters_number(self.n_clusters_first_level)
        self.first_level.prepare_data()
        self.first_level.clusterize()
        self.first_level.save_clusters('attributes')

    def clusterize_second_level(self):
        clusters = pd.read_csv(self.first_level_cluster_path)
        for i, n_cluster in enumerate(self.n_clusters_second_level):
            cluster_trace_idx = filter_dataframe_on_cluster(clusters, i)['CASE_ID'].tolist()
            self.second_level.case_ids = cluster_trace_idx
            self.second_level.set_clusters_number(n_cluster)
            self.second_level.prepare_data()
            self.second_level.clusterize()
            self.second_level.save_clusters(f'cluster{i}')


class SequencesClustering(Clustering):
    def __init__(self, log_path: str, save_dir: str, time_stamp: bool=True, filter_lifecycle: bool=True):
        self.time_stamp = None
        log = pm4py.read_xes(log_path)
        # select only "lifecycle:transition" = 'start' 
        if filter_lifecycle:
            self.log = pm4py.filter_event_attribute_values(
                    log, 'lifecycle:transition', ['start'], level='event', retain=True)
        else:
            self.log = log
        activities = pm4py.get_event_attribute_values(self.log, 'concept:name')
        self.map_activities = create_map_activities(activities)
        self.case_ids = []
        if not 'sequences' in os.listdir(save_dir):
            os.mkdir(os.path.join(save_dir, 'sequences'))
        if time_stamp:
            timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            self.save_dir = os.path.join(save_dir, 'sequences', timestamp)
            os.mkdir(self.save_dir)
            self.time_stamp = timestamp
        else:
            self.save_dir = os.path.join(save_dir, 'sequences')

    def set_clusters_number(self, n_clusters: int):
        self.n_clusters = n_clusters

    def prepare_data(self):
        log = self.log
        if self.case_ids:
            log = filter_log_by_case_id(log, self.case_ids)
        variants_dict = pm4py.get_variants_as_tuples(log)
        self.variants = list(variants_dict.keys())
        tokenized_variants = from_act_to_num(self.variants, self.map_activities)
        self.distance_matrix = create_distance_matrix(tokenized_variants, custom_distance)

    def filter_data(self, date_col: str, start_date_str: str, end_date_str: str):
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
        self.log = filter_log_by_time(self.log, date_col, start_date, end_date)

    def clusterize(self):
        print('\nClustering sequences: we are working for you...')
        if not self.n_clusters:
            raise ValueError('Unkwnown number of clusters.')
        if not len(self.distance_matrix):
            raise ValueError('Clustering data empty. Prepare data before fitting.')
        self.km = kmedoids.fasterpam(self.distance_matrix, self.n_clusters)

    def save_clusters(self, cluster_name: str, cluster_dir: str=""):
        # cluster_dir = os.path.join(self.save_dir, cluster_name)
        config = {}
        if "config.json" in os.listdir(os.path.join(self.save_dir, "..")):
            with open(os.path.join(self.save_dir, "..", "config.json"), 'r') as file:
                config = json.load(file)
        if not "sequences-clusters" in config:
            config['sequences-clusters'] = []
        config['sequences-clusters'].append(self.n_clusters)
        with open(os.path.join(self.save_dir, "..", "config.json"), 'w') as file:
            json.dump(config, file)
        save_dir = self.save_dir
        if cluster_dir:
            save_dir = os.path.join(save_dir, cluster_dir)
        if not 'clusters' in os.listdir(save_dir):
            os.mkdir(os.path.join(save_dir, 'clusters'))
        save_dir = os.path.join(save_dir, 'clusters')
        if cluster_name:
            cluster_name = f'{cluster_name}-'
        with open(os.path.join(save_dir, f'{cluster_name}medoids.pkl'), 'wb') as file:
            pkl.dump(self.km, file)
        with open(os.path.join(save_dir, f'{cluster_name}variants.pkl'), 'wb') as file:
            pkl.dump(self.variants, file)

    def elbow_curve(self, n_clusters: int, view: bool=False, name: str=""):
        losses = []
        for n_clstr in tqdm(range(1, n_clusters)):
            self.set_clusters_number(n_clstr)
            self.prepare_data()
            self.clusterize()
            losses.append(self.km.loss)
        fig = go.Figure(go.Scatter(x=list(range(1,n_clusters)), y=losses, mode='lines+markers'))
        fig.update_layout(title='Trace attributes clustering elbow curve')
        figures_dir = os.path.join(self.save_dir, 'figures')
        if not 'figures' in os.listdir(self.save_dir):
            os.mkdir(figures_dir)
        if name:
            name = f'{name}-'
        fig.write_html(os.path.join(figures_dir, f'{name}elbow_curve.html'))
        if view:
            fig.show()


class AttributesClustering(Clustering):
    def __init__(self, log_path: str, num_clmns: list[str], cat_clmns: list[str], save_dir: str, time_stamp: bool=True, parse_dates=False, dates_col=None):
        self.time_stamp = None
        if not 'attributes' in os.listdir(save_dir):
            os.mkdir(os.path.join(save_dir, 'attributes'))
        if time_stamp:
            timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            self.save_dir = os.path.join(save_dir, 'attributes', timestamp)
            os.mkdir(self.save_dir)
            self.time_stamp = timestamp
        else:
            self.save_dir = os.path.join(save_dir, 'attributes')
        if parse_dates and dates_col:
            data_frame = pd.read_csv(
                    log_path,
                    parse_dates=[dates_col],
                    date_parser=lambda col: pd.to_datetime(col, utc=True))
        else:
            data_frame = pd.read_csv(log_path)
        self.data = data_frame
        self.categorical_columns = cat_clmns
        self.numerical_columns = num_clmns
        self.data_clustering = []

    def filter_data(self, date_col: str, start_date_str: str, end_date_str: str):
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
        self.data = filter_df_by_time(self.data, date_col, start_date, end_date)

    def set_clusters_number(self, n_clusters: int):
        self.n_clusters = n_clusters

    def clusterize(self):
        print('Clustering attributes: we are working for you...')
        if not self.n_clusters:
            raise ValueError('Unkwnown number of clusters.')
        if len(self.data_clustering) == 0:
            raise ValueError('Clustering data empty. Prepare data before fitting.')
        # fit
        try:
            #breakpoint()
            print("Huang initialization")
            k_proto = KPrototypes(n_clusters=self.n_clusters, init='Huang', n_init=5, random_state=42)
            self.clusters = k_proto.fit_predict(
                    self.data_clustering, categorical=self.categorical_idxs)
        except:
            print("Random initialization")
            k_proto = KPrototypes(n_clusters=self.n_clusters, init='random', n_init=5, random_state=42)
            self.clusters = k_proto.fit_predict(
                    self.data_clustering, categorical=self.categorical_idxs)
        self.cost = k_proto.cost_

    def save_clusters(self, cluster_name: str, cluster_dir: str=""):
        config = {}
        if "config.json" in os.listdir(os.path.join(self.save_dir, "..")):
            with open(os.path.join(self.save_dir, "..", "config.json"), 'r') as file:
                config = json.load(file)
        if not "attributes-clusters" in config:
            config['attributes-clusters'] = []
        config['attributes-clusters'].append(self.n_clusters)
        with open(os.path.join(self.save_dir, "..", "config.json"), 'w') as file:
            json.dump(config, file)
        save_dir = self.save_dir
        if cluster_dir:
            save_dir = os.path.join(save_dir, cluster_dir)
        if not 'clusters' in os.listdir(save_dir):
            os.mkdir(os.path.join(save_dir, 'clusters'))
        save_dir = os.path.join(save_dir, 'clusters')
        data_clusters = self.data.copy()
        data_clusters['cluster'] = self.clusters
        # save cluster data
        data_clusters['cluster'].to_csv(os.path.join(
            save_dir, f'{cluster_name}.csv'))

    def prepare_data(self):
        data_frame_only_case = self.data.drop_duplicates('CASE_ID').set_index('CASE_ID')
        self.data = data_frame_only_case
        data_clustering = scale_data(self.data, self.numerical_columns, self.categorical_columns)
        self.data_clustering = data_clustering
        self.categorical_idxs = list(
                np.arange(len(self.categorical_columns)) + len(self.numerical_columns))

    def elbow_curve(self, n_clusters: int, view: bool=False, name: str=""):
        # if len(self.data_clustering):
        #     raise ValueError('Clustering data empty. Prepare data before fitting.')
        cost_values = []
        self.prepare_data()
        for n_clstr in tqdm(range(2, n_clusters+1)):
            self.set_clusters_number(n_clstr)
            self.clusterize()
            cost_values.append(self.cost)
        fig = go.Figure(go.Scatter(x=list(range(1,n_clusters)), y=cost_values, mode='lines+markers'))
        fig.update_layout(title='Trace attributes clustering elbow curve')
        figures_dir = os.path.join(self.save_dir, 'figures')
        if not 'figures' in os.listdir(self.save_dir):
            os.mkdir(figures_dir)
        if name:
            name = f'{name}-'
        fig.write_html(os.path.join(figures_dir, f'{name}elbow_curve.html'))
        if view:
            fig.show()
