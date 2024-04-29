""" 
This module contains all the functions linked to the different clustering algorithm implemented
"""

from abc import ABC, abstractmethod
from typing import Callable
import pandas as pd
from fastDamerauLevenshtein import damerauLevenshtein as dam_lev_dist
from sklearn.preprocessing import StandardScaler
import numpy as np
from pm4py.objects.log.obj import Trace
from utils.generals import extract_variant
from utils.distance_computing import continuous_distance, categorical_distance, sequence_distance 


class DistanceFunction(ABC):

    @abstractmethod
    def compute_distance(self, trace_x: Trace, trace_y: Trace):
        pass


class MultiviewDistanceFunction(DistanceFunction):
    def __init__(self, extract_cont_trace_attrs: Callable[[Trace, dict], list],
                 extract_cat_trace_attrs: Callable[[Trace], list], maxs: dict, maxs_cont: dict):
        self.extract_cont = extract_cont_trace_attrs
        self.extract_cat = extract_cat_trace_attrs
        self.maxs = maxs
        self.maxs_cont = maxs_cont

    def compute_distance(self, trace_x: Trace, trace_y: Trace) -> float:
        cont_attrs_x = self.extract_cont(trace_x, self.maxs_cont)
        cont_attrs_y = self.extract_cont(trace_y, self.maxs_cont)
        cont_distance = continuous_distance(cont_attrs_x, cont_attrs_y)
        cat_attrs_x = self.extract_cat(trace_x)
        cat_attrs_y = self.extract_cat(trace_y)
        cat_distance = categorical_distance(cat_attrs_x, cat_attrs_y)
        variant_x = extract_variant(trace_x)
        variant_y = extract_variant(trace_y)
        seq_distance = sequence_distance(variant_x, variant_y)
        return cont_distance/self.maxs['cont'] + cat_distance/self.maxs['cat'] + seq_distance/self.maxs['seq']

def scale_data(data_frame: pd.DataFrame, numerical_clmns: list[str], categorical_clmns: list[str]) -> pd.DataFrame:
    """ Scale only numerical data in the dataframe """
    scaler = StandardScaler().fit(data_frame[numerical_clmns].to_numpy())
    scaled_data = scaler.transform(data_frame[numerical_clmns].to_numpy())
    scaled_data_frame = pd.DataFrame()
    for i, num_col in enumerate(numerical_clmns):
        scaled_data_frame[num_col] = scaled_data[:, i]
    scaled_data_frame = scaled_data_frame.set_index(data_frame.index)
    scaled_data_frame[categorical_clmns] = data_frame[categorical_clmns]
    return scaled_data_frame

def custom_distance(x, y):
    return dam_lev_dist(x, y, swapWeight=0, replaceWeight=2, similarity=False)

def create_map_activities(activities):
    return {act: i for i,act in enumerate(activities)}

def from_act_to_num(variants, map_acts):
    converted_variants = []
    for variant in variants:
        converted_variants.append(list(
            map(lambda x: map_acts[x], variant)
            ))
    return converted_variants

def create_distance_matrix(input_arr, distance):
    matrix = np.zeros((len(input_arr), len(input_arr)))
    for i, el_x in enumerate(input_arr):
        for j, el_y in enumerate(input_arr):
            matrix[i, j] = distance(el_x, el_y)
    return matrix

def filter_variants(cluster_variants_dict, cluster):
    return [variant for variant in cluster_variants_dict if cluster_variants_dict[variant] == cluster]
