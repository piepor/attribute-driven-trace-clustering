"""
This module contains other functions not belonging to other categories
"""

from typing import Tuple, Callable
import pandas as pd
import random
import pm4py
import copy
from pm4py.objects.log.obj import EventLog, Trace
# from fastDamerauLevenshtein import damerauLevenshtein as dam_lev_dist
from utils.filtering import filter_log_on_variants_coverage_v2, filter_log_on_variants_coverage
from utils.compliance import is_compliant_fines, is_compliant_incomplete_bpic2019

def convert_similarities_file(file: dict) -> pd.DataFrame:
    sim = {'cluster_x': [], 'cluster_y': [], 'value': [], 'distance_considered': []}
    for cluster_x in file['intracluster']:
        for dist in file['intracluster'][cluster_x]:
            sim['cluster_x'].append(cluster_x)
            sim['cluster_y'].append(cluster_x)
            sim['value'].append(file['intracluster'][cluster_x][dist])
            sim['distance_considered'].append(f"{dist}_intracluster")
    for cluster_x in file['intercluster']:
        for cluster_y in file['intercluster'][cluster_x]:
            for dist in file['intercluster'][cluster_x][cluster_y]:
                sim['cluster_x'].append(cluster_x)
                sim['cluster_y'].append(cluster_y)
                sim['value'].append(file['intercluster'][cluster_x][cluster_y][dist])
                sim['distance_considered'].append(f"{dist}_intercluster")
    return pd.DataFrame.from_dict(sim)

def attributes_names_traffic_fine() -> Tuple[list, list]:
    return ['amount', 'points'], ['vehicleClass', 'article'] 

def attributes_names_bpic2019() -> Tuple[list, list]:
    numerical_columns = ['Cumulative net worth (EUR)']
    categorical_columns = ['Spend area text', 'Document Type', 'Sub spend area text', 'Purch. Doc. Category name',
                           'Item Type', 'Item Category', 'Spend classification text', 'GR-Based Inv. Verif.', 'Goods Receipt']
    return numerical_columns, categorical_columns 

def attributes_names_bpic2017() -> Tuple[list, list]:
    numerical_columns = ['RequestedAmount']
    categorical_columns = ['LoanGoal', 'ApplicationType']
    return numerical_columns, categorical_columns 

def attributes_names_bpic2012() -> Tuple[list, list]:
    numerical_columns = ['AMOUNT_REQ'] 
    categorical_columns = [ 'Categorical']
    return numerical_columns, categorical_columns 

def extract_case_ids(log: EventLog) -> list:
    return [trace.attributes['concept:name'] for trace in log]

def get_clusters_log(clusters: dict) -> dict:
    clusters_log = {}
    for cluster in clusters:
        cluster_name = cluster.split('-')[0] 
        if not cluster_name in clusters_log:
            clusters_log[cluster_name] = EventLog()
        for trace in clusters[cluster]:
            clusters_log[cluster_name].append(trace)
    return clusters_log

def get_map_categorical_attributes_inox(log: EventLog, attrs_map: dict={}) -> dict:
    if not attrs_map:
        attrs_map = {'CLUSTER_MATERIALE': {}, 'DESCR_GRADO': {}}
    for trace in log:
        for attr in attrs_map:
            value = trace.attributes[attr] 
            if value not in attrs_map[attr]:
                attrs_map[attr][value] = len(attrs_map[attr])
    return attrs_map

def get_map_categorical_attributes_bpic2019(log: EventLog, attrs_map: dict={}) -> dict:
    if not attrs_map:
        attrs_map = {'Spend area text': {}, 'Document Type': {}, 'Sub spend area text': {},
                      'Purch. Doc. Category name': {}, 'Item Type': {}, 'Item Category': {},
                      'Spend classification text': {}, 'GR-Based Inv. Verif.': {}, 'Goods Receipt': {}}
    for trace in log:
        for attr in attrs_map:
            value = trace.attributes[attr] 
            if value not in attrs_map[attr]:
                attrs_map[attr][value] = len(attrs_map[attr])
    return attrs_map

def get_map_categorical_attributes_bpic2017(log: EventLog, attrs_map: dict={}) -> dict:
    if not attrs_map:
        attrs_map = {'LoanGoal': {}, 'ApplicationType': {}}
    for trace in log:
        for attr in attrs_map:
            value = trace.attributes[attr] 
            if value not in attrs_map[attr]:
                attrs_map[attr][value] = len(attrs_map[attr])
    return attrs_map

def get_map_categorical_attributes_bpic2012(log: EventLog, attrs_map: dict={}) -> dict:
    if not attrs_map:
        attrs_map = {}
    for trace in log:
        for attr in attrs_map:
            value = trace.attributes[attr] 
            if value not in attrs_map[attr]:
                attrs_map[attr][value] = len(attrs_map)
    attrs_map['Categorical'] = {'Placeholder': 0}
    return attrs_map

def get_map_categorical_attributes_fine(log: EventLog, attrs_map: dict={}) -> dict:
    if not attrs_map:
        attrs_map = {'vehicleClass': {}, 'article': {}}
    for trace in log:
        for attr in attrs_map:
            value = trace[0][attr] 
            if value not in attrs_map[attr]:
                attrs_map[attr][value] = len(attrs_map[attr])
    return attrs_map

def get_sizes(clusters: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subcluster_df = {'cluster': [], 'size': []}
    sizes_clusters = {}
    for cluster in clusters:
        subcluster_df['cluster'].append(cluster)
        subcluster_df['size'].append(len(clusters[cluster]))
        cluster_name = cluster.split('-')[0]    
        if not cluster_name in sizes_clusters:
            sizes_clusters[cluster_name] = []
        sizes_clusters[cluster_name].append(len(clusters[cluster]))
    cluster_df = {'cluster': [], 'size': []}
    for cluster in sizes_clusters:
        cluster_df['cluster'].append(cluster)
        cluster_df['size'].append(sum(sizes_clusters[cluster]))
    return pd.DataFrame.from_dict(cluster_df), pd.DataFrame.from_dict(subcluster_df)

def get_percentage_shared_variants_subclusters(clusters: dict) -> pd.DataFrame:
    variants_sharing = {'cluster_x': [], 'cluster_y': [], 'sharing': []}
    for cluster_x in clusters:
        variants_x = pm4py.get_variants_as_tuples(clusters[cluster_x])
        for cluster_y in clusters:
            variants_y = pm4py.get_variants_as_tuples(clusters[cluster_y])
            intersection = set(variants_x).intersection(set(variants_y))
            sharing_perc = len(intersection) / len(variants_y)
            variants_sharing['cluster_x'].append(cluster_x)
            variants_sharing['cluster_y'].append(cluster_y)
            variants_sharing['sharing'].append(sharing_perc)
    return pd.DataFrame.from_dict(variants_sharing)

def get_percentage_shared_variants_clusters(clusters: dict) -> pd.DataFrame:
    variants_sharing = {'cluster_x': [], 'cluster_y': [], 'sharing': []}
    variants_clusters = {}
    for cluster in clusters:
        cluster_name = cluster.split('-')[0]
        if not cluster_name in variants_clusters:
            variants_clusters[cluster_name] = set()
        variants_clusters[cluster_name] = variants_clusters[cluster_name].union(set(pm4py.get_variants_as_tuples(clusters[cluster])))
    for cluster_x in variants_clusters:
        for cluster_y in variants_clusters:
            intersection = variants_clusters[cluster_x].intersection(variants_clusters[cluster_y])
            sharing_perc = len(intersection) / len(variants_clusters[cluster_y])
            variants_sharing['cluster_x'].append(cluster_x)
            variants_sharing['cluster_y'].append(cluster_y)
            variants_sharing['sharing'].append(sharing_perc)
    return pd.DataFrame.from_dict(variants_sharing)

def get_performance_cases_fines(clusters: dict) -> pd.DataFrame:
    compliant_cases = {'cluster': [], 'subcluster': [], 'throughput': [],
                       'compliance': [], 'start_date': [], 'compliance_type': []}
    for cluster in clusters:
        for trace in clusters[cluster]:
            start_date = trace[0]['time:timestamp']
            end_date = trace[-1]['time:timestamp']
            throughput = end_date - start_date
            throughput_days = throughput.days + throughput.seconds/(60*60*24)
            compliance, compliance_type = is_compliant_fines(trace)
            compliant_cases['cluster'].append(cluster.split('-')[0])
            compliant_cases['subcluster'].append(cluster)
            compliant_cases['throughput'].append(throughput_days)
            compliant_cases['compliance'].append(compliance)
            compliant_cases['compliance_type'].append(compliance_type)
            compliant_cases['start_date'].append(start_date)
    return pd.DataFrame.from_dict(compliant_cases)

def get_performance_cases_bpic2019(clusters: dict) -> pd.DataFrame:
    compliant_cases = {'cluster': [], 'subcluster': [], 'throughput': [],
                       'compliance': [], 'compliance_type': [], 'completeness': [],
                       'start_date': [], 'item_category': []}
    for cluster in clusters:
        for trace in clusters[cluster]:
            start_date = trace[0]['time:timestamp']
            end_date = trace[-1]['time:timestamp']
            throughput = end_date - start_date
            compliance, completeness = is_compliant_incomplete_bpic2019(trace)
            compliant_cases['cluster'].append(cluster.split('-')[0])
            compliant_cases['subcluster'].append(cluster)
            compliant_cases['throughput'].append(throughput)
            compliant_cases['compliance'].append(compliance)
            compliant_cases['compliance_type'].append(compliance)
            compliant_cases['completeness'].append(completeness)
            compliant_cases['start_date'].append(start_date)
            compliant_cases['item_category'].append(trace.attributes['Item Category'])
    return pd.DataFrame.from_dict(compliant_cases)

def get_performance_cases_bpic2017(clusters: dict) -> pd.DataFrame:
    compliant_cases = {'cluster': [], 'subcluster': [], 'throughput': [],
                       'compliance': [], 'compliance_type': [], 'completeness': [],
                       'start_date': [], 'item_category': []}
    for cluster in clusters:
        for trace in clusters[cluster]:
            start_date = trace[0]['time:timestamp']
            end_date = trace[-1]['time:timestamp']
            throughput = end_date - start_date
            #compliance, completeness = is_compliant_incomplete_bpic2019(trace)
            compliance = random.choice(['compliant', 'incompliant'])
            completeness = random.choice(['complete', 'incomplete'])

            compliant_cases['cluster'].append(cluster.split('-')[0])
            compliant_cases['subcluster'].append(cluster)
            compliant_cases['throughput'].append(throughput)
            compliant_cases['compliance'].append(compliance)
            compliant_cases['compliance_type'].append(compliance)
            compliant_cases['completeness'].append(completeness)
            compliant_cases['start_date'].append(start_date)
            compliant_cases['item_category'].append('None')
    return pd.DataFrame.from_dict(compliant_cases)

def get_performance_cases_bpic2012(clusters: dict) -> pd.DataFrame:
    compliant_cases = {'cluster': [], 'subcluster': [], 'throughput': [],
                       'compliance': [], 'compliance_type': [], 'completeness': [],
                       'start_date': [], 'item_category': []}
    for cluster in clusters:
        for trace in clusters[cluster]:
            start_date = trace[0]['time:timestamp']
            end_date = trace[-1]['time:timestamp']
            throughput = end_date - start_date
            #compliance, completeness = is_compliant_incomplete_bpic2019(trace)
            compliance = random.choice(['compliant', 'incompliant'])
            completeness = random.choice(['complete', 'incomplete'])

            compliant_cases['cluster'].append(cluster.split('-')[0])
            compliant_cases['subcluster'].append(cluster)
            compliant_cases['throughput'].append(throughput)
            compliant_cases['compliance'].append(compliance)
            compliant_cases['compliance_type'].append(compliance)
            compliant_cases['completeness'].append(completeness)
            compliant_cases['start_date'].append(start_date)
            compliant_cases['item_category'].append('None')
    return pd.DataFrame.from_dict(compliant_cases)

def compute_percentage(data: pd.DataFrame, level_0: str, level_1: str='') -> pd.DataFrame:
    data = copy.deepcopy(data)
    data.insert(len(data.columns), 'perc', 0)
    if level_1:
        data_perc = data.pivot_table(index=[level_0, level_1], aggfunc='count')
        data_perc = data_perc['perc'] / data_perc.groupby(level=0)['perc'].sum()
        return data_perc.reset_index()[[level_0, level_1, 'perc']]
    else:
        data_perc = data.pivot_table(index=[level_0], aggfunc='count')
        data_perc = data_perc['perc'] / data_perc['perc'].sum()
        return data_perc.reset_index()[[level_0, 'perc']]

def create_daily_time_series(data_in: pd.DataFrame, column: str) -> pd.DataFrame:
    data = copy.deepcopy(data_in)
    data.insert(len(data.columns), 'counter', 0)
    data['start_date'] = pd.to_datetime(data['start_date'], utc=True)
    data['start_date'] = data['start_date'].dt.date
    days = data.pivot_table(index=[column, 'start_date'], values='counter', aggfunc='count') 
    days = days.reset_index()
    return days

def extract_variant(trace: Trace) -> list:
    return [event['concept:name'] for event in trace]

def extract_attrs_fines_log(trace: Trace, maxs: dict, map_cat: dict) -> Tuple[list, int]:
    attrs_name = ['amount', 'points', 'vehicleClass', 'article']
    num_of_continuous = 2
    attrs = []
    for idx, attr in enumerate(attrs_name):
        if idx < num_of_continuous:
            value = trace[0][attr]
            if not maxs[attr] == 0:
                value = value/maxs[attr]
            attrs.append(value)
        else:
            attrs.append(map_cat[attr][trace[0][attr]])
    return attrs, num_of_continuous

def extract_continuous_attrs_fines_log(trace: Trace, maxs: dict) -> list:
    attrs = ['amount', 'points']
    return [trace[0][attr]/maxs[attr] for attr in attrs]

def extract_categorical_attrs_fines_log(trace: Trace) -> list:
    attrs = ['vehicleClass', 'article']
    return [trace[0][attr] for attr in attrs]

def extract_attrs_bpic2019_log(trace: Trace, maxs: dict, map_cat: dict) -> Tuple[list, int]:
    attrs_name = ['Cumulative net worth (EUR)', 'Spend area text', 'Document Type',
             'Sub spend area text', 'Purch. Doc. Category name', 'Item Type', 'Item Category',
             'Spend classification text', 'GR-Based Inv. Verif.', 'Goods Receipt']
    num_of_continuous = 1
    attrs = []
    for idx, attr in enumerate(attrs_name):
        if idx < num_of_continuous:
            value = trace[0][attr]
            if not maxs[attr] == 0:
                value = value/maxs[attr]
            attrs.append(value)
        else:
            attrs.append(map_cat[attr][trace.attributes[attr]])
    return attrs, num_of_continuous

def extract_continuous_attrs_bpic2019_log(trace: Trace, maxs: dict) -> list:
    attrs = ['Cumulative net worth (EUR)']
    return [trace[0][attr]/maxs[attr] for attr in attrs]

def extract_categorical_attrs_bpic2019_log(trace: Trace) -> list:
    attrs = {'Spend area text': [], 'Document Type': [], 'Sub spend area text': [],
                  'Purch. Doc. Category name': [], 'Item Type': [], 'Item Category': [],
                  'Spend classification text': [], 'GR-Based Inv. Verif.': [], 'Goods Receipt': []}
    return [trace.attributes[attr] for attr in trace.attributes if attr in attrs]

def extract_attrs_bpic2017_log(trace: Trace, maxs: dict, map_cat: dict) -> Tuple[list, int]:
    attrs_name = ['RequestedAmount', 'LoanGoal', 'ApplicationType']
    num_of_continuous = 1
    attrs = []
    for idx, attr in enumerate(attrs_name):
        if idx < num_of_continuous:
            value = trace.attributes[attr]
            if not maxs[attr] == 0:
                value = value/maxs[attr]
            attrs.append(value)
        else:
            attrs.append(map_cat[attr][trace.attributes[attr]])
    return attrs, num_of_continuous

def extract_continuous_attrs_bpic2017_log(trace: Trace, maxs: dict) -> list:
    attrs = ['RequestedAmount']
    return [trace.attributes[attr]/maxs[attr] for attr in attrs]

def extract_categorical_attrs_bpic2017_log(trace: Trace) -> list:
    attrs = {'LoanGoal': [], 'ApplicationType': []}
    return [trace.attributes[attr] for attr in trace.attributes if attr in attrs]

def extract_attrs_bpic2012_log(trace: Trace, maxs: dict, map_cat: dict) -> Tuple[list, int]:
    attrs_name = ['AMOUNT_REQ', 'Categorical']
    num_of_continuous = 1
    attrs = []
    for idx, attr in enumerate(attrs_name):
        if idx < num_of_continuous:
            value = trace.attributes[attr]
            if not maxs[attr] == 0:
                value = value/maxs[attr]
            attrs.append(value)
        else:
            attrs.append(map_cat[attr][trace.attributes[attr]])
    return attrs, num_of_continuous

def extract_continuous_attrs_bpic2012_log(trace: Trace, maxs: dict) -> list:
    attrs = ['AMOUNT_REQ']
    return [trace.attributes[attr]/maxs[attr] for attr in attrs]

def extract_categorical_attrs_bpic2012_log(trace: Trace) -> list:
    #attrs = {}
    return ['Placeholder']

def complete_log_from_clusters(clusters: dict) -> EventLog:
    complete_log = EventLog()
    for cluster in clusters:
        for trace in clusters[cluster]:
            complete_log.append(trace)
    return complete_log

def extract_max_continuous_values_fines_log(log: EventLog) -> dict:
    max_attrs = {'amount': 0, 'points': 0}
    for trace in log:
        for attr in max_attrs:
            if trace[0][attr] > max_attrs[attr]:
                max_attrs[attr] = trace[0][attr]
    return max_attrs

def extract_max_continuous_values_bpic2019_log(log: EventLog) -> dict:
    max_attrs = {'Cumulative net worth (EUR)': 0}
    for trace in log:
        for attr in max_attrs:
            if trace[0][attr] > max_attrs[attr]:
                max_attrs[attr] = trace[0][attr]
    return max_attrs

def extract_max_continuous_values_bpic2017_log(log: EventLog) -> dict:
    max_attrs = {'RequestedAmount': 0}
    for trace in log:
        for attr in max_attrs:
            if trace.attributes[attr] > max_attrs[attr]:
                max_attrs[attr] = trace.attributes[attr]
    return max_attrs

def extract_max_continuous_values_bpic2012_log(log: EventLog) -> dict:
    max_attrs = {'AMOUNT_REQ': 0}
    for trace in log:
        for attr in max_attrs:
            if trace.attributes[attr] > max_attrs[attr]:
                max_attrs[attr] = trace.attributes[attr]
    return max_attrs

def extract_max_continuous_values_from_clusters(clusters: dict, extract_max_cont_attrs: Callable[[EventLog], dict]) -> dict:
    maxs = {}
    for cluster in clusters:
        maxs_cluster = extract_max_cont_attrs(clusters[cluster])
        for attr in maxs_cluster:
            if not attr in maxs:
                maxs[attr] = maxs_cluster[attr]
            elif maxs_cluster[attr] > maxs[attr]:
                maxs[attr] = copy.copy(maxs_cluster[attr])
    return maxs

def get_map_cat_attrs_from_clusters(clusters: dict, get_map_cat_attrs: Callable[[EventLog, dict], dict]) -> dict:
    map_cat = {}
    for cluster in clusters:
        map_cat = get_map_cat_attrs(clusters[cluster], map_cat)
    return map_cat

def model_performance(results: dict, filtered_log: EventLog, coverage: float,
                      model_name: str, variants: dict, num_variants_tot: int) -> dict:
    net, im, fm = pm4py.discover_petri_net_inductive(filtered_log, noise_threshold=0.6)
    fitness_dict = pm4py.fitness_alignments(filtered_log, net, im, fm)
    fitness = fitness_dict['log_fitness']
    precision = pm4py.precision_alignments(filtered_log, net, im, fm)
    f1_score = 2 * (fitness * precision) / (fitness + precision)
    results['model_clstr'].append(model_name)
    results['fitness'].append(fitness)
    results['precision'].append(precision)
    results['num_traces'].append(len(filtered_log))
    results['num_variants'].append(len(variants))
    results['perc_variants'].append(len(variants)/num_variants_tot)
    results['coverage'].append(coverage)
    results['f1_score'].append(f1_score)
    return results

def compute_models_performance(clusters: dict, coverages: list=[1], version_filter: str='v2') -> pd.DataFrame:
    results = {'model_clstr': [], 'fitness': [], 'precision': [], 'f1_score': [],
               'num_traces': [], 'num_variants': [], 'perc_variants': [], 'coverage': []}
    tot_num_variants = {}
    complete_log = complete_log_from_clusters(clusters)
    for coverage in coverages:
        if coverage != 1:
            if version_filter == 'v2':
                filtered_log = filter_log_on_variants_coverage_v2(complete_log, coverage)
            else:
                filtered_log = filter_log_on_variants_coverage(complete_log, coverage)
        else:
            filtered_log = copy.deepcopy(complete_log)
        variants = pm4py.get_variants_as_tuples(filtered_log)
        if coverage == 1:
            tot_num_variants['complete'] = len(variants)
        num_variants_tot = tot_num_variants['complete']
        results = model_performance(results, filtered_log, coverage, 'complete', variants, num_variants_tot)
        for cluster in clusters:
            if coverage != 1:
                if version_filter == 'v2':
                    filtered_log = filter_log_on_variants_coverage_v2(clusters[cluster], coverage)
                else:
                    filtered_log = filter_log_on_variants_coverage(clusters[cluster], coverage)
            else:
                filtered_log = copy.deepcopy(clusters[cluster])
            variants = pm4py.get_variants_as_tuples(filtered_log)
            if coverage == 1:
                tot_num_variants[cluster] = len(variants)
            results = model_performance(results, filtered_log, coverage, cluster, variants, tot_num_variants[cluster])
    return pd.DataFrame.from_dict(results)

def create_timeseries_compliance_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    compliant_df = df[df['compliance'] == 'compliant']
    compliant_time_series = create_daily_time_series(copy.deepcopy(compliant_df), column)
    compliant_time_series.insert(len(compliant_time_series.columns), 'compliance', 'compliant')
    incompliant_df = df[df['compliance'] == 'incompliant']
    incompliant_time_series = create_daily_time_series(copy.deepcopy(incompliant_df), column)
    incompliant_time_series.insert(len(incompliant_time_series.columns), 'compliance', 'incompliant')
    time_series_compliance = pd.concat(
            [compliant_time_series[[column, 'compliance', 'counter', 'start_date']],
             incompliant_time_series[[column, 'compliance', 'counter', 'start_date']]], axis=0)
    return time_series_compliance
