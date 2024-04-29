import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog
from scipy.spatial.distance import jensenshannon
from utils.counting import count_activities

def get_activities_frequency_perc(log: EventLog) -> dict:
    total_cases = len(log)
    variants = pm4py.get_variants_as_tuples(log)
    activities = pm4py.get_event_attribute_values(log, "concept:name")
    activities_count = {}
    for activity in activities:
        activity_count = 0
        for variant in variants:
            if activity in variant:
                activity_count += len(variants[variant])
        activities_count[activity] = activity_count / total_cases
    return activities_count

def get_activities_perc_subclusters(activity_count: pd.DataFrame) -> pd.DataFrame:
    ptable = activity_count.pivot_table(index=['cluster', 'subcluster', 'activity'], values='total_num', aggfunc='sum') 
    perc_table = ptable['total_num'] / ptable.groupby(level=[0, 1])['total_num'].sum()
    return perc_table.reset_index().rename(columns={'total_num': 'perc'})

def get_activities_perc_clusters(activity_count: pd.DataFrame) -> pd.DataFrame:
    ptable = activity_count.pivot_table(index=['cluster', 'activity'], values='total_num', aggfunc='sum') 
    perc_table = ptable['total_num'] / ptable.groupby(level=0)['total_num'].sum()
    return perc_table.reset_index().rename(columns={'total_num': 'perc'})

# def get_attribute_perc(data: pd.DataFrame, column: str, attribute: str) -> pd.DataFrame:
#     data.insert(len(data.columns), "perc", 0)
#     ptable = data.pivot_table(index=[column, attribute], values='perc', aggfunc='count') 
#     perc_table = ptable['perc'] / ptable.groupby(level=0)['perc'].sum()
#     return perc_table.reset_index()

def get_activities_distribution_across_cluster(activity_count: pd.DataFrame) -> pd.DataFrame:
    ptable = activity_count.pivot_table(index=['activity', 'cluster'], values='total_num', aggfunc='sum') 
    perc_table = ptable['total_num'] / ptable.groupby(level=0)['total_num'].sum()
    return perc_table.reset_index().rename(columns={'total_num': 'perc'})

def get_activities_distribution(activities_count: pd.DataFrame, possible_activities: list) -> list:
    distribution = []
    for act in possible_activities:
        if act in activities_count['activity'].unique():
            distribution.append(activities_count[activities_count['activity'] == act]['total_num'].sum())
        else:
            distribution.append(0)
    total_num_act = sum(distribution)
    return [x/total_num_act for x in distribution]

def get_trace_attributes_distribution_traffic_fine(log: EventLog) -> pd.DataFrame:
    attributes = {'amount': [], 'points': [], 'vehicleClass': [], 'article': []}
    for trace in log:
        for attribute in attributes:
            attributes[attribute].append(trace[0][attribute])
    return pd.DataFrame.from_dict(attributes)

def get_trace_attributes_distribution_bpi19(log: EventLog) -> pd.DataFrame:
    attributes = {'Cumulative net worth (EUR)': [], 'Spend area text': [], 'Document Type': [], 'Sub spend area text': [],
                  'Purch. Doc. Category name': [], 'Item Type': [], 'Item Category': [],
                  'Spend classification text': [], 'GR-Based Inv. Verif.': [], 'Goods Receipt': []}
    for trace in log:
        for attribute in attributes:
            if attribute == 'Cumulative net worth (EUR)':
                attributes[attribute].append(trace[0][attribute])
            else:
                attributes[attribute].append(trace.attributes[attribute])
    return pd.DataFrame.from_dict(attributes)

def get_trace_attributes_distribution_bpi17(log: EventLog) -> pd.DataFrame:
    attributes = {'RequestedAmount': [], 'LoanGoal': [], 'ApplicationType': []}
    for trace in log:
        for attribute in attributes:
            attributes[attribute].append(trace.attributes[attribute])
    return pd.DataFrame.from_dict(attributes)

def get_trace_attributes_distribution_bpi12(log: EventLog) -> pd.DataFrame:
    attributes = {'AMOUNT_REQ': []}
    for trace in log:
        for attribute in attributes:
            attributes[attribute].append(trace.attributes[attribute])
    attributes['Categorical'] = len(log)*['Placeholder']
    return pd.DataFrame.from_dict(attributes)

def get_activities_distribution_divergence(clusters: dict, column: str) -> pd.DataFrame:
    act_distr_div = {'cluster_x': [], 'cluster_y': [], 'divergence': []}
    activities_count = count_activities(clusters)
    possible_activities = activities_count['activity'].unique().tolist()
    for cluster_x in activities_count[column].unique():
        act_count_x = activities_count[activities_count[column] == cluster_x]
        act_distr_x = get_activities_distribution(act_count_x, possible_activities)
        for cluster_y in activities_count[column].unique():
            act_count_y = activities_count[activities_count[column] == cluster_y]
            act_distr_y = get_activities_distribution(act_count_y, possible_activities)
            divergence = jensenshannon(act_distr_x, act_distr_y, 2.0)
            act_distr_div['cluster_x'].append(cluster_x)
            act_distr_div['cluster_y'].append(cluster_y)
            act_distr_div['divergence'].append(divergence)
    return pd.DataFrame.from_dict(act_distr_div)

def distribution_start_activities(start_end_activities: pd.DataFrame, level: str) -> pd.DataFrame:
    count_start_activities = (
            start_end_activities
            .groupby(by=[level, 'start_activity']).count().reset_index())
    perc_table = (
            count_start_activities
            .rename(columns={'end_activity': 'perc'})
            .pivot_table(index=[level, 'start_activity'], values='perc', aggfunc='sum'))
    perc_table = (
            perc_table['perc'] / perc_table.groupby(level=0)['perc'].sum())
    return perc_table.reset_index()

def distribution_end_activities(start_end_activities: pd.DataFrame, level: str) -> pd.DataFrame:
    count_end_activities = (
            start_end_activities
            .groupby(by=[level, 'end_activity']).count().reset_index())
    perc_table = (
            count_end_activities
            .rename(columns={'start_activity': 'perc'})
            .pivot_table(index=[level, 'end_activity'], values='perc', aggfunc='sum'))
    perc_table = (
            perc_table['perc'] / perc_table.groupby(level=0)['perc'].sum())
    return perc_table.reset_index()

def get_variants_distribution(clusters: dict, coverage: float=1.0) -> dict:
    variants_distr = {}
    for cluster in clusters:
        lengths_variants = []
        lengths_variants_perc = []
        traces_represented = 0
        colors = []
        tot_traces = len(clusters[cluster])
        coverage_traces = coverage*tot_traces
        variants = pm4py.get_variants_as_tuples(clusters[cluster])
        variants = dict(sorted(variants.items(), key=lambda x: len(x[1]), reverse=True))
        for variant in variants:
            lengths_variants.append(len(variants[variant]))
            lengths_variants_perc.append(len(variants[variant])/tot_traces)
            traces_represented += len(variants[variant])
            if traces_represented <= coverage_traces:
                colors.append('red')
            else:
                colors.append('blue')
        variants_distr[cluster] = {'lengths': lengths_variants, 'lengths_perc': lengths_variants_perc, 'colors': colors, 'coverage': coverage}
    return variants_distr
