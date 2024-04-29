from collections import Counter
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog

def count_activities_in_clusters(clusters: dict) -> pd.DataFrame:
    """
    count the number of activities occurrence in each cluster log.
    log_cluster is a dict {log: cluster_id}
    """
    clusters_activities_count = {'cluster': [], 'activity': []}
    for cluster in clusters:
        for trace in clusters[cluster]:
            for event in trace:
                clusters_activities_count['cluster'].append(cluster)
                clusters_activities_count['activity'].append(event['concept:name'])
    return pd.DataFrame.from_dict(clusters_activities_count)

def count_activities(clusters: dict) -> pd.DataFrame:
    activity_count = {'activity': [], 'cluster': [], 'total_num': [], 'subcluster': []}
    for cluster in clusters:
        variants = pm4py.get_variants_as_tuples(clusters[cluster])
        for variant in variants:
            act_count = Counter(variant)
            for act in act_count:
                activity_count['activity'].append(act)
                activity_count['total_num'].append(len(variants[variant])*act_count[act])
                activity_count['cluster'].append(cluster.split('-')[0])
                activity_count['subcluster'].append(cluster)
    activity_count = pd.DataFrame.from_dict(activity_count)
    return activity_count.pivot_table(
            index=['cluster', 'subcluster', 'activity'], values='total_num', aggfunc='sum').reset_index()

def count_start_end_activities(clusters: dict) -> pd.DataFrame:
    start_end_activities = {'start_activity': [], 'end_activity': [], 'cluster': [], 'subcluster': []}
    for cluster in clusters:
        for trace in clusters[cluster]:
            start_act = trace[0]['concept:name']
            end_act = trace[-1]['concept:name']
            start_end_activities['start_activity'].append(start_act)
            start_end_activities['end_activity'].append(end_act)
            start_end_activities['subcluster'].append(cluster)
            start_end_activities['cluster'].append(cluster.split('-')[0])
    return pd.DataFrame.from_dict(start_end_activities)

def count_activities_frequency_perc(log: EventLog) -> dict:
    acts_freq = {}
    total_acts = 0
    for trace in log:
        for event in trace:
            if not event['concept:name'] in acts_freq:
                acts_freq[event['concept:name']] = 1
            else:
                acts_freq[event['concept:name']] += 1
            total_acts += 1
    acts_freq_perc = {}
    for act in acts_freq:
        acts_freq_perc[act] = acts_freq[act] / total_acts
    return acts_freq_perc
