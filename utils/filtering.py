"""
This modules contains functions related to data filtering
"""

from typing import Tuple
from datetime import datetime
from dateutil import tz
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.conformance import EventStream
from utils.counting import count_activities_frequency_perc

def filter_dataframe_on_cluster(dataframe: pd.DataFrame, cluster: int) -> pd.DataFrame:
    """ filter a dataframe based on the cluster name """
    return dataframe[dataframe['cluster'] == cluster]

def filter_log_on_cluster(log: EventLog | pd.DataFrame, cluster_idx: list[int]) -> EventLog | EventStream:
    """ filter an event log based on the cluster index """
    # TODO: sometimes case ids are string others integer
    if type(cluster_idx[0]) == int:
        return pm4py.filter_log(lambda trace: int(trace.attributes['concept:name']) in cluster_idx, log)
    elif type(cluster_idx[0]) == str:
        return pm4py.filter_log(lambda trace: str(trace.attributes['concept:name']) in cluster_idx, log)
    else:
        raise TypeError('Case id type not understood')

def filter_variants(cluster_variants_dict, cluster):
    """ filter variants based on cluster """
    return [variant for variant in cluster_variants_dict if cluster_variants_dict[variant] == cluster]

def filter_log_on_variants(log: EventLog|EventStream, variants: list[Tuple]) -> EventLog:
    """ filter an event log based on variants """
    total_variants = pm4py.get_variants_as_tuples(log)
    filtered_log = EventLog()
    for variant in variants:
        for trace in total_variants[variant]:
            filtered_log.append(trace)
    return filtered_log

def filter_log_on_variants_coverage(log: EventLog, traces_perc: float) -> EventLog:
    """ filter an event log based on how many traces the variants represent """
    total_variants = pm4py.get_variants_as_tuples(log)
    total_variants = dict(reversed(sorted(total_variants.items(), key=lambda item: len(item[1]))))
    threshold_traces_number = traces_perc * len(log)
    filtered_log = EventLog()
    last_traces_added_num = -1
    for variant in total_variants:
        if len(filtered_log) < threshold_traces_number:
            for trace in total_variants[variant]:
                filtered_log.append(trace)
        elif last_traces_added_num == len(total_variants[variant]):
            for trace in total_variants[variant]:
                filtered_log.append(trace)
        # update only if the target is not reached
        if len(filtered_log) < threshold_traces_number:
            last_traces_added_num = len(total_variants[variant])
    return filtered_log

def filter_log_on_variants_coverage_v2(log: EventLog, traces_perc: float) -> EventLog:
    """ filter an event log based on how many traces the variants represent """
    total_variants = pm4py.get_variants_as_tuples(log)
    total_variants = dict(reversed(sorted(total_variants.items(), key=lambda item: len(item[1]))))
    threshold_traces_number = traces_perc * len(log)
    filtered_log = EventLog()
    for variant in total_variants:
        if len(filtered_log) < threshold_traces_number:
            for trace in total_variants[variant]:
                filtered_log.append(trace)
    return filtered_log

def filter_dataframe_by_value_substring(dataframe: pd.DataFrame, column:str, substring: str) -> pd.DataFrame:
    return dataframe[dataframe[column].str.contains(substring)]

def filter_log_by_case_id(log, ids):
    # TODO: sometimes case ids are string others integer
    return pm4py.filter_log(lambda trace: int(trace.attributes['concept:name']) in ids, log)

def filter_log_by_event_attribute(log: EventLog, attribute: str, values: list[str]) -> EventLog | pd.DataFrame:
    """ filters out case with values of the specified attribute in the 'values' list """
    return pm4py.filter_event_attribute_values(log, attribute, values, retain=False)

def filter_log_by_activity_freq(log: EventLog, freq: float) -> EventLog | pd.DataFrame:
    """ filters out cases containing an activity with frequency below the specified percentage """
    activities_freq = count_activities_frequency_perc(log)
    activities_to_filter =[activity for activity in activities_freq if activities_freq[activity] < freq]
    return filter_log_by_event_attribute(log, "concept:name", activities_to_filter)

def filter_log_by_time(log: EventLog, date_col:str, start_date: datetime, end_date: datetime) -> EventLog:
    # considering time zone if the log has it
    if log[0][0][date_col].tzinfo:
        tzinfo = tz.gettz('Europe/Rome')
        start_date = datetime(
                start_date.year, start_date.month, start_date.day, 
                start_date.hour, start_date.minute, start_date.second, tzinfo=tzinfo) 
        end_date = datetime(
                end_date.year, end_date.month, end_date.day, 
                end_date.hour, end_date.minute, end_date.second, tzinfo=tzinfo) 
    filtered_log = EventLog()
    for trace in log:
        if start_date and end_date:
            if trace[0][date_col] >= start_date and trace[0][date_col] < end_date:
                filtered_log.append(trace)
        elif start_date:
            if trace[0][date_col] >= start_date:
                filtered_log.append(trace)
        elif end_date:
            if trace[0][date_col] < end_date:
                filtered_log.append(trace)
    return filtered_log

def filter_df_by_time(df: pd.DataFrame, date_col: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    # considering time zone if the log has it
    # breakpoint()
    if df[date_col].dt.tz:
        # tzinfo = tz.gettz(df[date_col].dt.tz)
        tzinfo = df[date_col].dt.tz
        start_date = datetime(
                start_date.year, start_date.month, start_date.day, 
                start_date.hour, start_date.minute, start_date.second, tzinfo=tzinfo) 
        end_date = datetime(
                end_date.year, end_date.month, end_date.day, 
                end_date.hour, end_date.minute, end_date.second, tzinfo=tzinfo) 
    if start_date and end_date:
        df = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
    elif start_date:
        df = df[df[date_col] >= start_date]
    elif end_date:
        df = df[df[date_col] < end_date]
    return df
