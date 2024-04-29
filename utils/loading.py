"""
This module contains functions related to loading any kind of data
"""

import pm4py
from pm4py.objects.log.obj import EventLog, Trace
import copy

def load_xes(path_log: str, only_lifecycle_start=True):
    """ 
    import the event log stored as a XES file.
    If only_lifecycle_start is true than the log will be filtered 
    retaining only events that have the 'lifecycle:transition' property 
    equal to 'start'
    """
    log = pm4py.read_xes(path_log)
    if only_lifecycle_start:
        filtered_log = EventLog()
        for trace in log:
            filtered_trace = Trace(attributes=copy.deepcopy(trace.attributes))
            for event in trace:
                if event['lifecycle:transition'] == 'start':
                    filtered_trace.append(event)
            filtered_log.append(filtered_trace)
    else:
        filtered_log = copy.deepcopy(log)
    return filtered_log
