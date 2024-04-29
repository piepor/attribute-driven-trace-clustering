import pm4py
import pandas as pd

log = pm4py.read_xes('./data/road_traffic_fine_management_process.xes')
trace_attributes = {
        'amount': [], 'article': [], 'points': [], 
        'vehicleClass': [], 'CASE_ID': [], 'time:timestamp': []}

for trace in log:
    for attr in trace_attributes:
        if attr == 'CASE_ID':
            trace_attributes[attr].append(trace.attributes['concept:name'])
        else:
            trace_attributes[attr].append(trace[0][attr])
trace_attributes = pd.DataFrame.from_dict(trace_attributes).fillna('Unknown')
trace_attributes.to_csv('./data/road_traffic_fine_management_process-trace-attributes.csv', index=False)
