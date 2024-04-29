import pm4py
import pandas as pd

log = pm4py.read_xes('./data/BPI_Challenge_2019-max_length_45.xes')
trace_attributes = {
        'Spend area text': [], 'Document Type': [], 'Sub spend area text': [], 
        'Purch. Doc. Category name': [], 'Item Type': [], 'Item Category': [],
        'Spend classification text': [], 'GR-Based Inv. Verif.': [],
        'Goods Receipt': [], 'Cumulative net worth (EUR)': [], 'CASE_ID': [], 'time:timestamp': []}

for trace in log:
    for attr in trace_attributes:
        if attr in ['Cumulative net worth (EUR)', 'time:timestamp']:
            trace_attributes[attr].append(trace[0][attr])
        elif 'CASE_ID':
            trace_attributes[attr].append(trace.attributes['concept:name'])
        else:
            trace_attributes[attr].append(trace.attributes[attr])
trace_attributes = pd.DataFrame.from_dict(trace_attributes).fillna('Unknown')
trace_attributes.to_csv('./data/bpi_2019-max_length_45-trace-attributes.csv', index=False)
