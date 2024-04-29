import pm4py
import pandas as pd
from datetime import datetime 
import plotly.express as px

log = pm4py.read_xes('./logs/Road_Traffic_Fine_Management_Process.xes')
end_datetime = datetime.strptime('2012-06-01 00:00:00 +0000', '%Y-%m-%d %H:%M:%S %z') 
cases_filtered = log[(log['concept:name'] == 'Create Fine') & (log['time:timestamp'] < end_datetime)]['case:concept:name'].tolist()
log_filtered = log[log['case:concept:name'].isin(cases_filtered)]

cases_log_low = log_filtered[(log_filtered['concept:name'] == 'Create Fine') & (log_filtered['amount'] <= 70)]['case:concept:name'].tolist()
log_low = log_filtered[log_filtered['case:concept:name'].isin(cases_log_low)]

cases_log_medium_high = log_filtered[(log_filtered['concept:name'] == 'Create Fine') & (log_filtered['amount'] > 70)]['case:concept:name'].tolist()
log_medium_high = log_filtered[log_filtered['case:concept:name'].isin(cases_log_medium_high)]

# how many payments between low and medium-high
# breakpoint()
payments_low = log_low.groupby('case:concept:name').apply(lambda x: x['concept:name'].tolist().count('Payment'))
payments_low.name = 'payments'
payments_medium_high = log_medium_high.groupby('case:concept:name').apply(lambda x: x['concept:name'].tolist().count('Payment'))
payments_medium_high.name = 'payments'

payments_df = payments_low.to_frame().copy()
payments_df.insert(1, 'type', 'low')
payments_medium_high = payments_medium_high.to_frame()
payments_medium_high.insert(1, 'type', 'medium-high')
payments_df = pd.concat([payments_df, payments_medium_high])

fig = px.histogram(payments_df, 'payments', color='type', barmode='group', histnorm='probability density')
fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                 showline=True, linewidth=2, linecolor='black')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
fig.show()
fig.write_html('payments-low-vs-medium-high.html')
fig.write_image('payments-low-vs-medium-high.svg')

# Appeal between low and medium high
appeal_low = log_low.groupby('case:concept:name').apply(lambda x: len(set(['Appeal to Judge', 'Send Appeal to Prefecture']).intersection(set(x['concept:name'].tolist()))) > 0)
appeal_low.name = 'appeal'
appeal_medium_high = log_medium_high.groupby('case:concept:name').apply(lambda x: len(set(['Appeal to Judge', 'Send Appeal to Prefecture']).intersection(set(x['concept:name'].tolist()))) > 0)
appeal_medium_high.name = 'appeal'

appeal_df = appeal_low.to_frame().copy()
appeal_df.insert(1, 'type', 'low')
appeal_medium_high = appeal_medium_high.to_frame()
appeal_medium_high.insert(1, 'type', 'medium-high')
appeal_df = pd.concat([appeal_df, appeal_medium_high])

fig = px.histogram(appeal_df, 'appeal', color='type', barmode='group', histnorm='probability density')
fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                 showline=True, linewidth=2, linecolor='black')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
fig.show()

# Credit collection between low and medium high
credit_low = log_low.groupby('case:concept:name').apply(lambda x: 'Send for Credit Collection' in x['concept:name'].tolist())
credit_low.name = 'credit'
credit_medium_high = log_medium_high.groupby('case:concept:name').apply(lambda x: 'Send for Credit Collection' in x['concept:name'].tolist())
credit_medium_high.name = 'credit'

credit_df = credit_low.to_frame().copy()
credit_df.insert(1, 'type', 'low')
credit_medium_high = credit_medium_high.to_frame()
credit_medium_high.insert(1, 'type', 'medium-high')
credit_df = pd.concat([credit_df, credit_medium_high])

fig = px.histogram(credit_df, 'credit', color='type', barmode='group', histnorm='probability density')
fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                 showline=True, linewidth=2, linecolor='black')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
fig.show()

# how many payments in medium high with points deduction between judge and no judge
cases_points = log_medium_high[(log_medium_high['concept:name'] == 'Create Fine') & (log_medium_high['points'] > 0)]['case:concept:name'].tolist()
log_points = log_medium_high[log_medium_high['case:concept:name'].isin(cases_points)]
payments_points = log_points.groupby('case:concept:name').apply(lambda x: x['concept:name'].tolist().count('Payment'))
payments_points.name = 'payments'
judge_points = log_points.groupby('case:concept:name').apply(lambda x: 'Appeal to Judge' in x['concept:name'].tolist())
judge_points.name = 'judge'
judge_payments_df = pd.merge(payments_points.to_frame(), judge_points.to_frame(), on='case:concept:name')

fig = px.histogram(judge_payments_df, 'payments', color='judge', barmode='group', histnorm='probability density')
fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                 showline=True, linewidth=2, linecolor='black')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
fig.show()
fig.write_html('payments-points-judge-vs-no-judge.html')
fig.write_image('payments-points-judge-vs-no-judge.svg')

# appeal to judge without appeal to prefecture between points/no points in medium high
cases_no_points = log_medium_high[(log_medium_high['concept:name'] == 'Create Fine') & (log_medium_high['points'] == 0)]['case:concept:name'].tolist()
log_no_points = log_medium_high[log_medium_high['case:concept:name'].isin(cases_no_points)]
judge_no_points = log_no_points.groupby('case:concept:name').apply(lambda x: 'Appeal to Judge' in x['concept:name'].tolist())
judge_no_points.name = 'judge'

# only cases with judge
cases_judge_points = judge_points[judge_points == True].index.tolist()
cases_judge_no_points = judge_no_points[judge_no_points == True].index.tolist()
log_judge_points = log_points[log_points['case:concept:name'].isin(cases_judge_points)]
log_judge_no_points = log_no_points[log_no_points['case:concept:name'].isin(cases_judge_no_points)]

judge_prefecture_points = log_judge_points.groupby('case:concept:name').apply(lambda x: 'Send Appeal to Prefecture' in x['concept:name'].tolist())
judge_prefecture_points.name = 'prefecture'
judge_prefecture_no_points = log_judge_no_points.groupby('case:concept:name').apply(lambda x: 'Send Appeal to Prefecture' in x['concept:name'].tolist())
judge_prefecture_no_points.name = 'prefecture'

judge_prefecture_df = judge_prefecture_points.to_frame().copy()
judge_prefecture_df.insert(1, 'type', 'points')
judge_prefecture_no_points = judge_prefecture_no_points.to_frame()
judge_prefecture_no_points.insert(1, 'type', 'no-points')
judge_prefecture_df = pd.concat([judge_prefecture_df, judge_prefecture_no_points])

fig = px.histogram(judge_prefecture_df, 'prefecture', color='type', barmode='group', histnorm='probability density')
fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                 showline=True, linewidth=2, linecolor='black')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
fig.show()
