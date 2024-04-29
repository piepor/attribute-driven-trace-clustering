""" 
This module contains all the functions for data and models visualization
"""

from typing import Tuple
import copy
import copy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.directly_follow_graph import create_graphviz_dashed_arcs, create_graphviz_delete_arcs
from utils.generals import compute_percentage, create_daily_time_series, create_timeseries_compliance_df, compute_percentage
from graphviz import Digraph

def create_dot(dfg, start_activities, end_activities, filter_mode='dashed', filter_perc=0.0) -> Digraph:
    """
    creates the graphviz object. Two visualization available: with non frequent arcs
    deleted or represented as red dashed lines
    """
    if filter_mode == 'dashed':
        dot = create_graphviz_dashed_arcs(
                dfg, start_activities, end_activities, filter_perc
                )
    elif filter_mode == 'delete':
        dot = create_graphviz_delete_arcs(
                dfg, start_activities, end_activities, filter_perc
                )
    else:
        raise NotImplemented("Filter mode can only be 'dashed' or 'delete'")
    return dot

def draw_dfg(dfg, start_activities, end_activities, save_path,
             perc_filt=0.05, view=False, dfg_type='dashed'):
    """ 
    draw a directly follow graph and saves it. It can also be directly viewed.
    """
    dot = create_dot(dfg, start_activities, end_activities, dfg_type, perc_filt)
    dot.format = 'svg'
    dot.render(save_path, view=view)

def make_attributes_figures(cluster_df: pd.DataFrame, continuos_cols: list) -> go.Figure:
    # figs = {}
    fig = make_subplots(rows=len(cluster_df.columns), cols=1)
    for i, column in enumerate(cluster_df):
        if column in continuos_cols:
            fig.add_trace(go.Histogram(x=cluster_df[column].tolist(), name=column), row=i+1, col=1)
            # figs[f'{column}'] = go.Histogram(x=cluster_df[column].tolist(), name=column)
        else:
            if column == 'DESCR_GRADO':
                fig_col = go.Figure()
                for col_value in cluster_df["CLUSTER_MATERIALE"].unique():
                    filter_col = cluster_df[cluster_df["CLUSTER_MATERIALE"] == col_value]
                    fig.add_trace(go.Histogram(x=filter_col[column].tolist(), name=col_value), row=i+1, col=1)
                    fig_col.add_trace(go.Histogram(x=filter_col[column].tolist(), name=col_value))
            else:
                fig_col = fig.add_trace(go.Histogram(x=cluster_df[column].astype(str).tolist(), name=column), row=i+1, col=1)
            # figs[f'{column}'] = fig_col
    # figs['overall'] = fig
    return fig

def make_attributes_overall_figures(data: pd.DataFrame, continuos_cols: list, column: str) -> go.Figure:
    data_columns = data.drop(columns=column).columns
    fig = make_subplots(rows=len(data_columns), cols=1, subplot_titles=data_columns)
    default_colors = px.colors.qualitative.Plotly
    if len(data[column].unique()) > len(default_colors):
        default_colors = px.colors.qualitative.Alphabet
    clusters_colors = {}
    for i, cluster in enumerate(data[column].unique()):
        clusters_colors[cluster] = default_colors[i]
    for i, attr in enumerate(data_columns):
        showlegend = i == 0
        for cluster in data[column].unique():
            data_cluster = data[data[column] == cluster]
            if attr in continuos_cols:
                fig.add_trace(go.Histogram(
                    x=data_cluster[attr].tolist(), name=cluster,
                    marker=dict(color=clusters_colors[cluster]), showlegend=showlegend), row=i+1, col=1)
            else:
                fig.add_trace(go.Histogram(
                    x=data_cluster[attr].astype(str).tolist(), name=cluster,
                    marker=dict(color=clusters_colors[cluster]), showlegend=showlegend), row=i+1, col=1)
    fig.update_layout(barmode='group')
    return fig

def make_trace_attributes_figures(trace_attributes_df: pd.DataFrame, continuous_cols: list, column: str) -> dict:
    figures = {}
    column_order = list(trace_attributes_df[column].unique())
    column_order.sort()
    for cluster in trace_attributes_df[column].unique():
        trace_attributes_cluster = trace_attributes_df[trace_attributes_df[column] == cluster]
        fig = make_attributes_figures(trace_attributes_cluster, continuous_cols)
        # for fig in figs:
        fig.update_layout(title=f'{cluster}')
        figures[cluster] = copy.deepcopy(fig)
    fig = make_attributes_overall_figures(trace_attributes_df, continuous_cols, column)
    fig.update_layout(title='overall')
    figures['overall'] = copy.deepcopy(fig)
    for attr in continuous_cols:
        attr_order = list(trace_attributes_df[attr].unique())
        attr_order.sort()
        figures[f'{column}_{attr}'] = px.box(trace_attributes_df[[column, attr]], x=column, y=attr,
                                             category_orders={column: column_order})
        figures[f'{column}_{attr}'].update_layout(title=f'{column} {attr}')
        figures[f'{column}_{attr}_log'] = px.box(trace_attributes_df[[column, attr]], x=column, y=attr, 
                                                 log_y=True, category_orders={column: column_order})
        figures[f'{column}_{attr}_log'].update_layout(title=f'{column} {attr} logarithmic')
    columns = set(trace_attributes_df.columns)
    categorical_cols = columns.difference(continuous_cols)
    for attr in categorical_cols:
        if not attr in ['cluster', 'subcluster']:
            attr_order = list(trace_attributes_df[attr].unique())
            attr_order = attr_order.sort()
            trace_attributes_df[attr] = trace_attributes_df[attr].astype(str)
            figures[f'{column}_{attr}'] = px.histogram(trace_attributes_df, x=attr, color=column,
                                                       barmode='group', category_orders={attr: attr_order})
            figures[f'{column}_{attr}'].update_layout(title=f'{column} {attr}')
            data_perc = compute_percentage(trace_attributes_df[[column, attr]], column, attr)
            figures[f'{column}-{attr}-perc'] = px.bar(data_perc, x=attr, y='perc', color=column, 
                                                      barmode='group', category_orders={attr: attr_order})
    return figures

def make_amount_figure_bpic(trace_attributes_df: pd.DataFrame, column: str) -> go.Figure:
    amount = trace_attributes_df[['Cumulative net worth (EUR)', column]].pivot_table(index=column, aggfunc='sum').reset_index()
    column_order = list(trace_attributes_df[column].unique())
    column_order.sort()
    return px.bar(amount, x=column, y='Cumulative net worth (EUR)', category_orders={column: column_order})

def make_amount_per_trace_figure_bpic(trace_attributes_df: pd.DataFrame) -> go.Figure:
    trace_attributes_df = copy.deepcopy(trace_attributes_df)
    amount = trace_attributes_df[['Cumulative net worth (EUR)', 'cluster']].pivot_table(index='cluster', aggfunc='sum')
    size = trace_attributes_df[['Cumulative net worth (EUR)', 'cluster']].rename(
            columns={'Cumulative net worth (EUR)': 'size'})
    size = size.pivot_table(index='cluster', aggfunc='count')
    amount_per_trace = amount.join(size)
    amount_per_trace['ratio'] = amount_per_trace['Cumulative net worth (EUR)'] / amount_per_trace['size']
    amount_per_trace = amount_per_trace.reset_index()
    return px.bar(amount_per_trace, x='cluster', y='ratio')

def make_cluster_activities_distribution_figures(distribution: pd.DataFrame) -> go.Figure:
    act_order = list(distribution['activity'].unique())
    act_order.sort()
    return px.bar(distribution, x='activity', y='perc', color='cluster', barmode='group', category_orders={'activity': act_order})

def make_subcluster_activities_distribution_figures(distribution: pd.DataFrame) -> dict:
    figures = {}
    act_order = list(distribution['activity'].unique())
    act_order.sort()
    for cluster in distribution['cluster'].unique():
        data = distribution[distribution['cluster'] == cluster]
        fig = px.bar(data, x='activity', y='perc', color='subcluster', 
                     barmode='group', category_orders={'activity': act_order})
        fig.update_layout(title=f'{cluster}')
        figures[f'{cluster}_acivities_distribution'] = copy.deepcopy(fig)
    return figures

def make_variants_distribution_figures(variants_distr_clusters: dict) -> dict:
    figures = {}
    for cluster in variants_distr_clusters:
        lengths_variants = variants_distr_clusters[cluster]['lengths']
        lengths_variants_perc = variants_distr_clusters[cluster]['lengths_perc']
        colors = variants_distr_clusters[cluster]['colors']
        coverage = variants_distr_clusters[cluster]['coverage'] 
        fig = go.Figure()
        fig.add_trace(go.Bar(y=lengths_variants, marker_color=colors))
        fig.update_layout(title=cluster)
        figures[f'variants-length-{cluster}-{coverage}'] = fig
        fig = go.Figure()
        fig.add_trace(go.Bar(y=lengths_variants_perc, marker_color=colors))
        fig.update_layout(title=cluster)
        figures[f'variants-length-percentage-{cluster}-{coverage}'] = fig
    return figures

def make_sizes_figures(clusters_sizes: pd.DataFrame, subclusters_sizes: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    fig_subclusters = px.bar(subclusters_sizes, x='cluster', y='size')
    fig_subclusters.update_layout(title='subclusters size')
    fig_clusters = px.bar(clusters_sizes, x='cluster', y='size')
    fig_clusters.update_layout(title='clusters size')
    return fig_clusters, fig_subclusters

def make_variants_sharing_between_subcluster_figures(variants_sharing: pd.DataFrame) -> go.Figure:
    variants_sharing = variants_sharing[variants_sharing['cluster_x'] != variants_sharing['cluster_y']]
    fig = px.box(variants_sharing, x='cluster_x', y='sharing')
    fig.update_layout(title='percentage of shared variants')
    return fig

def make_variants_sharing_between_cluster_figures(variants_sharing: pd.DataFrame) -> go.Figure:
    variants_sharing = variants_sharing[variants_sharing['cluster_x'] != variants_sharing['cluster_y']]
    fig = px.box(variants_sharing, x='cluster_x', y='sharing')
    fig.update_layout(title='percentage of shared variants')
    return fig

def make_similarities_figures(similarities: pd.DataFrame) -> go.Figure:
    return px.box(similarities, x='cluster_x', y='value', color='distance_considered') 

def make_start_acts_distr_figures(start_activities: pd.DataFrame, level: str) -> go.Figure:
    return px.bar(start_activities, x='start_activity', y='perc', color=level, barmode='group')

def make_end_acts_distr_figures(end_activities: pd.DataFrame, level: str) -> go.Figure:
    return px.bar(end_activities, x='end_activity', y='perc', color=level, barmode='group')

def make_throughput_distribution_figures(performance: pd.DataFrame) -> dict:
    figures = {}
    figures['throughput_clusters'] = px.histogram(performance, x='throughput', color='cluster', barmode='group')
    figures['throughput_clusters_facet'] = px.histogram(performance, x='throughput', color='cluster', barmode='group', facet_row='cluster')
    figures['throughput_clusters_box'] = px.box(performance, x='cluster', y='throughput')
    figures['throughput_subclusters'] = px.histogram(performance, x='throughput', color='subcluster', barmode='group')
    figures['throughput_subclusters_box'] = px.box(performance, x='subcluster', y='throughput')
    for cluster in performance['cluster'].unique():
        cluster_perf = performance[performance['cluster']==cluster]
        fig = px.histogram(cluster_perf, x='throughput', color='subcluster', barmode='group', facet_row='subcluster')
        fig.update_yaxes(matches=None)
        figures[f'{cluster}-throughput_subclusters_facet'] = fig
    if 'compliance' in performance.columns:
        figures['throughput_clusters_compliance'] = px.histogram(
                performance, x='throughput', color='compliance_type', barmode='group', facet_col='cluster', facet_col_wrap=4)
        figures['throughput_clusters_compliance'].update_yaxes(matches=None, showticklabels=True)
        figures['throughput_clusters_compliance_boxplot'] = px.box(
                performance, x='cluster', y='throughput', color='compliance')
        figures['throughput_subclusters_compliance'] = px.histogram(
                performance, x='throughput', color='compliance_type', barmode='group', facet_col='subcluster', facet_col_wrap=4)
        figures['throughput_subclusters_compliance'].update_yaxes(matches=None, showticklabels=True)
        figures['throughput_subclusters_compliance_boxplot'] = px.box(
                performance, x='subcluster', y='throughput', color='compliance')
    return figures

def make_models_performances_figure(results: pd.DataFrame) -> go.Figure:
    return px.bar(results, x='model_clstr', y='f1_score', color='coverage')

def make_compliance_figures(performance: pd.DataFrame) -> dict:
    figures = {}
    perc_cluster = compute_percentage(copy.deepcopy(performance), 'cluster', 'compliance')
    figures['compliance_cluster'] = px.bar(perc_cluster, x='cluster', y='perc', color='compliance')
    perc_subcluster = compute_percentage(copy.deepcopy(performance), 'subcluster', 'compliance')
    figures['compliance_subcluster'] = px.bar(perc_subcluster, x='subcluster', y='perc', color='compliance')
    return figures

def make_temporal_distribution_figures(performance: pd.DataFrame) -> dict:
    figures = {}
    time_series = create_daily_time_series(copy.deepcopy(performance), 'cluster')
    fig = px.line(time_series, x='start_date', y='counter', facet_row='cluster', markers=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    figures['temporal_distribution_cluster'] = fig 
    time_series = create_daily_time_series(copy.deepcopy(performance), 'subcluster')
    fig = px.line(time_series, x='start_date', y='counter', facet_row='subcluster', markers=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    figures['temporal_distribution_subcluster'] = fig 
    # compliant_df = performance[performance['compliance'] == 'compliant']
    # compliant_time_series = create_daily_time_series(copy.deepcopy(compliant_df), 'cluster')
    # compliant_time_series.insert(len(compliant_time_series.columns), 'compliance', 'compliant')
    # incompliant_df = performance[performance['compliance'] == 'incompliant']
    # incompliant_time_series = create_daily_time_series(copy.deepcopy(incompliant_df), 'cluster')
    # incompliant_time_series.insert(len(incompliant_time_series.columns), 'compliance', 'incompliant')
    # time_series_compliance = pd.concat(
    #         [compliant_time_series[['cluster', 'compliance', 'counter', 'start_date']],
    #          incompliant_time_series[['cluster', 'compliance', 'counter', 'start_date']]], axis=0)
    if 'compliance' in performance.columns:
        time_series_compliance = create_timeseries_compliance_df(performance, 'cluster')
        fig = px.line(time_series_compliance, x='start_date', y='counter', color='compliance', facet_row='cluster', markers=True)
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_layout(title='cluster compliance - time series')
        figures['temporal_distribution_compliance_cluster'] = fig 
        time_series_compliance = create_timeseries_compliance_df(performance, 'subcluster')
        fig = px.line(time_series_compliance, x='start_date', y='counter', color='compliance', facet_row='subcluster', markers=True)
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_layout(title='subcluster compliance - time series')
        figures['temporal_distribution_compliance_subcluster'] = fig 
    return figures

def make_activities_divergence_figures(divergences: pd.DataFrame) -> go.Figure:
    return px.box(divergences, x='cluster_x', y='divergence')
