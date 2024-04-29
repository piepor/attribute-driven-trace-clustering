"""
This module contains classes to visualize results
"""

import os
import copy
import plotly.express as px
from utils.visualizing import make_activities_divergence_figures, make_amount_per_trace_figure_bpic, make_cluster_activities_distribution_figures, make_compliance_figures, make_end_acts_distr_figures, make_similarities_figures, make_sizes_figures, make_start_acts_distr_figures, make_temporal_distribution_figures, make_throughput_distribution_figures
from utils.visualizing import make_subcluster_activities_distribution_figures, make_trace_attributes_figures, make_variants_distribution_figures
from utils.visualizing import make_variants_sharing_between_cluster_figures, make_variants_sharing_between_subcluster_figures 
from utils.visualizing import make_models_performances_figure, make_amount_figure_bpic
from utils.generals import compute_percentage
from analyzer import Analyzer

class VisualizeClusters:
    def __init__(self, analyzer: Analyzer, save_dir: str) -> None:
        self.analyzer = analyzer
        self.save_dir = save_dir
        self.save_html = os.path.join(save_dir, 'html')
        self.save_svg = os.path.join(save_dir, 'svg')
        self.continuous_cols = self.analyzer.get_continuous_attrs()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            os.mkdir(self.save_html)
            os.mkdir(self.save_svg)

    def plot_trace_attributes_clusters(self, view: bool=False):
        figures = make_trace_attributes_figures(
                self.analyzer.get_traces_attributes().drop(columns='subcluster'), self.continuous_cols, 'cluster')
        for title in figures:
            if view:
                figures[title].show()
            figures[title].update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                                        showline=True, linewidth=2, linecolor='black')
            figures[title].update_xaxes(showline=True, linewidth=2, linecolor='black')
            figures[title].update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
            figures[title].update_layout(legend=dict(
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            figures[title].write_html(os.path.join(self.save_html, f"{title}.html"))
            figures[title].write_image(os.path.join(self.save_svg, f"{title}.svg"))
        if self.analyzer.log_name == 'BPI_Challenge_2019':
            fig = make_amount_figure_bpic(self.analyzer.get_traces_attributes(), 'cluster')
            if view:
                fig.show()
            fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                             showline=True, linewidth=2, linecolor='black')
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
            fig.update_layout(legend=dict(
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            fig.write_html(os.path.join(self.save_html, f"amount-bpic-clusters.html"))
            fig.write_image(os.path.join(self.save_svg, f"amount-bpic-clusters.svg"))
            fig = make_amount_per_trace_figure_bpic(self.analyzer.get_traces_attributes())
            if view:
                fig.show()
            fig.write_html(os.path.join(self.save_html, f"amount-per-trace-bpic-clusters.html"))
            fig.write_image(os.path.join(self.save_svg, f"amount-per-trace-bpic-clusters.svg"))

    def plot_trace_attributes_subclusters(self, view: bool=False):
        trace_attributes = self.analyzer.get_traces_attributes()
        for cluster in trace_attributes['cluster'].unique():
            trace_attrs = trace_attributes[trace_attributes['cluster'] == cluster]
            figures = make_trace_attributes_figures(
                    trace_attrs.drop(columns='cluster'), self.continuous_cols, 'subcluster')
            for title in figures:
                name_fig = copy.copy(f"{cluster}-{title}")
                if view:
                    figures[title].show()
                if title == 'overall':
                    name_fig = f'{cluster}-overall'
                figures[title].update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                                            showline=True, linewidth=2, linecolor='black')
                figures[title].update_xaxes(showline=True, linewidth=2, linecolor='black')
                figures[title].update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
                figures[title].update_layout(legend=dict(
                    orientation="h",
                    entrywidth=70,
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
                figures[title].write_html(os.path.join(self.save_html, f"{name_fig}.html"))
                figures[title].write_image(os.path.join(self.save_svg, f"{name_fig}.svg"))
        cat_cols = set(trace_attributes.columns).difference(set(self.continuous_cols))
        cat_cols = cat_cols.difference({'cluster', 'subcluster'})
        for cat_col in cat_cols:
            trace_attributes[cat_col] = trace_attributes[cat_col].astype(str)
            data_perc = compute_percentage(trace_attributes, 'subcluster', cat_col)
            fig = px.bar(data_perc, x=cat_col, y='perc', color='subcluster', barmode='group')
            fig.write_html(os.path.join(self.save_html, f"overall-subcluster-{cat_col}.html"))
            fig.write_image(os.path.join(self.save_svg, f"overall-subcluster-{cat_col}.svg"))
        if self.analyzer.log_name == 'BPI_Challenge_2019':
            fig = make_amount_figure_bpic(self.analyzer.get_traces_attributes(), 'subcluster')
            if view:
                fig.show()
            fig.write_html(os.path.join(self.save_html, f"amount-bpic-subclusters.html"))
            fig.write_image(os.path.join(self.save_svg, f"amount-bpic-subclusters.svg"))

    def plot_activities_distribution(self, view: bool=False):
        figures = {}
        distr_subclusters, distr_cluster = self.analyzer.get_activities_distribution()
        figures['overall_activities_distribution'] = make_cluster_activities_distribution_figures(distr_cluster)
        figures.update(make_subcluster_activities_distribution_figures(distr_subclusters))
        start_acts_distr_cluster = self.analyzer.distribution_start_activities_cluster()
        figures['clusters_start_activities_distribution'] = make_start_acts_distr_figures(start_acts_distr_cluster, 'cluster')
        start_acts_distr_subcluster = self.analyzer.distribution_start_activities_subcluster()
        figures['subclusters_start_activities_distribution'] = make_start_acts_distr_figures(start_acts_distr_subcluster, 'subcluster')
        end_acts_distr_cluster = self.analyzer.distribution_end_activities_cluster()
        figures['clusters_end_activities_distribution'] = make_end_acts_distr_figures(end_acts_distr_cluster, 'cluster')
        end_acts_distr_subcluster = self.analyzer.distribution_end_activities_subcluster()
        figures['subclusters_end_activities_distribution'] = make_end_acts_distr_figures(end_acts_distr_subcluster, 'subcluster')
        for title in figures:
            fig = figures[title]
            name = title.replace('_', '-')
            if view:
                fig.show()
            figures[title].update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                                        showline=True, linewidth=2, linecolor='black')
            figures[title].update_xaxes(showline=True, linewidth=2, linecolor='black')
            figures[title].update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
            figures[title].update_layout(legend=dict(
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            fig.write_html(os.path.join(self.save_html, f'{name}.html'))
            fig.write_image(os.path.join(self.save_svg, f'{name}.svg'))

    def plot_clusters_size(self, view: bool=False):
        clusters_sizes, subcluster_sizes = self.analyzer.get_sizes()
        fig_clusters, fig_subclusters = make_sizes_figures(clusters_sizes, subcluster_sizes)
        if view:
            fig_clusters.show()
            fig_subclusters.show()
        fig_clusters.write_html(os.path.join(self.save_html, f'clusters-sizes.html'))
        fig_subclusters.write_html(os.path.join(self.save_html, f'subclusters-sizes.html'))
        fig_clusters.write_image(os.path.join(self.save_svg, f'clusters-sizes.svg'))
        fig_subclusters.write_image(os.path.join(self.save_svg, f'subclusters-sizes.svg'))
        # if self.analyzer.log_name == 'BPI_Challenge_2019':
        #     fig = make_size_amount_figure_bpic()

    def plot_variants_sharing(self, view: bool=False):
        variants_sharing_clusters = self.analyzer.get_variants_sharing_clusters()
        fig_clusters = make_variants_sharing_between_cluster_figures(variants_sharing_clusters)
        variants_sharing_subclusters = self.analyzer.get_variants_sharing_subclusters()
        fig_subclusters = make_variants_sharing_between_subcluster_figures(variants_sharing_subclusters)
        if view:
            fig_clusters.show()
            fig_subclusters.show()
        fig_clusters.write_html(os.path.join(self.save_html, f'variants-sharing-clusters.html'))
        fig_subclusters.write_html(os.path.join(self.save_html, f'variants-sharing-subclusters.html'))
        fig_clusters.write_image(os.path.join(self.save_svg, f'variants-sharing-clusters.svg'))
        fig_subclusters.write_image(os.path.join(self.save_svg, f'variants-sharing-subclusters.svg'))

    def plot_variants_distribution(self, coverage: float=1.0, view: bool=False):
        divergences = self.analyzer.get_variants_distribution(coverage)
        figures = make_variants_distribution_figures(divergences)
        for title in figures:
            fig = figures[title]
            name = title.replace('_', '-')
            if view:
                fig.show()
            fig.write_html(os.path.join(self.save_html, f'{name}.html'))
            fig.write_image(os.path.join(self.save_svg, f'{name}.svg'))

    def plot_similarities(self, view: bool=False):
        similarities = self.analyzer.get_similarities()
        fig = make_similarities_figures(similarities)
        if view:
            fig.show()
        fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                         showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_layout(legend=dict(
            orientation="h",
            entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig.write_html(os.path.join(self.save_html, f'similarities.html'))
        fig.write_image(os.path.join(self.save_svg, f'similarities.svg'))

    def plot_performance(self, view: bool=False, compute_mod_perf: bool=False):
        performance = self.analyzer.get_performance()
        figures = make_throughput_distribution_figures(copy.deepcopy(performance))
        if 'compliance' in performance:
            figures.update(make_compliance_figures(copy.deepcopy(performance)))
        figures.update(make_temporal_distribution_figures(copy.deepcopy(performance)))
        for title in figures:
            fig = figures[title]
            name = title.replace('_', '-')
            if view:
                fig.show()
            figures[title].update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                                        showline=True, linewidth=2, linecolor='black')
            figures[title].update_xaxes(showline=True, linewidth=2, linecolor='black')
            figures[title].update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
            figures[title].update_layout(legend=dict(
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            fig.write_html(os.path.join(self.save_html, f'{name}.html'))
            fig.write_image(os.path.join(self.save_svg, f'{name}.svg'))
        models_performance = self.analyzer.get_models_performance(compute=compute_mod_perf)
        fig = make_models_performances_figure(models_performance)
        if view:
            fig.show()
        fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                         showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_layout(legend=dict(
            orientation="h",
            entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig.write_html(os.path.join(self.save_html, f'models_performance.html'))
        fig.write_image(os.path.join(self.save_svg, f'models_performance.svg'))

    def plot_divergence(self, view: bool=False):
        divergences = self.analyzer.get_activites_divergence()
        fig = make_activities_divergence_figures(divergences)
        if view:
            fig.show()
        fig.write_html(os.path.join(self.save_html, 'activities-divergences.html'))
        fig.write_image(os.path.join(self.save_svg, 'activities-divergences.svg'))

    # def draw_clusters_dfg(self):
    #     for cluster in self.clusters:
    #         log_dfg = copy.deepcopy(self.clusters[cluster])
    #         for trace in log_dfg:
    #             for event in trace:
    #                 event['concept:name'] = event['concept:name'].replace(':', ' -')
    #         dfg, start_activities, end_activities = pm4py.discover_dfg(log_dfg)
    #         draw_dfg(dfg, start_activities, end_activities, os.path.join(self.save_dir, cluster))
