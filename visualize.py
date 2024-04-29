import os
from visualizers import VisualizeClusters
from loaders import MultilevelTraceAttributesVariants
from analyzer import Analyzer
from distances import MultiviewDistanceFunction
from utils.logs import LogUtilities
from configs import ConfigVisualize

def visualize(configs: ConfigVisualize):
    if not configs.model_dir:
        raise ValueError("Model directory not specified.")
    save_dir = os.path.join(configs.model_dir, 'figures')
    print("Loading utilities")
    log_utils = LogUtilities(configs.log_path)
    print("Loading logs")
    loader = MultilevelTraceAttributesVariants(configs, log_utils)
    print("Loading distance")
    distance = MultiviewDistanceFunction(loader, log_utils)
    print("Loading analyzer")
    analyzer = Analyzer(loader, distance, log_utils, os.path.join(configs.model_dir, 'results'))
    print("Loading visualizer")
    visualizer = VisualizeClusters(analyzer, save_dir)
    print("Plotting")
    visualizer.plot_trace_attributes_clusters(configs.view)
    visualizer.plot_trace_attributes_subclusters(configs.view)
    visualizer.plot_activities_distribution(configs.view)
    visualizer.plot_clusters_size(configs.view)
    visualizer.plot_variants_sharing(configs.view)
    visualizer.plot_variants_distribution(configs.variant_coverage, configs.view)
    visualizer.plot_similarities(configs.view)
    visualizer.plot_performance(configs.view, configs.compute_models_performance)
    visualizer.plot_divergence(configs.view)
