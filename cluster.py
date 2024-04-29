import os
from algos import MultiLevelAttributesThenVariants
from configs import ConfigCluster
from utils.logs import LogUtilities

def cluster(configs: ConfigCluster):
    log_utils = LogUtilities(configs.log_path)
    log_sequence_path = configs.log_path
    log_attribute_path = "{}-trace-attributes.csv".format(configs.log_path.split('.xes')[0])
    if configs.elbow_curve_first_level:
        clustering = MultiLevelAttributesThenVariants(
                log_sequence_path, 
                log_attribute_path, 
                './cluster-models',
                attributes_dates=True,
                dates_col=log_utils.date_col,
                filter_lifecycle=log_utils.only_lifecycle_start
                )
    else:
        clustering = MultiLevelAttributesThenVariants(
                log_sequence_path, 
                log_attribute_path, 
                './cluster-models',
                model_dir=configs.model_dir,
                attributes_dates=True,
                dates_col=log_utils.date_col,
                filter_lifecycle=log_utils.only_lifecycle_start
                )
    if log_utils.filter_time:
        clustering.filter_first_level(log_utils.date_col, log_utils.start_date, log_utils.end_date)
        clustering.filter_second_level(log_utils.date_col, log_utils.start_date, log_utils.end_date)

    if configs.elbow_curve_first_level:
        clustering.elbow_curve_first_level(configs.num_points_elbow_curve_first_level)
    elif configs.elbow_curve_second_level:
        clustering.elbow_curve_second_level(
                configs.n_clusters_first_level, 
                os.path.join(clustering.save_dir, 'attributes', 'clusters', 'attributes.csv'),
                configs.num_points_elbow_curve_second_level)

    if configs.cluster_first_level:
        clustering.set_n_clusters_first_level(configs.n_clusters_first_level)
        clustering.clusterize_first_level()
    if configs.cluster_second_level:
        first_level_clstr = os.path.join(clustering.save_dir, 'attributes', 'clusters', 'attributes.csv')
        clustering.set_first_level_cluster_path(first_level_clstr)
        clustering.set_n_clusters_second_level(configs.n_clusters_second_level)
        clustering.clusterize_second_level()
