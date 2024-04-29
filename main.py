"""
Main functions
"""

import argparse
from visualize import visualize
from cluster import cluster
from configs import ConfigVisualize, ConfigCluster, ConfigMain, Mode


DATASET_NAME = "bpic19"
#MODE = 'cluster'
MODE = 'vis'
mode_map = {'cluster': Mode.CLUSTER, 'vis': Mode.VISUALIZE}

# clustering options

ELBOW_CURVE_FIRST_LEVEL = False
NUM_POINTS_ELBOW_CURVE_FIRST_LEVEL = 10
NUM_POINTS_ELBOW_CURVE_SECOND_LEVEL = 10
ELBOW_CURVE_SECOND_LEVEL = False
CLUSTER_FIRST_LEVEL = False
CLUSTER_SECOND_LEVEL = False
N_CLUSTERS_FIRST_LEVEL = 4
N_CLUSTERS_SECOND_LEVEL = [4, 3, 3, 3]

# visualize options
COMPUTE_MODEL_PERFORMANCE = True
VIEW = False

MODEL_NAME_FINES = "2023-03-07T18-59-18" # traffic fines
MODEL_NAME_BPIC_19 = "2023-02-27T09-27-29" # bpic 2019 filtered
MODEL_NAME_BPIC_17 = "2024-03-12T12-26-07" # bpic 2017 filtered
MODEL_NAME_BPIC_12 = "2024-03-12T10-29-33" # bpic 2012 filtered
LOG_PATH_FINES = './data/road_traffic_fine_management_process.xes'
LOG_PATH_BPIC_19 = './data/BPI_Challenge_2019-max_length_45.xes'
LOG_PATH_BPIC_17 = './data/BPI_Challenge_2017-max_length_45.xes'
LOG_PATH_BPIC_12 = './data/BPI_Challenge_2012-max_length_45.xes'

dataset_map = {'fines': LOG_PATH_FINES, 'bpic12': LOG_PATH_BPIC_12,
               'bpic17': LOG_PATH_BPIC_17, 'bpic19': LOG_PATH_BPIC_19}

model_map = {'fines': MODEL_NAME_FINES, 'bpic2012': MODEL_NAME_BPIC_12,
             'bpic2017': MODEL_NAME_BPIC_17, 'bpic2019': MODEL_NAME_BPIC_19}

def main(config: ConfigMain):
    if config.dataset == "bpic19":
        MODEL_NAME = MODEL_NAME_BPIC_19
        LOG_PATH = LOG_PATH_BPIC_19
    elif config.dataset == "fines":
        MODEL_NAME = MODEL_NAME_FINES
        LOG_PATH = LOG_PATH_FINES
    elif config.dataset == "bpic17":
        MODEL_NAME = MODEL_NAME_BPIC_17
        LOG_PATH = LOG_PATH_BPIC_17
    elif config.dataset == "bpic12":
        MODEL_NAME = MODEL_NAME_BPIC_12
        LOG_PATH = LOG_PATH_BPIC_12
    else:
        raise NotImplementedError("Dataset not implemented")
    if config.mode == Mode.VISUALIZE:
        MODEL_DIR = f'./cluster-models/{MODEL_NAME}/'
        configs = ConfigVisualize(
                model_dir=MODEL_DIR, log_path=LOG_PATH, view=VIEW, 
                variant_coverage=1.0, compute_models_performance=COMPUTE_MODEL_PERFORMANCE)
        visualize(configs)
    elif config.mode == Mode.CLUSTER:
        if ELBOW_CURVE_FIRST_LEVEL:
            MODEL_DIR = ""
        else:
            MODEL_DIR = MODEL_NAME
        configs = ConfigCluster(
                model_dir=MODEL_DIR, log_path=LOG_PATH,
                elbow_curve_first_level=config.elbow_curve_first_level, 
                num_points_elbow_curve_first_level=NUM_POINTS_ELBOW_CURVE_FIRST_LEVEL,
                elbow_curve_second_level=config.elbow_curve_second_level,
                num_points_elbow_curve_second_level=NUM_POINTS_ELBOW_CURVE_SECOND_LEVEL,
                cluster_first_level=config.cluster_first_level,
                cluster_second_level=config.cluster_second_level,
                n_clusters_first_level=N_CLUSTERS_FIRST_LEVEL,
                n_clusters_second_level=N_CLUSTERS_SECOND_LEVEL,
                )
        cluster(configs)
    else:
        raise NotImplementedError('Mode not available.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset considered to process', 
                        choices=['bpic2012', 'bpic2017', 'bpic2019', 'fines'])
    parser.add_argument('mode', help='Mode to launch the program: clustering or visualize results',
                        choices=['cluster', 'vis'])
    parser.add_argument('--elbow_curve_first_level', help='Compute the elbow curve for the first level',
                        default=False, type=bool)
    parser.add_argument('--elbow_curve_second_level', help='Compute the elbow curve for every cluster in the second level',
                        default=False, type=bool)
    parser.add_argument('--cluster_first_level', help='clusters the first level',
                        default=False, type=bool)
    parser.add_argument('--cluster_second_level', help='clusters each first level cluster',
                        default=False, type=bool)
    args = parser.parse_args()
    mode = mode_map[args.mode]
    config = ConfigMain(
            dataset=args.dataset,
            mode=mode,
            elbow_curve_first_level=args.elbow_curve_first_level,
            elbow_curve_second_level=args.elbow_curve_second_level,
            cluster_first_level=args.cluster_first_level,
            cluster_second_level=args.cluster_second_level
            )
    main(config)
