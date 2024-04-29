import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import pm4py
from typing import Tuple
from pm4py.objects.log.obj import EventLog
from fastDamerauLevenshtein import damerauLevenshtein as dam_lev_dist
from tqdm import tqdm
from utils.generals import extract_variant

def get_total_distance(data: np.ndarray, maxs: dict) -> np.ndarray:
    cat_dist = data[:, 0]
    cont_dist = data[:, 1]
    seq_dist = data[:, 2]
    if not maxs['categorical'] == 0:
        cat_dist = data[:, 0]/maxs['categorical']
    if not maxs['continuous'] == 0:
        cont_dist = data[:, 1]/maxs['continuous']
    if not maxs['sequence'] == 0:
        seq_dist = data[:, 2]/maxs['sequence']
    distances = cat_dist + cont_dist + seq_dist
    return distances

def get_distances_matrix(cat_matrix: np.ndarray, cont_matrix: np.ndarray, seq_matrix: np.ndarray) -> np.ndarray:
    distances_matrix = np.zeros((len(cat_matrix), 3))
    distances_matrix[:, 0] = cat_matrix[:, 0]
    distances_matrix[:, 1] = cont_matrix[:, 0]
    distances_matrix[:, 2] = seq_matrix[:, 0]
    return distances_matrix

def get_centroid_id(distances: np.ndarray, case_ids: list) -> str:
    pos_min = np.argmin(distances)
    return case_ids[pos_min]

def categorical_distance(x: list, y: list) -> int:
    if len(x) != len(y):
        raise ValueError("Dimension mismatch. Provide two lists with the same number of elements")
    counter = 0
    for el_x, el_y in zip(x, y):
        if el_x != el_y:
            counter += 1
    return counter

def continuous_distance(x: list, y: list) -> int:
    if len(x) != len(y):
        raise ValueError("Dimension mismatch. Provide two lists with the same number of elements")
    return np.sqrt(np.sum(np.power(np.array(x) - np.array(y), 2)))

def sequence_distance(x, y):
    return dam_lev_dist(x, y, swapWeight=0, replaceWeight=2, similarity=False)

def map_variants_distances(set_x: EventLog, set_y: EventLog) -> dict:
    variants_x = pm4py.get_variants_as_tuples(set_x)
    variants_y = pm4py.get_variants_as_tuples(set_y)
    variants_distances = {}
    for variant_x in variants_x:
        for variant_y in variants_y:
            seq_distance = sequence_distance(variant_x, variant_y)
            variants_distances[(variant_x, variant_y)] = seq_distance
    return variants_distances

def compute_sequence_distances(set_x: EventLog, set_y: EventLog, intercluster=True) -> np.ndarray:
    variants_distances = map_variants_distances(set_x, set_y)
    sequences_distances = []
    variants_x = []
    size = len(set_y)
    if not intercluster:
        size = len(set_y) - 1
    for trace in set_x:
        variants_x.append(tuple(extract_variant(trace)))
    variants_y = []
    for trace in set_y:
        variants_y.append(tuple(extract_variant(trace)))
    for pos_x, _ in tqdm(enumerate(set_x), total=len(set_x)):
        avg_dist = 0
        max_dist = 0
        min_dist = -1
        # variant_x = extract_variant(trace_x)
        variant_x = variants_x[pos_x]
        for pos_y, _ in enumerate(set_y):
            # variant_y = extract_variant(trace_y)
            variant_y = variants_y[pos_y]
            distance = variants_distances[(variant_x, variant_y)]
            avg_dist += distance 
            if distance > max_dist:
                max_dist = distance
            if distance < min_dist or min_dist == -1:
                min_dist = distance
        sequences_distances.append([avg_dist/size, max_dist, min_dist]) 
    return np.asarray(sequences_distances)

def compute_distances_on_gpu(set_x: np.ndarray, set_y: np.ndarray, continuous_num: int, intercluster=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    computes continuous and categorical distances between two sets.
    The input matrix must have [continuous_attributes, categorical_attributes] as columns (in this order)
    The continuous part must be normalized between 0 and 1. 
    The categorical part must be integers indicating the category.
    Returns a tuple with (categorical_distances, continuous_distances). 
    Every matrix has the following columns: [average_distance, max_distance, min_distance]
    """
    SIZE_X = set_x.shape[0]
    SIZE_Y = set_y.shape[0]
    # rescale the continuous attributes to 0-100: this way the results were almost the same on cpu and gpu rounded to the 4th decimal
    set_x_cont = 100*set_x[:, :continuous_num]
    set_x_cont = set_x_cont.astype(np.single)
    set_y_cont = 100*set_y[:, :continuous_num]
    set_y_cont = set_y_cont.astype(np.single)
    set_x_cat = set_x[:, continuous_num:]
    set_x_cat = set_x_cat.astype(np.intc)
    set_y_cat = set_y[:, continuous_num:]
    set_y_cat = set_y_cat.astype(np.intc)
    DIM_CONT = continuous_num
    DIM_CAT = set_x_cat.shape[1]
    MAX_THREADS_PER_BLOCK = drv.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)
    # computation is 1D
    BLOCK_SIZE = int(MAX_THREADS_PER_BLOCK)
    BLOCK_NUMBER = int(SIZE_X / MAX_THREADS_PER_BLOCK) + 1
    GRID_DIM = (BLOCK_NUMBER, 1, 1)
    SIZE = SIZE_Y
    # subtract one if set_x is equal to set_y (intracluster) because we do not consider the distance from the same element
    if not intercluster:
        SIZE_AVG = SIZE_Y - 1
    else:
        SIZE_AVG = SIZE_Y
    variables = {
            'SIZE': SIZE,
            'SIZE_AVG': SIZE_AVG,
            'DIM_CAT': DIM_CAT,
            'DIM_CONT': DIM_CONT,
            }
    solution_cat = np.zeros((SIZE_X, 3))
    solution_cat = solution_cat.astype(np.single)
    solution_cont = np.zeros((SIZE_X, 3))
    solution_cont = solution_cont.astype(np.single)
    kernels_code = get_kernels_code(variables)
    mod = SourceModule(kernels_code)
    func = mod.get_function("categorical")
    func(drv.In(set_x_cat), drv.In(set_y_cat), drv.Out(solution_cat), block=(BLOCK_SIZE, 1, 1), grid=GRID_DIM)
    func = mod.get_function("euclidean")
    func(drv.In(set_x_cont), drv.In(set_y_cont), drv.Out(solution_cont), block=(BLOCK_SIZE, 1, 1), grid=GRID_DIM)
    return np.round(solution_cat, 4), np.round(solution_cont, 4) / 100

def get_kernels_code(variables: dict) -> str:
    kernels_code_template = """
    #include <math.h>
    __global__ void categorical(int *x, int *y, float *solution) {
            
            int idx = threadIdx.x + blockDim.x * blockIdx.x;

            if ( idx < %(SIZE)s ) {

                float total_distance = 0.0;
                float max_distance = 0.0;
                float min_distance = -1.0;

                for (int iter = 0; iter < %(SIZE)s; iter++) {

                    int total_sum = 0;
                    for (int iter_dim = 0; iter_dim < %(DIM_CAT)s; iter_dim++) {
                        int x_e = x[%(DIM_CAT)s*idx + iter_dim];
                        int y_e = y[%(DIM_CAT)s*iter + iter_dim];
                        if (x_e != y_e) {
                            total_sum += 1;
                            }
                        }
                    total_distance += total_sum;

                    if ( total_sum > max_distance) {
                        max_distance = total_sum;
                        }
                    if ( (total_sum < min_distance) || (min_distance == -1) ) {
                        min_distance = total_sum;
                        }

                    }
                solution[3*idx] = total_distance / %(SIZE_AVG)s;
                solution[3*idx + 1] = max_distance;
                solution[3*idx + 2] = min_distance;
                }
            }

    __global__ void euclidean(float *x, float *y, float *solution) {
            
            int idx = threadIdx.x + blockDim.x * blockIdx.x;

            if ( idx < %(SIZE)s ) {

                float total_distance = 0.0;
                float max_distance = 0.0;
                float min_distance = -1.0;

                for (int iter = 0; iter < %(SIZE)s; iter++) {

                    float total_sum = 0.0;
                    for (int iter_dim = 0; iter_dim < %(DIM_CONT)s; iter_dim++) {
                        float x_e = x[%(DIM_CONT)s*idx + iter_dim];
                        float y_e = y[%(DIM_CONT)s*iter + iter_dim];
                        total_sum += pow(x_e - y_e, 2);
                        }
                    total_distance += sqrt(total_sum);

                    if ( sqrt(total_sum) > max_distance) {
                        max_distance = sqrt(total_sum);
                        }
                    if ( (sqrt(total_sum) < min_distance) || (min_distance == -1.0) ) {
                        min_distance = sqrt(total_sum);
                        }

                    }
                solution[3*idx] = total_distance / %(SIZE_AVG)s;
                solution[3*idx + 1] = max_distance;
                solution[3*idx + 2] = min_distance;
                }
            }
    """
    kernels_code = kernels_code_template % variables
    return kernels_code
