from utils.distribution import get_trace_attributes_distribution_traffic_fine, get_trace_attributes_distribution_bpi19, get_trace_attributes_distribution_bpi17, get_trace_attributes_distribution_bpi12, get_trace_attributes_distribution_inoxtech
from utils.generals import extract_attrs_fines_log, extract_max_continuous_values_fines_log, get_map_categorical_attributes_fine, get_performance_cases_fines
from utils.generals import extract_attrs_bpic2019_log, extract_max_continuous_values_bpic2019_log, get_map_categorical_attributes_bpic2019, get_performance_cases_bpic2019
from utils.generals import extract_attrs_inox_log, extract_max_continuous_values_inox_log, get_map_categorical_attributes_inox, get_performance_cases_inox
from utils.generals import extract_attrs_bpic2017_log, extract_max_continuous_values_bpic2017_log, get_map_categorical_attributes_bpic2017, get_performance_cases_bpic2017
from utils.generals import extract_attrs_bpic2012_log, extract_max_continuous_values_bpic2012_log, get_map_categorical_attributes_bpic2012, get_performance_cases_bpic2012


class LogUtilities:
    def __init__(self, log_name: str):
        if 'road_traffic_fine_management_process' in log_name:
            self.max_continuous_values = extract_max_continuous_values_fines_log 
            self.trace_attributes_distribution = get_trace_attributes_distribution_traffic_fine
            self.map_categorical_attributes = get_map_categorical_attributes_fine
            self.trace_attributes = extract_attrs_fines_log
            self.performance = get_performance_cases_fines
            self.only_lifecycle_start = False
            self.log_name = "road_traffic_fine_management_process"
            self.filter_time = True
            self.date_col = 'time:timestamp'
            self.start_date = "2000-01-01 00:00:00"
            self.end_date = "2012-06-01 00:00:00"
        elif 'BPI_Challenge_2019' in log_name:
            self.max_continuous_values = extract_max_continuous_values_bpic2019_log 
            self.trace_attributes_distribution = get_trace_attributes_distribution_bpi19
            self.map_categorical_attributes = get_map_categorical_attributes_bpic2019
            self.trace_attributes = extract_attrs_bpic2019_log
            self.performance = get_performance_cases_bpic2019
            self.only_lifecycle_start = False
            self.log_name = "BPI_Challenge_2019"
            self.filter_time = True
            self.date_col = 'time:timestamp'
            self.start_date = "2018-01-01 00:00:00"
            self.end_date = "2019-01-01 00:00:00"
        elif 'BPI_Challenge_2017' in log_name:
            self.max_continuous_values = extract_max_continuous_values_bpic2017_log
            self.trace_attributes_distribution = get_trace_attributes_distribution_bpi17
            self.map_categorical_attributes = get_map_categorical_attributes_bpic2017
            self.trace_attributes = extract_attrs_bpic2017_log
            self.performance = get_performance_cases_bpic2017
            self.only_lifecycle_start = False
            self.log_name = "BPI_Challenge_2017"
            self.filter_time = True
            self.date_col = 'time:timestamp'
            self.start_date = "2016-01-01 00:00:00"
            # max duration 45 days
            self.end_date = "2017-01-15 00:00:00"
        elif 'BPI_Challenge_2012' in log_name:
            self.max_continuous_values = extract_max_continuous_values_bpic2012_log
            self.trace_attributes_distribution = get_trace_attributes_distribution_bpi12
            self.map_categorical_attributes = get_map_categorical_attributes_bpic2012
            self.trace_attributes = extract_attrs_bpic2012_log
            self.performance = get_performance_cases_bpic2012
            self.only_lifecycle_start = False
            self.log_name = "BPI_Challenge_2012"
            self.filter_time = True
            self.date_col = 'time:timestamp'
            self.start_date = "2011-10-01 00:00:00"
            # max duration 12 days
            self.end_date = "2012-03-01 00:00:00"
        else:
            raise NotImplementedError('Log not implemented')
