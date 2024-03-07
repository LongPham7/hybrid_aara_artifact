degree = 1

# Input data

input_data = \
    [
        [2, 1, 5, 1, 8, 6, 7, 1],
        [5, 5, 1, 6, 3, 4, 7, 6],
        [1, 11, 9, 10, 7, 2, 12, 6, 4, 4, 9, 6],
        [6, 6, 6, 7, 5, 10, 12, 5, 8, 4, 6, 6],
        [13, 9, 8, 12, 18, 2, 12, 12, 10, 12, 18, 8, 17, 15, 10, 9, 6, 2],
        [2, 3, 1, 8, 1, 12, 8, 12, 14, 13, 11, 14, 16, 13, 1, 17, 6, 17],
        [16, 2, 17, 19, 6, 21, 10, 7, 22, 5, 5, 18, 2, 26,
            24, 20, 24, 19, 23, 5, 21, 6, 4, 7, 8, 27, 8],
        [8, 5, 27, 23, 3, 18, 21, 12, 15, 1, 23, 26, 17, 13,
            7, 18, 19, 27, 7, 14, 20, 2, 16, 15, 23, 18, 20],
        [2, 22, 7, 15, 1, 25, 10, 32, 11, 17, 14, 34, 19, 13, 15, 39, 19, 36, 37, 11,
            1, 23, 36, 5, 1, 3, 11, 35, 2, 27, 9, 39, 25, 20, 4, 35, 38, 24, 3, 17],
        [26, 17, 31, 27, 27, 38, 28, 17, 25, 6, 1, 12, 19, 33, 2, 4, 14, 37, 30, 22,
            30, 24, 3, 17, 32, 15, 38, 40, 40, 1, 40, 14, 22, 16, 13, 33, 26, 20, 22, 28],
        [8, 10, 38, 5, 40, 35, 34, 6, 17, 11, 31, 46, 16, 7, 46, 59, 50, 24, 60, 55, 57, 38, 34, 15, 56, 11, 41, 40, 41, 24,
            32, 23, 2, 42, 41, 52, 52, 20, 26, 42, 7, 6, 9, 1, 55, 20, 48, 17, 50, 19, 50, 9, 46, 3, 29, 37, 22, 3, 28, 22],
        [28, 59, 36, 39, 16, 10, 16, 14, 42, 43, 2, 56, 18, 30, 14, 44, 9, 14, 53, 7, 4, 18, 22, 4, 5, 40, 4, 37, 16, 31, 39,
            12, 53, 3, 44, 47, 33, 43, 58, 37, 15, 58, 59, 59, 22, 33, 4, 24, 20, 59, 44, 38, 3, 18, 40, 45, 56, 25, 39, 37],
        [12, 72, 33, 20, 36, 17, 65, 82, 4, 74, 26, 52, 79, 30, 66, 57, 23, 60, 43, 90, 75, 66, 54, 29, 78, 74, 78, 3, 25, 44, 40, 35, 87, 52, 60, 24, 48, 77, 37, 28, 20, 25, 43, 85,
            19, 32, 41, 83, 63, 44, 6, 62, 84, 11, 36, 2, 70, 39, 29, 61, 50, 29, 54, 46, 3, 65, 2, 66, 33, 56, 39, 76, 89, 1, 37, 31, 8, 62, 13, 50, 29, 1, 42, 71, 66, 81, 60, 67, 6, 43],
        [54, 80, 30, 84, 65, 71, 77, 42, 73, 15, 79, 68, 83, 6, 11, 37, 39, 28, 74, 39, 73, 84, 45, 39, 46, 87, 39, 45, 67, 84, 85, 47, 15, 71, 73, 45, 1, 29, 51, 51, 18, 6, 20, 1, 62,
            82, 57, 72, 42, 61, 4, 85, 79, 86, 58, 36, 35, 34, 61, 64, 3, 52, 25, 10, 7, 57, 11, 89, 39, 53, 68, 29, 52, 56, 8, 83, 74, 49, 5, 21, 84, 67, 22, 6, 28, 7, 79, 53, 24, 5],
        [92, 129, 35, 57, 38, 32, 58, 50, 33, 50, 112, 130, 101, 77, 134, 15, 13, 117, 99, 70, 112, 73, 97, 86, 127, 100, 27, 67, 32, 115, 73, 59, 59, 76, 112, 17, 89, 105, 14, 37, 43, 55, 87, 10, 123, 51, 29, 96, 132, 53, 117, 40, 52, 65, 45, 70, 9, 45, 39, 115, 58, 133, 66, 1, 126, 123, 90, 61,
            116, 66, 91, 61, 86, 126, 39, 13, 14, 28, 81, 37, 20, 67, 17, 42, 103, 62, 111, 29, 46, 19, 128, 126, 114, 91, 30, 90, 94, 103, 31, 121, 13, 98, 64, 56, 73, 20, 102, 27, 95, 114, 19, 46, 75, 132, 56, 47, 69, 116, 90, 89, 56, 129, 37, 34, 58, 59, 23, 119, 33, 98, 99, 23, 124, 94, 62],
        [111, 78, 76, 61, 38, 1, 2, 20, 6, 103, 124, 18, 113, 86, 100, 109, 45, 124, 45, 50, 70, 59, 32, 27, 45, 131, 102, 126, 102, 78, 87, 109, 62, 35, 89, 134, 116, 106, 8, 93, 34, 16, 87, 114, 51, 64, 127, 15, 80, 24, 106, 64, 85, 57, 38, 34, 21, 81, 55, 88, 127, 83, 45, 4, 64, 13, 93, 60,
            25, 130, 64, 55, 103, 68, 18, 112, 104, 46, 81, 107, 56, 67, 126, 24, 37, 132, 87, 116, 50, 19, 92, 64, 134, 47, 44, 11, 48, 81, 26, 48, 29, 27, 132, 3, 51, 80, 57, 55, 74, 95, 113, 72, 14, 132, 123, 74, 28, 66, 44, 115, 38, 99, 131, 125, 36, 28, 60, 32, 93, 49, 95, 44, 41, 106, 33]
    ]

# Function names

function_name_data_driven = "insertion_sort_second_time2"

function_name_hybrid = "insertion_sort_second_time"

function_name_dict = {"data_driven": function_name_data_driven,
                      "hybrid": function_name_hybrid}

# Configurations

config_data_driven_opt = {"data_analysis_mode": "opt"}

config_data_driven_hybrid_bayeswc = {"data_analysis_mode": "bayeswc",
                                     "num_samples": 400}

lp = (10.0, False, True)
warmup = ("rdhr", 36.0, 100, 100)
coefficient_distribution_sigma = 1.122
cost_model_alpha = 1.0
cost_model_sigma = 108.571
cost_model = (cost_model_alpha, cost_model_sigma)
num_samples = 400
walk_length = 400
step_size = 0.1
config_data_driven_bayespc = {"lp": lp,
                              "warmup": warmup,
                              "coefficient_distribution": coefficient_distribution_sigma,
                              "cost_model": cost_model,
                              "num_samples": num_samples,
                              "walk_length": walk_length,
                              "step_size": step_size}

output_lp_vars = [7, 6, 3, 2]
config_hybrid_opt = {"data_analysis_mode": "opt",
                     "output_lp_vars": output_lp_vars}

lp = (10.0, False, True)
warmup = ("rdhr", 18.0, 100, 100)
coefficient_distribution_sigma = 1.253
cost_model_alpha = 1.0
cost_model_sigma = 102.040
cost_model = (cost_model_alpha, cost_model_sigma)
num_samples = 400
walk_length = 400
step_size = 0.4
config_hybrid_bayespc = {"lp": lp,
                         "warmup": warmup,
                         "coefficient_distribution": coefficient_distribution_sigma,
                         "cost_model": cost_model,
                         "num_samples": num_samples,
                         "walk_length": walk_length,
                         "step_size": step_size,
                         "output_lp_vars": output_lp_vars}

config_data_driven_dict = {"opt": config_data_driven_opt,
                           "bayeswc": config_data_driven_hybrid_bayeswc,
                           "bayespc": config_data_driven_bayespc}

config_hybrid_dict = {"opt": config_hybrid_opt,
                      "bayeswc": config_data_driven_hybrid_bayeswc,
                      "bayespc": config_hybrid_bayespc}

config_dict = {"data_driven": config_data_driven_dict,
               "hybrid": config_hybrid_dict}
