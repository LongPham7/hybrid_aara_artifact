degree = 1

# Input data

input_data = \
    [
        [87, 86, 63, 11, 35, 25, 33, 22],
        [58, 29, 49, 28, 13, 84, 37, 69],
        [11, 77, 76, 74, 0, 27, 3, 78],
        [12, 16, 38, 2, 83, 84, 11, 57],
        [14, 10, 90, 40, 13, 17, 42, 27],
        [0, 37, 36, 40, 60, 53, 4, 75, 51, 80, 16, 48],
        [28, 87, 88, 59, 86, 25, 8, 11, 5, 68, 55, 2],
        [81, 52, 19, 18, 16, 29, 2, 75, 20, 55, 96, 47],
        [57, 16, 87, 77, 48, 5, 43, 62, 36, 14, 0, 21],
        [34, 37, 84, 8, 91, 53, 32, 57, 19, 36, 97, 23],
        [42, 46, 51, 24, 77, 98, 99, 26, 2, 62, 33, 33, 38, 70, 26, 60, 86, 80],
        [98, 32, 24, 5, 28, 15, 13, 4, 28, 89, 26, 58, 16, 74, 28, 72, 82, 41],
        [64, 57, 9, 52, 30, 37, 25, 7, 99, 98, 87, 39, 59, 93, 68, 76, 90, 97],
        [76, 54, 14, 32, 43, 52, 13, 28, 84, 63, 41, 91, 58, 90, 48, 71, 71, 68],
        [9, 38, 62, 29, 50, 36, 30, 66, 66, 85, 15, 32, 64, 55, 85, 24, 41, 61],
        [42, 57, 68, 50, 42, 38, 73, 93, 58, 31, 37, 86, 53, 11,
            81, 88, 59, 17, 44, 2, 78, 13, 15, 24, 27, 83, 60],
        [60, 13, 41, 19, 2, 80, 12, 91, 29, 30, 48, 65, 89, 45,
            8, 58, 87, 68, 49, 39, 31, 72, 21, 22, 29, 62, 35],
        [32, 12, 29, 77, 21, 99, 30, 44, 14, 40, 47, 15, 26, 61,
            92, 86, 74, 23, 15, 92, 90, 68, 34, 23, 74, 92, 97],
        [19, 91, 48, 50, 89, 23, 69, 43, 85, 34, 67, 20, 52, 79,
            95, 58, 81, 17, 32, 68, 20, 0, 19, 14, 2, 71, 51],
        [55, 28, 60, 47, 28, 30, 7, 41, 91, 17, 46, 25, 79, 79,
            98, 1, 99, 96, 61, 60, 29, 84, 34, 59, 10, 72, 81],
        [72, 70, 33, 21, 61, 46, 88, 3, 25, 57, 80, 98, 57, 1, 41, 51, 16, 84, 20, 12,
            71, 8, 31, 73, 75, 74, 30, 86, 40, 96, 98, 68, 0, 47, 58, 81, 16, 59, 80, 34],
        [72, 18, 45, 25, 82, 35, 15, 4, 34, 15, 42, 32, 11, 49, 26, 46, 22, 93, 37, 99,
            14, 13, 62, 92, 79, 95, 70, 33, 82, 37, 69, 77, 9, 34, 26, 42, 30, 10, 42, 38],
        [92, 80, 62, 32, 69, 80, 18, 88, 24, 79, 38, 1, 17, 16, 29, 96, 57, 60, 96, 19,
            76, 76, 63, 95, 87, 52, 64, 48, 60, 29, 26, 1, 91, 31, 31, 78, 61, 42, 44, 90],
        [68, 21, 68, 29, 16, 11, 68, 52, 73, 70, 92, 59, 69, 99, 48, 57, 85, 69, 54, 63,
            93, 26, 89, 20, 8, 92, 18, 95, 48, 34, 25, 90, 87, 8, 72, 74, 46, 58, 39, 98],
        [85, 42, 58, 45, 30, 1, 40, 97, 82, 80, 18, 45, 70, 46, 14, 5, 18, 36, 89, 28,
            4, 42, 51, 43, 26, 45, 17, 49, 74, 36, 20, 72, 27, 53, 43, 97, 69, 92, 91, 41],
        [96, 34, 30, 86, 10, 17, 82, 57, 35, 97, 42, 35, 91, 22, 26, 97, 86, 0, 48, 34, 10, 21, 53, 9, 36, 65, 7, 4, 62, 17,
            59, 29, 1, 24, 23, 13, 5, 33, 65, 68, 77, 44, 98, 43, 67, 49, 11, 66, 44, 58, 91, 82, 72, 1, 8, 34, 35, 67, 30, 74],
        [11, 21, 62, 60, 90, 5, 28, 62, 60, 44, 63, 11, 81, 19, 19, 71, 37, 71, 47, 17, 7, 34, 93, 26, 9, 25, 7, 48, 82, 75,
            69, 14, 79, 4, 20, 95, 71, 19, 73, 71, 86, 37, 4, 36, 93, 3, 47, 27, 50, 99, 26, 58, 89, 92, 61, 8, 21, 6, 39, 46],
        [27, 50, 28, 72, 96, 82, 73, 72, 71, 44, 56, 83, 22, 19, 19, 47, 92, 24, 28, 81, 21, 22, 55, 60, 78, 66, 45, 25, 82, 98,
            38, 19, 52, 24, 94, 19, 63, 45, 53, 59, 53, 1, 84, 90, 44, 26, 74, 7, 96, 63, 28, 41, 94, 33, 94, 89, 36, 79, 71, 5],
        [9, 85, 65, 84, 5, 41, 39, 85, 55, 27, 27, 58, 4, 1, 60, 94, 17, 95, 18, 89, 36, 86, 50, 54, 63, 35, 88, 75, 5, 79, 40,
            9, 26, 20, 32, 86, 93, 63, 79, 38, 14, 0, 33, 82, 86, 80, 76, 20, 83, 13, 83, 81, 98, 30, 80, 97, 79, 26, 49, 56],
        [87, 88, 6, 72, 2, 63, 91, 0, 75, 17, 54, 99, 12, 57, 40, 29, 30, 33, 76, 4, 11, 89, 22, 38, 40, 23, 84, 31, 9, 11,
            75, 66, 75, 36, 43, 80, 16, 37, 62, 84, 84, 50, 84, 39, 5, 85, 60, 59, 19, 77, 90, 94, 4, 26, 4, 53, 55, 66, 4, 14],
        [7, 76, 29, 58, 22, 93, 10, 49, 84, 37, 83, 6, 57, 41, 48, 88, 27, 3, 9, 37, 77, 50, 81, 52, 1, 29, 29, 77, 2, 79, 59, 68, 18, 34, 32, 96, 79, 38, 91, 1, 37, 45, 58, 62, 1, 53,
            43, 18, 59, 38, 4, 98, 69, 62, 92, 17, 88, 46, 17, 74, 75, 34, 88, 17, 44, 59, 48, 90, 93, 28, 44, 73, 93, 42, 28, 73, 66, 44, 82, 88, 62, 82, 75, 29, 14, 82, 19, 21, 74, 60],
        [82, 47, 58, 44, 31, 27, 92, 24, 60, 26, 34, 40, 69, 69, 36, 85, 85, 47, 83, 75, 48, 69, 12, 88, 46, 97, 77, 20, 78, 80, 95, 3, 88, 10, 12, 99, 26, 88, 7, 38, 36, 69, 40, 36,
            45, 77, 88, 9, 31, 21, 94, 26, 46, 67, 70, 79, 1, 9, 96, 53, 87, 83, 8, 48, 2, 70, 59, 6, 71, 43, 62, 38, 69, 46, 76, 62, 39, 59, 3, 35, 35, 90, 36, 81, 49, 26, 9, 54, 24, 44],
        [23, 26, 80, 37, 45, 95, 41, 56, 1, 22, 59, 52, 89, 3, 65, 89, 65, 61, 90, 82, 33, 0, 42, 89, 94, 82, 64, 17, 8, 80, 37, 16, 34, 32, 46, 31, 6, 8, 16, 50, 73, 4, 97, 17, 96,
            68, 89, 48, 40, 29, 7, 38, 30, 90, 46, 56, 49, 14, 57, 55, 5, 87, 23, 41, 43, 99, 8, 6, 44, 75, 54, 52, 6, 25, 37, 88, 74, 57, 19, 31, 7, 59, 87, 85, 81, 59, 31, 33, 35, 46],
        [46, 10, 33, 75, 86, 93, 46, 23, 77, 80, 79, 29, 63, 88, 21, 57, 2, 23, 4, 90, 26, 63, 69, 91, 5, 97, 80, 72, 29, 35, 41, 14, 75, 36, 28, 68, 50, 53, 49, 34, 74, 81, 88, 91,
            11, 43, 77, 0, 44, 6, 98, 40, 3, 81, 26, 2, 1, 86, 2, 95, 85, 44, 25, 84, 66, 14, 53, 51, 44, 86, 58, 83, 62, 63, 59, 52, 29, 31, 11, 8, 62, 67, 9, 15, 49, 2, 38, 87, 69, 42],
        [48, 93, 15, 75, 17, 0, 76, 57, 83, 90, 63, 46, 67, 6, 46, 0, 82, 25, 4, 61, 95, 25, 31, 12, 67, 87, 73, 99, 31, 58, 80, 78, 49, 49, 65, 65, 95, 59, 84, 51, 42, 91, 32, 12, 12,
            53, 16, 50, 53, 90, 47, 40, 97, 94, 73, 87, 68, 72, 93, 25, 16, 96, 79, 82, 58, 78, 38, 62, 95, 54, 72, 96, 0, 21, 50, 55, 57, 5, 57, 71, 5, 10, 21, 40, 16, 12, 57, 81, 24, 18]
    ]

# Function names

function_name_data_driven = "z_algorithm2"

function_name_hybrid = "z_algorithm"

function_name_dict = {"data_driven": function_name_data_driven,
                      "hybrid": function_name_hybrid}

# Configurations

config_data_driven_opt = {"data_analysis_mode": "opt"}

config_data_driven_hybrid_bayeswc = {"data_analysis_mode": "bayeswc",
                                     "num_samples": 400}

lp = (10.0, False, True)
warmup = ("rdhr", 36.0, 100, 100)
coefficient_distribution_sigma = 1.136
# cost_model_alpha = 1.0
cost_model_alpha = 1.5
cost_model_sigma = 114.790
cost_model = (cost_model_alpha, cost_model_sigma)
num_samples = 400
walk_length = 400
# step_size = 0.2
step_size = 0.4
config_data_driven_bayespc = {"lp": lp,
                              "warmup": warmup,
                              "coefficient_distribution": coefficient_distribution_sigma,
                              "cost_model": cost_model,
                              "num_samples": num_samples,
                              "walk_length": walk_length,
                              "step_size": step_size}

output_lp_vars = [55, 53, 54, 51]
config_hybrid_opt = {"data_analysis_mode": "opt",
                     "output_lp_vars": output_lp_vars}

lp = (10.0, True, True)
warmup = ("cdhr", 100.0, 1000, 1200)
coefficient_distribution_sigma = 1.787
# cost_model_alpha = 1.0
cost_model_alpha = 1.25
cost_model_sigma = 104.955
cost_model = (cost_model_alpha, cost_model_sigma)
num_samples = 400
walk_length = 400
# step_size = 0.3
step_size = 0.1
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
