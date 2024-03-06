import os
import subprocess
import json
import time
from json_manipulation import write_input_data_json, write_execution_time_json
from pathnames import bin_directory


# Create configuration files


def create_config_opt(output_lp_vars_params):
    if output_lp_vars_params is not None:
        output_lp_vars_python, output_lp_vars_file = output_lp_vars_params
        output_params_python = {
            "output_lp_vars": output_lp_vars_python, "output_file": output_lp_vars_file}
        return {"mode": "opt",
                "lp_objective": {"cost_gaps_optimization": True, "coefficients_optimization": "Equal_weights"},
                "output_params": output_params_python}
    else:
        return {"mode": "opt",
                "lp_objective": {"cost_gaps_optimization": True, "coefficients_optimization": "Equal_weights"}}


def create_config_bayeswc(num_samples):
    stan_params_python = {
        "scale_beta": 5.0,
        "scale_s": 5.0,
        "num_chains": 4,
        "num_stan_samples": num_samples
    }
    lp_objective_python = {"cost_gaps_optimization": True,
                           "coefficients_optimization": "Equal_weights"}
    config_dictionary = {"mode": "bayeswc",
                         "stan_params": stan_params_python,
                         "lp_objective": lp_objective_python}
    return config_dictionary


def create_config_bayespc(lp, warmup, coefficient_distribution, cost_model,
                          num_samples, walk_length, step_size, output_lp_vars_params):
    upper_bound, implicit_equality_removal, output_potential_set_to_zero = lp
    lp_params_python = {"box_constraint": {"upper_bound": upper_bound},
                        "implicit_equality_removal": implicit_equality_removal,
                        "output_potential_set_to_zero": output_potential_set_to_zero}

    algorithm_type, variance, warmup_num_samples, warmup_walk_length = warmup
    if algorithm_type == "rdhr":
        algorithm = "Gaussian_rdhr"
    else:
        algorithm = "Gaussian_cdhr"
    warmup_params_python = {"algorithm": algorithm,
                            "variance": variance,
                            "num_samples": warmup_num_samples,
                            "walk_length": warmup_walk_length}

    coefficient_distribution_sigma = coefficient_distribution
    coefficient_distribution_python = {
        "distribution_type": "Gaussian", "mu": 0.0, "sigma": coefficient_distribution_sigma}

    cost_model_alpha, cost_model_sigma = cost_model
    coefficient_distribution_with_target_python = {
        "distribution": coefficient_distribution_python, "target": "Individual_coefficients"}
    cost_model_python = {"distribution_type": "Weibull",
                         "alpha": cost_model_alpha, "sigma": cost_model_sigma}
    cost_model_with_target_python = {
        "distribution": cost_model_python, "target": "Individual_coefficients"}

    hmc_params_python = {"coefficient_distribution_with_target": coefficient_distribution_with_target_python,
                         "cost_model_with_target": cost_model_with_target_python,
                         "num_samples": num_samples,
                         "walk_length": walk_length,
                         "step_size": step_size}

    if output_lp_vars_params is not None:
        output_lp_vars_python, output_lp_vars_file = output_lp_vars_params
        output_params_python = {
            "output_lp_vars": output_lp_vars_python, "output_file": output_lp_vars_file}
        return {"mode": "bayespc",
                "lp_params": lp_params_python,
                "warmup_params": warmup_params_python,
                "hmc_params": hmc_params_python,
                "output_params": output_params_python}
    else:
        return {"mode": "bayespc",
                "lp_params": lp_params_python,
                "warmup_params": warmup_params_python,
                "hmc_params": hmc_params_python}


def create_config(config, analysis_info):
    data_analysis_mode = analysis_info["data_analysis_mode"]

    if data_analysis_mode == "opt":
        # If config does not have the key "output_lp_vars", then the value is
        # None
        output_lp_vars = config.get("output_lp_vars")
        if output_lp_vars is not None:
            bin_path = bin_directory(analysis_info)
            output_lp_vars_file = os.path.join(bin_path, "output_lp_vars.json")
            output_lp_vars_params = (output_lp_vars, output_lp_vars_file)
        else:
            output_lp_vars_params = None
        return create_config_opt(output_lp_vars_params)
    elif data_analysis_mode == "bayeswc":
        num_samples = config["num_samples"]
        return create_config_bayeswc(num_samples)
    elif data_analysis_mode == "bayespc":
        lp = config["lp"]
        warmup = config["warmup"]
        coefficient_distribution = config["coefficient_distribution"]
        cost_model = config["cost_model"]
        num_samples = config["num_samples"]
        walk_length = config["walk_length"]
        step_size = config["step_size"]

        # If config does not have the key "output_lp_vars", then the value is
        # None.
        output_lp_vars = config.get("output_lp_vars")
        if output_lp_vars is not None:
            bin_path = bin_directory(analysis_info)
            output_lp_vars_file = os.path.join(bin_path, "output_lp_vars.json")
            output_lp_vars_params = (output_lp_vars, output_lp_vars_file)
        else:
            output_lp_vars_params = None

        return create_config_bayespc(lp, warmup, coefficient_distribution,
                                     cost_model,
                                     num_samples, walk_length, step_size, output_lp_vars_params)
    else:
        raise Exception("The given config is invalid")


# Perform AARA


def run_aara(ocaml_code, input_data, degree, config, analysis_info):
    function_name = analysis_info["function_name"]

    # Create a bin directory if necessary
    bin_path = bin_directory(analysis_info)

    # Store OCaml code
    file_name = "{}.raml".format(function_name)
    file_path = os.path.join(bin_path, file_name)
    with open(file_path, "w") as f:
        f.write(ocaml_code)

    # Create a JSON configuration file
    hybrid_aara_config_python = create_config(config, analysis_info)
    config_name = "config.json"
    config_path = os.path.join(bin_path, config_name)
    with open(config_path, "w") as write_file:
        json.dump(hybrid_aara_config_python, write_file)

    # Run RaML and measure its execution time
    raml_main_path = os.path.expanduser(
        os.path.join("/home", "hybrid_aara", "raml", "main"))
    output_file = os.path.join(bin_path, "inference_result.json")
    start_time = time.perf_counter()
    subprocess.run([raml_main_path, "stat_analyze", "ticks", str(
        degree), "-m", file_path, function_name, "-config", config_path, "-o", output_file])
    end_time = time.perf_counter()

    # Record the execution time
    execution_time = end_time - start_time
    execution_time_file_path = os.path.join(bin_path, "execution_time.json")
    write_execution_time_json(
        execution_time, hybrid_aara_config_python, execution_time_file_path)

    # Write input data
    input_data_file_path = os.path.join(bin_path, "input_data.json")
    write_input_data_json(input_data, input_data_file_path)
