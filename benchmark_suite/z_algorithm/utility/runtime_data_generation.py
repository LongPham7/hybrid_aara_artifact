import random


def create_random_list(max_size, num_possible_elements, max_achieved=False):
    if (max_size == 0 or max_achieved):
        list_length = max_size
    else:
        # In this branch, the list is guaranteed to be non-empty.
        list_length = random.randrange(1, max_size+1)

    # We only have three possible elements: 0, 1, and 2.
    if num_possible_elements == 1:
        all_possible_elements = [0]
        weights = [1]
    elif num_possible_elements == 2:
        all_possible_elements = [0, 1]
        weights = [8.0, 2.0]
    elif num_possible_elements == 3:
        all_possible_elements = list(range(0, 3))
        weights = [7.5, 1.8, 0.7]
    else:
        all_possible_elements = list(range(0, num_possible_elements))
        weights = [1 / num_possible_elements] * num_possible_elements
    return random.choices(all_possible_elements, weights=weights, k=list_length)


def create_random_input_data(num_samples, max_input_size, num_possible_elements):
    # So we only generate non-empty input lists.
    assert (max_input_size > 0)
    input_data = []
    for _ in range(num_samples):
        input_list = create_random_list(
            max_input_size, num_possible_elements, False)
        input_data.append(input_list)
    return input_data


def create_exponentially_growing_input_data(initial_size, exponential_base,
                                            num_size_categories, num_samples_per_category, num_possible_elements):
    input_data = []
    input_size = initial_size
    for _ in range(0, num_size_categories):
        for _ in range(0, num_samples_per_category):
            input_list = create_random_list(
                input_size, num_possible_elements, True)
            input_data.append(input_list)
        input_size = int(input_size * exponential_base)
    return input_data


def create_input_data(input_data_generation_params):
    input_size_distribution = input_data_generation_params["input_size_distribution"]
    if input_size_distribution == "random":
        num_inputs = input_data_generation_params["num_inputs"]
        max_input_size = input_data_generation_params["max_input_size"]
        num_possible_elements = input_data_generation_params["num_possible_elements"]
        input_data_python = create_random_input_data(
            num_inputs, max_input_size, num_possible_elements)
    elif input_size_distribution == "exponential_growth":
        initial_size = input_data_generation_params["initial_size"]
        exponential_base = input_data_generation_params["exponential_base"]
        num_size_categories = input_data_generation_params["num_size_categories"]
        num_samples_per_category = input_data_generation_params["num_samples_per_category"]
        num_possible_elements = input_data_generation_params["num_possible_elements"]
        input_data_python = create_exponentially_growing_input_data(initial_size, exponential_base,
                                                                    num_size_categories, num_samples_per_category, num_possible_elements)
    else:
        raise Exception("The given input size distribution is invalid.")
    return input_data_python


def convert_list_python_to_ocaml(list_python):
    list_ocaml = "["
    for i, element in enumerate(list_python):
        if i == 0:
            list_ocaml += str(element)
        else:
            list_ocaml += "; " + str(element)
    return list_ocaml + "]"


def convert_input_data_python_to_ocaml(input_data_python, split_line=True):
    input_data_ocaml = "["
    for i, input_list_python in enumerate(input_data_python):
        input_list_ocaml = convert_list_python_to_ocaml(input_list_python)
        if i == 0:
            input_data_ocaml += input_list_ocaml
        else:
            if split_line:
                input_data_ocaml += ";\n" + input_list_ocaml
            else:
                input_data_ocaml += ";" + input_list_ocaml
    return input_data_ocaml + "]"
