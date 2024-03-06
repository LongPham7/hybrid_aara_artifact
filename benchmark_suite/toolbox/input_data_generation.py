import random

# Create a random list


def create_random_list(max_size, max_achieved=False):
    if (max_size == 0 or max_achieved):
        list_length = max_size
    else:
        # In this branch, the list is guaranteed to be non-empty.
        list_length = random.randrange(1, max_size+1)

    max_element_size = max_size

    # We randomly sample list elements without replacement.
    # We don't allow replacement so that the complexity analysis of median of medians works.
    return random.sample(list(range(0, max_element_size)), list_length)
    # return random.choices(list(range(1, max_element_size+1)), k=list_length)

# Create lists.


def create_random_lists(num_inputs, max_input_size):
    input_data = []
    for _ in range(num_inputs):
        input_list = create_random_list(max_input_size)
        input_data.append(input_list)
    return input_data


def create_exponential_lists(initial_size, exponential_base,
                             num_size_categories, num_samples_per_category):
    input_data = []
    input_size = initial_size
    for _ in range(0, num_size_categories):
        for _ in range(0, num_samples_per_category):
            input_list = create_random_list(input_size, True)
            input_data.append(input_list)
        input_size = int(input_size * exponential_base)
    return input_data


def create_input_data_lists(input_data_generation_params):
    input_size_distribution = input_data_generation_params["input_size_distribution"]
    if input_size_distribution == "random":
        num_inputs = input_data_generation_params["num_inputs"]
        max_input_size = input_data_generation_params["max_input_size"]
        input_data_python = create_random_lists(
            num_inputs, max_input_size)
    elif input_size_distribution == "exponential_growth":
        initial_size = input_data_generation_params["initial_size"]
        exponential_base = input_data_generation_params["exponential_base"]
        num_size_categories = input_data_generation_params["num_size_categories"]
        num_samples_per_category = input_data_generation_params["num_samples_per_category"]
        input_data_python = create_exponential_lists(initial_size, exponential_base,
                                                     num_size_categories, num_samples_per_category)
    else:
        raise Exception("The given input size distribution is invalid.")
    return input_data_python

# Create pairs of lists. This is used in the benchmark append.


def create_random_pairs_lists(num_samples, max_input_size1, max_input_size2):
    input_data = []
    for _ in range(num_samples):
        input1 = create_random_list(max_input_size1)
        input2 = create_random_list(max_input_size2)
        input_pair = (input1, input2)
        input_data.append(input_pair)
    return input_data


def create_exponential_pairs_lists(initial_input_size1, exponential_base1, num_size_categories1,
                                   initial_input_size2, exponential_base2, num_size_categories2,
                                   num_samples_per_category):
    input_data = []
    for i in range(0, num_size_categories1):
        for j in range(0, num_size_categories2):
            input_size1 = int(initial_input_size1 * exponential_base1**i)
            input_size2 = int(initial_input_size2 * exponential_base2**j)

            for _ in range(0, num_samples_per_category):
                input1 = create_random_list(input_size1, max_achieved=True)
                input2 = create_random_list(input_size2, max_achieved=True)
                input_pair = (input1, input2)
                input_data.append(input_pair)
    return input_data


def create_input_data_pairs_lists(input_data_generation_params):
    input_size_distribution = input_data_generation_params["input_size_distribution"]
    if input_size_distribution == "random":
        num_inputs = input_data_generation_params["num_inputs"]
        max_input_size1 = input_data_generation_params["max_input_size1"]
        max_input_size2 = input_data_generation_params["max_input_size2"]
        input_data_python = create_random_pairs_lists(
            num_inputs, max_input_size1, max_input_size2)
    elif input_size_distribution == "exponential_growth":
        initial_input_size1 = input_data_generation_params["initial_input_size1"]
        exponential_base1 = input_data_generation_params["exponential_base1"]
        num_size_categories1 = input_data_generation_params["num_size_categories1"]
        initial_input_size2 = input_data_generation_params["initial_input_size2"]
        exponential_base2 = input_data_generation_params["exponential_base2"]
        num_size_categories2 = input_data_generation_params["num_size_categories2"]
        num_samples_per_category = input_data_generation_params["num_samples_per_category"]
        input_data_python = create_exponential_pairs_lists(initial_input_size1, exponential_base1, num_size_categories1,
                                                           initial_input_size2, exponential_base2, num_size_categories2,
                                                           num_samples_per_category)
    else:
        raise Exception("The given input size distribution is invalid.")
    return input_data_python

# Create nested lists. This is used in the benchmark concat.


def create_random_nested_list(max_num_inner_lists, max_inner_list_size, max_num_inner_lists_achieved=False, max_inner_list_size_achieved=False):
    if (max_num_inner_lists == 0 or max_num_inner_lists_achieved):
        num_inner_lists = max_num_inner_lists
    else:
        # In this branch, the number of inner lists is at least one.
        num_inner_lists = random.randrange(1, max_num_inner_lists + 1)

    nested_list = []
    for _ in range(num_inner_lists):
        inner_list = create_random_list(max_inner_list_size)
        nested_list.append(inner_list)
    return nested_list


def create_random_nested_lists(num_samples, max_num_inner_lists, max_inner_list_size):
    input_data = []
    for _ in range(num_samples):
        nested_list = create_random_nested_list(
            max_num_inner_lists, max_inner_list_size)
        input_data.append(nested_list)
    return input_data


def create_random_nested_list_with_fixed_combined_size_split_equally(combined_size, num_inner_lists):
    div_result = combined_size // num_inner_lists
    remainder = combined_size % num_inner_lists
    nested_list = []
    for _ in range(0, num_inner_lists - remainder):
        inner_list = create_random_list(div_result, True)
        nested_list.append(inner_list)
    for _ in range(0, remainder):
        inner_list = create_random_list(div_result+1, True)
        nested_list.append(inner_list)
    return nested_list


def create_exponential_nested_lists(initial_combined_size, exponential_base,
                                    num_size_categories, num_inner_lists, num_samples_per_category):
    input_data = []
    for i in range(0, num_size_categories):
        combined_size = int(initial_combined_size * exponential_base**i)
        for _ in range(0, num_samples_per_category):
            nested_list = create_random_nested_list_with_fixed_combined_size_split_equally(
                combined_size, num_inner_lists)
            input_data.append(nested_list)
    return input_data


def create_input_data_nested_lists(input_data_generation_params):
    input_size_distribution = input_data_generation_params["input_size_distribution"]
    if input_size_distribution == "random":
        num_inputs = input_data_generation_params["num_inputs"]
        max_num_inner_lists = input_data_generation_params["max_num_inner_lists"]
        max_inner_list_size = input_data_generation_params["max_inner_list_size"]
        input_data_python = create_random_nested_lists(
            num_inputs, max_num_inner_lists, max_inner_list_size)
    elif input_size_distribution == "exponential_growth":
        initial_combined_size = input_data_generation_params["initial_combined_size"]
        exponential_base = input_data_generation_params["exponential_base"]
        num_size_categories = input_data_generation_params["num_size_categories"]
        num_inner_lists = input_data_generation_params["num_inner_lists"]
        num_samples_per_category = input_data_generation_params["num_samples_per_category"]
        input_data_python = create_exponential_nested_lists(initial_combined_size, exponential_base,
                                                            num_size_categories, num_inner_lists, num_samples_per_category)
    else:
        raise Exception("The given input size distribution is invalid.")
    return input_data_python

# Convert a list from Python to OCaml


def convert_list_python_to_ocaml(list_python):
    list_ocaml = "["
    for i, element in enumerate(list_python):
        if i == 0:
            list_ocaml += str(element)
        else:
            list_ocaml += "; " + str(element)
    return list_ocaml + "]"

# Convert input data of lists from Python to OCaml


def convert_lists_python_to_ocaml(input_data_python, split_line=True):
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

# Convert input data of pairs of lists from Python to OCaml


def convert_pairs_lists_python_to_ocaml(input_data_python, split_line=True):
    input_data_ocaml = "["
    for i, input_pair in enumerate(input_data_python):
        input1, input2 = input_pair
        input_pair_ocaml = "(" + convert_list_python_to_ocaml(input1) + \
            ", " + convert_list_python_to_ocaml(input2) + ")"
        if i == 0:
            input_data_ocaml += input_pair_ocaml
        else:
            if split_line:
                input_data_ocaml += ";\n" + input_pair_ocaml
            else:
                input_data_ocaml += ";" + input_pair_ocaml
    return input_data_ocaml + "]"

# Convert input data of nested lists from Python to OCaml


def convert_nested_list_python_to_ocaml(nested_list_python):
    nested_list_ocaml = "["
    for i, inner_list_python in enumerate(nested_list_python):
        inner_list_ocaml = convert_list_python_to_ocaml(inner_list_python)
        if i == 0:
            nested_list_ocaml += inner_list_ocaml
        else:
            nested_list_ocaml += "; " + inner_list_ocaml
    return nested_list_ocaml + "]"


def convert_nested_lists_python_to_ocaml(input_data_python, split_line=True):
    input_data_ocaml = "["
    for i, nested_list_python in enumerate(input_data_python):
        nested_list_ocaml = convert_nested_list_python_to_ocaml(
            nested_list_python)
        if i == 0:
            input_data_ocaml += nested_list_ocaml
        else:
            if split_line:
                input_data_ocaml += ";\n" + nested_list_ocaml
            else:
                input_data_ocaml += "; " + nested_list_ocaml
    return input_data_ocaml + "]"

# Convert input data of integer-list pairs from Python to OCaml

def convert_integer_list_pair_python_to_ocaml(input_data_python, split_line=True):
    input_data_ocaml = "["
    for i, input_pair in enumerate(input_data_python):
        index, input_list_python = input_pair
        input_pair_ocaml = "(" + str(index) + \
            ", " + convert_list_python_to_ocaml(input_list_python) + ")"
        if i == 0:
            input_data_ocaml += input_pair_ocaml
        else:
            if split_line:
                input_data_ocaml += ";\n" + input_pair_ocaml
            else:
                input_data_ocaml += ";" + input_pair_ocaml
    return input_data_ocaml + "]"
