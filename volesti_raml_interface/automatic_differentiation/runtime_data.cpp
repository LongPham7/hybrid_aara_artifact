#include "runtime_data.h"

#include <iostream>

void print_runtime_data_sample(const runtime_data_sample& sample) {
  int* array_cindices = sample.array_cindices;
  double* potential_of_cindices = sample.potential_of_cindices;
  int num_cindices = sample.num_cindices;
  int* array_indices = sample.array_indices;
  double* potential_of_indices = sample.potential_of_indices;
  int num_indices = sample.num_indices;
  double cost = sample.cost;

  std::cout << "Input potential: (index, potential) = ";
  for (auto i = 0; i != num_cindices; i++) {
    std::cout << "(" << array_cindices[i] << ", " << potential_of_cindices[i]
              << ") ";
  }
  std::cout << std::endl;

  std::cout << "Output potential: (index, potential) = ";
  for (auto i = 0; i != num_indices; i++) {
    std::cout << "(" << array_indices[i] << ", " << potential_of_indices[i]
              << ") ";
  }
  std::cout << std::endl;

  std::cout << "Cost = " << cost << std::endl;
}

void print_runtime_data(runtime_data_sample* runtime_data,
                        int const num_samples) {
  for (auto i = 0; i != num_samples; i++) {
    print_runtime_data_sample(runtime_data[i]);
  }
}

runtime_data_sample* create_runtime_data_for_testing() {
  runtime_data_sample* runtime_data = new runtime_data_sample[5];

  for (auto i = 0; i != 5; i++) {
    int* array_cindices = new int[3];
    double* potential_of_cindices = new double[3];
    array_cindices[0] = 0;
    array_cindices[1] = 1;
    array_cindices[2] = 2;
    potential_of_cindices[0] = 1;
    potential_of_cindices[1] = i;
    potential_of_cindices[2] = 5;

    runtime_data_sample new_sample{
        array_cindices, potential_of_cindices, 3, nullptr, nullptr, 0,
        (double)i};
    runtime_data[i] = new_sample;
  }
  return runtime_data;
}
