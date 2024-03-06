#ifndef RUNTIME_DATA
#define RUNTIME_DATA

#ifdef __cplusplus
extern "C" {
#endif

struct runtime_data_sample {
  int* array_cindices;
  double* potential_of_cindices;
  int num_cindices;
  int* array_indices;
  double* potential_of_indices;
  int num_indices;
  double cost;
};

void print_runtime_data_sample(const runtime_data_sample& sample);
void print_runtime_data(runtime_data_sample* runtime_data,
                        int const num_samples);
runtime_data_sample* create_runtime_data_for_testing(void);

#ifdef __cplusplus
}
#endif

#endif
