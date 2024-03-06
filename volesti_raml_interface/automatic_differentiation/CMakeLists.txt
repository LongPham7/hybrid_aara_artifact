add_library(runtime_data SHARED runtime_data.cpp)
add_library(probabilistic_programming SHARED probability_distributions.cpp probabilistic_programming.cpp)
set_property(TARGET probabilistic_programming PROPERTY CXX_STANDARD 17) # enable C++17 standard
target_link_libraries(probabilistic_programming runtime_data)
