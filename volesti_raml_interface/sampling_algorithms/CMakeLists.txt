# In order for OCaml's Ctype-based C binding to work properly, we must generate
# dynamic libraries, instead of static ones.
add_library(logconcave_hmc SHARED logconcave_hmc.cpp create_polytope.cpp)
add_library(hit_and_run SHARED hit_and_run.cpp create_polytope.cpp)
set_property(TARGET logconcave_hmc PROPERTY CXX_STANDARD 17) # enable C++17 standard
