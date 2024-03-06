# VolEsti (volume computation and sampling library)
# Copyright (c) 2012-2018 Vissarion Fisikopoulos
# Copyright (c) 2018 Apostolos Chalkis
# Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.
# Further modified by Long Pham.
# Licensed under GNU LGPL.3, see LICENCE file

CMAKE_MINIMUM_REQUIRED(VERSION 3.22.1)

project( logconcave_HMC )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(VOLESTIROOT /home/hybrid_aara/volesti)
set(MKLROOT /opt/intel/oneapi/mkl/latest)

set(DEBUG_MODE False)

if(COMMAND cmake_policy)
       cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


option(BUILTIN_EIGEN "Use eigen from volesti/external" OFF) 
option(USE_MKL "Use MKL library to build eigen" ON)

add_definitions(-DDISABLE_NLP_ORACLES)

# Pass a debugging flag to Volesti. The debugging mdoe is useful for getting a
# more detailed error message when the computation of the Chebyshev centers
# fails.
add_compile_definitions(VOLESTI_DEBUG)

# include("${VOLESTIROOT}/external/cmake-files/Eigen.cmake")
# GetEigen()

# include("${VOLESTIROOT}/external/cmake-files/Boost.cmake")
# GetBoost()

include("${VOLESTIROOT}/external/cmake-files/LPSolve.cmake")
GetLPSolve()

if (BUILTIN_EIGEN)
    include_directories (BEFORE ${VOLESTIROOT}/external/_deps/Eigen)
else ()
    include_directories(BEFORE /usr/include/eigen3)
endif(BUILTIN_EIGEN)

if (USE_MKL)
  find_library(BLAS NAME libblas.so PATHS /usr/lib/x86_64-linux-gnu /usr/lib/i386-linux-gnu)
  include_directories (BEFORE ${MKLROOT}/include)
  # The following line in the original code contains a wrong path to the directory. 
  # set(MKL_LINK "-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
  set(MKL_LINK "-L${MKLROOT}/lib/intel64 -Wl,-rpath=${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
  add_definitions(-DEIGEN_USE_MKL_ALL)
endif(USE_MKL)

# Find lpsolve library
find_library(LP_SOLVE NAMES liblpsolve55.so PATHS /usr/lib/lp_solve)

if (NOT LP_SOLVE)
  message(FATAL_ERROR "This program requires the lp_solve library, and will not be compiled.")
else ()
  message(STATUS "Library lp_solve found: ${LP_SOLVE} ${BLAS}")

  set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

  include_directories (BEFORE ${VOLESTIROOT}/external)
  include_directories (BEFORE ${VOLESTIROOT}/external/minimum_ellipsoid)
  include_directories (BEFORE ${VOLESTIROOT}/include/generators)
  include_directories (BEFORE ${VOLESTIROOT}/include/volume)
  include_directories (BEFORE ${VOLESTIROOT}/include)
  include_directories (BEFORE ${VOLESTIROOT}/include/lp_oracles)
  include_directories (BEFORE ${VOLESTIROOT}/include/nlp_oracles)
  include_directories (BEFORE ${VOLESTIROOT}/include/convex_bodies)
  include_directories (BEFORE ${VOLESTIROOT}/include/random_walks)
  include_directories (BEFORE ${VOLESTIROOT}/include/annealing)
  include_directories (BEFORE ${VOLESTIROOT}/include/ode_solvers)
  include_directories (BEFORE ${VOLESTIROOT}/include/root_finders)
  include_directories (BEFORE ${VOLESTIROOT}/include/samplers)
  include_directories (BEFORE ${VOLESTIROOT}/include/misc)
  include_directories (BEFORE ${VOLESTIROOT}/include/optimization)

  # for Eigen
  if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
    add_compile_options(-D "EIGEN_NO_DEBUG")
  else ()
    add_compile_definitions("EIGEN_NO_DEBUG")
  endif ()

  # add_definitions(${CMAKE_CXX_FLAGS} "-std=c++11")  # enable C++11 standard
  add_definitions(${CMAKE_CXX_FLAGS} "-O3")  # optimization of the compiler
  # add_definitions(${CMAKE_CXX_FLAGS} "-O3 -g")  # Debug mode for perf report
  #add_definitions(${CXX_COVERAGE_COMPILE_FLAGS} "-lgsl")
  add_definitions(${CXX_COVERAGE_COMPILE_FLAGS} "-lm")
  add_definitions(${CXX_COVERAGE_COMPILE_FLAGS} "-ldl")
  add_definitions(${CXX_COVERAGE_COMPILE_FLAGS} "-DBOOST_NO_AUTO_PTR")
  add_definitions(${CMAKE_CXX_FLAGS} "-DMKL_ILP64")
  #add_definitions(${CXX_COVERAGE_COMPILE_FLAGS} "-lgslcblas")
  #add_definitions( "-O3 -lgsl -lm -ldl -lgslcblas" )

  include_directories(BEFORE "${PROJECT_SOURCE_DIR}")
  include_directories(BEFORE "${PROJECT_SOURCE_DIR}/sampling_algorithms")
  include_directories(BEFORE "${PROJECT_SOURCE_DIR}/automatic_differentiation")

  add_subdirectory(sampling_algorithms)
  add_subdirectory(automatic_differentiation)
  add_executable(sample sample.cpp)
  # add_executable(sample sample.c)
  set_property(TARGET sample PROPERTY CXX_STANDARD 17) # enable C++17 standard

  target_link_libraries(logconcave_hmc probabilistic_programming ${LP_SOLVE} ${BLAS} ${MKL_LINK})
  target_link_libraries(hit_and_run ${LP_SOLVE} ${BLAS} ${MKL_LINK})
  target_link_libraries(sample logconcave_hmc hit_and_run runtime_data probabilistic_programming)
  
endif()
