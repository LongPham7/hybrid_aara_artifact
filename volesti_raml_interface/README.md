# Interface between volesti and Hybrid RaML

This project creates shared libraries (implemented in C++) that call the
hit-and-run algorithm and reflective HMC implemented in the C++ library
[volesti](https://github.com/GeomScale/volesti). In turn, the shared libraries
are called by Hybrid RaML through the OCaml-C binding.

To build the shared libraries, first go to the root of this project directory.
Then run
```
mkdir build && cd build
cmake ..
cmake --build .
```
