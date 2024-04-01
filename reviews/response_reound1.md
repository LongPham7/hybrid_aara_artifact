Thank you all for the feedback. We have revised the artifact and `README.pdf`.
They are available on [Zenodo](https://doi.org/10.5281/zenodo.10901662).

# Review A

Thank you for taking the time to run all Python scripts. We are happy that you
did not encounter any technical issues with the artifact.

# Review B

Thank you for going through the getting-started guide. We are happy that you did
not encounter any technical issues with the artifact.

# Review C

> The binary included in the docker image crashes immediately (Illegal
> instruction (core dumped)) on my architecture (described below to ease
> debugging).

Thank you for reporting this issue. The same error message has been reported in
other Docker images (e.g.,
[here](https://github.com/ultralytics/ultralytics/issues/7085)). We suspect that
the prebuilt Docker image requires an instruction that your computer does not
support. We built the Docker image on the OS Ubuntu 22.04 LTS and the CPU
Intel(R) Core(TM) i7-7600U CPU @ 2.80GHz. We also tested the prebuilt image on
macOS with Apple silicon, and it ran successfully.

We could not identify the root cause of the issue. We believe it is a technical
issue of Docker, rather than our code. As stated on the website of PLDI 2024
Artifact Evaluation, it is recommended to provide prebuilt Docker images (or
VMs), instead of providing scripts for building them. Hence, we will keep the
artifact in the form of a prebuilt Docker image, even though it is not
self-contained anymore if the user has trouble with the prebuilt image.

If you do not mind, could you please run the artifact by cloning the GitHub
repository and building a Docker image locally on your machine? We are happy
that at least you successfully built the image on your machine.

In Section 2 of the revised `README.pdf`, we refer the reader to Section 4.2 if
they have trouble running the prebuilt image.

> It would be good to add instructions in the README to recompile the code
> directly inside the docker, and avoid those crashes. This would additionally
> allow user to tweak the code directly inside the docker, and observe changes
> there.

Thank you for the suggestion.

In Section 4.4 of the revised `README.pdf`, we provide details of how to
recompile (i) the C++ code of the volesti-RaML interface and (i) the OCaml code
of Hybrid RaML. The remaining code, such as the Python scripts, does not need
recompilation after modification.

> Being able to test to analyser on new benchmarks is required to grant the
> "Reusable" badge. It would be helpful to add a small paragraph in the README
> instructing how to do so, and include tools in the docker image to make this
> process easier.

If the user wishes to analyze their custom OCaml program using Hybrid RaML, the
user should first prepare (i) the source file of their OCaml program, augmented
with a collection of inputs for generating runtime cost data and (ii) a
configuration file specifying hyperparameters of Hybrid AARA. To learn the
formats of these files, the user can look at the examples inside the artifact.
Finally, the user runs Hybrid RaML as demonstrated in Section 1 of `README.pdf`.

In Section 4.3 of the revised `README.pdf`, we explain this point.

> It would improve the artifact to include an editor.

We will install the text editor Vim in the updated Docker image. Alternatively,
if the user wishes to install a different text editor, while the Docker
container is running, the user can run `apt update && apt install <editor-name>
-y` inside the container.

> Should input_dataset be entered manually by the users on each new benchmark?
> Are scripts provided to generate them automatically?

The artifact already comes with a Python script (`input_data_generation.py`
inside the directory `home/hybrid_aara/benchmark_suite/toolbox/`) for generating
lists, pairs of lists (for the benchmark MapAppend), and nested lists (for the
benchmark Concat). For example, the function `create_exponential_lists` inside
this script generates lists whose sizes grow exponentially (e.g., 1, 2, 4, 8,
16, etc.). Here, the initial size (i.e., 1) and exponent (i.e., 2) are specified
by the user. The content of each list is determined randomly. The function
`convert_lists_python_to_ocaml` then converts a list from the Python format to
the OCaml format.

In Section 4.3 of the revised `README.pdf`, we mention this Python script. Also,
we will revise the script such that, when the user runs `python3
input_data_generation.py`, the script prints out an example collection of input
lists used for runtime cost data generation.
