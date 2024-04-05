PLDI 2024 Artifacts Paper #57 Reviews and Comments
===========================================================================
Paper #57 Robust Resource Bounds with Static Analysis and Bayesian
Inference


Review #57A
===========================================================================

Overall merit
-------------
5. Reusable

Reviewer expertise
------------------
1. Outsider: I know nothing about the area at all

Paper summary
-------------
This paper targets the problem of automatically analyzing resource consumption in programs. It proposes two novel and robust statistical analysis algorithms leveraging Bayesian inference techniques and Hamiltonian Monte Carlo sampling. The paper then devices a hybrid approach combining static analysis based on AARA and data-driven analysis based the aforementioned techniques. The proposed approach is evaluated on a range of functional programs.

Comments for authors
--------------------
The `README.md` is well-rewritten and easy to follow, thanks!

I managed to follow the step-by-step instructions on an example `append.ml` program and on the benchmark suite mentioned in the paper. All scripts ran perfectly. The source code looks very solid too.

The authors have mentioned in the `README.md` that some hyperparameters to the Bayesian inference were changed. I obtained results that are similar to what were described in the paper, and I think the differences are in an acceptable range and still support all the claims in the paper.

I also managed to run the reproduction scripts (`python3 run_experiment.py all`), which terminate within 2 hours as claimed in the `README.md`. I then follow steps described in Section 3.2, and again I obtained similar results.

I would like to consider a Reusable badge.



Review #57B
===========================================================================

Overall merit
-------------
2. Pass Phase 1, Pending Phase 2

Reviewer expertise
------------------
2. Knowledgeable: I know something about this area

Paper summary
-------------
The paper introduces two Bayesian resource analysis techniques: BayesWC and BayesPC; and combines these data-driven approaches with static resource analysis techniques. The authors provided a Docker image along with a README file. The Docker image is self-contained with all the dependencies installed and scripts to reproduce the tables and figures of the paper. The linked GitHub repo contains the source code and instructions on the installation and custom usage of the resulting tool Hybrid AARA.

Comments for authors
--------------------
The authors carefully packaged the artifact with a detailed documentation. I found it easy to follow the getting started guide and was able to reproduce the initial results without any issues.



Review #57C
===========================================================================
* Updated: Apr 3, 2024, 3:56:48 PM UTC

Overall merit
-------------
5. Reusable

Reviewer expertise
------------------
3. Expert: this is my area

Paper summary
-------------
# Phase 2 review

The artifact provided by the authors contains a recent version of the
paper, a docker image, as well as a well-structured README.pdf
containing explanations on the analyses performed, detailed step-by-step
instructions to interact easily with the docker image, test the analyser
on simple examples, and reproduce all experiments and tables presented
in the paper.

Additionally, a link to a GitHub repository is provided, containing the
source code and instructions to rebuild a docker image, which is
required for some architectures.


Scripts to reproduce the experiments are very complete, well explained,
and convenient: output tables and images are provided following the
format of the paper.

The code itself is very well organised and commented, making it easy to
tweak. Instructions to recompile the necessary parts of the source after
a tweak are provided in the README.

To test new experiments, explanations are given in the README. This
requires writing both the source of a bench, and to include a set of
input data in the source. Such input data can be generated using a
script provided by the authors and described in the README.

Comments for authors
--------------------
This is a very solid artifact.
I would like to recommend awarding both `Functional` and `Reusable`
badge.

Details, following the required features for both badges, are given
below.

## `Functional`

### `Documented` (Presence of a README)

A very complete and very helpful README is present in the paper
providing all details to understand and use the artifact (including
interaction with the docker image itself) perform and understand
experiments of the paper, rebuild and tweak the source from the docker
image, and test custom benchmarks for new usages.

### `Consistent`

The result of the experiments follow those reported in the paper,
within expected deviations.

### `Complete`

Every experimental claim of the paper is directly backed up by the
artifact.

### `Exercisable`

The artifact contains all scripts to reproduce experiments of the paper.
This is done to a very helpful extent, with scripts reproducing tables
and graphs in the same format as in the paper.
This can be done both for the original results obtained by the authors,
and for new (similar) results obtained after rerunning the experiments.

## `Reusable`

### Code structure and documentation

As said above,
> The code itself is very well organised and commented, making it easy to
> tweak. Instructions to recompile the necessary parts of the source after
> a tweak are provided in the README.

### Using the artifact of new inputs

Explanations for this are given in the README,
and scripts are provided to help in this task.

As said above,
> This requires writing both the source of a bench, and to include a set
> of input data in the source. Such input data can be generated using a
> script provided by the authors and described in the README.

As discussed with the authors, to perfect the tools for users, this
process could be made a bit more convenient, in particular regarding
inclusion of input data in benchmarks, although this is beyond the level
of expectation for an artifact.
Authors have declared they would update one of the scripts intended for
this task.

# Conclusion

Once again, this is a very solid benchmark, for which I thank the
authors, and would like to recommend awarding both `Functional` and
`Reusable` badge.



Comment @A1 by <longp@andrew.cmu.edu> (Author)
---------------------------------------------------------------------------
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


Comment @A2 by Reviewer C
---------------------------------------------------------------------------
Dear authors,

Thanks for the update and extra info!

## Overall comment

I am satisfied with the modifications.
Experiments with rebuilding the code (on a docker image rebuilt for my architecture) and testing new benchmarks went fine (it will probably be easier for users with your revised `input_data_generation.py` script, which is great). The source code appears well-written and organised.

I will quickly check that I do not identify experimental claims in the papers that are not supported by the artifact, but I have a priori no further concerns with recommending both `Functional` and `Reusable` badges.

Minor comments are included below.

## Minor comments

+ > We could not identify the root cause of the issue. We believe it is a technical issue of Docker, rather than our code.

   I believe this an expected behaviour from Docker, more than an issue: docker containers, unlike VMs, are not a priori portable to foreign hardware, unless they have been explicitly built for several hardware including them.
   Given the policy stated of PLDI's website, and that you provide information to rebuild both the image and internal code if necessary, I have no further issues with this point.

+ Typos?
    - Section 4.4 of the README, I believe there is no `home/hybrid aara/volesti raml interface/bin`, but there is a `[...]/build`.

+ Designing new benchmarks.
   - It seems to me that your updated `input_data_generation.py` will be quite useful for users, e.g. to compare with other tools on predefined benchmarks (just translated to RaML). Great!
     It is a bit inconvenient for users to have to include input_data manually (or write a script for it) in the bench source (after generation by the python script) rather than this being done by the solver itself, but this is more of a convenience question than something expected for an artifact.
   - Some knowledge of RaML and its limitations appears useful in order to write new benchmarks in a way supported by the solver, but you explain key points in both the paper and the README, and users can use your set of benchmarks to infer key points, so I have no issues with that.

A solid artifact, thanks for submitting it!


Comment @A3 by <longp@andrew.cmu.edu> (Author)
---------------------------------------------------------------------------
Dear Reviewer C,

Thank you for updating your review. We are happy that you find our artifact
reusable. Also, thanks for pointing out the typo in Section 4.4 of `README.pdf`.
We have fixed it in the GitHub repository, and will update the file on Zenodo as
well.