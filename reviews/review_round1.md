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

Overall merit
-------------
2. Pass Phase 1, Pending Phase 2

Reviewer expertise
------------------
3. Expert: this is my area

Paper summary
-------------
The artifact provided by the authors contains a recent version of the paper,
a docker image, as well as a well-structured README.pdf containing
explanations on the analyses performed, detailed step-by-step instructions
to interact easily with the docker image, test the analyser on simple
examples, and reproduce all experiments and tables presented in the paper.

Additionally, a link to a GitHub repository is provided, containing the
source code and instructions to rebuild a docker image.

The code itself, at a first glance, seems well organised, well commented,
and easy to tweak. (This will be evaluated in more detail during Phase 2.)

Comments for authors
--------------------
## Overall comment

In the context of evaluating the Functional badge, the artifact and
corresponding step-by-step instructions are well-organised, precise and
helpful. Scripts are easy to use and reproduce experimental results of
the paper, up to small expected deviations.

I have an overall very good impression of this artifact, although a few
key updates, described below, are either required to complete it or
could improve it further.

In short,
1) The artifact is not currently entirely self-contained, as I need to
      rebuild the docker image from a github repo to avoid crashes.
   Detail on supported architectures and/or instruction to rebuild
    internally would be appreciated.
2) For the reusable badge, instructions in the README on running new
	benchmarks would be appreciated.

## Portability. Internal build script/instructions.

The binary included in the docker image crashes immediately (`Illegal
instruction (core dumped)`) on my architecture (described below to ease
debugging).

No instructions are provided to rebuild the code *inside the docker
container*, although a github repo is provided to rebuild the image,
which allowed me to perform the experiments.

The artifact being self-contained is helpful to ensure its future
availability.

It would be good to add instructions in the README to recompile the code
directly inside the docker, and avoid those crashes.
This would additionally allow user to tweak the code directly inside the
docker, and observe changes there.

## Running new benchmarks.

Being able to test to analyser on new benchmarks is required to grant
the "Reusable" badge.
It would be helpful to add a small paragraph in the README instructing
how to do so, and include tools in the docker image to make this process
easier.

### Editor?

I do not find a text editor installed in the docker (`nano`, `vi`, ...),
making it required to either copy new benchs from outside, or use the
impractical `cat > new_bench.ml` approach.

It would improve the artifact to include an editor.

### input_dataset?

Should input_dataset be entered manually by the users on each new
benchmark? Are scripts provided to generate them automatically?

## Architecture (for debugging)

I run experiments on the following architecture.
On this architecture, the precompiled binary in the docker image crashes.

- OS: Debian/GNU Linux 11 (amd64)
- RAM: 128 GB DDR4 @2666 MHz
- CPU: Intel Xeon Gold 6154 @3GHz (72 cores)