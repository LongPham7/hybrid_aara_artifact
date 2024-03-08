# Docker Image of Hybrid Resource-Aware ML (RaML)

This repository contains the code for building a Docker image of Hybrid
Resource-Aware ML (RaML). Hybrid RaML was submitted to PLDI 2024 Artifact
Evaluation. The directory `README` contains LaTex source files for the
instructions of the artifact.

To build a Docker image, clone this repository and run (in
the root directory)
```
docker build -t hybrid_aara .
```
It creates a Docker image named `hybrid_aara`. We need a period at the end of
the command to indicate that Dockerfile exists in the current working directory.
The build will take 20-30 minutes.

To run the image `hybrid_aara`, run
```
docker run --name hybrid_aara -it --rm
```
It creates a Docker container (i.e., a runnable instance of a Docker image)
named `hybrid_aara`. The container then starts a shell, through which the user
can interact and experiment with Hybrid RaML.

To save the Docker image as a tar archive and compress it, run
```
docker save hybrid_aara | gzip > hybrid_aara.tar.gz
```
