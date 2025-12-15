# eagle-tools

Tools for processing and evaluating anemoi based EAGLE ML models

## ⚠️  Disclaimer ⚠️

This package is pip-installable, but it is more in the form of research code
rather than well-documented and tested software.
There are likely better and more efficient ways to accomplish the main
functionality of this package, but this gets the job done.

## Installation

For more discussion on installing the right versions of torch and
flash-attention, see
[this discussion](https://github.com/NOAA-PSL/eagle-tools/discussions/29).

### Install as a user

Since some dependencies are only available on conda, it's recommended to create
a conda environment for all dependencies.
Note that this package is not (yet) available on conda, but it can still be
installed via pip.

Note also that the module load statements are for working on Perlmutter, and
would need to be changed for different machines.

```
module load cudnn nccl
conda create -n eagle -c conda-forge python=3.12 ufs2arco
conda activate eagle
pip install git+https://github.com/timothyas/xmovie.git@feature/gif-scale
pip install anemoi-datasets anemoi-graphs anemoi-models anemoi-training[azure] anemoi-inference anemoi-utils anemoi-transform
pip install eagle-tools
pip install "torch<2.7" torchvision
pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1
pip install "mlflow-skinny<3.0"
```

Note that it is no longer necessary to `module load gcc` since `gcc-native` is a
loaded default.
Also, it is possible to install ufs2arco without mpich as detailed
[here](https://ufs2arco.readthedocs.io/en/latest/installation.html#install-from-conda-forge-without-mpi),
since this may be necessary to hook up to prebuilt MPI distributions on different HPC machines.

### Install as a developer (Perlmutter example)

It is sometimes necessary to install anemoi, ufs2arco, and eagle-tools repos so
that they are modifiable.
This requires a slightly different path than the one outlined above.
The following are steps that worked on Perlmutter on Dec 9, 2025.
Unfortunately some packages (e.g. flash-attn, torch) through different errors
based on how the machine is configured, so your mileage may vary.

Note that here we set the environment `repo_path`, which assumes that all
repositories are located in that location.
This will need to be changed as necessary based on your repo locations.
Also, developers may not need to install editable versions of every single repo
as is done here, it's up to you.

```
module load cudnn nccl
export repo_path=$HOME
conda create -n eagle -c conda-forge python=3.12 xesmf esmf=*=nompi* jupyter seaborn
conda activate eagle
MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
pip install git+https://github.com/timothyas/xmovie.git@feature/gif-scale
pip install -e $repo_path/anemoi-utils
pip install -e $repo_path/anemoi-transform
pip install -e $repo_path/anemoi-datasets
pip install -e $repo_path/anemoi-core/graphs
pip install -e $repo_path/anemoi-core/models
pip install -e $repo_path/anemoi-core/training[azure]
pip install -e $repo_path/anemoi-core/inference
pip install -e $repo_path/ufs2arco
pip install -e $repo_path/eagle-tools
pip install "torch<2.7" torchvision
pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1
pip install "mlflow-skinny<3.0"
```


## Usage

This provides the following functionality.
Note that each command uses a configuration yaml, and documentation of the yaml
contents can be found by running `eagle-tools <command> --help`.
For example, one can run `eagle-tools inference --help` to get documentation.

### Inference

Run
[anemoi-inference](https://anemoi.readthedocs.io/projects/inference/en/latest/)
over many initial conditions

```
eagle-tools inference config.yaml
```

### Averaged Error Metrics

Compute Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), preserving the initial
condition dimension (t0).

```
eagle-tools metrics config.yaml
```

### Spatial Error Metrics

Compute the spatial distribution of RMSE and MAE for each lead time.
By default, these are averaged over all initial conditions used.

```
eagle-tools spatial config.yaml
```

### Power Spectra

Compute the power spectrum, averaged of initial conditions.

```
eagle-tools spectra config.yaml
```


### Visualize Predictions Compared to Targets

Make figures or movies, showing the targets and predictions.
Note that the argument `end_date` has different meanings for each.
For figures, `end_date` is the date plotted, whereas for movies, all timestamps
between `start_date` and `end_date` get shown in the movie.

```
eagle-tools figures config.yaml
eagle-tools movies config.yaml
```
