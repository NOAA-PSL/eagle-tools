# eagle-tools

Tools for processing and evaluating anemoi based EAGLE ML models

## ⚠️  Disclaimer ⚠️

This package is pip-installable, but it is more in the form of research code
rather than well-documented and tested software.
There are likely better and more efficient ways to accomplish the main
functionality of this package, but this gets the job done.

## Installation

Since some dependencies are only available on conda, it's recommended to create
a conda environment for all dependencies.
Note that this package is not (yet) available on conda, but it can still be
installed via pip.

Note also that the module load statements are for working on Perlmutter, and
would need to be changed for different machines.

```
conda env create -n eagle
conda install -c conda-forge ufs2arco
module load gcc cudnn nccl
pip install anemoi-datasets anemoi-graphs anemoi-models anemoi-training anemoi-inference anemoi-utils anemoi-transform
pip install flash-attn --no-build-isolation
pip install git+https://github.com/timothyas/xmovie.git@feature/gif-scale
pip install eagle-tools
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

Compute the power spectra, averaged of initial conditions.

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
