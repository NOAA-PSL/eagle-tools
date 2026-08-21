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

### Compare Model Performance

Create only two scorecard-style model performance plot types from metric
NetCDF files: regional improvement heatmaps and all-response violin plots.

```
eagle-tools performance-heatmap config.yaml
eagle-tools performance-violin config.yaml
```

Example config files are included in `src/eagle/tools/config/performance_heatmap.yaml`
and `src/eagle/tools/config/performance_violin.yaml`.
These configs use one `input_path` root and one `output_path`; each model only
needs a directory name for the common scorecard layout. Standard model labels,
colors, filename patterns, regions, variables, and levels have built-in
defaults, and can be overridden in YAML when needed.

Expected input layout:

```
new_data/
  nested_eagle_global_2025/
    rmse.convobs.nested-global.nc
    rmse.convobs.nested-global.conus.nc
  gfs_2025/
    rmse.convobs.global.nc
    rmse.convobs.global.conus.nc
  aifs_2025/
  aigfs_2025/
  nested_eagle_lam_2025/
    rmse.convobs.nested-lam.nc
  hrrr_2025/
    rmse.convobs.lam.nc
```

Common built-in model keys are `nested_eagle_global`, `nested_eagle_lam`,
`gfs`, `aifs`, `aigfs`, `ecmwf_ifs`, and `hrrr`.

Minimal config edits:

```
metric: rmse
input_path: /path/to/new_data
output_path: /path/to/plots
```

By default, all selected models are evaluated on their exact overlapping
initialization times (`t0`) and forecast hours (`fhr`). This keeps model
performance comparisons one-to-one even when one model has only a month of data
and another has a full year.

Optional temporal filters can be added to either config:

```
require_exact_time_match: true
start_date: "2025-01-01"
end_date: "2025-12-31"
years: [2025]
months: [1, 2, 12]
```

Use `years` for one or more years, `months` for one or more months, or
`start_date` / `end_date` for a precise date window. Filters are applied before
matching models.

If your model directory names match the defaults, no `models` block is needed.
If they differ, add only the directory overrides:

```
models:
  gfs:
    directory: my_gfs_scores
  aifs:
    directory: my_aifs_scores
```

For heatmaps, add or remove comparisons by editing `candidate_model` and
`baseline_model`:

```
plots:
  - candidate_model: nested_eagle_global
    baseline_model: gfs
```

For violin plots, add or remove models by editing the `models` list:

```
plots:
  - regions: [global, conus]
    lead_hours: [24, 240]
    models: [nested_eagle_global, gfs, aigfs, aifs]
```

Output names include plot type, regions, models, metric, and lead range, for
example `heatmap_regions-global-nh-sh-conus_models-nested_eagle_global-vs-gfs_rmse_d1-d10.png`.
Violin plots also write a small summary CSV.
