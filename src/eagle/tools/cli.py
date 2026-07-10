import click

@click.group()
def cli():
    """A CLI for the Eagle Tools suite."""
    pass


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def inference(config_file):
    """
    Run inference.
    """
    from eagle.tools.inference import main
    main(config_file)

inference.help = """Runs Anemoi inference pipeline over many initialization dates.

    \b
    Note:
        There may be ways to do this directly with anemoi-inference, and
        there might be more efficient ways to parallelize inference by
        better using anemoi-inference.
        However, this works, especially for low resolution applications.

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        start_date (str): The first initial condition date to process.
        \b
        end_date (str): The last initial condition date to process.
        \b
        freq (str): Frequency string for the date range (e.g., "6h").
        \b
        lead_time (int): Forecast lead time in hours (e.g., 240 = 240h = 10days).
        \b
        checkpoint_path (str): Path to the trained model checkpoint for inference.
        \b
        input_dataset_kwargs (dict): A dictionary of arguments passed to
            anemoi-dataset to open an anemoi dataset for initial conditions.
        \b
        output_path (str): Directory where the output NetCDF files will be saved in the format
            f"{output_path}/{t0}.{lead_time}h.nc", or
            if extract_lam=True, then f"{output_path}/{t0}.{lead_time}h.lam.nc"
        \b
        runner (str, optional): The name of the anemoi-inference runner to use.
            Defaults to "default".
        \b
        extract_lam (bool, optional): If True, extracts and saves only the LAM
            (Limited Area Model) domain from the output. Only used for Nested model configurations.
            Defaults to False.
        \b
        use_mpi (bool, optional): If True, distribute initialization dates across MPI ranks.
            Each rank loads its own copy of the model onto its GPU. Launch with
            ``srun --ntasks=N --gpus-per-task=1`` to bind one GPU per rank.
            Cannot be combined with ``runner: parallel``. Defaults to False.
        \b
        log_path (str, optional): When using MPI, the directory where per-rank log files
            are saved. Defaults to "eagle-logs/inference".

        \b
        base_seed (int, optional): base seed for running a stochastic model
        \b
        overwrite_existing (bool, optional): If True, re-run inference even when
            the output NetCDF file already exists. If False, skip initialization
            dates whose output files are already present. Defaults to False.
        \b
        vars_of_interest (list[str], optional): A subselection of variables to store
        from the forecast. Note that variable names need the pressure level prefix here,
        unlike all other eagle-tools workflows (e.g., t_850, t_500 rather than selecting
        "t" and levels: [500, 850]). If None, all variables are used. Defaults to None.
    """


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def postprocess(config_file):
    """
    Run postprocessing.
    """
    from eagle.tools.postprocess import main
    main(config_file)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def metrics(config_file):
    """
    Compute error metrics.
    """
    from eagle.tools.metrics import main
    main(config_file)

metrics.help = """Compute grid cell area weighted RMSE and MAE.

    \b
    This function processes forecast and verification datasets over a specified
    date range, computes the Root Mean Square Error (RMSE) and Mean Absolute
    Error (MAE) between them, and saves the results to NetCDF files.

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        model_type (str): The type of model grid, one of: "global", "lam",
            "nested-lam", "nested-global".
            This determines how grid cell area weights, edge trimming, and coordinates are handled.
        \b
        verification_dataset_path (str): The path to the anemoi dataset with target data
            used for comparison.
        \b
        forecast_path (str): The directory path containing the forecast datasets.
        \b
        output_path (str): The directory where the output NetCDF files will be saved, as
            f"{output_path}/rmse.{model_type}.nc" and
            f"{output_path}/mae.{model_type}.nc"
            Subregion metrics (see ``subregions`` below) are written with a
            region suffix, e.g. f"{output_path}/rmse.{model_type}.{region}.nc".
        \b
        subregions (dict, optional): Geographic subregions for regional metrics, in
            addition to the full-field ("global") metrics. Each entry maps a name to
            latitude and/or longitude bounds, e.g.
            ``{conus: {latitude: [25, 50], longitude: [-125, -65]}}``. Longitude
            bounds are given in [-180, 180]. A mask file is written to
            f"{output_path}/subregions.{model_type}.nc". Defaults to None.
        \b
        start_date (str): The first initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        end_date (str): The last initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        freq (str): The frequency string for generating the date range between
            start_date and end_date (e.g., "6h"), passed to pandas.date_range.
        \b
        lead_time (int): Length of forecast in hours.
        \b
        from_anemoi (bool, optional): If True, opens forecast data using the
            anemoi inference dataset format. Otherwise, assumes layout of dataset
            created by ufs2arco using a base target layout. Defaults to True.
        \b
        lam_index (int, optional): For nested models (e.g., model_type="nested-lam"), this integer
            specifies the number of grid points belonging to the LAM domain.
            Defaults to None.
        \b
        levels (list, optional): A list of vertical levels to subset from the
            datasets. If None, all levels are used. Defaults to None.
        \b
        vars_of_interest (list[str], optional): A list of variable names to
            include in the analysis. If None, all variables are used. Defaults to None.
        \b
        trim_edge (int, optional): Specifies the number of grid points to trim
            from the edges of the verification dataset. Only used for LAM or Nested configurations.
            Defaults to None.
        \b
        trim_forecast_edge (int, optional): Specifies the number of grid points to
            trim from the edges of the forecast dataset. Defaults to None.
        \b
        forecast_regrid_kwargs (dict, optional): options passed to ufs2arco.transforms.horizontal_regrid
        \b
        target_regrid_kwargs (dict, optional): options passed to ufs2arco.transforms.horizontal_regrid
        \b
        use_mpi (bool, optional): if True, use a separate MPI process per initial condition
        \b
        log_path (str, optional): if using MPI, provide a path to where the logs get saved (one per MPI process)
    """


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def fss(config_file):
    """
    Compute the Fractions Skill Score.
    """
    from eagle.tools.fss import main
    main(config_file)

fss.help = """Compute the deterministic Fractions Skill Score (FSS).

    \b
    Implements the neighborhood-based FSS of Roberts & Lean (2008, MWR). For each
    threshold and neighborhood radius, forecast and verification fields are
    converted to binary exceedance fields, the fraction of exceeding grid points
    within a square neighborhood is computed for each grid cell, and
    FSS = 1 - MSE / MSE_ref, where MSE = mean((M-O)^2) and MSE_ref = mean(M^2+O^2)
    over the domain (M, O = forecast/observed neighborhood fractions).

    \b
    Results are stored per initial condition and forecast hour (no averaging over
    initial conditions), so downstream code can both aggregate correctly
    (FSS = 1 - sum(mse)/sum(mse_ref) over t0) and estimate statistical significance.

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        forecast_path (str): Directory containing the forecast NetCDF files, named
            f"{forecast_path}/{t0}.{lead_time}h.nc".
        \b
        output_path (str): Directory where the output is saved, as
            f"{output_path}/fss.{model_type}.nc". The result has data_vars
            fss, mse, mse_ref with dims (t0, fhr, threshold, radius). If
            percentiles are configured, a second file
            f"{output_path}/fss.percentile.{model_type}.nc" is written with
            data_vars pfss, pmse, pmse_ref and dims (t0, fhr, percentile, radius).
        \b
        lead_time (int): Forecast length in hours, used only to build the forecast
            filename.
        \b
        forecast_hours (list[int]): The specific lead times (hours) at which to
            compute FSS.
        \b
        thresholds (list[float]): Exceedance thresholds, in the units of the fields
            (e.g. mm of precip).
        \b
        percentiles (list[float], optional): If present, additionally compute a
            percentile-threshold FSS (Roberts & Lean, 2008). Each field is
            binarized at its own P-th percentile value, quantile(P/100), computed
            separately for the forecast and observations over valid, wet (> 0) grid
            points at each lead time (wet-only avoids the dry-mass degeneracy where
            low percentiles land at 0 mm); deriving the thresholds per field removes
            rainfall-amount bias to isolate spatial accuracy. If omitted, only
            threshold FSS is done.
        \b
        radius (float | list[float]): Neighborhood half-width radius in km. A window
            of side 2*round(radius/grid_spacing_km)+1 grid points is used. May be a
            scalar or a list; the output carries a radius dimension.
        \b
        grid_spacing_km (float, optional): Grid spacing in km, used to convert radius
            to grid points. Defaults to 6.0.
        \b
        verification_dataset (dict): Config for the gridded verification dataset. Keys:
            path (str): Path to the zarr store (plain gridded (time, y, x) dataset).
            precip_varname (str, optional): Name of the precip variable. Defaults to
                the sole data_var if the dataset has exactly one.
            trim_edge (dict, optional): Per-axis grid points to trim, e.g.
                {x: [lo, hi], y: [lo, hi]}, to align with the forecast grid.
        \b
        forecast_dataset (dict): Config for the forecast dataset. Keys:
            model_type (str): The model grid type, e.g. "nested-lam".
            precip_varname (str): Name of the precip variable in the forecast.
            lam_index (int, optional): For nested models, the number of grid points
                belonging to the LAM domain.
            lcc_info (dict, optional): Lambert Conformal Conic details {n_x, n_y}.
            trim_forecast_edge (list[int], optional): Additional edge trimming.
            from_anemoi (bool, optional): Included for parity with other workflows;
                forecasts are opened via the anemoi inference format.
        \b
        start_date (str): The first initial condition date to process.
        \b
        end_date (str): The last initial condition date to process.
        \b
        freq (str): Frequency string for the date range (e.g., "6h").
        \b
        use_mpi (bool, optional): If True, distribute initial conditions across MPI ranks.
        \b
        log_path (str, optional): When using MPI, the directory for per-rank log files.
    """


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def spatial(config_file):
    """
    Compute spatial error metrics.
    """
    from eagle.tools.spatial import main
    main(config_file)

spatial.help = """Compute spatial maps of RMSE and MAE

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        keep_t0 (bool, optional): If True, keeps the initial condition time (t0)
            as a separate dimension in the output file. This can produce very large results
            and requires a lot of memory. If False, the metrics
            are averaged over all initial conditions. Defaults to False.

    \b
    Config Args common to metrics.py:
        model_type (str): The type of model grid, one of: "global", "lam",
            "nested-lam", "nested-global".
            This determines how grid cell area weights, edge trimming, and coordinates are handled.
        \b
        verification_dataset_path (str): The path to the anemoi dataset with target data
            used for comparison.
        \b
        forecast_path (str): The directory path containing the forecast datasets.
        \b
        output_path (str): The directory where the output NetCDF files will be saved, as
            f"{output_path}/spatial.rmse.{model_type}.nc" and
            f"{output_path}/spatial.mae.{model_type}.nc"
            or if keep_t0=True, then as
            f"{output_path}/spatial.rmse.perIC.{model_type}.nc" and
            f"{output_path}/spatial.mae.perIC.{model_type}.nc"
        \b
        start_date (str): The first initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        end_date (str): The last initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        freq (str): The frequency string for generating the date range between
            start_date and end_date (e.g., "6h"), passed to pandas.date_range.
        \b
        lead_time (int): Length of forecast in hours.
        \b
        from_anemoi (bool, optional): If True, opens forecast data using the
            anemoi inference dataset format. Otherwise, assumes layout of dataset
            created by ufs2arco using a base target layout. Defaults to True.
        \b
        lam_index (int, optional): For nested models (e.g., model_type="nested-lam"), this integer
            specifies the number of grid points belonging to the LAM domain.
            Defaults to None.
        \b
        levels (list, optional): A list of vertical levels to subset from the
            datasets. If None, all levels are used. Defaults to None.
        \b
        vars_of_interest (list[str], optional): A list of variable names to
            include in the analysis. If None, all variables are used. Defaults to None.
        \b
        trim_edge (int, optional): Specifies the number of grid points to trim
            from the edges of the verification dataset. Only used for LAM or Nested configurations.
            Defaults to None.
        \b
        trim_forecast_edge (int, optional): Specifies the number of grid points to
            trim from the edges of the forecast dataset. Defaults to None.
    """


@cli.command("spatial-forecast")
@click.argument('config_file', type=click.Path(exists=True))
def spatial_forecast(config_file):
    """
    Compute spatial error metrics between two forecasts.
    """
    from eagle.tools.spatial_forecast import main
    main(config_file)

spatial_forecast.help = """Compute spatial maps of RMSE and MAE between two forecast datasets.

    \b
    forecast1 is treated as the reference (target) and forecast2 as the prediction.
    Each forecast can independently be opened as an anemoi inference dataset or a
    forecast zarr dataset via its ``from_anemoi`` flag, and may have a different
    model_type (e.g. one nested-global and one global). Area weights are derived
    from forecast1's grid.

    \b
    Output filenames encode both model types:
        f"{output_path}/spatial.rmse.{fc1_model_type}v{fc2_model_type}.fc1vfc2.nc"
        f"{output_path}/spatial.mae.{fc1_model_type}v{fc2_model_type}.fc1vfc2.nc"
        or if keep_t0=True, then as
        f"{output_path}/spatial.rmse.perIC.{fc1_model_type}v{fc2_model_type}.fc1vfc2.nc"
        f"{output_path}/spatial.mae.perIC.{fc1_model_type}v{fc2_model_type}.fc1vfc2.nc"

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        forecast1 (dict): Config for the reference forecast. Keys:
            model_type (str): The type of model grid, one of: "global", "lam",
                "nested-lam", "nested-global".
            path (str): Directory containing per-date NetCDF files (from_anemoi=True)
                or path to the zarr store (from_anemoi=False).
            from_anemoi (bool, optional): If True, opens with the anemoi inference
                dataset format. Defaults to True.
            lam_index (int, optional): For nested models, the number of grid points
                belonging to the LAM domain. Defaults to None.
            trim_edge (int, optional): Number of grid points to trim from the edges.
                Defaults to None.
            lcc_info (dict, optional): Lambert Conformal Conic projection details.
                Required for LAM and nested-lam model types.
            horizontal_regrid_kwargs (dict, optional): Options passed to
                ufs2arco.transforms.horizontal_regrid. Required when
                model_type="nested-global".
            anemoi_reference_dataset_kwargs (dict, optional): kwargs passed to
                anemoi.datasets.open_dataset to retrieve the global mask needed
                for conservative regridding. Required when model_type="nested-global"
                and the target grid file does not already contain a mask.
        \b
        forecast2 (dict): Config for the prediction forecast. Same keys as forecast1.
        \b
        keep_t0 (bool, optional): If True, keeps the initial condition time (t0)
            as a separate dimension in the output file. Defaults to False.

    \b
    Config Args common to spatial.py:
        output_path (str): The directory where the output NetCDF files will be saved.
        \b
        start_date (str): The first initial condition date to process.
        \b
        end_date (str): The last initial condition date to process.
        \b
        freq (str): Frequency string for the date range (e.g., "6h").
        \b
        lead_time (int): Length of forecast in hours.
        \b
        levels (list, optional): A list of vertical levels to subset from both
            forecasts. Defaults to None.
        \b
        vars_of_interest (list[str], optional): A list of variable names to include
            from both forecasts. Defaults to None.
        \b
        use_mpi (bool, optional): If True, distribute initialization dates across MPI ranks.
            Launch with ``srun --ntasks=N`` to use N ranks. Defaults to False.
        \b
        log_path (str, optional): When using MPI, the directory where per-rank log files
            are saved. Defaults to "eagle-logs/spatial_forecast".
    """


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def spectra(config_file):
    """
    Compute power spectra.
    """
    from eagle.tools.spectra import main
    main(config_file)

spectra.help = """Compute the Power Spectrum averaged over all initial conditions

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        min_delta_lat (float, optional): The minimum delta latitude used as a
            parameter for the power spectrum computation. Defaults to 0.0003.

    \b
    Config Args common to metrics.py:
        model_type (str): The type of model grid, one of: "global", "lam",
            "nested-lam", "nested-global".
            This determines how grid cell area weights, edge trimming, and coordinates are handled.
        \b
        verification_dataset_path (str): The path to the anemoi dataset with target data
            used for comparison.
        \b
        forecast_path (str): The directory path containing the forecast datasets.
        \b
        output_path (str): The directory where the output NetCDF files will be saved, as
            f"{output_path}/spectra.{model_type}.nc"
        \b
        start_date (str): The first initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        end_date (str): The last initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        freq (str): The frequency string for generating the date range between
            start_date and end_date (e.g., "6h"), passed to pandas.date_range.
        \b
        lead_time (int): Length of forecast in hours.
        \b
        from_anemoi (bool, optional): If True, opens forecast data using the
            anemoi inference dataset format. Otherwise, assumes layout of dataset
            created by ufs2arco using a base target layout. Defaults to True.
        \b
        lam_index (int, optional): For nested models (e.g., model_type="nested-lam"), this integer
            specifies the number of grid points belonging to the LAM domain.
            Defaults to None.
        \b
        levels (list, optional): A list of vertical levels to subset from the
            datasets. If None, all levels are used. Defaults to None.
        \b
        vars_of_interest (list[str], optional): A list of variable names to
            include in the analysis. If None, all variables are used. Defaults to None.
        \b
        trim_edge (int, optional): Specifies the number of grid points to trim
            from the edges of the verification dataset. Only used for LAM or Nested configurations.
            Defaults to None.
        \b
        trim_forecast_edge (int, optional): Specifies the number of grid points to
            trim from the edges of the forecast dataset. Defaults to None.
    """

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def figures(config_file):
    """
    Visualize the fields as figures
    """
    from eagle.tools.visualize import main
    main(config_file, mode="figure")

figures.help = """Create figures or movies visually comparing predictions to targets

    \b
    Note:
        All temperature fields are converted from K to degrees Celsius

    \b
    Note:
        The following variables can be computed, even though they may not be in the original dataset:
        ``["wind_speed", "10m_wind_speed", "80m_wind_speed", "100m_wind_speed"]``.
        These are computed from the vector valued quantities.

    \b
    Config Args:
        end_date (str): For figures, this is the timestamp that gets plotted.
            For movies, all timestamps between start_date and end_date get plotted.
        \b
        model_name (str, optional): A display name for the prediction dataset
            in plot titles. Defaults to "".
        \b
        target_name (str, optional): A display name for the target dataset in
            plot titles. Defaults to "".
        \b
        fig_kwargs (dict, optional): A dictionary of global figure settings to
            override defaults, such as `dpi`, `width`, `height`, and `projection`.
        \b
        per_variable_kwargs (dict, optional): A dictionary to override plotting
            options for specific variables. Keys are variable names (e.g.,
            "2m_temperature"), and values are dictionaries of options (e.g.,
            `{"vmin": -10, "vmax": 30}`).
        \b
        units (dict, optional): A dictionary to override the units displayed for
            specific variables.

    \b
    Config Args common to metrics.py
        model_type (str): The type of model grid, one of: "global", "lam",
            "nested-lam", "nested-global".
            This determines how grid cell area weights, edge trimming, and coordinates are handled.
        \b
        verification_dataset_path (str): The path to the anemoi dataset with target data
            used for comparison.
        \b
        forecast_path (str): The directory path containing the forecast datasets.
        \b
        output_path (str): The directory where the output NetCDF files will be saved, as
            f"{output_path}/{variable_name}.{t0}.{tf}.jpeg/gif/mp4" for surface variables and
            f"{output_path}/{variable_name}.level{level}.{t0}.{tf}.jpeg/gif/mp4" for 3D variables, per level
        \b
        start_date (str): The first initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        lead_time (int): Length of forecast in hours.
        \b
        lam_index (int, optional): For nested models (e.g., model_type="nested-lam"), this integer
            specifies the number of grid points belonging to the LAM domain.
            Defaults to None.
        \b
        levels (list, optional): A list of vertical levels to subset from the
            datasets. If None, all levels are used. Defaults to None.
            Note that all 3D variables will be plotted at all levels provided.
        \b
        vars_of_interest (list[str], optional): A list of variable names to
            include in the analysis. If None, all variables are used. Defaults to None.
        \b
        trim_edge (int, optional): Specifies the number of grid points to trim
            from the edges of the verification dataset. Only used for LAM or Nested configurations.
            Defaults to None.
    """


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def movies(config_file):
    """
    Visualize the fields as figures
    """
    from eagle.tools.visualize import main
    main(config_file, mode="movie")

movies.help = figures.help


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def prewxvx(config_file):
    """
    Postprocess forecast files for wxvx
    """
    from eagle.tools.prewxvx import main
    main(config_file)

prewxvx.help = """Postprocess forecast files for wxvx.

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        forecast_path (str): The directory path containing the forecast datasets.
        \b
        output_path (str): The directory where output NetCDF files will be saved.
        \b
        model_type (str): The model type identifier.
        \b
        lead_time (int): Length of forecast in hours.
        \b
        start_date (str): The first initial condition date to process.
        \b
        end_date (str): The last initial condition date to process.
        \b
        freq (str): Frequency string for the date range (e.g., "6h").
        \b
        from_anemoi (bool, optional): If True, opens forecasts using the
            anemoi inference dataset format. Defaults to True.
        \b
        chunks (dict, optional): A dictionary mapping dimension names to chunk
            sizes (e.g., ``{time: 1}``). When provided, the output dataset is
            chunked accordingly before writing to NetCDF. Dimensions not listed
            are left unchunked. Defaults to None (no chunking).
        \b
        rename_curvilinear_coords_to_latlon (bool, optional): For LAM model types,
            if True, renames the curvilinear ``x``/``y`` coordinates to
            ``longitude``/``latitude``. Set to False to keep the original
            coordinate names. Defaults to True for backward compatibility.
    """


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def postwxvx(config_file):
    """
    Gather wxvx stats
    """
    from eagle.tools.postwxvx import main
    main(config_file)

@cli.command("obs-metrics")
@click.argument('config_file', type=click.Path(exists=True))
def obs_metrics(config_file):
    """Verify forecasts against observations."""
    from eagle.tools.obs_metrics import main
    main(config_file)

obs_metrics.help = """Verify forecasts against real observations (e.g., radiosondes).

    \b
    This workflow loads forecast data and compares it against irregular
    observation data (station-based, not gridded). Forecasts are interpolated
    to observation locations, and RMSE, MAE, bias, and count are computed
    per forecast hour. Observation datasets and variable mappings are defined
    in config/obs_metrics.yaml.

    \b
    Note:
        Observations are loaded from nnja_ai.DataCatalog (parquet from GCS).
        Quality control filtering is applied per-variable using NCEP PREPBUFR codes.

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        forecast_path (str): The directory path containing the forecast datasets.
        \b
        lead_time (int): Length of forecast in hours.
        \b
        output_path (str): The directory where output NetCDF files will be saved, as
            f"{output_path}/{metric}.convobs.{model_type}.nc"
        \b
        start_date (str): The first initial condition date to process.
        \b
        end_date (str): The last initial condition date to process.
        \b
        freq (str): Frequency string for the date range (e.g., "12h").
        \b
        vars_of_interest (list[str]): Variable names to verify. Names are mapped to
            canonical names via config/rename.yaml. Derived variables like wind_speed
            and 10m_wind_speed are supported (include their components too, e.g. u, v).
        \b
        levels (list[int], optional): Pressure levels (hPa) for upper-air variables.
            If not provided, all levels present in the forecast dataset are used.
        \b
        model_type (str, optional): The type of model grid. Defaults to "global".
        \b
        temporal_window (str, optional): Time window (+/-) for matching obs to forecast
            valid times. Defaults to "30min".
        \b
        max_qc_value (int, optional): Keep obs with QC flag <= this value. NaN QC flags
            are always kept. Defaults to 2.
        \b
        n_members (int, optional): Number of ensemble members. If > 1, ensemble metrics
            (spread, CRPS, ensemble mean RMSE/MAE/bias) are also computed. Defaults to 1.
        \b
        subregions (dict, optional): Geographic subregions for regional metrics. Each entry
            maps a name to latitude and/or longitude bounds.
        \b
        from_anemoi (bool, optional): If True, opens forecast data using the anemoi
            inference dataset format. Defaults to True.
        \b
        lam_index (int, optional): Index for LAM selection in nested models.
        \b
        lcc_info (dict, optional): Lambert Conformal Conic projection details for LAM grids.
        \b
        trim_forecast_edge (list[int], optional): Additional edge trimming for LAM grids.
        \b
        use_mpi (bool, optional): If True, distribute initialization dates across MPI ranks.
        \b
        log_path (str, optional): When using MPI, the directory for per-rank log files.
    """


@cli.command()
@click.option('--offline_path', required=True, type=click.Path(exists=True), help='Path to the experiment mlflow logs')
@click.option('--local_id', required=True, type=str, help='The generated 18 character experiment ID')
@click.option('--remote_name', required=True, type=str, help='The experiment group to show up in on AML')
def amlsync(offline_path, local_id, remote_name):
    """
    Sync offline MLflow logs to Azure Machine Learning (AML).

    Note:
        Users must have the following credentials defined as environment variables:
        * AZURE_TENANT_ID
        * AZURE_SUBSCRIPTION_ID
        * AZURE_CLIENT_ID
        * AZURE_CLIENT_SECRET
    """

    from eagle.tools.amlsync import main
    main(
        offline_path=offline_path,
        local_id=local_id,
        remote_name=remote_name,
    )

if __name__ == "__main__":
    cli()
