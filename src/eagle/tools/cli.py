import click
import yaml

from eagle.tools.utils import open_yaml_config

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
    config = open_yaml_config(config_file)
    main(config)

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
    """


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def postprocess(config_file):
    """
    Run postprocessing.
    """
    from eagle.tools.postprocess import main
    config = open_yaml_config(config_file)
    main(config)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def metrics(config_file):
    """
    Compute error metrics.
    """
    from eagle.tools.metrics import main
    config = open_yaml_config(config_file)
    main(config)

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
        lead_time (str): A string representing the forecast lead time (e.g., "240h")
            used as part of the forecast input filename.
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
def spatial(config_file):
    """
    Compute spatial error metrics.
    """
    from eagle.tools.spatial import main
    config = open_yaml_config(config_file)
    main(config)

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
        lead_time (str): A string representing the forecast lead time (e.g., "240h")
            used as part of the forecast input filename.
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
def spectra(config_file):
    """
    Compute spectra error metrics.
    """
    from eagle.tools.spectra import main
    config = open_yaml_config(config_file)
    main(config)

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
        lead_time (str): A string representing the forecast lead time (e.g., "240h")
            used as part of the forecast input filename.
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
    config = open_yaml_config(config_file)
    main(config, mode="figure")

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
        lead_time (str): A string representing the forecast lead time (e.g., "240h")
            used as part of the forecast input filename.
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
    config = open_yaml_config(config_file)
    main(config, mode="movie")

movies.help = figures.help


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def prewxvx(config_file):
    """
    Postprocess forecast files for wxvx
    """
    from eagle.tools.prewxvx import main
    config = open_yaml_config(config_file)
    main(config)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def postwxvx(config_file):
    """
    Gather wxvx stats
    """
    from eagle.tools.postwxvx import main
    config = open_yaml_config(config_file)
    main(config)


if __name__ == "__main__":
    cli()
