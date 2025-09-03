import click
import yaml

from eagle.tools.utils import open_yaml_config

def common_options(func):
    """A decorator that applies all common command-line options."""
    options = [
        click.option(
            "--model-type",
            help="The type of model grid.",
        ),
        click.option(
            "--verification-dataset-path",
            type=click.Path(),
            help="Path to the zarr verification dataset.",
        ),
        click.option(
            "--forecast-path",
            type=click.Path(),
            help="Directory path for forecast datasets.",
        ),
        click.option(
            "--output-path",
            type=click.Path(),
            help="Directory path for output NetCDF files.",
        ),
        click.option(
            "--start-date",
            help="First initial condition date to process.",
        ),
        click.option(
            "--end-date",
            help="Last initial condition date to process.",
        ),
        click.option(
            "--freq",
            help="Frequency string for the date range (e.g., '6h').",
        ),
        click.option(
            "--lead-time",
            help="Forecast lead time string (e.g., '240h').",
        ),
        click.option(
            "--from-anemoi/--no-from-anemoi",
            default=None,
            help="Toggle the anemoi inference dataset format.",
        ),
        click.option(
            "--lam-index",
            type=int,
            help="Integer for the number of grid points in the LAM domain.",
        ),
        click.option(
            "--level",
            "levels",
            multiple=True,
            type=int,
            help="A vertical level to subset. Can be specified multiple times.",
        ),
        click.option(
            "--var",
            "vars_of_interest",
            multiple=True,
            type=str,
            help="A variable name to include. Can be specified multiple times.",
        ),
        click.option(
            "--trim-edge",
            type=int,
            help="Grid points to trim from verification dataset edges.",
        ),
        click.option(
            "--trim-forecast-edge",
            type=int,
            help="Grid points to trim from forecast dataset edges.",
        ),
    ]
    # Apply decorators in reverse order
    for option in reversed(options):
        func = option(func)
    return func

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
    from eagle.tools.inference import main as inference_main

    config = open_yaml_config(config_file)
    inference_main(config)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def postprocess(config_file):
    """
    Run postprocessing.
    """
    from eagle.tools.postprocess import main as postprocess_main

    config = open_yaml_config(config_file)
    postprocess_main(config)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def metrics(config_file):
    """
    Compute error metrics.
    """
    from eagle.tools.metrics import main as metrics_main

    config = open_yaml_config(config_file)
    metrics_main(config)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def spatial(config_file):
    """
    Compute spatial error metrics.
    """
    from eagle.tools.spatial import main as spatial_main

    config = open_yaml_config(config_file)
    spatial_main(config)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def spectra(config_file):
    """
    Compute spectra error metrics.
    """
    from eagle.tools.spectra import main as spectra_main

    config = open_yaml_config(config_file)
    spectra_main(config)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def figures(config_file):
    """
    Visualize the fields as figures
    """
    from eagle.tools.visualize import main as visualize_main

    config = open_yaml_config(config_file)
    visualize_main(config, mode="figure")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def movies(config_file):
    """
    Visualize the fields as figures
    """
    from eagle.tools.visualize import main as visualize_main

    config = open_yaml_config(config_file)
    visualize_main(config, mode="movie")

if __name__ == "__main__":
    cli()
