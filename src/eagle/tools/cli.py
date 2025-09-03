import click
import yaml

from eagle.tools.utils import open_yaml_config
from eagle.tools.inference import main as inference_main

from eagle.tools.metrics import main as metrics_main
from eagle.tools.spatial import main as spatial_main
from eagle.tools.spectra import main as spectra_main
from eagle.tools.visualize import main as visualize_main

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
    config = open_yaml_config(config_file)
    inference_main(config)

inference.help = inference_main.__doc__


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
    config = open_yaml_config(config_file)
    metrics_main(config)

metrics.help = metrics_main.__doc__


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def spatial(config_file):
    """
    Compute spatial error metrics.
    """
    config = open_yaml_config(config_file)
    spatial_main(config)

spatial.help = spatial_main.__doc__


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def spectra(config_file):
    """
    Compute spectra error metrics.
    """
    config = open_yaml_config(config_file)
    spectra_main(config)

spectra.help = spectra_main.__doc__


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def figures(config_file):
    """
    Visualize the fields as figures
    """
    config = open_yaml_config(config_file)
    visualize_main(config, mode="figure")

figures.help = visualize_main.__doc__


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def movies(config_file):
    """
    Visualize the fields as figures
    """
    config = open_yaml_config(config_file)
    visualize_main(config, mode="movie")

movies.help = visualize_main.__doc__

if __name__ == "__main__":
    cli()
