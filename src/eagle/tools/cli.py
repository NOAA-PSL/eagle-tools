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


if __name__ == "__main__":
    cli()
