import click
import yaml

# Import the main functions from your other modules
from eagle.tools.metrics import main as metrics_main
from eagle.tools.spatial import main as spatial_main
from eagle.tools.utils import open_yaml_config

@click.group()
def cli():
    """A CLI for the Eagle Tools suite."""
    pass

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def metrics(config_file):
    """
    Run metrics operations using a configuration file.
    """
    click.echo(f"Running metrics with config: {config_file}")

    config = open_yaml_config(config_file)
    metrics_main(config)

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def spatial(config_file):
    """
    Run spatial operations using a configuration file.
    """
    click.echo(f"Running spatial with config: {config_file}")

    config = open_yaml_config(config_file)
    spatial_main(config)

if __name__ == "__main__":
    cli()
