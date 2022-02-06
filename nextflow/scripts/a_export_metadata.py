#!/usr/bin/env python
from aicsimageio import AICSImage
import click
import yaml

@click.command()
@click.argument('input_czi', type=click.Path(exists=True))
@click.argument("output_metadata_yaml", type=click.Path())
def main(input_czi, output_metadata_yaml):
    aics_image = AICSImage(input_czi)
    metadata=dict(
        channels = aics_image.channel_names,
        dims = aics_image.dims,
        physical_pixel_sizes = aics_image.physical_pixel_sizes
    )
    with open(output_metadata_yaml, "w") as f:
        yaml.dump(metadata, f)