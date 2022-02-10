#!/usr/bin/env python
from aicsimageio import AICSImage
import click
import yaml

@click.command()
@click.argument('input_czi', type=click.Path(exists=True))
@click.argument("output_metadata_yaml", type=click.Path())
def main(input_czi, output_metadata_yaml):
    aics_image = AICSImage(input_czi,reconstruct_mosaic=False)
    dims=aics_image.dims

    # Export metadata to YAML
    metadata=dict(
        channel_names = list(map(str,aics_image.channel_names)), # channel name strings
        dims = dict(dims.items()), # dimensions of the image
        mosaic_positions = [list(aics_image.get_mosaic_tile_position(i)) 
                            for i in range(dims.M)], # Y and X mosaic positions in pixel
        physical_pixel_sizes = [
            aics_image.physical_pixel_sizes.Z,
            aics_image.physical_pixel_sizes.Y,
            aics_image.physical_pixel_sizes.X
        ] # physical pixel sizes in microns, in order of Z, Y, X
    )
    with open(output_metadata_yaml, "w") as f:
        yaml.dump(metadata, f)


if __name__ == '__main__':
    main()