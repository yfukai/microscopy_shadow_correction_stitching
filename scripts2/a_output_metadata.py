#!/usr/bin/env python
from aicsimageio import AICSImage
from os import path
import click
import yaml
from tqdm import tqdm
try:
    from aicspylibczi import CziFile
except:
    pass

def output_metadata(input_czi, output_metadata_yaml):
    aics_image = AICSImage(input_czi,reconstruct_mosaic=False)
    dims=aics_image.dims

    try:
        with open(input_czi) as f:
            czi = CziFile(f)
            bboxes = czi.get_all_mosaic_tile_bounding_boxes()
            bbox = list(bboxes.values())
            mosaic_positions = [[b.y, b.x] for b in bbox]
    except:
        mosaic_positions = [list(aics_image.get_mosaic_tile_position(i)) 
                            for i in tqdm(range(dims.M))], # Y and X mosaic positions in pixel
    # Export metadata to YAML
    metadata=dict(
        filename=path.abspath(input_czi),
        channel_names = list(map(str,aics_image.channel_names)), # channel name strings
        dims = dict(dims.items()), # dimensions of the image
        mosaic_positions = mosaic_positions,
        physical_pixel_sizes = [
            aics_image.physical_pixel_sizes.Z,
            aics_image.physical_pixel_sizes.Y,
            aics_image.physical_pixel_sizes.X
        ] # physical pixel sizes in microns, in order of Z, Y, X
    )
    with open(output_metadata_yaml, "w") as f:
        yaml.dump(metadata, f)


@click.command()
@click.argument('input_czi', type=click.Path(exists=True))
@click.argument("output_metadata_yaml", type=click.Path())
def main(input_czi, output_metadata_yaml):
    output_metadata(input_czi,output_metadata_yaml)

if __name__ == '__main__':
    main()