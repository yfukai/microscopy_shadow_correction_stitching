#! /usr/bin/env python3
import os
from os import path
import numpy as np
import click
import zarr
from tqdm import tqdm
from matplotlib import pyplot as plt
import yaml

@click.command()
@click.argument('stitched_zarr', type=click.Path(exists=True))
@click.argument("metadata_yaml", type=click.Path(exists=True))
@click.argument("report_path", type=click.Path())
def main(stitched_zarr,
         metadata_yaml,
         report_path,
         export_only_first=True):
    os.makedirs(report_path,exist_ok=True)
    zarr_file=zarr.open(stitched_zarr,mode="r")["image"]
    assert len(zarr_file.shape)==5 # assume TCZYX
    sizeT,sizeC,sizeZ=zarr_file.shape[:3]
    if export_only_first:
        sizeT=1
    thumbnail_path=path.join(report_path,"thumbnails")
    os.makedirs(thumbnail_path,exist_ok=True)
    for iT,iC,iZ in tqdm(np.ndindex(sizeT,sizeC,sizeZ),
                         total=sizeT*sizeC*sizeZ):
        plt.figure(figsize=(10,10))
        qs=np.quantile(zarr_file[iT,iC,iZ],[0.01,0.99])
        plt.imshow(zarr_file[iT,iC,iZ],
                   cmap="gray",vmin=qs[0],vmax=qs[1])
        plt.colorbar()
        plt.savefig(
            path.join(thumbnail_path,
                     f"image_T{iT:03d}_C{iC:03d}_Z{iZ:03d}.pdf"),
                     bbox_inches="tight")
    with open(metadata_yaml, "r") as f:
        metadata=yaml.safe_load(f)
    mosaic_poss=np.array(metadata["stitching_result"]["mosaic_positions"])
    plt.figure(figsize=(10,10))
    plt.plot(mosaic_poss[:,0],mosaic_poss[:,1],"-")
    plt.savefig(path.join(report_path,"mosaic_positions.pdf"))

if __name__ == "__main__":
    main()