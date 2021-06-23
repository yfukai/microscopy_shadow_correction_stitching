#! /usr/bin/env python3
from os import path

import numpy as np
import fire
import zarr
import tifffile

from matplotlib import pyplot as plt

def zarr_to_tiff(zarr_path,tiff_dir_path=None):
    if tiff_dir_path is None:
        tiff_dir_path=path.splitext(zarr_path)[0]+"_thumbnail"
    zarr_file=zarr.open(zarr_path,mode="r")
    assert len(zarr_file.shape)==5 # assume TCZYX
    sizeT,sizeC,sizeZ=zarr_file.shape[:3]
    for iT,iC,iZ in np.ndindex(sizeT,sizeC,sizeZ):
        plt.figure(figsize=(10,10))
        plt.imshow(zarr_file[iT,iC,iZ])
        plt.colorbar()
        plt.savefig(
            path.join(tiff_dir_path,
                     f"image_T{iT:03d}_C{iC:03d}_Z{iZ:03d}.pdf"),
                     bbox_to_inches="tight")


if __name__ == "__main__":
    try:
        zarr_to_tiff(
            path.dirname(snakemake.input["zarr_path"]),
        )
    except NameError as e:
        if "snakemake" in str(e):
            raise ej
        fire.Fire(zarr_to_tiff)
