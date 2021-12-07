#! /usr/bin/env python3
import os
from os import path

import numpy as np
import fire
import zarr
from tqdm import tqdm

from matplotlib import pyplot as plt

def create_thumbnails(zarr_path,
                 thumbnail_dir_path=None,
                 export_only_first=True):
    if thumbnail_dir_path is None:
        thumbnail_dir_path=path.splitext(zarr_path)[0]+"_thumbnail"
    os.makedirs(thumbnail_dir_path,exist_ok=True)
    zarr_file=zarr.open(zarr_path,mode="r")["image"]
    assert len(zarr_file.shape)==5 # assume TCZYX
    sizeT,sizeC,sizeZ=zarr_file.shape[:3]
    if export_only_first:
        sizeT=1
    for iT,iC,iZ in tqdm(np.ndindex(sizeT,sizeC,sizeZ),total=sizeT*sizeC*sizeZ):
        plt.figure(figsize=(10,10))
        plt.imshow(zarr_file[iT,iC,iZ])
        plt.colorbar()
        plt.savefig(
            path.join(thumbnail_dir_path,
                     f"image_T{iT:03d}_C{iC:03d}_Z{iZ:03d}.pdf"),
                     bbox_inches="tight")


if __name__ == "__main__":
    try:
        create_thumbnails(
            snakemake.input["zarr_path"], #type: ignore
        )
    except NameError as e:
        if not "snakemake" in str(e):
            raise e
        fire.Fire(create_thumbnails)
