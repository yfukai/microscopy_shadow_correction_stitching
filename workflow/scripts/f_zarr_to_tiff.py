#! /usr/bin/env python3
import os
import re
from os import path

import fire
import zarr

def zarr_to_tiff(zarr_path,tiff_dir_path=None):
    if tiff_dir_path is None:
        tiff_path=path.splitext(zarr_path)[0]+"_tiff"
    zarr_file=zarr.open(zarr_path,mode="r")
    assert len(zarr_file.shape)==5 # assume TCZYX
    sizeT,sizeC,sizeZ=zarr_file.shape[:3]
    

if __name__ == "__main__":
    try:
        zarr_to_tiff(
            path.dirname(snakemake.input["zarr_path"]),
        )
    except NameError as e:
        if "snakemake" in str(e):
            raise e
        fire.Fire(zarr_to_tiff)
