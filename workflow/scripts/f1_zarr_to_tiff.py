#! /usr/bin/env python3
import os
from os import path

import numpy as np
import fire
import zarr
import tifffile
from tqdm import tqdm
from dask import array as da

type_strings=[
    "uint8",
    "uint16",
    "uint32",
    "float32",
    "float64",
]

def zarr_to_tiff(zarr_path,
                 tiff_dir_path=None,
                 *,
                 type_string="float32",
                 intensity_rescale=False,
                 intensity_rescale_quantiles=(0.001,0.999)
    ):

    assert type_string in type_strings, f"type_string must be in {type_strings}"

    if tiff_dir_path is None:
        tiff_dir_path=path.splitext(zarr_path)[0]+"_tiff"
    os.makedirs(tiff_dir_path,exist_ok=True)
    zarr_file=zarr.open(zarr_path,mode="r")["image"]
    image=da.from_zarr(zarr_file)
    
    if intensity_rescale==True:
        q1,q2=np.quantile(image,intensity_rescale_quantiles)
        image=np.clip((image-q1)/(q2-q1))

    dtype=np.dtype(type_string)
    if "int" in type_strings:
        dtype_max=np.iinfo(dtype).max
        image=image*dtype_max
    image=image.astype(dtype)

    assert len(zarr_file.shape)==5 # assume TCZYX
    sizeT,sizeC,sizeZ=zarr_file.shape[:3]
    for iT,iC,iZ in tqdm(np.ndindex(sizeT,sizeC,sizeZ),total=sizeT*sizeC*sizeZ):
        subimage=image[iT,iC,iZ].compute()
        tifffile.imsave(
            path.join(tiff_dir_path,
                     f"image_T{iT:03d}_C{iC:03d}_Z{iZ:03d}.tiff"),
                     subimage)


if __name__ == "__main__":
    try:
        zarr_to_tiff(
            snakemake.input["zarr_path"], # type:ignore pylint:disable=undefined-variable
            **snakemake.config["f1_zarr_to_tiff"], # type:ignore pylint:disable=undefined-variable
        )
    except NameError as e:
        if "snakemake" in str(e):
            raise e
        fire.Fire(zarr_to_tiff)
