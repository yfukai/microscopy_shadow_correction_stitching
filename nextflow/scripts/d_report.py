#! /usr/bin/env python3
import os
from os import path

import numpy as np
import click
import zarr
from tqdm import tqdm

from matplotlib import pyplot as plt

def main(zarr_path,
         report_path,
         export_only_first=True):
    os.makedirs(report_path,exist_ok=True)
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
            path.join(report_path,
                     f"image_T{iT:03d}_C{iC:03d}_Z{iZ:03d}.pdf"),
                     bbox_inches="tight")


