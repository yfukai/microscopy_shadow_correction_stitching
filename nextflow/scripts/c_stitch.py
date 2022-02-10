#!/usr/bin/env python3
import pandas as pd
from glob import glob
from os import path
import click
import zarr
import yaml
import numpy as np
from m2stitch import stitch_images
from itertools import product
from tqdm import tqdm

"""
(only for test)
basedir="/work/fukai/2021-03-04-timelapse_analyzed/210311-HL60-ctrl-staining_analyzed/"
rescaled_zarr=path.join(basedir,"rescaled.zarr")
metadata_yaml=path.join(basedir,"metadata.yaml")
stitched_zarr=path.join(basedir,"stitched.zarr")
only_first_timepoint=False,
stitching_channels=("Phase",)
stitching_result_csv=path.join(basedir,"stitching_result.csv")
"""


@click.command()
@click.argument("rescaled_zarr", type=click.Path(exists=True))
@click.argument("metadata_yaml", type=click.Path(exists=True))
@click.argument("stitched_zarr", type=click.Path())
@click.argument("stitching_result_csv", type=click.Path())
def main(
    rescaled_zarr,
    metadata_yaml,
    stitched_zarr,
    stitching_result_csv,
    only_first_timepoint=False,
    stitching_channels=("Phase",),
    export_only_full_tile=True,
):
    print(" loading data ")
    rescaled = zarr.open(rescaled_zarr, mode="r")
    with open(metadata_yaml, "r") as f:
        metadata=yaml.safe_load(f)
    channel_names=metadata["channel_names"]
    mosaic_positions=np.array(metadata["mosaic_positions"])

    # calculate the tile index in each direction
    mosaic_indices=[]
    for i in range(mosaic_positions.shape[1]):
        unique_poss=np.sort(np.unique(mosaic_positions[:,i]))
        map_poss=dict(zip(unique_poss,range(len(unique_poss))))
        mosaic_indices.append(np.array([map_poss[p] for p in mosaic_positions[:,i]]))
    mosaic_indices=np.array(mosaic_indices).T

    # calculate stitching positions
    stitching_props=dict()
    stitching_df=pd.DataFrame()
    for stitching_channel in stitching_channels:
        c=channel_names.index(stitching_channel)
        if only_first_timepoint:
            T_indices=[0]
        else:
            T_indices = np.arange(rescaled.shape[1])
        Z_indices = np.arange(rescaled.shape[3])
        for t,z in product(T_indices,Z_indices):
            print(t,z)
            images=np.array(rescaled[:,t,c,z,:,:])
            _grid, stitching_props[(c,t,z)] = stitch_images(images, 
                                                position_indices=mosaic_indices)
            stitching_df=stitching_df.append(pd.DataFrame(dict(
                m=np.arange(len(_grid)),
                t=t,c=c,z=z,
                pos_y=_grid["x_pos"], # confusing but to sustain the same order as the original code
                pos_x=_grid["y_pos"],
            )))
    for j,d in enumerate("yx"):
        stitching_df[f"original_pos_{d}"]=mosaic_positions[stitching_df["m"],j]
    stitching_df.to_csv(stitching_result_csv)

    stitching_df_summarized = (
        stitching_df[["m", "pos_y", "pos_x"]]
        .groupby(["m"]).agg([np.median,np.std])
        .fillna(0)
    )
    for d in "yx":
        vals=stitching_df_summarized[(f"pos_{d}","median")]
        stitching_df_summarized[f"pos_{d}2"]=(vals-np.min(vals)).astype(pd.Int64Dtype())
    stitching_mosaic_positions=stitching_df_summarized\
            .loc[np.arange(rescaled.shape[0]),["pos_y2","pos_x2"]]\
            .values.tolist()
    metadata["stitching_result"]=dict(
        mosaic_positions=stitching_mosaic_positions
    )
    with open(metadata_yaml, "w") as f:
        yaml.safe_dump(metadata,f)
#%%
    ### saving stitched images
    print(" saving data ")
    sizeY,sizeX=rescaled.shape[-2:]
    max_y=np.max(stitching_df_summarized["pos_y2"])
    max_x=np.max(stitching_df_summarized["pos_x2"])
    stitched_image_size = [*rescaled.shape[1:-2],max_y+sizeY,max_x+sizeX]
    output_zarr_file = zarr.open(
        stitched_zarr,
        mode="w",
    )
    output_zarr = output_zarr_file.create_dataset(
        "image",
        shape=stitched_image_size,
        chunks=(1, 1, 1, 2048, 2048),
        dtype=np.float32,
        overwrite=True,
    )

    for t in range(rescaled.shape[1]):
        stitched_image = np.zeros(stitched_image_size[1:], dtype=np.float32)
        for m in range(rescaled.shape[0]):
            y,x=stitching_mosaic_positions[m]
            window = (slice(y, y + sizeY), slice(x, x + sizeX))
            image = np.array(rescaled[m, t, :, :, :, :])
            stitched_image[:, :, window[0], window[1]] = image
        output_zarr[t, :, :, :, :] = stitched_image

#%%

if __name__ == "__main__":
    main()
# %%
