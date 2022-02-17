#!/usr/bin/env python3
import os
import re
from glob import glob
from os import path

import fire
import numpy as np
import pandas as pd
import pycziutils
import yaml
import zarr
from m2stitch import stitching
from tqdm import tqdm


def main(
    output_dir,
    *,
    only_first_timepoint=False,
    stitching_channels=("Phase",),
    stitching_mode="divide",
    n_procs=1,
):

    assert stitching_mode in ["divide", "subtract", "none"]

    output_dir = path.abspath(output_dir)
    image_directory = path.join(output_dir, "rescaled_images")
    image_key = f"rescaled_image_{stitching_mode}"
    image_directory2 = path.join(image_directory, image_key)
    image_props_path = path.join(output_dir, "image_props.yaml")

    assert path.isdir(output_dir)
    assert path.isdir(image_directory2)
    assert path.isfile(image_props_path)

    stitching_log_directory = path.join(output_dir, "stitching_log")
    os.makedirs(stitching_log_directory, exist_ok=True)

    params_dict = locals()

    planes_df = pd.read_csv(path.join(output_dir, "planes_df2.csv"))
    with open(image_props_path, "r") as f:
        image_props = yaml.safe_load(f)
    channel_names = image_props["channel_names"]

    stitching_df = pd.DataFrame()
    props_dicts = {}

    planes_df["choosepos_full"] = True
    for c_name in stitching_channels:
        c_indices = [j for j, c in enumerate(channel_names) if c_name in c]
        assert len(c_indices) == 1
        c_index = c_indices[0]
        selected_planes_df = planes_df[planes_df["C_index"] == c_index]

        indices_groups = list(selected_planes_df.groupby(["T_index", "Z_index"]))
        if only_first_timepoint:
            indices_groups = [indices_groups[0]]
        full_choosepos_size = len(indices_groups[0][1])
        for (t, z), grp in tqdm(indices_groups):
            if len(grp) < full_choosepos_size:
                planes_df.loc[planes_df["T_index"] == t, "choosepos_full"] = False
                continue  # only stitch full tiles
            images = []
            rows = []
            cols = []
            for j, row in grp.iterrows():
                c, t, z, s = (
                    int(row["C_index"]),
                    int(row["T_index"]),
                    int(row["Z_index"]),
                    int(row["S_index"]),
                )
                rescaled_image_path = path.join(
                    image_directory2,
                    f"S{s:03d}_{row['row_col_label']}.zarr",
                )
                img = zarr.open(rescaled_image_path, mode="r")["image"][t, c, z, :, :]
                images.append(img)
                rows.append(row["X_index"])
                cols.append(row["Y_index"])

            grid, props_dict = stitching.stitch_images(images, rows, cols)
            props_dicts[(c_index, t, z)] = props_dict
            grid["c_name"] = c_name
            grid["C_index"] = c_index
            grid["T_index"] = t
            grid["Z_index"] = z
            stitching_df = stitching_df.append(grid)

    # interpolate stitching results from full grids

    props_path = path.join(stitching_log_directory, "stitching_result_props.yaml")
    with open(props_path, "w") as f:
        yaml.dump(props_dicts, f)

    stitching_df.to_csv(path.join(output_dir, "stitching_result_raw.csv"))
    stitching_df2_median = (
        stitching_df[["row", "col", "x_pos", "y_pos"]]
        .groupby(["row", "col"])
        .median()
        .reset_index()
    )
    stitching_df2_std = (
        stitching_df[["row", "col", "x_pos", "y_pos"]]
        .groupby(["row", "col"])
        .std()
        .reset_index()
    )
    stitching_df2_std = stitching_df2_std.fillna(0)
    stitching_df2 = pd.merge(
        stitching_df2_median,
        stitching_df2_std,
        on=["row", "col"],
        suffixes=["_median", "_std"],
    )
    stitching_df2["row"] = stitching_df2["row"].astype(np.int32)
    stitching_df2["col"] = stitching_df2["col"].astype(np.int32)
    stitching_df2.to_csv(path.join(output_dir, "stitching_result_sumamrized.csv"))

    planes_df.to_csv(path.join(output_dir, "planes_df3.csv"))

    print(params_dict)
    params_path = path.join(output_dir, "stitching_params.yaml")
    with open(params_path, "w") as f:
        yaml.dump(params_dict, f)


if __name__ == "__main__":
    try:
        main(
            path.dirname(snakemake.input["output_dir_created"]), #type: ignore
            **snakemake.config["d_stitch_images"], #type: ignore
            n_procs=snakemake.threads, #type: ignore
        )
    except NameError as e:
        if not "snakemake" in str(e):
            raise e
        fire.Fire(main)
