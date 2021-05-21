#!/usr/bin/env python3
import fire
import os
from os import path
from glob import glob
import re
import numpy as np
import pandas as pd

import pycziutils
from m2stitch import stitching
from tqdm import tqdm
import zarr
import yaml

def main(output_dir,
         show_result=False,
         only_first_timepoint=False,
         stitching_channels=("Phase",),
         stitching_mode="divide",
         n_procs=1):

    assert stitching_mode in ["divide","subtract","none"]

    output_dir=path.abspath(output_dir)
    image_directory = path.join(output_dir, "rescaled_images")
    metadata_xml_name = path.join(output_dir, "metadata.xml")
    image_key=f"rescaled_image_{stitching_mode}"
    image_directory2=path.join(image_directory,image_key)

    planes_df=pd.read_csv(path.join(output_dir,"planes_df2.csv"))

    stitching_log_directory=path.join(output_dir,"stitching_log")
    os.makedirs(stitching_log_directory,exist_ok=True)

    params_dict=locals()

    stitching_df=pd.DataFrame()
    props_dicts={}

    with open(metadata_xml_name, "r") as f:
        meta = "".join(f.readlines())
    channels=pycziutils.parse_channels(meta)
 
    for c_name in stitching_channels:
        c_indices=[j for j,c in enumerate(channels) if c_name in c["@Fluor"]]
        assert len(c_indices)==1
        c_index=c_indices[0]
        selected_planes_df=planes_df[planes_df["C_index"]==c_index]

        for (t,z),grp in tqdm(list(selected_planes_df.groupby(["T_index","Z_index"]))):
            images=[]
            rows=[]
            cols=[]
            for j, row in grp.iterrows():
                c,t,z,s=int(row["C_index"]),int(row["T_index"]),int(row["Z_index"]),int(row["S_index"]) 
                rescaled_image_path=path.join(image_directory2,
                        f"S{s:03d}_{row['row_col_label']}.zarr",)
                img=zarr.open(rescaled_image_path,mode="r")[t,c,z,:,:]
                images.append(img)
                rows.append(row["X_index"])
                cols.append(row["Y_index"])
            grid,props_dict=stitching.stitch_images(images,rows,cols)
            props_dicts[(c_index,t,z)]=props_dict
            grid["c_name"]=c_name
            grid["C_index"]=c_index
            grid["T_index"]=t
            grid["Z_index"]=z
            stitching_df=stitching_df.append(grid)

    props_path=path.join(stitching_log_directory,"stitching_result_props.yaml")
    with open(props_path,"w") as f:
        yaml.dump(props_dicts,f)           



    stitching_df.to_csv(path.join(output_dir,"stitching_result_raw.csv"))
    stitching_df2_median=stitching_df[["row","col","x_pos","y_pos"]]\
                    .groupby(["row","col"]).median().reset_index()
    stitching_df2_std=stitching_df[["row","col","x_pos","y_pos"]]\
                    .groupby(["row","col"]).std().reset_index()
    stitching_df2_std=stitching_df2_std.fillna(0)
    stitching_df2=pd.merge(stitching_df2_median,stitching_df2_std,
                           on=["row","col"],suffixes=["_median","_std"])
    stitching_df2["row"]=stitching_df2["row"].astype(np.int32)
    stitching_df2["col"]=stitching_df2["col"].astype(np.int32)
    stitching_df2.to_csv(path.join(output_dir,"stitching_result_sumamrized.csv"))

    print(params_dict)
    params_path=path.join(output_dir,"stitching_params.yaml")
    with open(params_path,"w") as f:
        yaml.dump(params_dict,f)           


if __name__ == "__main__":
    try:
        main(path.dirname(snakemake.input["output_dir_created"]),
                          only_first_timepoint=snakemake.config["stitching_only_first_timepoint"],
                          n_procs=snakemake.threads)
    except NameError:
        fire.Fire(main)
