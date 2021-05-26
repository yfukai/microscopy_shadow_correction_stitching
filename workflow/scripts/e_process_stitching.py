#! /usr/bin/env python3
import numpy as np
import os
from os import path
import re
import h5py
import zarr
import yaml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import fire
from tqdm import tqdm
from dask import bag as db
from dask.diagnostics import ProgressBar

def process_stitching(output_dir,
                      stitching_csv_path=None,
                      *,
                      export_only_full_tile=True,
                      rescale_methods=["divide"]):
    
    if stitching_csv_path is None:
        stitching_csv_path=path.join(output_dir,"stitching_result_sumamrized.csv")
    rescaled_image_directory=path.join(output_dir,"rescaled_images")
    stitching_log_directory=path.join(output_dir,"stitching_log")
    os.makedirs(stitching_log_directory,exist_ok=True)

    params_dict=locals()

    planes_df=pd.read_csv(path.join(output_dir,"planes_df3.csv"))
    stitching_df=pd.read_csv(stitching_csv_path,index_col=0)

    stitching_df2=pd.merge(stitching_df,planes_df,
                                  left_on=("row","col"),
                                  right_on=("X_index","Y_index"),
                                  how="right")
    stitching_df2["x_pos"]=np.round(stitching_df2["x_pos_median"]).astype(np.int32)
    stitching_df2["y_pos"]=np.round(stitching_df2["y_pos_median"]).astype(np.int32)
    shift_x=stitching_df2["x_pos"].min()
    shift_y=stitching_df2["y_pos"].min()

    stitching_df2["x_pos"]=stitching_df2["x_pos"]-shift_x
    stitching_df2["y_pos"]=stitching_df2["y_pos"]-shift_y
    stitching_df2["x_pos2"]=stitching_df2["y_pos"]
    stitching_df2["y_pos2"]=stitching_df2["x_pos"]
        

    fig,axes=plt.subplots(1,2,figsize=(10,5))
    im=axes[0].scatter(stitching_df2["x_pos2"],
                stitching_df2["y_pos2"],
                c=stitching_df2["row"],
                cmap=plt.cm.Paired)
    fig.colorbar(im,ax=axes[0],label="row")
    im=axes[1].scatter(stitching_df2["x_pos2"],
                stitching_df2["y_pos2"],
                c=stitching_df2["col"],
                cmap=plt.cm.Paired)
    fig.colorbar(im,ax=axes[1],label="col")
    fig.savefig(path.join(stitching_log_directory,
                          "stitching1_row_and_col_position.pdf"))

    fig,ax=plt.subplots(1,1)
    ax.plot(stitching_df2["x_pos2"]-stitching_df2["X"],
            label="x")
    ax.plot(stitching_df2["y_pos2"]-stitching_df2["Y"],
            label="y")
    ax.set_xlabel("series")
    ax.set_ylabel("difference b/w original and assigned position (px)")
    ax.legend()
    fig.savefig(path.join(stitching_log_directory,
                          "stitching2_position_difference.pdf"))

    fig,ax=plt.subplots(1,1)
    ax.plot(stitching_df2["col"],stitching_df2["x_pos2"]-stitching_df2["X"],".",label="x")
    ax.plot(stitching_df2["row"],stitching_df2["y_pos2"]-stitching_df2["Y"],".",label="y")
    ax.set_xlabel("series")
    ax.set_ylabel("deviation per row and col (px)")
    ax.legend()
    fig.savefig(path.join(stitching_log_directory,
                          "stitching3_deviation_per_row_col.pdf"))
 
    for rescale_method in rescale_methods:
        rescaled_image_key=f"rescaled_image_{rescale_method}"
        to_input_zarr_path=lambda row : \
            path.join(rescaled_image_directory,
                      rescaled_image_key,
                      f"S{row['S_index']:03d}_{row['row_col_label']}.zarr")
        stitching_df2["input_zarr_path"]=stitching_df2.apply(to_input_zarr_path,axis=1)

        input_zarr_shape=None
        for input_zarr_path in stitching_df2["input_zarr_path"]:
            input_zarr=zarr.open(input_zarr_path,mode="r")
            if input_zarr_shape is None:
                input_zarr_shape=input_zarr.shape
            else:
                assert np.array_equal(input_zarr_shape,input_zarr.shape)
        sizeT,sizeC,sizeZ,sizeY,sizeX=input_zarr_shape
        if export_only_full_tile:
            T_indices=planes_df[planes_df["tile_full"]]["T_index"].unique()
            sizeT=len(T_indices)
            assert np.array_equal(np.arange(sizeT),np.sort(T_indices))

        stitched_image_size=(
            sizeT,
            sizeC,
            sizeZ,
            stitching_df2["y_pos2"].max()+sizeY,
            stitching_df2["x_pos2"].max()+sizeX
        )

        output_zarr_path=path.join(output_dir,
                      f"stitched_image_{rescale_method}.zarr")
        output_zarr=zarr.open(output_zarr_path,
                              mode="w",
                              shape=stitched_image_size,
                              chunks=(1,1,1,2048,2048),
                              dtype=np.float32)

        def execute_stitching_for_single_plane(args):
            t,grp=args
            stitched_image=np.zeros(stitched_image_size[1:],
                        dtype=np.float16)
            for i, row in tqdm(grp.iterrows(),total=len(grp)):
                input_zarr_path=row["input_zarr_path"]
                input_zarr=zarr.open(input_zarr_path,mode="r")
                x2=int(row["x_pos2"])
                y2=int(row["y_pos2"])
                window=(slice(y2,y2+sizeY),slice(x2,x2+sizeX))
                image=np.array(input_zarr[t,:,:,:,:])
                stitched_image[:,:,window[0],window[1]]=image
            output_zarr[t,:,:,:,:]=stitched_image

        if export_only_full_tile:
            df=stitching_df2[stitching_df2["tile_full"]]
        else:
            df=stitching_df2

        for args in df.groupby("T_index"):
            execute_stitching_for_single_plane(args)

    print(params_dict)
    params_path=path.join(output_dir,"process_stitching_params.yaml")
    with open(params_path,"w") as f:
        yaml.dump(params_dict,f)           

if __name__ == "__main__":
    try:
        process_stitching(path.dirname(snakemake.input["output_dir_created"]),
                          **snakemake.config["e_process_stitching"],
                          )
    except NameError:
        fire.Fire(process_stitching)
