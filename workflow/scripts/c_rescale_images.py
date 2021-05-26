#!/usr/bin/env python3
"""
rescale_images.py
rescale czi images according to the saved backgrounds and export to HDF5

prerequisites:  
- 9_rescaled_background.hdf5 exists for each image in analyzed_dir
"""

import os
from os import path
import itertools
import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import fire
from tqdm import tqdm
from skimage import transform, filters
from skimage.io import imsave, imread
from skimage.morphology import disk
import zarr
from dask import bag as db

import pycziutils

@pycziutils.with_javabridge
def rescale_images(filename,
                   output_dir,
                   *,
                   background_method="median",
                   background_smoothing=True,
                   nonuniform_background_subtract_channels=[],#("Phase",),
                   nonuniform_background_shrink_factor=0.05,
                   nonuniform_background_median_disk_size=5,
                   modes=("divide","subtract")):
    if isinstance(modes,str):
        modes=[modes]
    assert all([m in ["divide","subtract","none"] for m in modes])

    image_props_path=path.join(output_dir,"image_props.yaml")
    bg_directory = path.join(output_dir, "rescaled_background")
    metadata_xml_name = path.join(output_dir, "metadata.xml")
    planes_df_csv_name = path.join(output_dir, "planes_df.csv")
#    print(planes_df_csv_name)
    assert path.isfile(image_props_path)
    assert path.isdir(bg_directory)
    assert path.isfile(metadata_xml_name)
    assert path.isfile(planes_df_csv_name)

    with open(image_props_path,"r") as f:
        image_props=yaml.safe_load(f)
        channel_names=image_props["channel_names"]
        sizeS=image_props["sizeS"]
        sizeT=image_props["sizeT"]
        sizeC=image_props["sizeC"]
        sizeZ=image_props["sizeZ"]
        sizeY=image_props["sizeY"]
        sizeX=image_props["sizeX"]

    image_directory = path.join(output_dir, "rescaled_images")

    log_dir=path.join(output_dir,"rescale_images_log")
    os.makedirs(log_dir,exist_ok=True)

    params_dict = locals()
    del params_dict["f"]

    def savefig(fig, name):
        fig.savefig(path.join(log_dir, name), bbox_inches="tight")
 
    ############## read and process metadata ################
    with open(metadata_xml_name, "r") as f:
        meta = "".join(f.readlines())
    channels=pycziutils.parse_channels(meta)
    px_sizes=[float(s) for s in pycziutils.parse_pixel_size(meta)[::2]]
    print(px_sizes)

    planes_df=pd.read_csv(planes_df_csv_name)
    Xs=np.sort(planes_df["X"].unique())
    Xs_dict=dict(zip(Xs,range(len(Xs))))
    Ys=np.sort(planes_df["Y"].unique())
    Ys_dict=dict(zip(Ys,range(len(Ys))))
    planes_df["X_index"]=planes_df["X"].map(Xs_dict)
    planes_df["Y_index"]=planes_df["Y"].map(Ys_dict)

    df=planes_df[planes_df["T_index"]==0]
    fig,ax=plt.subplots(1,1)
    ax.plot(df["X_index"],df["Y_index"])
    savefig(fig,"11_stage_motion.pdf")

    planes_df["X_pixel"]=planes_df["X"]/px_sizes[0]
    planes_df["Y_pixel"]=planes_df["Y"]/px_sizes[1]

    reader=pycziutils.get_tiled_reader(filename)
#    sizeS,sizeT,sizeC,sizeX,sizeY,sizeZ=pycziutils.summarize_image_size(reader)
    assert all(planes_df["image"].unique()==np.arange(sizeS))
    assert all(planes_df["T_index"].unique()==np.arange(sizeT))
    assert all(planes_df["C_index"].unique()==np.arange(sizeC))
    assert all(planes_df["Z_index"].unique()==np.arange(sizeZ))

    ############## Get corresponding camera dark image ################
    camera_dark_img=np.load(path.join(output_dir,"camera_dark_image.npy"))

    background_key=f"{'smoothed' if background_smoothing else 'rescaled'}_"+\
                   f"{background_method}"
    params_dict["background_key"]=background_key
    backgroundss = {}
    for iC,iZ in itertools.product(range(sizeC),range(sizeZ)):
        bg_image_path=path.join(bg_directory,
            f"{background_key}_C{iC}_{channel_names[iC]}_Z{iZ}.tiff")

        background=imread(bg_image_path)
        while np.any(background<=0):
            trunc_range=lambda l,size : slice(max(l-1,0),min(l+1,size))
            for i,j in zip(*np.where(background<=0)):
                background[i,j]=np.median(
                    background[trunc_range(i,background.shape[0]),
                               trunc_range(j,background.shape[1])
                ])
            if np.all(background<=0):
                raise RuntimeError("background intensity is <= 0")
        backgroundss[(iC,iZ)]=background

    ############## save images into TIFF ################

    planes_df["row_col_label"]=planes_df.apply(lambda row : 
        f'row{int(row["Y_index"])+1:03d}_col{int(row["X_index"])+1:03d}',
        axis=1)
    
    rescaled_image_pathss=[]

    nonuniform_background_subtract_c_indices=[
        [j for j,c in enumerate(channels) if c_name in c["@Fluor"]]
            for c_name in nonuniform_background_subtract_channels]
    assert all([len(js)==1 for js in nonuniform_background_subtract_c_indices])
    nonuniform_background_subtract_c_indices=\
        [js[0] for js in nonuniform_background_subtract_c_indices]

    for mode in modes:
        rescaled_image_paths={}
        image_key=f"rescaled_image_{mode}"
        image_directory2=path.join(image_directory,image_key)
#        os.makedirs(image_directory2,exist_ok=True)

        for s, grp in tqdm(list(planes_df.groupby("image"))):
            rescaled_image_path=path.join(image_directory2,
                    f"S{s:03d}_{grp.iloc[0]['row_col_label']}.zarr",)
            rescaled_image=zarr.open(rescaled_image_path,
                                     mode="w",
                                     shape=(sizeT,sizeC,sizeZ,sizeY,sizeX),
                                     chunks=(1,sizeC,sizeZ,sizeY,sizeX),
                                     dtype=np.float32)
            for _, row in grp.iterrows():
                c,t,z=int(row["C_index"]),int(row["T_index"]),int(row["Z_index"])
                background=backgroundss[(c,z)]
                img=reader.read(c=c,t=t,z=z,series=s,rescale=False)
#                print(img.shape)
#                print(camera_dark_img.shape)
#                print(background.shape)
                if mode=="divide":
                    img=(img-camera_dark_img[c])/background
                elif mode=="subtract":
                    img=img-camera_dark_img[c]-background
                if c in nonuniform_background_subtract_c_indices:
                    img_small=transform.rescale(img,nonuniform_background_shrink_factor,preserve_range=True,)
                    img_small_bg=filters.median(img_small,disk(nonuniform_background_median_disk_size))
                    img_bg=transform.resize(img_small_bg,img.shape,preserve_range=True)
                    img=img-img_bg
                rescaled_image[t,c,z,:,:]=img
            assert not s in rescaled_image_paths.keys()
            rescaled_image_paths[s]=rescaled_image_path
        rescaled_image_pathss.append(rescaled_image_paths)

    planes_df.to_csv(path.join(output_dir,"planes_df2.csv"))

    print(params_dict)
    params_path=path.join(bg_directory,"rescale_images_params.yaml")
    with open(params_path,"w") as f:
        yaml.dump(params_dict,f)           


if __name__ == "__main__":
    try:
        rescale_images(snakemake.input["filename"],
                       path.dirname(snakemake.input["output_dir_created"]),
                       **snakemake.config["c_rescale_images"])
    except NameError as e:
        raise e
        if "snakemake" in str(e):
            raise e
        fire.Fire(rescale_images)
