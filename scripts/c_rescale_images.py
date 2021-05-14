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
from skimage.io import imsave, imread
import xtiff

import pycziutils

@pycziutils.with_javabridge
def rescale_images(filename,
                         output_dir,
                         background_method="median",
                         background_smoothing=True,
                         image_export_channels=("Phase",),
                         image_export_mode_index=0,
                         modes=("divide","subtract")):
    if isinstance(modes,str):
        modes=[modes]
    assert all([m in ["divide","subtract","none"] for m in modes])
    assert image_export_mode_index < len(modes)

    bg_directory = path.join(output_dir, "rescaled_background")
    metadata_xml_name = path.join(output_dir, "metadata.xml")
    planes_df_csv_name = path.join(output_dir, "planes_df.csv")
    assert path.isdir(bg_directory)
    assert path.isfile(metadata_xml_name)
    assert path.isdir(planes_df_csv_name)

    image_directory = path.join(output_dir, "rescaled_images")
    stitching_tiff_directory = path.join(output_dir, "rescaled_images_for_stitching_tiff")
    os.makedirs(stitching_tiff_directory,exist_ok=True)

    log_dir=path.join(output_dir,"rescale_images_log")
    os.makedirs(log_dir,exist_ok=True)

    params_dict = locals()

    def savefig(fig, name):
        fig.savefig(path.join(log_dir, name), bbox_inches="tight")
 
    ############## read and process metadata ################
    with open(metadata_xml_name, "r") as f:
        meta = "".join(f.readlines())
    channels=pycziutils.parse_channels(meta)
    px_sizes=[float(s) for s in pycziutils.parse_pixel_size(meta)[::2]]

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
    sizeS,sizeT,sizeC,sizeX,sizeY,sizeZ=pycziutils.summarize_image_size(reader)
    assert all(planes_df["image"].unique()==np.arange(sizeS))
    assert all(planes_df["T_index"].unique()==np.arange(sizeT))
    assert all(planes_df["C_index"].unique()==np.arange(sizeC))
    assert all(planes_df["Z_index"].unique()==np.arange(sizeZ))

    ############## Get corresponding camera dark image ################
    camera_dark_img=imread(path.join(output_dir,"camera_dark_image.tiff"))

    background_key=f"{'smoothed' if background_smoothing else 'rescaled'}_"+\
                   f"{background_method}"
    params_dict["background_key"]=background_key
    backgroundss = {}
    for iC,iZ in itertools.product(range(sizeC),range(sizeZ)):
        bg_image_path=path.join(output_dir,"averaged_background",
                           f"{background_key}_C{iC}_{channels[iC]}_Z{iZ}.tiff")

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
    for mode in modes:
        rescaled_image_paths={}
        image_key=f"rescaled_image_{mode}"
        image_directory2=path.join(image_directory,image_key)
        os.makedirs(image_directory2,exist_ok=True)

        for s, grp in tqdm(list(planes_df.groupby("image"))):
            rescaled_image=np.zeros(sizeT,sizeC,sizeZ,sizeY,sizeX,dtype=np.float32)
            for _, row in grp.iterrows():
                c,t,z=int(row["C_index"]),int(row["T_index"]),int(row["Z_index"])
                background=backgroundss[(c,z)]
                img=reader.read(c=c,t=t,z=z,series=s,rescale=False)
                if mode=="divide":
                    rescaled_image[t,c,z,:,:]=(img-camera_dark_img)/background
                elif mode=="subtract":
                    rescaled_image[t,c,z,:,:]=img-camera_dark_img-background
            rescaled_image_path=path.join(image_directory2,
                                   f"S{s:03d}_{row['row_col_label']}.ome.tiff",)
            xtiff.to_tiff(rescaled_image,
                          rescaled_image_path, 
                          channel_names=channels,
                          profile=xtiff.TiffProfile.OME_TIFF)
            assert not s in rescaled_image_paths.keys()
            rescaled_image_paths[s]=rescaled_image_path
        rescaled_image_pathss.append(rescaled_image_paths)

    planes_df.to_csv(output_dir,"planes_df2.csv")

    ############## save images into TIFFs ################

    for c_name in image_export_channels:
        c_indices=[j for j,c in enumerate(channels) if c_name in c["@Fluor"]]
        assert len(c_indices)==1
        c_index=c_indices[0]
        rescaled_image_path=rescaled_image_pathss[image_export_mode_index][0]
        first_img=imread(rescaled_image_path)[0,c_index,0,:,:]
        img_min=np.min(first_img)
        img_max=np.max(first_img)
        selected_planes_df=planes_df[planes_df["C_index"]==c_index]

        for (t,z),grp in tqdm(list(selected_planes_df.groupby(["T_index","Z_index"]))):
            configuration_csv_path=path.join(stitching_tiff_directory,
                f"TileConfiguration_t{t+1:03d}_z{z+1:03d}_c{c_index+1:03d}.txt")
            with open(configuration_csv_path,"w") as fc:
                fc.write("dim = 2\n")
                for j, row in grp.iterrows():
                    c,t,z,s=int(row["C_index"]),int(row["T_index"]),int(row["Z_index"]),int(row["image"]) 
                    rescaled_image_path=rescaled_image_pathss[image_export_mode_index][s]
                    img=imread(rescaled_image_path)[t,c,z,:,:]
                    img=((img-img_min)/(img_max-img_min)*256).astype(np.uint8)
                    imgname=f'rescaled_t{t+1:03d}'\
                            f'_row{int(row["Y_index"])+1:03d}'\
                            f'_col{int(row["X_index"])+1:03d}'\
                            f'_color{int(row["C_index"]):03d}'\
                            ".tiff"
                    imsave(path.join(stitching_tiff_directory,imgname),
                           img,check_contrast=False)
                    fc.write(f'{imgname}; ; ({row["X_pixel"]}, {row["Y_pixel"]})\n')

    print(params_dict)
    params_path=path.join(bg_directory,"rescale_images_params.yaml")
    with open(params_path,"w") as f:
        yaml.dump(params_dict,f)           

if __name__ == "__main__":
    try:
        rescale_images(snakemake.input["filename"],
                       path.dirname(snakemake.input["output_dir_created"]))
    except NameError:
        fire.Fire(rescale_images)