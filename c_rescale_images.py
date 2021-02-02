#!/usr/bin/env python3
"""
rescale_images.py
rescale czi images according to the saved backgrounds and export to HDF5

prerequisites:  
- 9_rescaled_background.hdf5 exists for each image in analyzed_dir
"""

import sys
import os
from os import path

import numpy as np
import h5py
from matplotlib import pyplot as plt
import fire
from tqdm import tqdm
from skimage.io import imsave

script_path = path.dirname(path.abspath(__file__))
cziutils_path = path.abspath(path.join(script_path, "../../"))
sys.path.append(cziutils_path)
import cziutils

@cziutils.with_javabridge
def calculate_background(filename,
                         analyzed_dir=None,
                         image_export_channels=("Phase",)):
    if analyzed_dir is None:
        analyzed_dir = filename[:-4] + "_analyzed"

    bg_h5f_name = path.join(analyzed_dir, "10_rescaled_background.hdf5")
    img_h5f_name = path.join(analyzed_dir, "rescaled_image.hdf5")
    tiff_output_dir_name = path.join(analyzed_dir, "rescaled_images_tiff")
    os.makedirs(tiff_output_dir_name,exist_ok=True)
    metadata_xml_name = path.join(analyzed_dir, "metadata.xml")
    assert path.isfile(bg_h5f_name)
    assert path.isfile(metadata_xml_name)

    params_dict = locals()

    def savefig(fig, name):
        fig.savefig(path.join(analyzed_dir, name), bbox_inches="tight")

 
    ############## read and process metadata ################
    with open(metadata_xml_name, "r") as f:
        meta = "".join(f.readlines())
    channels=cziutils.get_channels(meta)

    planes_df=cziutils.get_planes(meta)
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

    px_sizes=[float(s) for s in cziutils.get_pixel_size(meta)[::2]]
    planes_df["X_pixel"]=planes_df["X"]/px_sizes[0]
    planes_df["Y_pixel"]=planes_df["Y"]/px_sizes[1]

    reader=cziutils.get_tiled_reader(filename)
    sizeS,sizeT,sizeC,sizeX,sizeY,sizeZ=cziutils.summarize_image_size(reader)
    assert all(planes_df["image"].unique()==np.arange(sizeS))
    assert all(planes_df["T_index"].unique()==np.arange(sizeT))
    assert all(planes_df["C_index"].unique()==np.arange(sizeC))
    assert all(planes_df["Z_index"].unique()==np.arange(sizeZ))

    ############## Get corresponding camera dark image ################
    with h5py.File(bg_h5f_name,"r") as bg_h5f:
        camera_dark=np.array(bg_h5f["camera_dark_img"])

        backgroundss=[]
        for j,c in enumerate(channels):
            channel_key=str(j)+"_"+c["@Fluor"]
            background_key=f"smoothed_background_{channel_key}_median_img"
            background=np.array(bg_h5f[background_key])

            while np.any(background<=0):
                trunc_range=lambda l,size : slice(max(l-1,0),min(l+1,size))
                for i,j in zip(*np.where(background<=0)):
                    background[i,j]=np.median(
                        background[trunc_range(i,background.shape[0]),
                                   trunc_range(j,background.shape[1])
                    ])
                if np.all(background<=0):
                    raise RuntimeError("background intensity is <= 0")
            backgroundss.append(background)

    ############## save images into HDF5 ################
    
    with h5py.File(img_h5f_name,"w") as h5f:
        h5f.create_dataset("rescaled_image",shape=(sizeS,sizeT,sizeZ,sizeC,sizeY,sizeX),dtype=np.float32)
        h5f["rescaled_image"].attrs["dimension_order"]="stzcyx"
        h5f["rescaled_image"].attrs["channels"]=np.array([c["@Fluor"] for c in channels],dtype="S")
        for _, row in tqdm(list(planes_df.iterrows())):
            c,t,z,s=int(row["C_index"]),int(row["T_index"]),int(row["Z_index"]),int(row["image"])
            img=reader.read(c=c,t=t,z=z,series=s,rescale=False)
            img=(img-camera_dark)/backgroundss[c]
            h5f["rescaled_image"][s,t,z,c,:,:]=img
        for k, v in params_dict.items():
            print(k,v)
            try:
                h5f.attrs[k] = v
            except TypeError:
                h5f.attrs[k] = np.array(v,dtype="S")

    planes_df.to_hdf(img_h5f_name,"planes_df")

    ############## save images into TIFFs ################

    with h5py.File(img_h5f_name,"r") as h5f:
        for c_name in image_export_channels:
            c_indices=[j for j,c in enumerate(channels) if c_name in c["@Fluor"]]
#            print(c_indices)
            assert len(c_indices)==1; c_index=c_indices[0]#; print(c_index)
            first_img=np.array(h5f["rescaled_image"][0,0,0,c_index,:,:])
            img_min=np.min(first_img)
            img_max=np.max(first_img)
            selected_planes_df=planes_df[planes_df["C_index"]==c_index]
            for (t,z),grp in tqdm(list(selected_planes_df.groupby(["T_index","Z_index"]))):
                configuration_csv_path=path.join(tiff_output_dir_name,
                    f"TileConfiguration_t{t+1:03d}_z{z+1:03d}_c{c_index+1:03d}.txt")
                with open(configuration_csv_path,"w") as fc:
                    fc.write("dim = 2\n")
                    for j, row in grp.iterrows():
                        c,t,z,s=int(row["C_index"]),int(row["T_index"]),int(row["Z_index"]),int(row["image"]) 
                        img=np.array(h5f["rescaled_image"][s,t,z,c,:,:])
                        img=((img-img_min)/(img_max-img_min)*256).astype(np.uint8)
                        imgname=f'rescaled_t{t+1:03d}'\
                                f'_row{int(row["Y_index"])+1:03d}'\
                                f'_col{int(row["X_index"])+1:03d}'\
                                f'_color{int(row["C_index"]):03d}'\
                                ".tiff"
                        imsave(path.join(tiff_output_dir_name,imgname),img,check_contrast=False)
                        fc.write(f'{imgname}; ; ({row["X_pixel"]}, {row["Y_pixel"]})\n')
            
if __name__ == "__main__":
    fire.Fire(calculate_background)
