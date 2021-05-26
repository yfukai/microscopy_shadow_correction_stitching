#!/usr/bin/env python3
"""
rescale_background.py
select and subtract the camera fixed background from the estimated background

"""

import sys
import warnings
import os
from os import path
from glob import glob

import fire
import numpy as np
import pandas as pd
import xmltodict
import h5py
import yaml
import itertools

from matplotlib import pyplot as plt
from skimage import filters, io
io.use_plugin("tifffile")
from skimage.morphology import disk

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import pycziutils 

from utils import wrap_list

match_keys_ops={
    "LUT" : lambda meta : list(pycziutils.parse_camera_LUT(meta)),
    "binning" : pycziutils.parse_binning,
    "bit_depth" : pycziutils.parse_camera_bits
}

def get_camera_props(meta,channel,
                     wf_match_keys=["LUT","binning","bit_depth"],
                     cf_match_keys=[],
                     ):
    meta_dict = xmltodict.parse(meta)
    detectors = wrap_list(meta_dict["OME"]["Instrument"]["Detector"])
    acq_mode=channel["@AcquisitionMode"]
    detector_id=channel["DetectorSettings"]["@ID"]
    detector = [d for d in detectors if d["@ID"] == detector_id]
    assert len(detector) == 1
    detector=detector[0]

    if acq_mode == "WideField":
        prop_dict={k:match_keys_ops[k](meta) for k in wf_match_keys}
        boundary = pycziutils.parse_camera_roi_slice(meta)
    elif acq_mode == "LaserScanningConfocalMicroscopy":
        prop_dict={k:match_keys_ops[k](meta) for k in cf_match_keys}
        boundary = (slice(None),slice(None))
    else:
        raise RuntimeError(f"Acquisition mode {acq_mode} not supported")
    
    if detector["@Model"]!="":
        prop_dict["camera"]=detector["@Model"]
    else:
        prop_dict["camera"]=detector["@ID"]
    return prop_dict,boundary


def rescale_background(output_dir,
                       camera_dark_path,
                       *,
                       match_keys=["LUT","binning","bit_depth"],
                       smoothing="gaussian",
                       sigma=10):

    log_dir=path.join(output_dir,"rescale_background_log")
    os.makedirs(log_dir,exist_ok=True)
    rescaled_bg_directory = path.join(output_dir, "rescaled_background")
    os.makedirs(rescaled_bg_directory,exist_ok=True)

    bg_directory = path.join(output_dir, "averaged_background")
    metadata_xml_name = path.join(output_dir, "metadata.xml")
    planes_df_csv_name = path.join(output_dir, "planes_df.csv")
    image_props_path=path.join(output_dir,"image_props.yaml")
    assert path.isdir(bg_directory)
    assert path.isfile(metadata_xml_name)
    assert path.isfile(planes_df_csv_name)
    assert path.isfile(image_props_path)
    params_dict = locals()
    def savefig(fig, name):
        fig.savefig(path.join(log_dir, name), bbox_inches="tight")

    ############## Load files ################
    with open(metadata_xml_name, "r") as f:
        meta = "".join(f.readlines())
    with open(image_props_path,"r") as f:
        image_props=yaml.safe_load(f)
        channel_names=image_props["channel_names"]
        sizeS=image_props["sizeS"]
        sizeT=image_props["sizeT"]
        sizeC=image_props["sizeC"]
        sizeZ=image_props["sizeZ"]
        sizeY=image_props["sizeY"]
        sizeX=image_props["sizeX"]

    planes_df=pd.read_csv(planes_df_csv_name)

    channels = pycziutils.parse_channels(meta)
    camera_propss=[get_camera_props(meta,channel,wf_match_keys=match_keys) for channel in channels]
    #id_keys=["camera","binning_str","bit_depth","exposure","LUT"]
    params_dict.update({"camera_propss":[p[0] for p in camera_propss]})
    params_dict.update({"boundary" : [[[b.start,b.stop] for b in p[1]] for p in camera_propss]})
    print(params_dict)

    ############## Get corresponding camera dark image ################

    if camera_dark_path:
        if path.isfile(camera_dark_path):
            camera_dark_img=io.imread(camera_dark_path)
        elif path.isdir(camera_dark_path):
            camera_dark_img=[]
            for channel,(camera_props,boundary) in zip(channels,camera_propss):
                candidate_files=[]
                propss=[]
                exposures=[]
                path_pattern=path.join(camera_dark_path,f"mean_*{camera_props['camera']}*.tiff")
                print(path_pattern)
                image_files=glob(path_pattern)
                assert len(image_files) > 0
                for f in image_files:
                    print(f)
                    with open(f.replace(".tiff",".yaml")) as f2:
                        props=yaml.safe_load(f2)
                    if all([np.array_equal(props[k], camera_props[k]) 
                            for k in camera_props.keys()]):
                        candidate_files.append(f)
                        propss.append(props)
                        exposures.append(props["exposure"])
                dark_image_file = candidate_files[np.argmax(exposures)]
                props=propss[np.argmax(exposures)]
                del props["meta"]
                params_dict.update({
                    "camera_dark_image_file": path.abspath(dark_image_file),
                    "camera_dark_image_props": props,
                })
                print(dark_image_file)
                img=io.imread(dark_image_file)
                camera_dark_img.append(img[boundary[1],boundary[0]])
        else :
            raise RuntimeError("corresponding dark image must exist")
       
    else:
        camera_dark_img=[]
        for channel,(camera_props,boundary) in zip(channels,camera_propss):
            def get_size(b,limit):
                x0=b.start if b.start else 0
                x1=b.stop if b.start else limit
                return x1-x0
            camera_dark_img.append(np.zeros((get_size(boundary[1],sizeY),
                                             get_size(boundary[0],sizeX))))
    camera_dark_img=np.array(camera_dark_img)
    
    np.save(path.join(output_dir,"camera_dark_image.npy"),camera_dark_img)

    print(camera_dark_img.shape)
    print(params_dict)

    ############## rescale background and save ##############
    smoothing_ops={
        "gaussian":lambda im : filters.gaussian(im,sigma=sigma),
        "median":lambda im : filters.median(im,disk(sigma)),
    }

    for iC,iZ,img_key in itertools.product(range(sizeC),
                                           range(sizeZ),
                                           ["median","mean"]):
        c_name = channel_names[iC]
        filename=f"{img_key}_C{iC}_{c_name}_Z{iZ}"
        image_path=path.join(bg_directory,filename+".tiff")
        bg_img= io.imread(image_path)
 
        assert np.array_equal(bg_img.shape,camera_dark_img.shape[1:])

        rescaled = bg_img-camera_dark_img[iC]
        rescaled_smoothed = smoothing_ops[smoothing](rescaled)
        ims = []
        imgs = [camera_dark_img[iC], bg_img,
                rescaled, rescaled_smoothed]
        names = ["camera_dark_img", "raw", "rscaled", "smoothed"]

        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
        for j, (img, name) in enumerate(zip(imgs, names)):
            if j > 1:
                io.imsave(path.join(rescaled_bg_directory,
                          f"{name}_{path.basename(image_path)}"),img)
            ims.append(axes[j].imshow(img))
            axes[j].set_title(name)
        for ax, im in zip(axes, ims):
            fig.colorbar(im, ax=ax)
        savefig(fig, f"9_rescaled_background_{path.basename(image_path)}.pdf")

    print(params_dict)
    params_path=path.join(rescaled_bg_directory,
                          "rescale_background_params.yaml")
    with open(params_path,"w") as f:
        yaml.dump(params_dict,f)

if __name__ == "__main__":
    try:
        rescale_background(path.dirname(snakemake.input["output_dir_created"]),
                           snakemake.config["camera_dark_path"],
                           **snakemake.config["b_rescale_background"]
                           )
    except NameError as e:
        if "snakemake" in str(e):
            raise e
        fire.Fire(rescale_background)
