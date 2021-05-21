#!/usr/bin/env python3
"""
rescale_background.py
select and subtract the camera fixed background from the estimated background
prerequisites:  
- 8_averaged_background.hdf5 exists for each image inoutput_diranalyzed_dir

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

from matplotlib import pyplot as plt
from skimage import filters, io
io.use_plugin("tifffile")
from skimage.morphology import disk

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import pycziutils # pylint: disable=import-error

#script_path=path.dirname(os.path.realpath(__file__))
#camera_dark_directory = path.abspath(
#    path.join(script_path, "../../../camera-dark/analyzed/"))
#microscope_dict = {
#    'HDCamC13440-20CU': "AxioObserver",
#    'Axiocam503m': "LSM800"
#}


def get_camera_props(meta):
    meta_dict = xmltodict.parse(meta)
    camera = meta_dict["OME"]["Instrument"]["Detector"]["@Model"]
    camera_props={
        "camera" : camera,
        "binning" : pycziutils.parse_binning(meta),
        "LUT" : list(pycziutils.parse_camera_LUT(meta)),
        "bit_depth" : pycziutils.parse_camera_bits(meta),
    }
    return camera_props


def rescale_background(output_dir,
                       camera_dark_path,
                       match_keys=["LUT","binning","bit_depth"],
                       smoothing="gaussian",
                       sigma=10,
                       camera_dark_average_method="mean"):
    assert camera_dark_average_method in ["mean","median"]

    log_dir=path.join(output_dir,"rescale_background_log")
    os.makedirs(log_dir,exist_ok=True)
    rescaled_bg_directory = path.join(output_dir, "rescaled_background")
    os.makedirs(rescaled_bg_directory,exist_ok=True)

    bg_directory = path.join(output_dir, "averaged_background")
    metadata_xml_name = path.join(output_dir, "metadata.xml")
    assert path.isdir(bg_directory)
    assert path.isfile(metadata_xml_name)
    params_dict = locals()
    def savefig(fig, name):
        fig.savefig(path.join(log_dir, name), bbox_inches="tight")

    ############## Load files ################
    with open(metadata_xml_name, "r") as f:
        meta = "".join(f.readlines())

    camera_props=get_camera_props(meta)
    #id_keys=["camera","binning_str","bit_depth","exposure","LUT"]
    boundary = pycziutils.parse_camera_roi_slice(meta)
    params_dict.update(camera_props)
    params_dict.update({"boundary" : [[b.start,b.stop] for b in boundary]})
    print(params_dict)

    ############## Get corresponding camera dark image ################

    if camera_dark_path:
        if path.isfile(camera_dark_path):
            camera_dark_img=io.imread(camera_dark_path)
        elif path.isdir(camera_dark_path):
            candidate_files=[]
            propss=[]
            exposures=[]
            path_pattern=path.join(camera_dark_path,f"mean_camera_{camera_props['camera']}*.tiff")
            print(path_pattern)
            image_files=glob(path_pattern)
            assert len(image_files) > 0
            for f in image_files:
                print(f)
                with open(f.replace(".tiff",".yaml")) as f2:
                    props=yaml.safe_load(f2)
                if all([np.array_equal(props[k], camera_props[k]) for k in match_keys]):
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
            camera_dark_img=io.imread(dark_image_file)
        else :
            raise RuntimeError("corresponding dark image must exist")
       
        camera_dark_img=camera_dark_img[boundary[1],boundary[0]]
    else:
        camera_dark_img=np.zeros((boundary[1].stop-boundary[1].start,
                                  boundary[0].stop-boundary[0].start))
    
    io.imsave(path.join(output_dir,"camera_dark_image.tiff"),camera_dark_img)

    print(camera_dark_img.shape)
    print(boundary)
    print(params_dict)

    ############## rescale background and save ##############
    smoothing_ops={
        "gaussian":lambda im : filters.gaussian(im,sigma=sigma),
        "median":lambda im : filters.median(im,disk(sigma)),
    }
 
    for image_path in glob(path.join(bg_directory,"*.tiff")):
        bg_img = io.imread(image_path)
        assert np.array_equal(bg_img.shape,camera_dark_img.shape)

        rescaled = bg_img-camera_dark_img
        rescaled_smoothed = smoothing_ops[smoothing](rescaled)
        ims = []
        imgs = [camera_dark_img, bg_img,
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
                           snakemake.config["camera_dark_path"])
    except NameError:
        fire.Fire(rescale_background)
